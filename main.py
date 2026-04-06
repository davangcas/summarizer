import base64
import os
import pathlib
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypeVar

import pymupdf
from markdown_pdf import MarkdownPdf, Section
from openai import OpenAI
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer

TModel = TypeVar("TModel", bound=BaseModel)

files_directory = pathlib.Path(r"D:\Ingenieria Mecatronica").resolve()
summarized_texts = pathlib.Path(__file__).resolve().parent / "summarized_texts"
summarized_texts.mkdir(parents=True, exist_ok=True)
completed_texts = pathlib.Path(__file__).resolve().parent / "completed_texts"
completed_texts.mkdir(parents=True, exist_ok=True)
summary_pdfs = pathlib.Path(__file__).resolve().parent / "summary_pdfs"
summary_pdfs.mkdir(parents=True, exist_ok=True)

completion_model = "mistralai/ministral-3-3b"
# Token counting for chunking. Prefer a Gemma tokenizer if you have HF access (see GEMMA_TOKENIZER_ID).
# Default chain ends with gpt2 (public, no Hugging Face login) so the pipeline works offline.
_TOKENIZER_FALLBACKS = (
    "google/gemma-3-12b-it",
    "google/gemma-3-270m",
    "mistralai/ministral-3-3b",
    "gpt2",
)
MAX_CONTEXT_TOKENS = 150000
OUTPUT_TOKEN_RESERVE = 10000
# Input budget per API call (prompt + document chunk); output reserved separately by the server.
MAX_INPUT_TOKENS = MAX_CONTEXT_TOKENS - OUTPUT_TOKEN_RESERVE


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


# Paralelismo (ajustar si LM Studio/GPU se satura: variables SUMMARIZER_*).
# Se usa ThreadPoolExecutor; AsyncOpenAI no es necesario salvo que midas cuello de botella distinto.
MAX_PARALLEL_PDFS = _env_int("SUMMARIZER_MAX_PARALLEL_PDFS", 4)
MAX_PARALLEL_SUMMARIES = _env_int("SUMMARIZER_MAX_PARALLEL_SUMMARIES", 4)
MAX_PARALLEL_CHUNKS = _env_int("SUMMARIZER_MAX_PARALLEL_CHUNKS", 4)
MAX_PARALLEL_OCR_PAGES = _env_int("SUMMARIZER_MAX_PARALLEL_OCR_PAGES", 4)

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

_tokenizer: AutoTokenizer | None = None
_tokenizer_lock = threading.Lock()


def get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    with _tokenizer_lock:
        if _tokenizer is None:
            env_id = os.environ.get("GEMMA_TOKENIZER_ID", "").strip()
            candidates: tuple[str, ...] = (env_id,) if env_id else _TOKENIZER_FALLBACKS
            last_err: Exception | None = None
            for model_id in candidates:
                try:
                    _tokenizer = AutoTokenizer.from_pretrained(model_id)
                    if model_id == "gpt2" and not env_id:
                        print(
                            "Tokenizer: using gpt2 for token counts (public). "
                            "Set GEMMA_TOKENIZER_ID to a Gemma repo if you use huggingface-cli login."
                        )
                    break
                except Exception as ex:
                    last_err = ex
            if _tokenizer is None:
                assert last_err is not None
                raise last_err
        return _tokenizer


def count_tokens(text: str) -> int:
    tok = get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


class OCRPageOutput(BaseModel):
    """Salida estricta para extracción de una página escaneada."""

    model_config = ConfigDict(extra="forbid")

    markdown_text: str = Field(
        description=(
            "Transcripción completa del texto visible en la imagen, en Markdown. "
            "Orden de lectura natural (columnas y bloques como en la página). "
            "Usa #/##/### si hay títulos claros, listas con - o 1., tablas en Markdown si se distinguen. "
            "Conserva fórmulas y símbolos lo más fielmente posible (LaTeX entre $ si aplica). "
            "Mantén el idioma original. Si algo es ilegible, marca [ilegible]. "
            "Sin introducción, sin conclusiones, sin 'aquí está el texto': solo el contenido de la página."
        )
    )


class CornellTopicBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        description="Nombre breve y descriptivo del tema (sin prefijos meta tipo 'Tema:')."
    )
    cues: list[str] = Field(
        description=(
            "Lista de pistas tipo Cornell: palabras clave o preguntas cortas de repaso (una idea por elemento). "
            "Evita frases largas; 3 a 10 ítems según densidad del tema."
        )
    )
    notes: str = Field(
        description=(
            "Síntesis de ideas, definiciones, relaciones y datos del tema en prosa clara. "
            "No pegues párrafos literales del origen. "
            "Si este bloque proviene de un fragmento y el tema continúa después, indica al final: "
            "(continúa en el siguiente fragmento)."
        )
    )
    topic_summary: str = Field(
        description="Cierre del tema: 2 a 5 frases que integren la idea central y utilidad o aplicación."
    )


class CornellSummaryStructured(BaseModel):
    """Resumen por temas estilo Cornell; coincide con el esquema JSON enviado al modelo."""

    model_config = ConfigDict(extra="forbid")

    topics: list[CornellTopicBlock] = Field(
        description=(
            "Temas coherentes del material: cada uno agrupa contenido relacionado. "
            "Prioriza la estructura lógica del autor (capítulos, unidades) cuando exista. "
            "Si el texto es muy breve, un solo tema puede bastar. "
            "Todo el contenido en español claro salvo términos técnicos habituales en el original."
        )
    )


OCR_PROMPT = """Rol: transcriptor OCR de documentos académicos.
Tarea: rellena únicamente el campo del esquema con el texto de la imagen.

Reglas:
- Transcribe todo lo legible; conserva jerarquía (títulos, listas, enumeraciones) en Markdown.
- No añadas explicaciones, saludos, comentarios sobre la imagen ni resúmenes.
- No inventes texto donde no haya; usa [ilegible] en huecos.
- Idioma: el mismo que aparece en la página."""

SUMMARY_CORNELL_USER_PREFIX = """Rol: tutor de estudio y síntesis para textos académicos en español.
Salida: cumple EXACTAMENTE el esquema JSON indicado (solo claves permitidas; lista `topics` con objetos title, cues, notes, topic_summary).

Instrucciones:
1. Tras el separador --- está el documento fuente. Identifica temas principales y agrupa ideas afines; evita temas duplicados o solapados.
2. Por tema: título informativo; cues como repaso (preguntas o keywords); notes como síntesis propia (no copiar párrafos extensos); topic_summary para cerrar el tema.
3. Prioriza hechos y definiciones del texto; no inventes datos, citas o referencias que no aparezcan implícitamente en el material.
4. Si el documento mezcla idiomas, sintetiza en español salvo nombres propios o términos técnicos estándar.
5. No incluyas texto fuera del JSON (sin markdown envolvente, sin comentarios)."""

SUMMARY_CHUNK_WRAPPER = """Contexto: este bloque es el fragmento {part} de {total} de un documento largo (no tienes el resto).

Qué hacer:
- Extrae solo los temas que se apoyen en el contenido de ESTE fragmento.
- Si un tema empieza aquí y seguramente sigue después, en `notes` indica al final: (continúa en el siguiente fragmento).
- No inventes contenido de otras partes del documento.

---
{body}"""

UNIFY_SUMMARIES_PROMPT = """Rol: editor de apuntes. Recibes varios resúmenes parciales del MISMO documento (Markdown) tras ---.

Objetivo: producir un único JSON del esquema con una lista `topics` coherente para todo el documento.

Reglas:
1. Fusiona temas duplicados o muy similares; unifica cues y notes sin repetir ideas.
2. Ordena los temas en secuencia lógica (orden del libro o del razonamiento, no orden de fragmentos).
3. Mantén el estilo Cornell (title, cues, notes, topic_summary) en cada tema.
4. Elimina contradicciones; prioriza consistencia.
5. Salida: solo el JSON del esquema, en español.

---
{combined}"""


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def completion_parsed_or_validate(
    completion: ParsedChatCompletion[Any],
    model: type[TModel],
) -> TModel:
    msg = completion.choices[0].message
    if msg.parsed is not None:
        return msg.parsed
    content = (msg.content or "").strip()
    if content:
        return model.model_validate_json(_strip_json_fence(content))
    raise ValueError("El modelo no devolvió contenido parseable")


def format_cornell_markdown(
    summary: CornellSummaryStructured, *, document_title: bool = True
) -> str:
    if not summary.topics:
        return ""
    blocks: list[str] = []
    for t in summary.topics:
        cues_md = "\n".join(f"- {c}" for c in t.cues) if t.cues else "-"
        blocks.append(
            f"## {t.title}\n\n### Pistas (cue)\n{cues_md}\n\n### Notas\n{t.notes}\n\n### Resumen del tema\n{t.topic_summary}"
        )
    body = "\n\n".join(blocks)
    # markdown_pdf → PyMuPDF: la TOC exige que el primer encabezado del documento sea nivel 1 (#).
    if document_title:
        return f"# Resumen\n\n{body}"
    return body


def ensure_markdown_h1_for_pdf(markdown: str) -> str:
    """PyMuPDF falla con set_toc si el primer heading no es # (ver 'hierarchy level of item 0 must be 1')."""
    text = markdown.strip()
    if not text:
        return "# Resumen\n"
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") and not s.startswith("##"):
            return markdown
        break
    return f"# Resumen\n\n{text}"


def _split_oversized_piece(piece: str, max_tokens: int) -> list[str]:
    tok = get_tokenizer()
    if len(tok.encode(piece, add_special_tokens=False)) <= max_tokens:
        return [piece]
    words = piece.split()
    if len(words) == 1:
        step = max(256, max_tokens * 3)
        return [piece[i : i + step] for i in range(0, len(piece), step)]
    lines = piece.split("\n")
    if len(lines) > 1:
        out: list[str] = []
        for line in lines:
            out.extend(_split_oversized_piece(line, max_tokens))
        return out
    words = piece.split()
    if not words:
        return [piece[: max_tokens * 4]]
    chunks: list[str] = []
    current: list[str] = []
    for w in words:
        trial = (" ".join(current + [w])).strip()
        if len(tok.encode(trial, add_special_tokens=False)) <= max_tokens:
            current.append(w)
        else:
            if current:
                chunks.append(" ".join(current))
            current = [w]
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_text_by_tokens(body: str, max_content_tokens: int) -> list[str]:
    """Split body so each piece encodes to at most max_content_tokens (approximate via tokenizer)."""
    body = body.strip()
    if not body:
        return []
    tok = get_tokenizer()
    if len(tok.encode(body, add_special_tokens=False)) <= max_content_tokens:
        return [body]
    parts: list[str] = []
    paragraphs = re.split(r"\n\s*\n", body)
    current = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        trial = (current + "\n\n" + p).strip() if current else p
        if len(tok.encode(trial, add_special_tokens=False)) <= max_content_tokens:
            current = trial
        else:
            if current:
                parts.append(current)
                current = ""
            if len(tok.encode(p, add_special_tokens=False)) <= max_content_tokens:
                current = p
            else:
                parts.extend(_split_oversized_piece(p, max_content_tokens))
    if current:
        parts.append(current)
    return parts


def _chat_cornell_structured(user_content: str) -> CornellSummaryStructured:
    completion = client.chat.completions.parse(
        model=completion_model,
        messages=[{"role": "user", "content": user_content}],
        response_format=CornellSummaryStructured,
    )
    return completion_parsed_or_validate(completion, CornellSummaryStructured)


def summarize_cornell_single(full_text: str) -> str:
    user_content = f"{SUMMARY_CORNELL_USER_PREFIX}\n\n---\n\n{full_text}"
    return format_cornell_markdown(_chat_cornell_structured(user_content))


def _summarize_one_chunk(part: int, total: int, body: str) -> tuple[int, str]:
    wrapped = SUMMARY_CHUNK_WRAPPER.format(part=part, total=total, body=body)
    user_content = f"{SUMMARY_CORNELL_USER_PREFIX}\n\n{wrapped}"
    md = format_cornell_markdown(
        _chat_cornell_structured(user_content), document_title=False
    )
    return part, md


def summarize_cornell_chunked(full_text: str, max_chunk_content_tokens: int) -> str:
    chunks = chunk_text_by_tokens(full_text, max_chunk_content_tokens)
    total = len(chunks)
    if total == 0:
        return ""
    workers = min(MAX_PARALLEL_CHUNKS, total)
    partial_by_part: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_summarize_one_chunk, i, total, ch)
            for i, ch in enumerate(chunks, start=1)
        ]
        for fut in as_completed(futures):
            part, md = fut.result()
            partial_by_part[part] = md
    partials = [partial_by_part[i] for i in range(1, total + 1)]
    combined_md = "# Resumen (fragmentos)\n\n" + "\n\n".join(
        f"## Fragmento {i} de {total}\n\n{md}" for i, md in enumerate(partials, start=1)
    )
    unify_user = UNIFY_SUMMARIES_PROMPT.format(combined=combined_md)
    if count_tokens(unify_user) <= MAX_INPUT_TOKENS:
        try:
            return format_cornell_markdown(_chat_cornell_structured(unify_user))
        except Exception:
            return combined_md
    return combined_md


def summarize_document(full_text: str) -> str:
    single_user = f"{SUMMARY_CORNELL_USER_PREFIX}\n\n---\n\n{full_text}"
    prompt_tokens = count_tokens(single_user)
    if prompt_tokens <= MAX_INPUT_TOKENS:
        return summarize_cornell_single(full_text)

    wrapper_empty = SUMMARY_CHUNK_WRAPPER.format(part=1, total=1, body="")
    overhead = count_tokens(wrapper_empty)
    max_chunk_content = max(512, MAX_INPUT_TOKENS - overhead - 200)
    return summarize_cornell_chunked(full_text, max_chunk_content)


def completed_md_path_for_pdf(src: pathlib.Path) -> pathlib.Path:
    rel = src.relative_to(files_directory)
    return completed_texts / rel.with_suffix(".md")


def _nonempty_utf8_file(path: pathlib.Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip() != ""
    except OSError:
        return False


def _nonempty_pdf_file(path: pathlib.Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def write_completed_text(src: pathlib.Path, file_text: str) -> None:
    out_path = completed_md_path_for_pdf(src)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(file_text, encoding="utf-8")


def write_summary_markdown(md_source_rel: pathlib.Path, summary_md: str) -> None:
    out_md = summarized_texts / md_source_rel
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(summary_md, encoding="utf-8")


def write_summary_pdf(md_source_rel: pathlib.Path, summary_md: str) -> None:
    out_pdf = summary_pdfs / md_source_rel.with_suffix(".pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf = MarkdownPdf()
    pdf.add_section(Section(text=ensure_markdown_h1_for_pdf(summary_md)))
    pdf.save(out_pdf)


def _extract_single_pdf(src: pathlib.Path) -> None:
    """Un PDF por tarea (hilo): abre su propio documento; no compartir fitz entre hilos."""
    try:
        print(src)
        file_text = extract_text_get_text_only(src)
        if file_text.strip():
            write_completed_text(src, file_text)
        # else:
        #     file_text = pdf_pages_to_vision_text(src)
        #     if file_text.strip():
        #         write_completed_text(src, file_text)
    except Exception as ex:
        print(f"Error processing {src}: {ex}")


def run_pdf_extraction() -> None:
    """Idempotent: skips PDFs that already have a non-empty completed_texts .md."""
    pending: list[pathlib.Path] = []
    for root, _, files in os.walk(files_directory):
        for file in files:
            if not file.endswith(".pdf"):
                continue
            src = pathlib.Path(root) / file
            if _nonempty_utf8_file(completed_md_path_for_pdf(src)):
                print(f"Skip extract (already in completed_texts): {src}")
                continue
            pending.append(src)
    if not pending:
        return
    workers = min(MAX_PARALLEL_PDFS, len(pending))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pool.map(_extract_single_pdf, pending)


def _summarize_single_md(md_path: pathlib.Path) -> None:
    try:
        rel = md_path.relative_to(completed_texts)
        out_summary_md = summarized_texts / rel
        out_summary_pdf = summary_pdfs / rel.with_suffix(".pdf")

        if _nonempty_utf8_file(out_summary_md) and _nonempty_pdf_file(out_summary_pdf):
            print(f"Skip summarize (summary + PDF done): {rel}")
            return

        if _nonempty_utf8_file(out_summary_md) and not _nonempty_pdf_file(
            out_summary_pdf
        ):
            print(f"Resume PDF from summary: {rel}")
            write_summary_pdf(rel, out_summary_md.read_text(encoding="utf-8"))
            return

        print(f"Summarizing: {md_path}")
        full_text = md_path.read_text(encoding="utf-8")
        if not full_text.strip():
            return
        summary_md = summarize_document(full_text)
        if summary_md.strip():
            write_summary_markdown(rel, summary_md)
            write_summary_pdf(rel, summary_md)
    except Exception as ex:
        print(f"Error summarizing {md_path}: {ex}")


def run_summarization_pipeline() -> None:
    """Idempotent: skips when summary .md and PDF exist; rebuilds PDF only if .md exists but PDF missing."""
    paths = sorted(completed_texts.rglob("*.md"))
    if not paths:
        return
    workers = min(MAX_PARALLEL_SUMMARIES, len(paths))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pool.map(_summarize_single_md, paths)


def extract_text_get_text_only(pdf_path: pathlib.Path) -> str:
    parts: list[str] = []
    with pymupdf.open(pdf_path) as pdf:
        for page in pdf:
            text = page.get_text()
            if text.strip():
                parts.append(text)
    return "\n\n".join(parts)


def _ocr_single_page(pdf_path: pathlib.Path, page_index: int) -> str:
    """Abre el PDF en el hilo actual; una página por llamada (seguro en paralelo)."""
    mat = pymupdf.Matrix(2, 2)
    with pymupdf.open(pdf_path) as pdf:
        page = pdf[page_index]
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        b64 = base64.standard_b64encode(png_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        completion = client.chat.completions.parse(
            model=completion_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
            response_format=OCRPageOutput,
        )
        return completion_parsed_or_validate(completion, OCRPageOutput).markdown_text


def pdf_pages_to_vision_text(pdf_path: pathlib.Path) -> str:
    with pymupdf.open(pdf_path) as pdf:
        n = len(pdf)
    if n == 0:
        return ""
    workers = min(MAX_PARALLEL_OCR_PAGES, n)
    by_index: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_ocr_single_page, pdf_path, i): i for i in range(n)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                by_index[idx] = fut.result()
            except Exception as ex:
                print(f"OCR page {idx} of {pdf_path}: {ex}")
                by_index[idx] = ""
    ordered = [by_index[i] for i in range(n)]
    return "\n\n".join(p for p in ordered if p.strip())


if __name__ == "__main__":
    get_tokenizer()
    run_pdf_extraction()
    run_summarization_pipeline()
