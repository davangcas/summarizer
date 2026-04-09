import base64
import hashlib
import json
import os
import pathlib
import re
import signal
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypeVar

import httpx
import pymupdf
from markdown_pdf import MarkdownPdf, Section
from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer

TModel = TypeVar("TModel", bound=BaseModel)

# Carpeta raíz con los PDF originales: se define en configure_source_directory() al ejecutar el script,
# o con la variable de entorno SUMMARIZER_FILES_DIRECTORY (sin diálogo).
files_directory: pathlib.Path | None = None
# PDFs sin capa de texto: si True, se usa el modelo de visión por página; si False, se omiten.
use_vision_for_scanned_pdfs: bool = False
summarized_texts = pathlib.Path(__file__).resolve().parent / "summarized_texts"
summarized_texts.mkdir(parents=True, exist_ok=True)
completed_texts = pathlib.Path(__file__).resolve().parent / "completed_texts"
completed_texts.mkdir(parents=True, exist_ok=True)
summary_pdfs = pathlib.Path(__file__).resolve().parent / "summary_pdfs"
summary_pdfs.mkdir(parents=True, exist_ok=True)
# Sub-resúmenes por ventana de páginas (reanudables); una carpeta por documento .md en completed_texts.
summary_partials = pathlib.Path(__file__).resolve().parent / "summary_partials"
summary_partials.mkdir(parents=True, exist_ok=True)

# LM Studio: http://localhost:1234 — API de listado: GET /api/v1/models
# (https://lmstudio.ai/docs/developer/rest/list). Se elige LLM cargado con visión y mayor contexto.
LM_STUDIO_HOST = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234").rstrip("/")

# Identificador del modelo para chat (se asigna en configure_lm_studio_model).
completion_model: str = ""
# Límite de contexto del modelo elegido (tokens); se actualiza al detectar el modelo.
MAX_CONTEXT_TOKENS: int = 0
# Input budget por llamada (prompt + documento); salida reservada aparte en el servidor.
MAX_INPUT_TOKENS: int = 0
PROMPT_CONTEXT_RATIO_TARGET = float(
    os.environ.get("SUMMARIZER_PROMPT_CONTEXT_RATIO", "0.40")
)
PROMPT_CONTEXT_RATIO_START = float(
    os.environ.get("SUMMARIZER_PROMPT_CONTEXT_RATIO_START", "0.30")
)
PROMPT_CONTEXT_RATIO_MIN = float(
    os.environ.get("SUMMARIZER_PROMPT_CONTEXT_RATIO_MIN", "0.10")
)
PROMPT_CONTEXT_RATIO_STEP = float(
    os.environ.get("SUMMARIZER_PROMPT_CONTEXT_RATIO_STEP", "0.05")
)

# Token counting for chunking. Prefer a Gemma tokenizer if you have HF access (see GEMMA_TOKENIZER_ID).
# Default chain ends with gpt2 (public, no Hugging Face login) so the pipeline works offline.
_TOKENIZER_FALLBACKS = (
    "google/gemma-3-12b-it",
    "google/gemma-3-270m",
    "mistralai/ministral-3-3b",
    "gpt2",
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _env_optional_timeout_seconds(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"none", "null", "off", "infinite", "inf", "0", "-1"}:
        return None
    try:
        value = float(raw)
        return None if value <= 0 else value
    except ValueError:
        return default


# Paralelismo (ajustar si LM Studio/GPU se satura: variables SUMMARIZER_*).
# Se usa ThreadPoolExecutor; AsyncOpenAI no es necesario salvo que midas cuello de botella distinto.
MAX_PARALLEL_PDFS = _env_int("SUMMARIZER_MAX_PARALLEL_PDFS", 4)
MAX_PARALLEL_SUMMARIES = _env_int("SUMMARIZER_MAX_PARALLEL_SUMMARIES", 4)
MAX_PARALLEL_CHUNKS = _env_int("SUMMARIZER_MAX_PARALLEL_CHUNKS", 4)
# OCR por imagen: varias páginas en paralelo dentro del mismo PDF (LM Studio suele serializar; subir con precaución).
MAX_PARALLEL_OCR_PAGES = _env_int("SUMMARIZER_MAX_PARALLEL_OCR_PAGES", 4)
MAX_PARALLEL_WINDOW_SUMMARIES = _env_int("SUMMARIZER_MAX_PARALLEL_WINDOW_SUMMARIES", 2)
SUMMARY_PAGE_OVERLAP = _env_int("SUMMARIZER_SUMMARY_PAGE_OVERLAP", 1)
ASSEMBLE_DEDUP_BORDER = os.environ.get(
    "SUMMARIZER_ASSEMBLE_DEDUP_BORDER", ""
).strip().lower() in ("1", "true", "yes", "sí", "si", "on")

REQUEST_TIMEOUT_SECONDS = _env_optional_timeout_seconds(
    "SUMMARIZER_REQUEST_TIMEOUT_SECONDS", None
)
REQUEST_RETRIES = _env_int("SUMMARIZER_REQUEST_RETRIES", 4)
REQUEST_RETRY_BACKOFF_SECONDS = float(
    os.environ.get("SUMMARIZER_REQUEST_RETRY_BACKOFF_SECONDS", "2.0")
)

MAX_PARALLEL_ONEDRIVE_UPLOADS = max(
    1, _env_int("SUMMARIZER_MAX_PARALLEL_ONEDRIVE_UPLOADS", 3)
)
_ONEDRIVE_UPLOAD_SEM = threading.BoundedSemaphore(MAX_PARALLEL_ONEDRIVE_UPLOADS)

client = OpenAI(
    base_url=f"{LM_STUDIO_HOST}/v1",
    api_key=os.environ.get("LM_API_TOKEN", os.environ.get("LM_API_KEY", "lm-studio")),
)

_stop_event = threading.Event()


class StopRequested(Exception):
    """Parada cooperativa solicitada por usuario."""


def _request_stop(reason: str) -> None:
    if _stop_event.is_set():
        return
    _stop_event.set()
    print(f"\n[STOP] {reason}")


def _check_stop_requested() -> None:
    if _stop_event.is_set():
        raise StopRequested("Proceso detenido por solicitud del usuario.")


def _sleep_with_stop(total_seconds: float) -> None:
    deadline = time.monotonic() + max(0.0, total_seconds)
    while True:
        _check_stop_requested()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.25, remaining))


def _install_stop_handlers() -> None:
    def _on_sigint(_signum: int, _frame: Any) -> None:
        _request_stop("SIGINT recibido (Ctrl+C).")

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except Exception:
        pass


def _stop_listener_tty_unix() -> None:
    """Linux / macOS: tecla sin Enter (cbreak + select)."""
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except (OSError, AttributeError, termios.error):
        _stop_listener_line()
        return

    def _char_stops(ch: str) -> bool:
        return len(ch) == 1 and ch.lower() in ("x", "q", "s")

    try:
        tty.setcbreak(fd)
        while not _stop_event.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not readable:
                continue
            ch = sys.stdin.read(1)
            if not ch:
                return
            if _char_stops(ch):
                _request_stop("Tecla rápida (x, q o s).")
                return
    except (EOFError, OSError, ValueError):
        return
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except (OSError, termios.error):
            pass


def _stop_listener_tty_windows() -> None:
    """Windows (consola): tecla sin Enter vía msvcrt."""
    import msvcrt

    while not _stop_event.is_set():
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch.lower() in (b"x", b"q", b"s") or ch == b"\x03":
                _request_stop("Tecla rápida (x, q o s) o Ctrl+C (consola).")
                return
        time.sleep(0.05)


def _stop_listener_line() -> None:
    """Sin TTY interactivo: una línea + Enter (p. ej. redirección de stdin)."""
    print("Control (modo línea): escriba 'stop' (o Enter) y pulse Enter para detener.")
    while not _stop_event.is_set():
        try:
            cmd = input().strip().lower()
        except EOFError:
            return
        except Exception:
            return
        if cmd in ("", "stop", "salir", "exit", "quit"):
            _request_stop("Parada solicitada desde consola.")
            return
        print("Comando no reconocido. Use 'stop' (o Enter) para detener.")


def _start_stop_listener() -> None:
    """
    Hilo daemon para detener el pipeline: tecla rápida si hay consola interactiva.

    - Windows / Linux / macOS (stdin TTY): pulse **x**, **q** o **s** sin Enter.
    - También **Ctrl+C** (SIGINT) y Ctrl+C en consola Windows (byte 0x03) donde aplique.
    - Sin TTY: igual que antes, comando por línea + Enter.
    """
    print(
        "Control activo: pulse x, q o s (sin Enter) para detener; "
        "o Ctrl+C. Sin consola interactiva, use 'stop' + Enter."
    )

    def _runner() -> None:
        if sys.stdin.isatty():
            if os.name == "nt" or sys.platform == "win32":
                _stop_listener_tty_windows()
            else:
                _stop_listener_tty_unix()
        else:
            _stop_listener_line()

    threading.Thread(target=_runner, daemon=True, name="stop-listener").start()


def _effective_context_tokens(model_entry: dict[str, Any]) -> int:
    """Contexto real de instancias cargadas (fallback: max_context_length)."""
    best = 0
    for inst in model_entry.get("loaded_instances") or []:
        cfg = inst.get("config") or {}
        cl = int(cfg.get("context_length") or 0)
        best = max(best, cl)
    if best <= 0:
        best = int(model_entry.get("max_context_length") or 0)
    return best


def configure_lm_studio_model() -> None:
    """
    Consulta GET {LM_STUDIO_HOST}/api/v1/models y elige un LLM cargado con vision=True
    y el mayor contexto cargado. Ajusta MAX_CONTEXT_TOKENS y MAX_INPUT_TOKENS.

    Override: SUMMARIZER_COMPLETION_MODEL + SUMMARIZER_MAX_CONTEXT_TOKENS (sin llamar a la API).
    """
    global completion_model, MAX_CONTEXT_TOKENS, MAX_INPUT_TOKENS
    manual_model = os.environ.get("SUMMARIZER_COMPLETION_MODEL", "").strip()
    manual_ctx = os.environ.get("SUMMARIZER_MAX_CONTEXT_TOKENS", "").strip()
    if manual_model and manual_ctx:
        completion_model = manual_model
        MAX_CONTEXT_TOKENS = max(1024, int(manual_ctx))
        MAX_INPUT_TOKENS = max(
            512, int(MAX_CONTEXT_TOKENS * PROMPT_CONTEXT_RATIO_TARGET)
        )
        print(
            f"Modelo (override entorno): {completion_model}, contexto {MAX_CONTEXT_TOKENS} tokens"
        )
        return

    url = f"{LM_STUDIO_HOST}/api/v1/models"
    headers: dict[str, str] = {}
    token = (
        os.environ.get("LM_API_TOKEN", "").strip()
        or os.environ.get("LM_API_KEY", "").strip()
    )
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = httpx.get(url, headers=headers, timeout=60.0)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise SystemExit(
            f"No se pudo listar modelos en LM Studio ({url}). ¿Está el servidor activo? {e}"
        ) from e

    payload = resp.json()
    models = payload.get("models") or []
    best: tuple[str, int] | None = None

    for m in models:
        if m.get("type") != "llm":
            continue
        caps = m.get("capabilities") or {}
        if not caps.get("vision"):
            continue
        if not (m.get("loaded_instances") or []):
            continue
        key = str(m.get("key") or "").strip()
        if not key:
            continue
        ctx = _effective_context_tokens(m)
        if ctx <= 0:
            ctx = 8192
        if best is None or ctx > best[1]:
            best = (key, ctx)

    if best is None:
        raise SystemExit(
            "No hay ningún LLM cargado en LM Studio con capacidad de visión (capabilities.vision). "
            "Cargue uno en la app o defina SUMMARIZER_COMPLETION_MODEL y "
            "SUMMARIZER_MAX_CONTEXT_TOKENS."
        )

    completion_model, MAX_CONTEXT_TOKENS = best
    MAX_INPUT_TOKENS = max(512, int(MAX_CONTEXT_TOKENS * PROMPT_CONTEXT_RATIO_TARGET))
    print(
        f"Modelo LM Studio (visión, mayor contexto cargado): {completion_model} — "
        f"contexto {MAX_CONTEXT_TOKENS} tokens (límite prompt objetivo "
        f"{int(PROMPT_CONTEXT_RATIO_TARGET * 100)}% ~{MAX_INPUT_TOKENS})"
    )


_tokenizer: AutoTokenizer | None = None
_tokenizer_lock = threading.Lock()
_prompt_ratio_lock = threading.Lock()
_adaptive_prompt_context_ratio: float = max(
    PROMPT_CONTEXT_RATIO_MIN,
    min(PROMPT_CONTEXT_RATIO_TARGET, PROMPT_CONTEXT_RATIO_START),
)


def _get_adaptive_prompt_ratio() -> float:
    with _prompt_ratio_lock:
        return _adaptive_prompt_context_ratio


def _record_prompt_ratio_success() -> float:
    global _adaptive_prompt_context_ratio
    with _prompt_ratio_lock:
        _adaptive_prompt_context_ratio = min(
            PROMPT_CONTEXT_RATIO_TARGET,
            _adaptive_prompt_context_ratio + max(0.01, PROMPT_CONTEXT_RATIO_STEP),
        )
        return _adaptive_prompt_context_ratio


def _record_prompt_ratio_overflow() -> float:
    global _adaptive_prompt_context_ratio
    with _prompt_ratio_lock:
        _adaptive_prompt_context_ratio = max(
            PROMPT_CONTEXT_RATIO_MIN,
            _adaptive_prompt_context_ratio - max(0.01, PROMPT_CONTEXT_RATIO_STEP),
        )
        return _adaptive_prompt_context_ratio


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
                    # Solo se usa para contar tokens; evita warning de longitudes del modelo base (ej. gpt2=1024).
                    _tokenizer.model_max_length = 10**9
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


# Encabezados de página en completed_texts (extracción nativa y visión).
_PAGE_HEADER_START_RE = re.compile(r"^## Página (\d+)(?:[^\n]*)?$", re.MULTILINE)


def split_markdown_by_page_headers(text: str) -> list[tuple[int, str]]:
    """Devuelve (número_página, cuerpo) ordenado. Sin marcadores: todo el texto como página 1."""
    text = text.strip()
    if not text:
        return []
    if not _PAGE_HEADER_START_RE.search(text):
        return [(1, text)]
    pieces = re.split(r"(?m)^(?=## Página \d+)", text)
    out: list[tuple[int, str]] = []
    header_re = re.compile(r"^## Página (\d+)(?:[^\n]*)?$", re.MULTILINE)
    for raw in pieces:
        raw = raw.strip()
        if not raw:
            continue
        first = raw.split("\n", 1)[0].strip()
        m = header_re.match(first)
        if not m:
            continue
        pn = int(m.group(1))
        body = header_re.sub("", raw, count=1).strip()
        out.append((pn, body))
    return sorted(out, key=lambda x: x[0])


def pdf_has_selectable_text(pdf_path: pathlib.Path) -> bool:
    with pymupdf.open(pdf_path) as pdf:
        return any(page.get_text().strip() for page in pdf)


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _last_page_from_completed_md(content: str) -> int | None:
    nums = [int(m) for m in _PAGE_HEADER_START_RE.findall(content)]
    return max(nums) if nums else None


def _slugify_anchor(label: str, *, fallback: str) -> str:
    s = unicodedata.normalize("NFKD", label)
    s = "".join(c if c.isalnum() or c in " -_" else "" for c in s)
    s = "-".join(s.lower().split())
    return s if s else fallback


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
        description=(
            "Nombre del tema alineado con la estructura del documento: capítulo, sección, subtítulo o temática "
            "cuando el texto las muestre (#/##/###, numeración, títulos destacados). Sin prefijos meta tipo 'Tema:'. "
            "No uses números de página, rangos ('páginas X–Y'), ni 'fragmento' en el título."
        )
    )
    cues: list[str] = Field(
        description=(
            "Lista de pistas tipo Cornell: palabras clave o preguntas cortas de repaso (una idea por elemento). "
            "Evita frases largas; 3 a 10 ítems según densidad del tema."
        )
    )
    notes: str = Field(
        description=(
            "Síntesis académica densa: definiciones, hipótesis, procedimientos, fórmulas (en texto o LaTeX ligero), "
            "relaciones causa-efecto y condiciones límite del modelo o experimento cuando el original las mencione. "
            "No pegues párrafos literales extensos. "
            "Solo si el fragmento termina antes de cerrar el tema y el origen no lo ciere aquí, indica al final: "
            "(continúa en el siguiente fragmento)."
        )
    )
    topic_summary: str = Field(
        description=(
            "Cierre del tema: 2 a 5 frases que integren la idea central, supuestos clave y utilidad en el contexto del texto."
        )
    )


class CornellSummaryStructured(BaseModel):
    """Resumen por temas estilo Cornell; coincide con el esquema JSON enviado al modelo."""

    model_config = ConfigDict(extra="forbid")

    topics: list[CornellTopicBlock] = Field(
        description=(
            "Lista ordenada según la secuencia del documento en este fragmento: cada elemento corresponde a una "
            "temática, sección o subtítulo explícito o implícito del texto (no al orden artificial de ## Página N). "
            "Unifica en un solo topic lo que el autor trata como la misma sección repartida en varias páginas. "
            "Si el texto es muy breve, un solo tema puede bastar. "
            "Todo en español claro salvo términos técnicos habituales en el original."
        )
    )


class _WindowSummaryCheckpoint(BaseModel):
    """Una ventana ya resumida (archivo en summary_partials)."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    start_p: int
    end_p: int
    body_sha256: str
    structured: CornellSummaryStructured


def _summary_partials_enabled() -> bool:
    raw = os.environ.get("SUMMARIZER_SUMMARY_PARTIALS", "").strip().lower()
    if raw in ("0", "false", "no", "n", "off"):
        return False
    return True


def summary_partials_dir_for_completed_rel(md_source_rel: pathlib.Path) -> pathlib.Path:
    """Carpeta dedicada al documento: summary_partials/<misma_jerarquía>/<stem>/"""
    stem = md_source_rel.stem
    parent = md_source_rel.parent
    return summary_partials / parent / stem


def _window_body_fingerprint(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _atomic_write_json(path: pathlib.Path, obj: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = obj.model_dump_json(indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _try_load_window_checkpoint(
    path: pathlib.Path,
    *,
    start_p: int,
    end_p: int,
    body: str,
) -> CornellSummaryStructured | None:
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        ck = _WindowSummaryCheckpoint.model_validate_json(raw)
    except (OSError, ValueError):
        return None
    if ck.start_p != start_p or ck.end_p != end_p:
        return None
    if ck.body_sha256 != _window_body_fingerprint(body):
        return None
    return ck.structured


def _save_window_checkpoint(
    path: pathlib.Path,
    *,
    start_p: int,
    end_p: int,
    body: str,
    structured: CornellSummaryStructured,
) -> None:
    ck = _WindowSummaryCheckpoint(
        start_p=start_p,
        end_p=end_p,
        body_sha256=_window_body_fingerprint(body),
        structured=structured,
    )
    _atomic_write_json(path, ck)


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
1. Tras el separador --- está el documento fuente. Particiona y resume por la estructura discursiva del autor: capítulos, secciones, subtítulos, apartados numerados o temáticas claras (incluidas en encabezados Markdown del propio texto o en negritas/títulos implícitos). Los marcadores `## Página N` solo delimitan el contenido disponible, no son títulos de salida.
2. En `topics`, ordena los elementos en el mismo orden en que aparecen esas secciones/temáticas en el fragmento. Si la misma sección continúa en varias páginas del bloque, unifica en un único topic.
3. Por tema: `title` debe reflejar esa sección o temática (adaptado o acortado si hace falta); nunca pongas en `title` números de página, rangos tipo "páginas X–Y", "Pág.", ni metadatos de fragmento. cues como repaso; notes densa (definiciones, pasos, supuestos, fórmulas cuando existan); topic_summary cierra la idea y utilidad.
4. Prioriza rigor: hechos, definiciones, datos y razonamiento del texto; no inventes citas, referencias ni detalles inexistentes en el material.
5. Si el documento mezcla idiomas, sintetiza en español salvo nombres propios o términos técnicos estándar.
6. No incluyas texto fuera del JSON (sin markdown envolvente, sin comentarios)."""

SUMMARY_CHUNK_WRAPPER = """Contexto: este bloque es el fragmento {part} de {total} de un documento largo (no tienes el resto).

Qué hacer:
- Extrae solo los temas que se apoyen en el contenido de ESTE fragmento.
- Si un tema empieza aquí y seguramente sigue después, en `notes` indica al final: (continúa en el siguiente fragmento).
- No inventes contenido de otras partes del documento.

---
{body}"""

SUMMARY_WINDOW_WRAPPER = """Contexto: el bloque siguiente contiene el texto de las páginas {start}–{end} del PDF original, separadas por marcadores `## Página N`. Eso es solo delimitación de contexto (no repitas ni uses esos rangos o números de página en los campos `title` del JSON).

Qué hacer:
- Extrae `topics` siguiendo títulos, subtítulos, secciones y temáticas del propio contenido (no una lista por página).
- Si la misma sección continúa en varias páginas dentro de este bloque, unifica en un solo `topic` con un único `title`.
- Resalta definiciones, fórmulas, procedimientos, hipótesis y límites que aparezcan en el texto (sin inventar).
- Si el texto está en otro idioma, sintetiza en español salvo nombres propios y términos técnicos habituales.
- Evita duplicar el mismo tema cerrado solo porque cambia el marcador `## Página`; en solape con otra ventana, prioriza información nueva sin repetir el mismo `title` si el contenido es redundante.

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


def _chat_parse_with_retry(**kwargs: Any) -> ParsedChatCompletion[Any]:
    last_error: Exception | None = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        _check_stop_requested()
        try:
            return client.chat.completions.parse(
                timeout=REQUEST_TIMEOUT_SECONDS,
                **kwargs,
            )
        except (APITimeoutError, APIConnectionError) as ex:
            last_error = ex
            if attempt >= REQUEST_RETRIES:
                break
            wait_seconds = REQUEST_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(
                "LM request timeout/conexión; "
                f"reintento {attempt}/{REQUEST_RETRIES - 1} en {wait_seconds:.1f}s..."
            )
            _sleep_with_stop(wait_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Fallo inesperado al ejecutar chat completion")


def _is_context_overflow_error(ex: Exception) -> bool:
    if isinstance(ex, BadRequestError):
        text = str(ex).lower()
        return (
            "context length" in text
            or "context size has been exceeded" in text
            or "context size exceeded" in text
            or "context has been exceeded" in text
            or "n_keep" in text
            or "n_ctx" in text
            or "too many tokens" in text
        )
    return False


def _format_cornell_topic_markdown(t: CornellTopicBlock, *, slug: str) -> str:
    cues_md = "\n".join(f"- {c}" for c in t.cues) if t.cues else "-"
    return (
        f"### {t.title} {{#{slug}}}\n\n"
        f"#### Pistas (cue)\n{cues_md}\n\n"
        f"#### Notas\n{t.notes}\n\n"
        f"#### Resumen del tema\n{t.topic_summary}"
    )


def format_cornell_markdown(
    summary: CornellSummaryStructured, *, document_title: bool = True
) -> str:
    if not summary.topics:
        return ""
    blocks: list[str] = []
    for i, t in enumerate(summary.topics):
        slug = _slugify_anchor(t.title, fallback=f"tema-{i + 1}")
        blocks.append(_format_cornell_topic_markdown(t, slug=slug))
    body = "\n\n".join(blocks)
    if document_title:
        return f"# Resumen\n\n{body}"
    return body


def _window_tokens_for_body(
    start_p: int, end_p: int, body: str, *, overhead_reserve: int
) -> int:
    wrapped = SUMMARY_WINDOW_WRAPPER.format(start=start_p, end=end_p, body=body)
    user_content = f"{SUMMARY_CORNELL_USER_PREFIX}\n\n{wrapped}"
    return count_tokens(user_content) + overhead_reserve


def _summary_window_token_budget() -> int:
    current_ratio = _get_adaptive_prompt_ratio()
    return max(512, int(MAX_CONTEXT_TOKENS * current_ratio))


def build_page_windows(
    pages: list[tuple[int, str]],
    *,
    overlap: int,
    budget_tokens: int,
    strict_one_page: bool,
) -> list[tuple[int, int, str]]:
    """Agrupa páginas consecutivas hasta budget_tokens; solape entre ventanas = overlap páginas."""
    if not pages:
        return []
    if strict_one_page:
        out: list[tuple[int, int, str]] = []
        for pnum, ptext in pages:
            out.append(
                (
                    pnum,
                    pnum,
                    f"## Página {pnum}\n\n{ptext}",
                )
            )
        return out

    n = len(pages)
    windows: list[tuple[int, int, str]] = []
    overhead_reserve = 64
    idx = 0
    while idx < n:
        start_p = pages[idx][0]
        end_idx = idx
        acc_body = ""
        while end_idx < n:
            pnum, ptext = pages[end_idx]
            sep = "\n\n" if acc_body else ""
            trial_body = acc_body + sep + f"## Página {pnum}\n\n{ptext}"
            if (
                _window_tokens_for_body(
                    start_p, pnum, trial_body, overhead_reserve=overhead_reserve
                )
                > budget_tokens
            ):
                break
            acc_body = trial_body
            end_idx += 1

        if end_idx == idx:
            pnum, ptext = pages[idx]
            inner_overhead = count_tokens(
                f"{SUMMARY_CORNELL_USER_PREFIX}\n\n"
                + SUMMARY_WINDOW_WRAPPER.format(start=pnum, end=pnum, body="")
            )
            room = max(256, budget_tokens - inner_overhead - overhead_reserve)
            pieces = chunk_text_by_tokens(ptext, room)
            for k, piece in enumerate(pieces):
                label = f"## Página {pnum} (parte {k + 1}/{len(pieces)})\n\n{piece}"
                windows.append((pnum, pnum, label))
            idx += 1
            continue

        end_p = pages[end_idx - 1][0]
        windows.append((start_p, end_p, acc_body))
        idx = max(idx + 1, end_idx - max(0, overlap))
    return windows


def _normalize_topic_title(title: str) -> str:
    return " ".join(title.lower().split())


def assemble_cornell_windows_markdown(
    ordered: list[tuple[int, int, CornellSummaryStructured]],
) -> str:
    """Une ventanas en un único Markdown con índice; temas Cornell seguidos sin agrupar por rango de páginas."""
    index_entries: list[tuple[str, str]] = []
    topic_blocks: list[str] = []
    global_i = 0
    used_slugs: set[str] = set()
    last_norm: str | None = None

    for win_i, (_start_p, _end_p, structured) in enumerate(ordered):
        if not structured.topics:
            continue
        for t in structured.topics:
            norm = _normalize_topic_title(t.title)
            if ASSEMBLE_DEDUP_BORDER and last_norm is not None and norm == last_norm:
                continue
            last_norm = norm
            global_i += 1
            slug = _slugify_anchor(t.title, fallback=f"tema-{global_i}")
            if slug in used_slugs:
                slug = f"{slug}-w{win_i}-t{global_i}"
            used_slugs.add(slug)
            index_entries.append((t.title, slug))
            topic_blocks.append(_format_cornell_topic_markdown(t, slug=slug))

    if not index_entries:
        return ""

    index_lines = "\n".join(f"- [{title}](#{slug})" for title, slug in index_entries)
    body = "\n\n".join(topic_blocks)
    return f"# Resumen\n\n## Índice\n\n{index_lines}\n\n---\n\n{body}"


def _strict_one_page_from_env() -> bool:
    raw = os.environ.get("SUMMARIZER_SUMMARY_STRICT_ONE_PAGE", "").strip().lower()
    return raw in ("1", "true", "yes", "sí", "si", "on")


def _chat_cornell_window(
    start_p: int, end_p: int, body: str
) -> CornellSummaryStructured:
    wrapped = SUMMARY_WINDOW_WRAPPER.format(start=start_p, end=end_p, body=body)
    user_content = f"{SUMMARY_CORNELL_USER_PREFIX}\n\n{wrapped}"
    completion = _chat_parse_with_retry(
        model=completion_model,
        messages=[{"role": "user", "content": user_content}],
        response_format=CornellSummaryStructured,
    )
    return completion_parsed_or_validate(completion, CornellSummaryStructured)


def summarize_document_paged_windows(
    full_text: str,
    *,
    partials_dir: pathlib.Path | None = None,
) -> str:
    """
    Resumen Cornell por ventanas de páginas (presupuesto + solape); ensamblado sin modelo.

    Si ``partials_dir`` está definido y SUMMARIZER_SUMMARY_PARTIALS no desactiva la función,
    cada ventana se guarda en ``window_NNNN.json`` (reanudable si el contenido de la ventana
    no cambia). El Markdown final es el mismo que sin checkpoints.
    """
    pages = split_markdown_by_page_headers(full_text)
    if not pages:
        return ""
    _check_stop_requested()

    budget = _summary_window_token_budget()
    print(
        f"Resumen por ventanas: {len(pages)} páginas detectadas; "
        f"presupuesto ~{budget} tokens; solape {SUMMARY_PAGE_OVERLAP} páginas."
    )

    strict = _strict_one_page_from_env()
    windows = build_page_windows(
        pages,
        overlap=SUMMARY_PAGE_OVERLAP,
        budget_tokens=budget,
        strict_one_page=strict,
    )
    if not windows:
        return ""

    use_partials = partials_dir is not None and _summary_partials_enabled()
    if use_partials:
        partials_dir.mkdir(parents=True, exist_ok=True)
        man_path = partials_dir / "_manifest.json"
        man_tmp = partials_dir / "_manifest.json.tmp"
        man_tmp.write_text(
            json.dumps(
                {
                    "version": 1,
                    "page_count": len(pages),
                    "window_count": len(windows),
                    "budget_tokens": budget,
                    "overlap": SUMMARY_PAGE_OVERLAP,
                    "strict_one_page": strict,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(man_tmp, man_path)

    def run_window(
        task_i: int, triplet: tuple[int, int, str]
    ) -> tuple[int, tuple[int, int, CornellSummaryStructured]]:
        _check_stop_requested()
        sp, ep, body = triplet
        part_path = (
            (partials_dir / f"window_{task_i:04d}.json")
            if use_partials and partials_dir is not None
            else None
        )
        if part_path is not None:
            cached = _try_load_window_checkpoint(
                part_path, start_p=sp, end_p=ep, body=body
            )
            if cached is not None:
                return task_i, (sp, ep, cached)
        try:
            structured = _chat_cornell_window(sp, ep, body)
        except BadRequestError as ex:
            if _is_context_overflow_error(ex):
                _record_prompt_ratio_overflow()
            raise
        if part_path is not None:
            _save_window_checkpoint(
                part_path, start_p=sp, end_p=ep, body=body, structured=structured
            )
        return task_i, (sp, ep, structured)

    workers = min(MAX_PARALLEL_WINDOW_SUMMARIES, len(windows))
    results: dict[int, tuple[int, int, CornellSummaryStructured]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(run_window, i, w) for i, w in enumerate(windows)]
        for fut in as_completed(futs):
            _check_stop_requested()
            i, triple = fut.result()
            results[i] = triple

    ordered_struct = [results[i] for i in range(len(windows))]
    # Un solo Markdown agregado (entrada lógica al PDF final).
    combined_md_path = (
        (partials_dir / "_combined_windows.md")
        if use_partials and partials_dir
        else None
    )
    md = assemble_cornell_windows_markdown(ordered_struct)
    if combined_md_path is not None and md.strip():
        _atomic_write_text(combined_md_path, md)
    if md.strip():
        _record_prompt_ratio_success()
    return md


def _markdown_for_pymupdf_pdf(markdown: str) -> str:
    """
    Ajusta el Markdown para el pipeline markdown_it → PyMuPDF Story.

    En modo commonmark, los atributos tipo Pandoc ``### Título {#slug}`` no se traducen a
    ``id=`` en el HTML; el índice manual con ``[texto](#slug)`` sí genera enlaces internos.
    Story falla luego con: No destination with id=...
    """
    text = markdown
    text = re.sub(
        r"(?ms)^## Índice\s*\n.*?\n---\s*\n+",
        "",
        text,
        count=1,
    )
    text = re.sub(
        r"^(#{1,6}\s+.+?)\s*\{#([^}]+)\}\s*$",
        r"\1",
        text,
        flags=re.MULTILINE,
    )
    return text


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


_HEADING_LINE_RE = re.compile(r"^(#{1,6})(\s+.*)$")


def normalize_markdown_heading_hierarchy_for_pdf(markdown: str) -> str:
    """
    PyMuPDF set_toc exige que los niveles del índice no salten (p. ej. h1 → h3),
    lo que provoca «bad hierarchy level in row …» al guardar el PDF.

    Tras quitar el bloque «## Índice» en _markdown_for_pymupdf_pdf, los temas Cornell
    (### …) quedan colgando directamente de # Resumen; aquí se reajustan los «#».
    """
    last_level = 0
    out: list[str] = []
    in_fence = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        m = _HEADING_LINE_RE.match(line)
        if not m:
            out.append(line)
            continue
        raw_level = len(m.group(1))
        rest = m.group(2)
        if last_level == 0:
            adj = 1
        elif raw_level > last_level + 1:
            adj = last_level + 1
        else:
            adj = raw_level
        last_level = adj
        out.append("#" * adj + rest)
    return "\n".join(out)


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
    completion = _chat_parse_with_retry(
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
    _check_stop_requested()
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
            _check_stop_requested()
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
    _check_stop_requested()
    single_user = f"{SUMMARY_CORNELL_USER_PREFIX}\n\n---\n\n{full_text}"
    prompt_tokens = count_tokens(single_user)
    current_ratio = _get_adaptive_prompt_ratio()
    effective_input_budget = max(512, int(MAX_CONTEXT_TOKENS * current_ratio))
    print(
        "Presupuesto de prompt (estimado tokenizer local): "
        f"{prompt_tokens} tokens; límite adaptativo: {effective_input_budget} "
        f"({int(current_ratio * 100)}% de contexto)"
    )
    if prompt_tokens <= effective_input_budget:
        try:
            result = summarize_cornell_single(full_text)
            new_ratio = _record_prompt_ratio_success()
            print(
                f"Resumen OK; elevando ratio adaptativo a {int(new_ratio * 100)}% para próximos documentos."
            )
            return result
        except Exception as ex:
            if not _is_context_overflow_error(ex):
                raise
            new_ratio = _record_prompt_ratio_overflow()
            print(
                "Overflow de contexto en resumen único; "
                f"bajando ratio adaptativo a {int(new_ratio * 100)}%."
            )

    wrapper_empty = SUMMARY_CHUNK_WRAPPER.format(part=1, total=1, body="")
    overhead = count_tokens(wrapper_empty)
    max_chunk_content = max(512, effective_input_budget - overhead - 200)
    for _ in range(4):
        try:
            result = summarize_cornell_chunked(full_text, max_chunk_content)
            new_ratio = _record_prompt_ratio_success()
            print(
                f"Resumen chunked OK; elevando ratio adaptativo a {int(new_ratio * 100)}%."
            )
            return result
        except Exception as ex:
            if not _is_context_overflow_error(ex):
                raise
            new_ratio = _record_prompt_ratio_overflow()
            if max_chunk_content <= 512:
                raise
            max_chunk_content = max(512, max_chunk_content // 2)
            print(
                "Context overflow detectado; reduciendo tamaño de fragmentos a "
                f"~{max_chunk_content} tokens y ratio adaptativo a {int(new_ratio * 100)}%..."
            )
    return summarize_cornell_chunked(full_text, max_chunk_content)


def completed_md_path_for_pdf(src: pathlib.Path) -> pathlib.Path:
    assert files_directory is not None
    rel = src.relative_to(files_directory)
    return completed_texts / rel.with_suffix(".md")


def configure_source_directory() -> None:
    """Asigna `files_directory` desde la variable de entorno o el diálogo del sistema."""
    global files_directory
    env = os.environ.get("SUMMARIZER_FILES_DIRECTORY", "").strip()
    if env:
        p = pathlib.Path(env).expanduser().resolve()
        if not p.is_dir():
            raise SystemExit(
                f"SUMMARIZER_FILES_DIRECTORY no es una carpeta válida: {p}"
            )
        files_directory = p
        print(f"Carpeta origen (entorno): {files_directory}")
        return
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as e:
        raise SystemExit(
            "No se pudo cargar tkinter para elegir carpeta. "
            "Instale tk o defina SUMMARIZER_FILES_DIRECTORY con la ruta a los PDF."
        ) from e
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    chosen = filedialog.askdirectory(
        title="Seleccione la carpeta donde están los archivos PDF originales",
    )
    root.destroy()
    if not chosen:
        raise SystemExit("No se seleccionó ninguna carpeta.")
    files_directory = pathlib.Path(chosen).resolve()
    print(f"Carpeta origen: {files_directory}")


def configure_vision_extraction_preference() -> None:
    """
    Pregunta una sola vez si se deben analizar por imagen los PDF sin texto extraíble (escaneados).

    Override sin diálogo: SUMMARIZER_USE_VISION_OCR=1 / true / yes / sí / si → sí;
    =0 / false / no → no.
    """
    global use_vision_for_scanned_pdfs
    raw = os.environ.get("SUMMARIZER_USE_VISION_OCR", "").strip().lower()
    if raw in ("1", "true", "yes", "sí", "si", "y", "on"):
        use_vision_for_scanned_pdfs = True
        print(
            "Extracción por imágenes (PDF escaneados): sí (SUMMARIZER_USE_VISION_OCR)"
        )
        return
    if raw in ("0", "false", "no", "n", "off"):
        use_vision_for_scanned_pdfs = False
        print(
            "Extracción por imágenes (PDF escaneados): no (SUMMARIZER_USE_VISION_OCR)"
        )
        return

    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError:
        ans = (
            input(
                "¿Analizar PDFs escaneados enviando cada página como imagen al modelo? "
                "Requiere modelo de visión y es más lento (S/N) [N]: "
            )
            .strip()
            .lower()
        )
        use_vision_for_scanned_pdfs = ans in ("s", "sí", "si", "y", "yes")
        print(
            f"Extracción por imágenes: {'sí' if use_vision_for_scanned_pdfs else 'no'}"
        )
        return

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    use_vision_for_scanned_pdfs = messagebox.askyesno(
        "Extracción por imágenes",
        "Algunos PDF no tienen texto seleccionable (escaneados o imágenes).\n\n"
        "¿Desea analizarlos enviando cada página como imagen al modelo con visión?\n\n"
        "• Sí: usa la GPU y tarda más.\n"
        "• No: solo se extrae texto normal; esos archivos quedarán sin contenido hasta que active la opción.",
        icon="question",
    )
    root.destroy()
    print(
        f"Extracción por imágenes para PDF sin texto: "
        f"{'sí' if use_vision_for_scanned_pdfs else 'no'}"
    )


def _nonempty_utf8_file(path: pathlib.Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip() != ""
    except OSError:
        return False


def _completed_md_needs_page_markers(out_md: pathlib.Path) -> bool:
    """True si hay texto completado pero sin ## Página (extracción antigua)."""
    if not _nonempty_utf8_file(out_md):
        return False
    try:
        txt = out_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return not _PAGE_HEADER_START_RE.search(txt)


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
    # toc=True: outline en el PDF según # / ## / ### (markdown_pdf → PyMuPDF).
    md_pdf = ensure_markdown_h1_for_pdf(_markdown_for_pymupdf_pdf(summary_md))
    md_pdf = normalize_markdown_heading_hierarchy_for_pdf(md_pdf)
    pdf.add_section(
        Section(
            text=md_pdf,
            toc=True,
        )
    )
    pdf.save(out_pdf)


def _ocr_page_body_or_empty(src: pathlib.Path, page_index: int) -> str:
    try:
        return _ocr_single_page(src, page_index).strip()
    except Exception as ex:
        print(f"OCR error página {page_index + 1} de {src}: {ex}")
        return ""


def extract_vision_pdf_incremental(src: pathlib.Path, out_path: pathlib.Path) -> None:
    """OCR de páginas en paralelo; escribe el .md al avanzar el prefijo consecutivo (atómico, reanudable)."""
    with pymupdf.open(src) as pdf:
        n = len(pdf)
    if n == 0:
        _atomic_write_text(out_path, "")
        return

    existing = ""
    if out_path.is_file():
        existing = out_path.read_text(encoding="utf-8", errors="replace")

    page_bodies: dict[int, str] = {}
    chunks_prefix: list[str] = []
    start_i = 0
    if existing.strip():
        if not _PAGE_HEADER_START_RE.search(existing):
            print(f"OCR reinicio (sin marcadores de página previos): {src.name}")
            page_bodies.clear()
            chunks_prefix.clear()
            start_i = 0
        else:
            parsed = split_markdown_by_page_headers(existing)
            if len(parsed) >= n:
                print(f"OCR ya completo ({n} páginas): {src.name}")
                return
            for pnum, body in parsed:
                idx = int(pnum) - 1
                body = body.strip()
                page_bodies[idx] = body
                chunks_prefix.append(f"## Página {pnum}\n\n{body}")
            start_i = len(parsed)

    print(
        f"OCR vis {src.name}: páginas {start_i + 1}..{n} de {n} "
        f"(paralelo ≤{MAX_PARALLEL_OCR_PAGES})"
    )
    if start_i >= n:
        _atomic_write_text(out_path, "\n\n".join(chunks_prefix))
        return
    _check_stop_requested()

    written_len = start_i
    write_lock = threading.Lock()

    def try_flush_extended_locked() -> None:
        nonlocal written_len
        while written_len < n and written_len in page_bodies:
            b = page_bodies[written_len]
            chunks_prefix.append(f"## Página {written_len + 1}\n\n{b}")
            written_len += 1
        if chunks_prefix:
            _atomic_write_text(out_path, "\n\n".join(chunks_prefix))

    pending = [i for i in range(start_i, n) if i not in page_bodies]
    workers = min(MAX_PARALLEL_OCR_PAGES, len(pending))
    workers = max(1, workers)

    def run_page(i: int) -> tuple[int, str]:
        return i, _ocr_page_body_or_empty(src, i)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_page, i): i for i in pending}
        for fut in as_completed(futures):
            _check_stop_requested()
            i, text = fut.result()
            with write_lock:
                page_bodies[i] = text
                try_flush_extended_locked()

    with write_lock:
        try_flush_extended_locked()


def _extract_single_pdf(src: pathlib.Path) -> None:
    """Un PDF por tarea (hilo): abre su propio documento; no compartir fitz entre hilos."""
    try:
        _check_stop_requested()
        print(src)
        out_md = completed_md_path_for_pdf(src)
        if pdf_has_selectable_text(src):
            file_text = extract_text_get_text_only(src)
            write_completed_text(src, file_text)
        elif use_vision_for_scanned_pdfs:
            extract_vision_pdf_incremental(src, out_md)
        else:
            print(
                f"Sin texto extraíble (get_text vacío); visión desactivada — omitido: {src}"
            )
    except Exception as ex:
        print(f"Error processing {src}: {ex}")


def run_pdf_extraction() -> None:
    """Idempotent: skips PDFs that already have a non-empty completed_texts .md."""
    assert files_directory is not None
    _check_stop_requested()
    pending: list[pathlib.Path] = []
    for root, _, files in os.walk(files_directory):
        for file in files:
            if not file.endswith(".pdf"):
                continue
            src = pathlib.Path(root) / file
            out_md = completed_md_path_for_pdf(src)
            if _nonempty_utf8_file(out_md) and not _completed_md_needs_page_markers(
                out_md
            ):
                print(f"Skip extract (already in completed_texts): {src}")
                continue
            if _completed_md_needs_page_markers(out_md):
                print(f"Re-extracción (marcadores de página): {src}")
            pending.append(src)
    if not pending:
        return
    workers = min(MAX_PARALLEL_PDFS, len(pending))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_extract_single_pdf, p) for p in pending]
        for fut in as_completed(futures):
            _check_stop_requested()
            fut.result()


def _summarize_single_md(md_path: pathlib.Path) -> None:
    try:
        _check_stop_requested()
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
        partials = summary_partials_dir_for_completed_rel(rel)
        summary_md = summarize_document_paged_windows(full_text, partials_dir=partials)
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
    _check_stop_requested()
    workers = min(MAX_PARALLEL_SUMMARIES, len(paths))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_summarize_single_md, p) for p in paths]
        for fut in as_completed(futures):
            _check_stop_requested()
            fut.result()


def extract_text_get_text_only(pdf_path: pathlib.Path) -> str:
    parts: list[str] = []
    with pymupdf.open(pdf_path) as pdf:
        for i, page in enumerate(pdf):
            text = page.get_text().strip()
            parts.append(f"## Página {i + 1}\n\n{text}")
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
        completion = _chat_parse_with_retry(
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


if __name__ == "__main__":
    _install_stop_handlers()
    try:
        configure_source_directory()
        configure_vision_extraction_preference()
        configure_lm_studio_model()
        get_tokenizer()
        _start_stop_listener()
        _check_stop_requested()
        run_pdf_extraction()
        _check_stop_requested()
        run_summarization_pipeline()
    except (StopRequested, KeyboardInterrupt):
        _request_stop("Ejecución interrumpida.")
        print("Proceso detenido por el usuario.")
