"""Resumen estilo Cornell: ventanas por página, troceo y unificación."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import BadRequestError

from summarizer import state as app_state
from summarizer.checkpoints import (
    save_window_checkpoint,
    summary_partials_enabled,
    try_load_window_checkpoint,
)
from summarizer.book_outline import chapter_outline_for_summary
from summarizer.config import (
    ASSEMBLE_DEDUP_BORDER,
    ASSEMBLE_DEDUP_GLOBAL,
    MAX_PARALLEL_CHUNKS,
    MAX_PARALLEL_WINDOW_SUMMARIES,
    SEMANTIC_DEDUP_THRESHOLD,
    SUMMARY_PAGE_OVERLAP,
    SUMMARY_MAX_PAGES_PER_WINDOW,
    SUMMARY_UNIFY_HIERARCHICAL,
    SUMMARY_UNIFY_MODE,
    SUMMARY_UNIFY_WINDOWS,
    cornell_depth_profile,
)
from summarizer.fs import atomic_write_text
from summarizer.llm import (
    chat_structured_with_retry,
    is_context_overflow_error,
)
from summarizer.markdown_utils import slugify_anchor, split_markdown_by_page_headers
from summarizer.models import CornellSummaryStructured, CornellTopicBlock
from summarizer.progress import progress_log
from summarizer.prompts import (
    BOOK_CHAPTER_OUTLINE_PREFIX,
    CORNELL_DEPTH_HIGH_SUFFIX,
    SUMMARY_CHUNK_WRAPPER,
    SUMMARY_CORNELL_SYSTEM_PROMPT,
    SUMMARY_WINDOW_WRAPPER,
    UNIFY_ASSEMBLED_CORNELL_BATCH_PROMPT,
    UNIFY_ASSEMBLED_CORNELL_PROMPT,
    UNIFY_SUMMARIES_PROMPT,
)
from summarizer.stop import check_stop_requested
from summarizer.tokenizer import chunk_text_by_tokens, count_tokens

_MAX_UNIFY_DEPTH = 12
_UNIFY_BATCH_OVERHEAD_TOKENS = 900


def _write_partial_assembly(
    results: dict[int, tuple[int, int, CornellSummaryStructured]],
    *,
    partial_md_path: Path | None,
    h1_title: str,
) -> None:
    """Vuelca el ensamblaje de las ventanas completadas hasta ahora.

    Pensado para llamar desde el handler de excepciones; nunca lanza.
    """
    if partial_md_path is None or not results:
        return
    try:
        ordered_so_far = [results[i] for i in sorted(results.keys())]
        partial_md = assemble_cornell_windows_markdown(
            ordered_so_far, h1_title=h1_title
        )
        if partial_md.strip():
            from summarizer.fs import atomic_write_text

            atomic_write_text(partial_md_path, partial_md)
            progress_log(
                f"Resumen parcial escrito ({len(results)} ventanas): {partial_md_path}"
            )
    except Exception as ex:
        progress_log(f"Aviso: no se pudo escribir resumen parcial: {ex}")


def _effective_cornell_system_prompt() -> str:
    """Mensaje system Cornell, con sufijo de profundidad si aplica."""
    if cornell_depth_profile() == "high":
        return f"{SUMMARY_CORNELL_SYSTEM_PROMPT}\n\n{CORNELL_DEPTH_HIGH_SUFFIX}"
    return SUMMARY_CORNELL_SYSTEM_PROMPT


def _topic_sections_from_assembled_markdown(md: str) -> list[str]:
    """Extrae bloques `### ...` del cuerpo tras el primer `---` (formato ensamblado)."""
    md = md.strip()
    sep = "\n---\n\n"
    if sep in md:
        body = md.split(sep, 1)[1].strip()
    else:
        body = md
    if not body:
        return []
    parts = re.split(r"(?=\n\n### )", body)
    return [p.strip() for p in parts if p.strip()]


def _group_topic_sections_into_batches(
    sections: list[str], *, max_batch_content_tokens: int
) -> list[list[str]]:
    """Agrupa secciones en lotes que caben en el presupuesto de tokens del prompt de unificación."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for s in sections:
        t = count_tokens(s)
        if t > max_batch_content_tokens:
            if current:
                batches.append(current)
                current = []
                current_tokens = 0
            for piece in chunk_text_by_tokens(s, max_batch_content_tokens):
                batches.append([piece])
            continue
        if current and current_tokens + t > max_batch_content_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(s)
        current_tokens += t
    if current:
        batches.append(current)
    return batches


def _format_cornell_topic_markdown(t: CornellTopicBlock, *, slug: str) -> str:
    cues_md = "\n".join(f"- {c}" for c in t.cues) if t.cues else "-"
    return (
        f"### {t.title} {{#{slug}}}\n\n"
        f"#### Pistas (claves)\n{cues_md}\n\n"
        f"#### Notas\n{t.notes}\n\n"
        f"#### Resumen del tema\n{t.topic_summary}"
    )


def format_cornell_markdown(
    summary: CornellSummaryStructured,
    *,
    document_title: bool = True,
    h1_title: str = "Resumen",
) -> str:
    if not summary.topics:
        return ""
    blocks: list[str] = []
    for i, t in enumerate(summary.topics):
        slug = slugify_anchor(t.title, fallback=f"tema-{i + 1}")
        blocks.append(_format_cornell_topic_markdown(t, slug=slug))
    body = "\n\n".join(blocks)
    if document_title:
        return f"# {h1_title}\n\n{body}"
    return body


def build_page_windows(
    pages: list[tuple[int, str]],
    *,
    overlap: int,
    max_pages_per_window: int,
) -> list[tuple[int, int, str]]:
    """Agrupa páginas por bloques con máximo fijo de páginas por ventana."""
    if not pages:
        return []
    max_pages = max(1, max_pages_per_window)
    step = max(1, max_pages - max(0, overlap))
    n = len(pages)
    out: list[tuple[int, int, str]] = []
    idx = 0
    while idx < n:
        chunk = pages[idx : idx + max_pages]
        if not chunk:
            break
        start_p = chunk[0][0]
        end_p = chunk[-1][0]
        body = "\n\n".join(f"## Página {pnum}\n\n{ptext}" for pnum, ptext in chunk)
        out.append((start_p, end_p, body))
        idx += step
    return out


def _chunk_single_page_if_needed(
    pnum: int, ptext: str, *, safety_margin: int = 200
) -> list[tuple[int, int, str]]:
    base_wrapper = SUMMARY_WINDOW_WRAPPER.format(start=pnum, end=pnum, body="")
    base_tokens = count_tokens(_effective_cornell_system_prompt()) + count_tokens(
        base_wrapper
    )
    ratio = app_state.get_adaptive_prompt_ratio()
    max_prompt = max(512, int(app_state.MAX_CONTEXT_TOKENS * ratio))
    room = max(256, max_prompt - base_tokens - safety_margin)
    chunks = chunk_text_by_tokens(ptext, room)
    if len(chunks) <= 1:
        return [(pnum, pnum, f"## Página {pnum}\n\n{ptext}")]
    out: list[tuple[int, int, str]] = []
    total = len(chunks)
    for i, piece in enumerate(chunks, start=1):
        out.append((pnum, pnum, f"## Página {pnum} (parte {i}/{total})\n\n{piece}"))
    return out


_SPANISH_STOPWORDS = frozenset(
    {
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "unos",
        "unas",
        "de",
        "del",
        "al",
        "a",
        "en",
        "y",
        "o",
        "u",
        "que",
        "por",
        "para",
        "con",
        "sin",
        "sobre",
        "entre",
        "su",
        "sus",
        "este",
        "esta",
        "estos",
        "estas",
        "ese",
        "esa",
        "esos",
        "esas",
        "aquel",
        "aquella",
        "es",
        "ser",
        "son",
        "como",
        "mas",
        "menos",
        "muy",
        "se",
        "lo",
        "le",
        "les",
        "me",
        "te",
        "nos",
        "os",
    }
)


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _title_tokens(title: str) -> frozenset[str]:
    """Tokens del título normalizado (minúsculas, sin acentos, sin stopwords)."""
    title_norm = _strip_accents(title.lower())
    return frozenset(
        t
        for t in re.findall(r"[a-z0-9]+", title_norm)
        if len(t) > 2 and t not in _SPANISH_STOPWORDS
    )


def _topic_signature(title: str, notes: str = "") -> frozenset[str]:
    """Firma extendida = tokens de título + bigramas del inicio de notas.

    Sirve para detectar paráfrasis cuando el título por sí solo no comparte
    suficientes tokens, pero el contenido de las notas converge.
    """
    title_set = _title_tokens(title)
    note_head = _strip_accents(notes.lower()[:200])
    note_tokens = re.findall(r"[a-z0-9]+", note_head)
    shingles: set[str] = set()
    for i in range(len(note_tokens) - 1):
        shingles.add(f"{note_tokens[i]}|{note_tokens[i + 1]}")
        if len(shingles) >= 16:
            break
    return frozenset(title_set | shingles)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _topic_similarity(a_title: str, a_notes: str, b_title: str, b_notes: str) -> float:
    """Similaridad combinada: máximo entre Jaccard de títulos y Jaccard de firmas.

    El Jaccard de títulos es la señal principal (paráfrasis del mismo asunto);
    el Jaccard de firmas completas atrapa casos con títulos divergentes pero
    contenido convergente.
    """
    title_jac = _jaccard(_title_tokens(a_title), _title_tokens(b_title))
    full_jac = _jaccard(
        _topic_signature(a_title, a_notes), _topic_signature(b_title, b_notes)
    )
    return max(title_jac, full_jac)


def _merge_notes(existing: str, new: str) -> str:
    """Concatena notas evitando líneas literalmente repetidas."""
    existing_norm = {line.strip() for line in existing.splitlines() if line.strip()}
    add_lines: list[str] = []
    for line in new.splitlines():
        stripped = line.strip()
        if not stripped or stripped in existing_norm:
            continue
        add_lines.append(line)
        existing_norm.add(stripped)
    if not add_lines:
        return existing
    sep = "\n\n" if existing.strip() else ""
    return existing.rstrip() + sep + "\n".join(add_lines)


def _merge_topic_blocks(
    existing: CornellTopicBlock, new: CornellTopicBlock
) -> CornellTopicBlock:
    """Fusiona dos temas: conserva título, une `cues` y `notes`."""
    cues_out = list(existing.cues)
    seen_cues = {c.strip().lower() for c in cues_out if c.strip()}
    for c in new.cues:
        cn = c.strip().lower()
        if cn and cn not in seen_cues:
            cues_out.append(c)
            seen_cues.add(cn)
    cues_out = cues_out[:10]
    notes = _merge_notes(existing.notes, new.notes)
    summary = existing.topic_summary or new.topic_summary
    return CornellTopicBlock(
        title=existing.title,
        cues=cues_out,
        notes=notes,
        topic_summary=summary,
    )


def format_cornell_structured_with_index(
    summary: CornellSummaryStructured, *, h1_title: str = "Resumen"
) -> str:
    """Un único CornellSummaryStructured a Markdown con ## Índice y bloques ###."""
    if not summary.topics:
        return ""
    index_entries: list[tuple[str, str]] = []
    topic_blocks: list[str] = []
    used_slugs: set[str] = set()
    for global_i, t in enumerate(summary.topics, start=1):
        slug = slugify_anchor(t.title, fallback=f"tema-{global_i}")
        if slug in used_slugs:
            slug = f"{slug}-t{global_i}"
        used_slugs.add(slug)
        index_entries.append((t.title, slug))
        topic_blocks.append(_format_cornell_topic_markdown(t, slug=slug))
    index_lines = "\n".join(f"- [{title}](#{slug})" for title, slug in index_entries)
    body = "\n\n".join(topic_blocks)
    return f"# {h1_title}\n\n## Índice\n\n{index_lines}\n\n---\n\n{body}"


def assemble_cornell_windows_markdown(
    ordered: list[tuple[int, int, CornellSummaryStructured]],
    *,
    h1_title: str = "Resumen",
) -> str:
    """Une ventanas en un único Markdown con índice y deduplicación semántica.

    A diferencia del modo legacy (igualdad exacta de títulos), este flujo usa
    similitud Jaccard sobre una firma del tema (título normalizado + shingles
    de notas). Cuando se detecta un duplicado, se **fusiona** el contenido
    (cues y notes) sobre el tema ya emitido para no perder detalle entre
    ventanas solapadas.
    """
    threshold = SEMANTIC_DEDUP_THRESHOLD
    used_slugs: set[str] = set()
    emitted: list[dict] = []

    for win_i, (_start_p, _end_p, structured) in enumerate(ordered):
        if not structured.topics:
            continue
        for t in structured.topics:
            merged_into: dict | None = None

            if ASSEMBLE_DEDUP_BORDER and emitted:
                prev = emitted[-1]
                if (
                    _topic_similarity(
                        t.title, t.notes, prev["topic"].title, prev["topic"].notes
                    )
                    >= threshold
                ):
                    merged_into = prev
            if merged_into is None and ASSEMBLE_DEDUP_GLOBAL:
                for state in emitted:
                    if (
                        _topic_similarity(
                            t.title,
                            t.notes,
                            state["topic"].title,
                            state["topic"].notes,
                        )
                        >= threshold
                    ):
                        merged_into = state
                        break

            if merged_into is not None:
                merged_topic = _merge_topic_blocks(merged_into["topic"], t)
                merged_into["topic"] = merged_topic
                continue

            global_i = len(emitted) + 1
            slug = slugify_anchor(t.title, fallback=f"tema-{global_i}")
            if slug in used_slugs:
                slug = f"{slug}-w{win_i}-t{global_i}"
            used_slugs.add(slug)
            emitted.append({"topic": t, "slug": slug})

    if not emitted:
        return ""

    index_entries = [(s["topic"].title, s["slug"]) for s in emitted]
    topic_blocks = [
        _format_cornell_topic_markdown(s["topic"], slug=s["slug"]) for s in emitted
    ]
    index_lines = "\n".join(f"- [{title}](#{slug})" for title, slug in index_entries)
    body = "\n\n".join(topic_blocks)
    return f"# {h1_title}\n\n## Índice\n\n{index_lines}\n\n---\n\n{body}"


def _strict_one_page_from_env() -> bool:
    raw = os.environ.get("SUMMARIZER_SUMMARY_STRICT_ONE_PAGE", "").strip().lower()
    return raw in ("1", "true", "yes", "sí", "si", "on")


def _chat_cornell_window(
    start_p: int,
    end_p: int,
    body: str,
    *,
    chapter_outline: list[str] | None = None,
) -> CornellSummaryStructured:
    user_content = SUMMARY_WINDOW_WRAPPER.format(start=start_p, end=end_p, body=body)
    if chapter_outline:
        outline_lines = "\n".join(f"- {c}" for c in chapter_outline)
        user_content += BOOK_CHAPTER_OUTLINE_PREFIX.format(outline_lines=outline_lines)
    return chat_structured_with_retry(
        model=app_state.completion_model,
        messages=[
            {"role": "system", "content": _effective_cornell_system_prompt()},
            {"role": "user", "content": user_content},
        ],
        response_format=CornellSummaryStructured,
    )


def summarize_document_paged_windows(
    full_text: str,
    *,
    partials_dir: Path | None = None,
    h1_title: str = "Resumen",
    partial_md_path: Path | None = None,
) -> tuple[str, str]:
    """Resumen Cornell por ventanas.

    Devuelve ``(markdown_final, markdown_ensamblado_pre_unificación)``.

    Si ``partial_md_path`` se indica y el proceso lanza una excepción tras
    haber procesado algunas ventanas, se escribe un ensamblado parcial en
    esa ruta antes de propagar la excepción (los checkpoints por ventana ya
    permiten reanudar en el siguiente run; este Markdown es para inspección
    del usuario y para señalizar 'incompleto').
    """
    pages = split_markdown_by_page_headers(full_text)
    if not pages:
        return "", ""
    check_stop_requested()

    chapter_outline = chapter_outline_for_summary(full_text)
    if chapter_outline:
        progress_log(
            f"Referencia de capítulos ({len(chapter_outline)} entradas): "
            "entorno o índice detectado en el texto."
        )

    max_pages_per_window = max(1, SUMMARY_MAX_PAGES_PER_WINDOW)
    progress_log(
        f"Resumen por ventanas: {len(pages)} páginas detectadas; "
        f"máximo {max_pages_per_window} páginas/ventana; solape {SUMMARY_PAGE_OVERLAP} páginas."
    )
    strict = _strict_one_page_from_env()
    if strict:
        max_pages_per_window = 1
    windows = build_page_windows(
        pages,
        overlap=SUMMARY_PAGE_OVERLAP,
        max_pages_per_window=max_pages_per_window,
    )
    if not windows:
        return "", ""

    use_partials = partials_dir is not None and summary_partials_enabled()
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
                    "max_pages_per_window": max_pages_per_window,
                    "overlap": SUMMARY_PAGE_OVERLAP,
                    "strict_one_page": strict,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(man_tmp, man_path)

    def summarize_with_fallback(
        sp: int, ep: int, body: str
    ) -> tuple[int, int, CornellSummaryStructured]:
        page_count = len(split_markdown_by_page_headers(body))
        try:
            structured = _chat_cornell_window(
                sp, ep, body, chapter_outline=chapter_outline
            )
            return sp, ep, structured
        except BadRequestError as ex:
            if not is_context_overflow_error(ex):
                raise
            app_state.record_prompt_ratio_overflow()
            if page_count > 1:
                progress_log(
                    f"Overflow en páginas {sp}-{ep}; reintentando con menor granularidad."
                )
                parsed = split_markdown_by_page_headers(body)
                next_max_pages = max(1, page_count - 1)
                retry_windows = build_page_windows(
                    parsed, overlap=0, max_pages_per_window=next_max_pages
                )
                merged_topics: list[CornellTopicBlock] = []
                retry_start = retry_windows[0][0] if retry_windows else sp
                retry_end = retry_windows[-1][1] if retry_windows else ep
                for rsp, rep, rbody in retry_windows:
                    _, _, partial = summarize_with_fallback(rsp, rep, rbody)
                    merged_topics.extend(partial.topics)
                return (
                    retry_start,
                    retry_end,
                    CornellSummaryStructured(topics=merged_topics),
                )
            parsed_page = split_markdown_by_page_headers(body)
            if not parsed_page:
                raise
            pnum, ptext = parsed_page[0]
            chunked = _chunk_single_page_if_needed(pnum, ptext)
            if len(chunked) <= 1:
                raise
            progress_log(
                f"Overflow en página {pnum}; reintentando por fragmentos internos."
            )
            merged_topics: list[CornellTopicBlock] = []
            for csp, cep, cbody in chunked:
                _, _, partial = summarize_with_fallback(csp, cep, cbody)
                merged_topics.extend(partial.topics)
            return sp, ep, CornellSummaryStructured(topics=merged_topics)

    def run_window(
        task_i: int, triplet: tuple[int, int, str]
    ) -> tuple[int, tuple[int, int, CornellSummaryStructured]]:
        check_stop_requested()
        sp, ep, body = triplet
        part_path = (
            (partials_dir / f"window_{task_i:04d}.json")
            if use_partials and partials_dir is not None
            else None
        )
        if part_path is not None:
            cached = try_load_window_checkpoint(
                part_path, start_p=sp, end_p=ep, body=body
            )
            if cached is not None:
                return task_i, (sp, ep, cached)
        _, _, structured = summarize_with_fallback(sp, ep, body)
        if part_path is not None:
            save_window_checkpoint(
                part_path, start_p=sp, end_p=ep, body=body, structured=structured
            )
        return task_i, (sp, ep, structured)

    results: dict[int, tuple[int, int, CornellSummaryStructured]] = {}
    n_win = len(windows)
    workers = min(MAX_PARALLEL_WINDOW_SUMMARIES, max(1, n_win))
    completed_count = 0

    def _log_window_progress() -> None:
        progress_log(f"Resumen ventana {completed_count}/{n_win}")

    try:
        if workers <= 1 or n_win == 1:
            for i, w in enumerate(windows):
                check_stop_requested()
                idx, triple = run_window(i, w)
                results[idx] = triple
                completed_count += 1
                _log_window_progress()
        else:
            progress_log(
                f"Resumen de ventanas en paralelo ({workers} workers, {n_win} ventanas)."
            )
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_map = {
                    pool.submit(run_window, i, w): i for i, w in enumerate(windows)
                }
                for fut in as_completed(future_map):
                    check_stop_requested()
                    task_i, triple = fut.result()
                    results[task_i] = triple
                    completed_count += 1
                    _log_window_progress()
    except BaseException:
        _write_partial_assembly(
            results, partial_md_path=partial_md_path, h1_title=h1_title
        )
        raise

    ordered_struct = [results[i] for i in range(n_win)]
    combined_md_path = (
        (partials_dir / "_combined_windows.md")
        if use_partials and partials_dir
        else None
    )
    assembled_md = assemble_cornell_windows_markdown(ordered_struct, h1_title=h1_title)
    if combined_md_path is not None and assembled_md.strip():
        atomic_write_text(combined_md_path, assembled_md)

    final_md = assembled_md
    if assembled_md.strip() and len(ordered_struct) > 1 and SUMMARY_UNIFY_WINDOWS:
        final_md = _try_unify_assembled(assembled_md, h1_title=h1_title)

    if final_md.strip():
        app_state.record_prompt_ratio_success()
    return final_md, assembled_md


def _single_pass_unify_if_fits(
    md: str, *, h1_title: str, max_budget: int
) -> str | None:
    """Devuelve Markdown unificado si el prompt cabe en el presupuesto; si no, None."""
    unify_prompt = UNIFY_ASSEMBLED_CORNELL_PROMPT.format(combined=md)
    prompt_tokens = count_tokens(unify_prompt)
    if prompt_tokens > max_budget:
        return None
    progress_log(
        "Unificando resumen de ventanas (fusión de duplicados y coherencia, un solo paso)…"
    )
    unified = _chat_cornell_structured(unify_prompt)
    unified_md = format_cornell_structured_with_index(unified, h1_title=h1_title)
    if unified_md.strip():
        progress_log("Resumen unificado correctamente.")
        return unified_md
    return None


def _hierarchical_unify_markdown(
    md: str, *, h1_title: str, max_budget: int, depth: int = 0
) -> str:
    """Reduce por lotes cuando el ensamblaje no cabe en un único prompt de unificación."""
    if depth > _MAX_UNIFY_DEPTH:
        progress_log(
            "Unificación jerárquica: profundidad máxima alcanzada; se conserva el último ensamblaje."
        )
        return md

    single = _single_pass_unify_if_fits(md, h1_title=h1_title, max_budget=max_budget)
    if single is not None:
        return single

    sections = _topic_sections_from_assembled_markdown(md)
    if not sections:
        progress_log(
            "Unificación jerárquica: no se detectaron bloques ###; se conserva ensamblaje."
        )
        return md

    batch_empty = UNIFY_ASSEMBLED_CORNELL_BATCH_PROMPT.format(
        part=1, total=1, combined=""
    )
    overhead = count_tokens(batch_empty) + _UNIFY_BATCH_OVERHEAD_TOKENS
    max_batch_content = max(4096, max_budget - overhead)

    batches = _group_topic_sections_into_batches(
        sections, max_batch_content_tokens=max_batch_content
    )
    progress_log(
        f"Unificación jerárquica: {len(sections)} bloques → {len(batches)} lote(s) "
        f"(nivel {depth + 1})."
    )

    merged_topics: list[CornellTopicBlock] = []
    total_batches = len(batches)
    for bi, batch in enumerate(batches, start=1):
        check_stop_requested()
        progress_log(
            f"Unificación jerárquica nivel {depth + 1}: lote {bi}/{total_batches}"
        )
        combined = "\n\n".join(batch)
        prompt = UNIFY_ASSEMBLED_CORNELL_BATCH_PROMPT.format(
            part=bi, total=total_batches, combined=combined
        )
        pt = count_tokens(prompt)
        if pt > max_budget:
            pieces = chunk_text_by_tokens(combined, max(max_batch_content // 2, 2048))
            for pj, piece in enumerate(pieces, start=1):
                sub_prompt = UNIFY_ASSEMBLED_CORNELL_BATCH_PROMPT.format(
                    part=pj, total=len(pieces), combined=piece
                )
                part_struct = _chat_cornell_structured(sub_prompt)
                merged_topics.extend(part_struct.topics)
        else:
            part_struct = _chat_cornell_structured(prompt)
            merged_topics.extend(part_struct.topics)

    if not merged_topics:
        return md

    next_md = format_cornell_structured_with_index(
        CornellSummaryStructured(topics=merged_topics), h1_title=h1_title
    )

    if len(merged_topics) >= len(sections) and depth >= 3:
        progress_log(
            "Unificación jerárquica: se detiene tras fusiones limitadas; "
            "resultado del último lote."
        )
        return _maybe_final_single_pass(
            next_md, h1_title=h1_title, max_budget=max_budget
        )

    unified_full = UNIFY_ASSEMBLED_CORNELL_PROMPT.format(combined=next_md)
    if count_tokens(unified_full) <= max_budget:
        return _maybe_final_single_pass(
            next_md, h1_title=h1_title, max_budget=max_budget
        )

    return _hierarchical_unify_markdown(
        next_md, h1_title=h1_title, max_budget=max_budget, depth=depth + 1
    )


def _maybe_final_single_pass(md: str, *, h1_title: str, max_budget: int) -> str:
    """Pasada final opcional sobre el Markdown ya unificado por lotes.

    Sólo se ejecuta cuando ``SUMMARY_UNIFY_MODE == "aggressive"`` (lo cual
    también aplica al flag legado ``SUMMARIZER_SUMMARY_FINAL_UNIFY_PASS``,
    que en ``config.py`` se mapea a ese modo). En los demás modos esta
    pasada comprimiría el resultado innecesariamente.
    """
    if SUMMARY_UNIFY_MODE != "aggressive":
        return md
    final_try = _single_pass_unify_if_fits(md, h1_title=h1_title, max_budget=max_budget)
    return final_try if final_try is not None else md


_TOPIC_HEADING_RE = re.compile(r"^###\s+(.+?)\s*(?:\{#[^}]+\})?\s*$")
_SECTION_HEADER_RE = re.compile(r"(?m)^####\s+")


def _parse_topic_section(section: str) -> CornellTopicBlock | None:
    """Reconstruye un ``CornellTopicBlock`` desde un bloque ensamblado.

    Espera el formato emitido por :func:`_format_cornell_topic_markdown`:
    título `### ... {#slug}` seguido de las sub-secciones `#### Pistas`,
    `#### Notas`, `#### Resumen del tema`. Devuelve ``None`` si el bloque
    no parsea (p. ej. heading vacío).
    """
    section = section.strip()
    if not section:
        return None
    lines = section.splitlines()
    title_match = _TOPIC_HEADING_RE.match(lines[0])
    if not title_match:
        return None
    title = title_match.group(1).strip()
    if not title:
        return None
    body = "\n".join(lines[1:])
    cues: list[str] = []
    notes = ""
    topic_summary = ""
    parts = _SECTION_HEADER_RE.split(body)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        header, _, content = part.partition("\n")
        header_norm = _strip_accents(header.strip().lower())
        content = content.strip()
        if header_norm.startswith("pistas") or header_norm.startswith("cue"):
            cues = [
                line.lstrip("-* ").strip()
                for line in content.splitlines()
                if line.lstrip("-* ").strip() and line.lstrip("-* ").strip() != "-"
            ]
        elif header_norm.startswith("notas") or header_norm == "notes":
            notes = content
        elif header_norm.startswith("resumen") or header_norm.startswith("summary"):
            topic_summary = content
    return CornellTopicBlock(
        title=title,
        cues=cues,
        notes=notes,
        topic_summary=topic_summary,
    )


def _lmless_second_pass(md: str, *, h1_title: str) -> str:
    """Aplica una segunda pasada de dedup Jaccard sobre el ensamblaje.

    No realiza llamadas LLM: parsea los bloques ``###`` del Markdown
    ensamblado, fusiona los temas con similitud Jaccard ≥ umbral relajado
    (``SEMANTIC_DEDUP_THRESHOLD - 0.1``) y re-emite el Markdown final.
    Pensado para preservar el máximo detalle, aceptando algunos duplicados
    blandos (sinónimos exactos que Jaccard no detecta).
    """
    sections = _topic_sections_from_assembled_markdown(md)
    if len(sections) <= 1:
        return md
    parsed: list[CornellTopicBlock] = []
    for section in sections:
        block = _parse_topic_section(section)
        if block is not None:
            parsed.append(block)
    if len(parsed) <= 1:
        return md
    threshold = max(0.1, SEMANTIC_DEDUP_THRESHOLD - 0.1)
    merged: list[CornellTopicBlock] = []
    for t in parsed:
        match_idx = -1
        for i, existing in enumerate(merged):
            if (
                _topic_similarity(t.title, t.notes, existing.title, existing.notes)
                >= threshold
            ):
                match_idx = i
                break
        if match_idx >= 0:
            merged[match_idx] = _merge_topic_blocks(merged[match_idx], t)
        else:
            merged.append(t)
    progress_log(
        f"Unificación LM-less: {len(parsed)} bloques → {len(merged)} tras "
        f"Jaccard (umbral {threshold:.2f})."
    )
    if not merged:
        return md
    return format_cornell_structured_with_index(
        CornellSummaryStructured(topics=merged), h1_title=h1_title
    )


def _try_unify_assembled(md: str, *, h1_title: str = "Resumen") -> str:
    """Aplica el modo de unificación seleccionado tras el ensamblaje.

    Modos (controlados por ``SUMMARIZER_SUMMARY_UNIFY_MODE``):

    - ``none``: devuelve el Markdown ensamblado sin tocar.
    - ``lmless``: segunda pasada Jaccard, cero llamadas LLM extra.
    - ``hierarchical`` (default): unificación por lotes con LLM.
    - ``aggressive``: hierarchical + pasada final single-pass LLM.
    """
    mode = SUMMARY_UNIFY_MODE
    progress_log(f"Modo de unificación: {mode}")

    if mode == "none":
        return md
    if mode == "lmless":
        try:
            return _lmless_second_pass(md, h1_title=h1_title)
        except Exception as ex:
            progress_log(f"Unificación LM-less fallida (se conserva ensamblaje): {ex}")
            return md

    try:
        ratio = app_state.get_adaptive_prompt_ratio()
        max_budget = max(512, int(app_state.MAX_CONTEXT_TOKENS * ratio))
        if SUMMARY_UNIFY_HIERARCHICAL:
            return _hierarchical_unify_markdown(
                md, h1_title=h1_title, max_budget=max_budget, depth=0
            )
        single = _single_pass_unify_if_fits(
            md, h1_title=h1_title, max_budget=max_budget
        )
        if single is not None:
            return single
        unify_prompt = UNIFY_ASSEMBLED_CORNELL_PROMPT.format(combined=md)
        progress_log(
            "Unificación omitida: el resumen ensamblado ("
            f"{count_tokens(unify_prompt)} tokens) excede el presupuesto "
            f"({max_budget} tokens). Puede activar la unificación jerárquica "
            "(SUMMARIZER_SUMMARY_UNIFY_HIERARCHICAL, por defecto activa) o "
            "desactivar la unificación (SUMMARIZER_SUMMARY_UNIFY_WINDOWS=false)."
        )
        return md
    except Exception as ex:
        progress_log(f"Unificación omitida (se conserva ensamblaje): {ex}")
    return md


def _chat_cornell_structured(user_content: str) -> CornellSummaryStructured:
    return chat_structured_with_retry(
        model=app_state.completion_model,
        messages=[
            {"role": "system", "content": _effective_cornell_system_prompt()},
            {"role": "user", "content": user_content},
        ],
        response_format=CornellSummaryStructured,
    )


def summarize_cornell_single(full_text: str, *, h1_title: str = "Resumen") -> str:
    user_content = f"---\n\n{full_text}"
    return format_cornell_markdown(
        _chat_cornell_structured(user_content), h1_title=h1_title
    )


def _summarize_one_chunk(part: int, total: int, body: str) -> tuple[int, str]:
    wrapped = SUMMARY_CHUNK_WRAPPER.format(part=part, total=total, body=body)
    md = format_cornell_markdown(
        _chat_cornell_structured(wrapped), document_title=False
    )
    return part, md


def summarize_cornell_chunked(
    full_text: str, max_chunk_content_tokens: int, *, h1_title: str = "Resumen"
) -> str:
    check_stop_requested()
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
            check_stop_requested()
            part, md = fut.result()
            partial_by_part[part] = md
    partials = [partial_by_part[i] for i in range(1, total + 1)]
    combined_md = f"# {h1_title} (fragmentos)\n\n" + "\n\n".join(
        f"## Fragmento {i} de {total}\n\n{md}" for i, md in enumerate(partials, start=1)
    )
    unify_user = UNIFY_SUMMARIES_PROMPT.format(combined=combined_md)
    if count_tokens(unify_user) <= app_state.MAX_INPUT_TOKENS:
        try:
            return format_cornell_markdown(
                _chat_cornell_structured(unify_user), h1_title=h1_title
            )
        except Exception:
            return combined_md
    return combined_md


def summarize_document(full_text: str, *, h1_title: str = "Resumen") -> str:
    check_stop_requested()
    single_user = f"---\n\n{full_text}"
    prompt_tokens = count_tokens(_effective_cornell_system_prompt()) + count_tokens(
        single_user
    )
    current_ratio = app_state.get_adaptive_prompt_ratio()
    effective_input_budget = max(512, int(app_state.MAX_CONTEXT_TOKENS * current_ratio))
    progress_log(
        "Presupuesto de prompt (estimado tokenizer local): "
        f"{prompt_tokens} tokens; límite adaptativo: {effective_input_budget} "
        f"({int(current_ratio * 100)}% de contexto)"
    )
    if prompt_tokens <= effective_input_budget:
        try:
            result = summarize_cornell_single(full_text, h1_title=h1_title)
            new_ratio = app_state.record_prompt_ratio_success()
            progress_log(
                f"Resumen OK; elevando ratio adaptativo a {int(new_ratio * 100)}% para próximos documentos."
            )
            return result
        except Exception as ex:
            if not is_context_overflow_error(ex):
                raise
            new_ratio = app_state.record_prompt_ratio_overflow()
            progress_log(
                "Overflow de contexto en resumen único; "
                f"bajando ratio adaptativo a {int(new_ratio * 100)}%."
            )

    wrapper_empty = SUMMARY_CHUNK_WRAPPER.format(part=1, total=1, body="")
    overhead = count_tokens(wrapper_empty)
    max_chunk_content = max(512, effective_input_budget - overhead - 200)
    for _ in range(4):
        try:
            result = summarize_cornell_chunked(
                full_text, max_chunk_content, h1_title=h1_title
            )
            new_ratio = app_state.record_prompt_ratio_success()
            progress_log(
                f"Resumen chunked OK; elevando ratio adaptativo a {int(new_ratio * 100)}%."
            )
            return result
        except Exception as ex:
            if not is_context_overflow_error(ex):
                raise
            new_ratio = app_state.record_prompt_ratio_overflow()
            if max_chunk_content <= 512:
                raise
            max_chunk_content = max(512, max_chunk_content // 2)
            progress_log(
                "Context overflow detectado; reduciendo tamaño de fragmentos a "
                f"~{max_chunk_content} tokens y ratio adaptativo a {int(new_ratio * 100)}%..."
            )
    return summarize_cornell_chunked(full_text, max_chunk_content, h1_title=h1_title)
