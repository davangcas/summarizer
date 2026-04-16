"""Resumen estilo Cornell: ventanas por página, troceo y unificación."""

from __future__ import annotations

import json
import os
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
    SUMMARY_PAGE_OVERLAP,
    SUMMARY_MAX_PAGES_PER_WINDOW,
)
from summarizer.fs import atomic_write_text
from summarizer.llm import (
    chat_parse_with_retry,
    completion_parsed_or_validate,
    is_context_overflow_error,
)
from summarizer.markdown_utils import slugify_anchor, split_markdown_by_page_headers
from summarizer.models import CornellSummaryStructured, CornellTopicBlock
from summarizer.prompts import (
    BOOK_CHAPTER_OUTLINE_PREFIX,
    SUMMARY_CHUNK_WRAPPER,
    SUMMARY_CORNELL_USER_PREFIX,
    SUMMARY_WINDOW_WRAPPER,
    UNIFY_SUMMARIES_PROMPT,
)
from summarizer.stop import check_stop_requested
from summarizer.tokenizer import chunk_text_by_tokens, count_tokens


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
    base_tokens = count_tokens(f"{SUMMARY_CORNELL_USER_PREFIX}\n\n{base_wrapper}")
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


def _normalize_topic_title(title: str) -> str:
    return " ".join(title.lower().split())


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
    """Une ventanas en un único Markdown con índice; temas Cornell seguidos sin agrupar por rango de páginas."""
    index_entries: list[tuple[str, str]] = []
    topic_blocks: list[str] = []
    global_i = 0
    used_slugs: set[str] = set()
    last_norm: str | None = None
    seen_global: set[str] = set()

    for win_i, (_start_p, _end_p, structured) in enumerate(ordered):
        if not structured.topics:
            continue
        for t in structured.topics:
            norm = _normalize_topic_title(t.title)
            if ASSEMBLE_DEDUP_BORDER and last_norm is not None and norm == last_norm:
                continue
            if ASSEMBLE_DEDUP_GLOBAL and norm in seen_global:
                continue
            seen_global.add(norm)
            last_norm = norm
            global_i += 1
            slug = slugify_anchor(t.title, fallback=f"tema-{global_i}")
            if slug in used_slugs:
                slug = f"{slug}-w{win_i}-t{global_i}"
            used_slugs.add(slug)
            index_entries.append((t.title, slug))
            topic_blocks.append(_format_cornell_topic_markdown(t, slug=slug))

    if not index_entries:
        return ""

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
    wrapped = SUMMARY_WINDOW_WRAPPER.format(start=start_p, end=end_p, body=body)
    user_content = f"{SUMMARY_CORNELL_USER_PREFIX}\n\n{wrapped}"
    if chapter_outline:
        outline_lines = "\n".join(f"- {c}" for c in chapter_outline)
        user_content += BOOK_CHAPTER_OUTLINE_PREFIX.format(outline_lines=outline_lines)
    completion = chat_parse_with_retry(
        model=app_state.completion_model,
        messages=[{"role": "user", "content": user_content}],
        response_format=CornellSummaryStructured,
    )
    return completion_parsed_or_validate(completion, CornellSummaryStructured)


def summarize_document_paged_windows(
    full_text: str,
    *,
    partials_dir: Path | None = None,
    h1_title: str = "Resumen",
) -> str:
    """Resumen Cornell por ventanas de páginas (máximo fijo por ventana, con fallback 2/1)."""
    pages = split_markdown_by_page_headers(full_text)
    if not pages:
        return ""
    check_stop_requested()

    chapter_outline = chapter_outline_for_summary(full_text)
    if chapter_outline:
        print(
            f"Referencia de capítulos ({len(chapter_outline)} entradas): "
            "entorno o índice detectado en el texto."
        )

    max_pages_per_window = max(1, SUMMARY_MAX_PAGES_PER_WINDOW)
    print(
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
        return ""

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
                print(
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
            print(f"Overflow en página {pnum}; reintentando por fragmentos internos.")
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
    for i, w in enumerate(windows):
        check_stop_requested()
        idx, triple = run_window(i, w)
        results[idx] = triple

    ordered_struct = [results[i] for i in range(len(windows))]
    combined_md_path = (
        (partials_dir / "_combined_windows.md")
        if use_partials and partials_dir
        else None
    )
    md = assemble_cornell_windows_markdown(ordered_struct, h1_title=h1_title)
    if combined_md_path is not None and md.strip():
        atomic_write_text(combined_md_path, md)
    if md.strip():
        app_state.record_prompt_ratio_success()
    return md


def _chat_cornell_structured(user_content: str) -> CornellSummaryStructured:
    completion = chat_parse_with_retry(
        model=app_state.completion_model,
        messages=[{"role": "user", "content": user_content}],
        response_format=CornellSummaryStructured,
    )
    return completion_parsed_or_validate(completion, CornellSummaryStructured)


def summarize_cornell_single(full_text: str, *, h1_title: str = "Resumen") -> str:
    user_content = f"{SUMMARY_CORNELL_USER_PREFIX}\n\n---\n\n{full_text}"
    return format_cornell_markdown(
        _chat_cornell_structured(user_content), h1_title=h1_title
    )


def _summarize_one_chunk(part: int, total: int, body: str) -> tuple[int, str]:
    wrapped = SUMMARY_CHUNK_WRAPPER.format(part=part, total=total, body=body)
    user_content = f"{SUMMARY_CORNELL_USER_PREFIX}\n\n{wrapped}"
    md = format_cornell_markdown(
        _chat_cornell_structured(user_content), document_title=False
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
    single_user = f"{SUMMARY_CORNELL_USER_PREFIX}\n\n---\n\n{full_text}"
    prompt_tokens = count_tokens(single_user)
    current_ratio = app_state.get_adaptive_prompt_ratio()
    effective_input_budget = max(512, int(app_state.MAX_CONTEXT_TOKENS * current_ratio))
    print(
        "Presupuesto de prompt (estimado tokenizer local): "
        f"{prompt_tokens} tokens; límite adaptativo: {effective_input_budget} "
        f"({int(current_ratio * 100)}% de contexto)"
    )
    if prompt_tokens <= effective_input_budget:
        try:
            result = summarize_cornell_single(full_text, h1_title=h1_title)
            new_ratio = app_state.record_prompt_ratio_success()
            print(
                f"Resumen OK; elevando ratio adaptativo a {int(new_ratio * 100)}% para próximos documentos."
            )
            return result
        except Exception as ex:
            if not is_context_overflow_error(ex):
                raise
            new_ratio = app_state.record_prompt_ratio_overflow()
            print(
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
            print(
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
            print(
                "Context overflow detectado; reduciendo tamaño de fragmentos a "
                f"~{max_chunk_content} tokens y ratio adaptativo a {int(new_ratio * 100)}%..."
            )
    return summarize_cornell_chunked(full_text, max_chunk_content, h1_title=h1_title)
