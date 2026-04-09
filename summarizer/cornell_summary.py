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
from summarizer.config import (
    ASSEMBLE_DEDUP_BORDER,
    MAX_PARALLEL_CHUNKS,
    MAX_PARALLEL_WINDOW_SUMMARIES,
    SUMMARY_PAGE_OVERLAP,
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
        slug = slugify_anchor(t.title, fallback=f"tema-{i + 1}")
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
    current_ratio = app_state.get_adaptive_prompt_ratio()
    return max(512, int(app_state.MAX_CONTEXT_TOKENS * current_ratio))


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
    return f"# Resumen\n\n## Índice\n\n{index_lines}\n\n---\n\n{body}"


def _strict_one_page_from_env() -> bool:
    raw = os.environ.get("SUMMARIZER_SUMMARY_STRICT_ONE_PAGE", "").strip().lower()
    return raw in ("1", "true", "yes", "sí", "si", "on")


def _chat_cornell_window(
    start_p: int, end_p: int, body: str
) -> CornellSummaryStructured:
    wrapped = SUMMARY_WINDOW_WRAPPER.format(start=start_p, end=end_p, body=body)
    user_content = f"{SUMMARY_CORNELL_USER_PREFIX}\n\n{wrapped}"
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
    check_stop_requested()

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
        try:
            structured = _chat_cornell_window(sp, ep, body)
        except BadRequestError as ex:
            if is_context_overflow_error(ex):
                app_state.record_prompt_ratio_overflow()
            raise
        if part_path is not None:
            save_window_checkpoint(
                part_path, start_p=sp, end_p=ep, body=body, structured=structured
            )
        return task_i, (sp, ep, structured)

    workers = min(MAX_PARALLEL_WINDOW_SUMMARIES, len(windows))
    results: dict[int, tuple[int, int, CornellSummaryStructured]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(run_window, i, w) for i, w in enumerate(windows)]
        for fut in as_completed(futs):
            check_stop_requested()
            i, triple = fut.result()
            results[i] = triple

    ordered_struct = [results[i] for i in range(len(windows))]
    combined_md_path = (
        (partials_dir / "_combined_windows.md")
        if use_partials and partials_dir
        else None
    )
    md = assemble_cornell_windows_markdown(ordered_struct)
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
    combined_md = "# Resumen (fragmentos)\n\n" + "\n\n".join(
        f"## Fragmento {i} de {total}\n\n{md}" for i, md in enumerate(partials, start=1)
    )
    unify_user = UNIFY_SUMMARIES_PROMPT.format(combined=combined_md)
    if count_tokens(unify_user) <= app_state.MAX_INPUT_TOKENS:
        try:
            return format_cornell_markdown(_chat_cornell_structured(unify_user))
        except Exception:
            return combined_md
    return combined_md


def summarize_document(full_text: str) -> str:
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
            result = summarize_cornell_single(full_text)
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
            result = summarize_cornell_chunked(full_text, max_chunk_content)
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
    return summarize_cornell_chunked(full_text, max_chunk_content)
