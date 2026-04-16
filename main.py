"""Punto de entrada: extracción de documentos → resúmenes Cornell → PDF final."""

import shutil

from summarizer import paths
from summarizer.extraction import collect_pending_sources, run_document_extraction
from summarizer.lm_studio import configure_lm_studio_model
from summarizer.output import summary_pdfs_output_dir
from summarizer.pipeline import (
    collect_completed_md_for_summary,
    run_summarization_pipeline,
)
from summarizer.progress import (
    close_global_progress,
    get_global_progress,
    init_global_progress,
    progress_log,
)
from summarizer.setup_flow import (
    configure_source_directory,
    configure_summary_pdfs_destination,
    configure_vision_extraction_preference,
)
from summarizer.stop import (
    StopRequested,
    check_stop_requested,
    install_stop_handlers,
    request_stop,
    start_stop_listener,
)
from summarizer.tokenizer import get_tokenizer


def sync_ocr_pdfs_to_summary_destination() -> None:
    """Copia PDFs OCR de texto completo al destino de resúmenes bajo OCR_PDFS."""
    src_root = paths.completed_texts_ocr
    if not src_root.is_dir():
        return
    dst_root = summary_pdfs_output_dir() / "OCR_PDFS"
    for src_pdf in src_root.rglob("*.pdf"):
        rel = src_pdf.relative_to(src_root)
        dst_pdf = dst_root / rel
        dst_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_pdf, dst_pdf)


if __name__ == "__main__":
    install_stop_handlers()
    try:
        configure_source_directory()
        configure_summary_pdfs_destination()
        configure_vision_extraction_preference()
        configure_lm_studio_model()
        get_tokenizer()
        start_stop_listener()
        check_stop_requested()
        pending_sources = collect_pending_sources()
        if pending_sources:
            init_global_progress(len(pending_sources), desc="Progreso global")
        run_document_extraction(pending_sources=pending_sources)
        sync_ocr_pdfs_to_summary_destination()
        check_stop_requested()
        summary_targets = collect_completed_md_for_summary()
        if summary_targets:
            progress = get_global_progress()
            if progress is None:
                init_global_progress(len(summary_targets), desc="Progreso global")
            else:
                progress.add_total(len(summary_targets))
        run_summarization_pipeline(file_paths=summary_targets)
    except (StopRequested, KeyboardInterrupt):
        request_stop("Ejecución interrumpida.")
        progress_log("Proceso detenido por el usuario.")
    finally:
        close_global_progress()
