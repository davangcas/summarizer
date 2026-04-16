"""Punto de entrada: extracción de documentos → resúmenes Cornell → PDF final."""

import shutil

from summarizer import paths
from summarizer.extraction import run_document_extraction
from summarizer.lm_studio import configure_lm_studio_model
from summarizer.output import summary_pdfs_output_dir
from summarizer.pipeline import run_summarization_pipeline
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
        run_document_extraction()
        sync_ocr_pdfs_to_summary_destination()
        check_stop_requested()
        run_summarization_pipeline()
    except (StopRequested, KeyboardInterrupt):
        request_stop("Ejecución interrumpida.")
        print("Proceso detenido por el usuario.")
