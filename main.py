"""Punto de entrada: extracción PDF → resúmenes Cornell → PDF de salida."""

from summarizer.extraction import run_pdf_extraction
from summarizer.lm_studio import configure_lm_studio_model
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
        run_pdf_extraction()
        check_stop_requested()
        run_summarization_pipeline()
    except (StopRequested, KeyboardInterrupt):
        request_stop("Ejecución interrumpida.")
        print("Proceso detenido por el usuario.")
