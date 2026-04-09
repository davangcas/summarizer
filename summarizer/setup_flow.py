"""Configuración inicial: carpeta de PDFs y preferencia de visión OCR."""

from __future__ import annotations

import os
from pathlib import Path

from summarizer import state as app_state


def configure_source_directory() -> None:
    """Asigna `files_directory` desde la variable de entorno o el diálogo del sistema."""
    env = os.environ.get("SUMMARIZER_FILES_DIRECTORY", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_dir():
            raise SystemExit(
                f"SUMMARIZER_FILES_DIRECTORY no es una carpeta válida: {p}"
            )
        app_state.files_directory = p
        print(f"Carpeta origen (entorno): {app_state.files_directory}")
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
    app_state.files_directory = Path(chosen).resolve()
    print(f"Carpeta origen: {app_state.files_directory}")


def configure_vision_extraction_preference() -> None:
    """
    Pregunta una sola vez si se deben analizar por imagen los PDF sin texto extraíble (escaneados).

    Override sin diálogo: SUMMARIZER_USE_VISION_OCR=1 / true / yes / sí / si → sí;
    =0 / false / no → no.
    """
    raw = os.environ.get("SUMMARIZER_USE_VISION_OCR", "").strip().lower()
    if raw in ("1", "true", "yes", "sí", "si", "y", "on"):
        app_state.use_vision_for_scanned_pdfs = True
        print(
            "Extracción por imágenes (PDF escaneados): sí (SUMMARIZER_USE_VISION_OCR)"
        )
        return
    if raw in ("0", "false", "no", "n", "off"):
        app_state.use_vision_for_scanned_pdfs = False
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
        app_state.use_vision_for_scanned_pdfs = ans in ("s", "sí", "si", "y", "yes")
        print(
            f"Extracción por imágenes: {'sí' if app_state.use_vision_for_scanned_pdfs else 'no'}"
        )
        return

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    app_state.use_vision_for_scanned_pdfs = messagebox.askyesno(
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
        f"{'sí' if app_state.use_vision_for_scanned_pdfs else 'no'}"
    )
