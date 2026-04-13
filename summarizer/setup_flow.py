"""Configuración inicial: carpeta de PDFs y preferencia de visión OCR."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from summarizer import paths
from summarizer import state as app_state


def _common_ancestor_directory(paths: list[Path]) -> Path:
    if len(paths) == 1:
        return paths[0].parent
    try:
        common = os.path.commonpath([str(p.resolve()) for p in paths])
    except ValueError as e:
        raise SystemExit(
            "Los PDF seleccionados deben compartir una ruta base común (p. ej. misma unidad)."
        ) from e
    return Path(common)


def _split_env_path_list(raw: str) -> list[str]:
    """Lista de rutas en una variable de entorno: ``;`` explícito o ``os.pathsep`` (``:`` en Unix, ``;`` en Windows)."""
    s = raw.strip()
    if not s:
        return []
    if ";" in s:
        return [p.strip() for p in s.split(";") if p.strip()]
    parts = [p.strip() for p in s.split(os.pathsep) if p.strip()]
    return parts if parts else [s]


def _parse_env_pdf_files_list(raw: str) -> list[Path]:
    out: list[Path] = []
    for s in _split_env_path_list(raw):
        p = Path(s).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"SUMMARIZER_PDF_FILES: no es un archivo válido: {p}")
        if p.suffix.lower() != ".pdf":
            raise SystemExit(f"SUMMARIZER_PDF_FILES: se esperaba .pdf: {p}")
        out.append(p)
    if not out:
        raise SystemExit("SUMMARIZER_PDF_FILES no contiene rutas válidas.")
    return out


def _configure_source_directory_gui() -> None:
    """Ventana con dos acciones: carpeta (todos los PDF recursivos) o archivos concretos."""
    import tkinter as tk
    from tkinter import filedialog, ttk

    choice: dict[str, str | None] = {"kind": None}

    root = tk.Tk()
    root.title("Origen de los PDF")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    frm = ttk.Frame(root, padding=20)
    frm.pack()

    ttk.Label(
        frm,
        text=(
            "Carpeta: se procesan todos los .pdf dentro (subcarpetas incluidas).\n"
            "Archivos: solo los PDF que elija."
        ),
        justify=tk.CENTER,
    ).pack(pady=(0, 14))

    def on_folder() -> None:
        choice["kind"] = "folder"
        root.quit()

    def on_files() -> None:
        choice["kind"] = "files"
        root.quit()

    def on_cancel() -> None:
        choice["kind"] = None
        root.quit()

    btn_f = ttk.Button(
        frm,
        text="Elegir carpeta…",
        command=on_folder,
    )
    btn_f.pack(fill=tk.X, pady=4)
    btn_a = ttk.Button(
        frm,
        text="Elegir archivos PDF…",
        command=on_files,
    )
    btn_a.pack(fill=tk.X, pady=4)
    ttk.Button(frm, text="Cancelar", command=on_cancel).pack(fill=tk.X, pady=(10, 0))

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.update_idletasks()
    w, h = root.winfo_reqwidth(), root.winfo_reqheight()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 3}")

    root.mainloop()
    root.destroy()

    kind = choice["kind"]
    if kind is None:
        raise SystemExit("Operación cancelada.")

    dlg = tk.Tk()
    dlg.withdraw()
    dlg.attributes("-topmost", True)
    try:
        if kind == "folder":
            chosen = filedialog.askdirectory(
                title="Seleccione la carpeta con los PDF a procesar",
                parent=dlg,
            )
            if not chosen:
                raise SystemExit("No se seleccionó ninguna carpeta.")
            app_state.files_directory = Path(chosen).resolve()
            app_state.source_pdf_paths = None
            print(
                f"Carpeta origen (todos los PDF en el árbol): {app_state.files_directory}"
            )
            return

        file_tuples = filedialog.askopenfilenames(
            title="Seleccione uno o varios archivos PDF",
            filetypes=[("PDF", "*.pdf"), ("Todos", "*.*")],
            parent=dlg,
        )
        if not file_tuples:
            raise SystemExit("No se seleccionó ningún archivo.")
        pdfs = [Path(f).resolve() for f in file_tuples]
        app_state.files_directory = _common_ancestor_directory(pdfs)
        app_state.source_pdf_paths = frozenset(pdfs)
        print(f"Archivos PDF seleccionados: {len(app_state.source_pdf_paths)}")
        print(f"Carpeta base derivada: {app_state.files_directory}")
    finally:
        dlg.destroy()


def configure_source_directory() -> None:
    """
    Asigna ``files_directory`` y opcionalmente ``source_pdf_paths``.

    Entorno:
    - ``SUMMARIZER_PDF_FILES``: uno o varios PDF; separador ``;`` o, en Unix,
      ``:`` (``os.pathsep``). Se deriva la carpeta base y solo se procesan esos archivos.
    - ``SUMMARIZER_FILES_DIRECTORY``: carpeta; se procesan todos los ``.pdf`` bajo ella (recursivo).
    """
    env_files = os.environ.get("SUMMARIZER_PDF_FILES", "").strip()
    if env_files:
        pdfs = _parse_env_pdf_files_list(env_files)
        app_state.files_directory = _common_ancestor_directory(pdfs)
        app_state.source_pdf_paths = frozenset(p.resolve() for p in pdfs)
        print(f"PDF indicados (entorno): {len(app_state.source_pdf_paths)} archivo(s)")
        print(f"Carpeta base derivada: {app_state.files_directory}")
        return

    env = os.environ.get("SUMMARIZER_FILES_DIRECTORY", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_dir():
            raise SystemExit(
                f"SUMMARIZER_FILES_DIRECTORY no es una carpeta válida: {p}"
            )
        app_state.files_directory = p
        app_state.source_pdf_paths = None
        print(f"Carpeta origen (entorno): {app_state.files_directory}")
        return
    if importlib.util.find_spec("tkinter") is None:
        raise SystemExit(
            "No se pudo cargar tkinter para elegir origen. "
            "Instale tk o defina SUMMARIZER_FILES_DIRECTORY o SUMMARIZER_PDF_FILES."
        )
    _configure_source_directory_gui()


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


def _configure_summary_pdfs_destination_gui() -> None:
    """Ventana: carpeta predeterminada del proyecto u otra carpeta."""
    import tkinter as tk
    from tkinter import filedialog, ttk

    choice: dict[str, str | None] = {"action": None}

    root = tk.Tk()
    root.title("Destino de los PDF de resumen")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    frm = ttk.Frame(root, padding=20)
    frm.pack()

    default_txt = str(paths.summary_pdfs.resolve())
    ttk.Label(
        frm,
        text=(
            "Los PDF finales (resúmenes) se pueden guardar en la carpeta\n"
            "del proyecto o en la ruta que elija."
        ),
        justify=tk.CENTER,
    ).pack(pady=(0, 8))
    ttk.Label(
        frm,
        text=f"Predeterminada:\n{default_txt}",
        justify=tk.CENTER,
        wraplength=420,
    ).pack(pady=(0, 14))

    def on_default() -> None:
        choice["action"] = "default"
        root.quit()

    def on_pick() -> None:
        choice["action"] = "pick"
        root.quit()

    def on_cancel() -> None:
        choice["action"] = None
        root.quit()

    ttk.Button(
        frm,
        text="Usar carpeta predeterminada",
        command=on_default,
    ).pack(fill=tk.X, pady=4)
    ttk.Button(
        frm,
        text="Elegir otra carpeta…",
        command=on_pick,
    ).pack(fill=tk.X, pady=4)
    ttk.Button(frm, text="Cancelar", command=on_cancel).pack(fill=tk.X, pady=(10, 0))

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.update_idletasks()
    w, h = root.winfo_reqwidth(), root.winfo_reqheight()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 3}")

    root.mainloop()
    root.destroy()

    act = choice["action"]
    if act is None:
        raise SystemExit("Operación cancelada.")
    if act == "default":
        app_state.summary_pdfs_directory = None
        print(
            f"Carpeta destino PDF de resumen (predeterminada): {paths.summary_pdfs.resolve()}"
        )
        return

    dlg = tk.Tk()
    dlg.withdraw()
    dlg.attributes("-topmost", True)
    try:
        chosen = filedialog.askdirectory(
            title="Seleccione la carpeta para los PDF de resumen",
            parent=dlg,
        )
        if not chosen:
            raise SystemExit("No se seleccionó ninguna carpeta.")
        app_state.summary_pdfs_directory = Path(chosen).resolve()
        print(f"Carpeta destino PDF de resumen: {app_state.summary_pdfs_directory}")
    finally:
        dlg.destroy()


def configure_summary_pdfs_destination() -> None:
    """
    Define ``summary_pdfs_directory`` (o None para la carpeta del proyecto).

    Entorno:
    - ``SUMMARIZER_SUMMARY_PDF_DIRECTORY``: ruta absoluta o relativa; debe ser
      una carpeta existente o creable (se crean subcarpetas al escribir).
    - ``SUMMARIZER_USE_DEFAULT_SUMMARY_PDF_DIR=1`` / ``true`` / ``yes`` / ``sí`` / ``si``:
      fuerza la carpeta predeterminada del proyecto (sin diálogo).
    """
    raw_default = (
        os.environ.get("SUMMARIZER_USE_DEFAULT_SUMMARY_PDF_DIR", "").strip().lower()
    )
    if raw_default in ("1", "true", "yes", "sí", "si", "y", "on"):
        app_state.summary_pdfs_directory = None
        print(
            "Carpeta destino PDF de resumen (predeterminada, entorno): "
            f"{paths.summary_pdfs.resolve()}"
        )
        return

    env = os.environ.get("SUMMARIZER_SUMMARY_PDF_DIRECTORY", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        app_state.summary_pdfs_directory = p
        print(f"Carpeta destino PDF de resumen (entorno): {p}")
        return

    if importlib.util.find_spec("tkinter") is None:
        app_state.summary_pdfs_directory = None
        print(
            "Sin tkinter: se usa la carpeta predeterminada para PDF de resumen: "
            f"{paths.summary_pdfs.resolve()}"
        )
        return
    _configure_summary_pdfs_destination_gui()
