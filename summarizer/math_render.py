"""Pre-render de fórmulas LaTeX a imágenes PNG via matplotlib.mathtext.

El motor de PDF (``markdown_pdf`` → PyMuPDF Story) no soporta matemáticas
nativamente. Este módulo detecta los spans ``$...$`` y ``$$...$$`` del
Markdown, renderiza cada fórmula a un PNG con ``matplotlib.mathtext`` y los
reemplaza por enlaces de imagen Markdown estándar (``![](ruta.png)``), que
PyMuPDF Story embebe sin problemas.

El renderer es puro Python, sin runtimes externos (Cairo/Pango, pandoc,
LaTeX), portable Windows/macOS/Linux. matplotlib.mathtext implementa un
subconjunto amplio de LaTeX: ``\\frac``, ``\\sqrt``, super/subíndices,
símbolos griegos, ``\\text``, integrales y sumas, etc.; no cubre paquetes
externos ni entornos como ``align``.
"""

from __future__ import annotations

import hashlib
import re
import urllib.parse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from summarizer.config import MATH_RENDER_DPI
from summarizer.progress import progress_log

_MATH_BLOCK_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_MATH_INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)([^\$\n]+?)\$(?!\$)")
_INLINE_BODY_HINT_RE = re.compile(r"[\\^_{}]")


def render_math_to_png(
    latex: str,
    *,
    inline: bool,
    out_dir: Path,
    dpi: int = MATH_RENDER_DPI,
) -> Path | None:
    """Renderiza ``latex`` a PNG y devuelve la ruta resultante.

    Idempotente: el nombre del archivo es ``math_<sha256[:24]>.png``, por lo
    que llamadas repetidas con la misma expresión reutilizan el cache en
    disco. Devuelve ``None`` si ``matplotlib.mathtext`` no puede parsear la
    expresión.
    """
    if not latex.strip():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        f"{'i' if inline else 'b'}:{dpi}:{latex}".encode("utf-8")
    ).hexdigest()[:24]
    out_path = out_dir / f"math_{key}.png"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    fontsize = 11 if inline else 16
    fig = Figure(figsize=(0.1, 0.1))
    FigureCanvasAgg(fig)
    try:
        fig.text(0.0, 0.0, f"${latex}$", fontsize=fontsize)
        fig.savefig(
            str(out_path),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.05,
            transparent=False,
            facecolor="white",
        )
    except Exception as ex:
        progress_log(f"math render fallido para '{latex[:60]}': {ex}")
        try:
            if out_path.exists():
                out_path.unlink()
        except OSError:
            pass
        return None
    return out_path


def replace_math_with_images(
    markdown: str,
    out_dir: Path,
    *,
    base_dir: Path | None = None,
) -> str:
    """Sustituye spans ``$$...$$`` y ``$...$`` por ``![](ruta_png)``.

    Ignora los bloques de código (entre ``\\`\\`\\```). Para spans inline
    sin pistas de LaTeX (``\\``, ``^``, ``_``, ``{``, ``}``), se preserva el
    texto literal para evitar capturar símbolos de moneda como ``$5 y $10``.

    Cuando ``base_dir`` se pasa, las URLs emitidas son **relativas** a esa
    carpeta y URL-encoded. Esto es requerido por ``fitz.Story`` (motor de
    PDF de ``markdown_pdf``), que sólo embebe imágenes vía su ``Archive``
    cuando la ruta es relativa a ``Section.root``. Si ``base_dir`` es
    ``None`` se emite la ruta absoluta URL-encoded (útil para tests y
    consumidores que renderizan Markdown con motores que sí soporten paths
    absolutos).
    """
    if "$" not in markdown:
        return markdown
    return _apply_outside_fences(
        markdown,
        lambda chunk: _replace_math_in_chunk(chunk, out_dir, base_dir),
    )


def _replace_math_in_chunk(markdown: str, out_dir: Path, base_dir: Path | None) -> str:
    markdown = _MATH_BLOCK_RE.sub(
        lambda m: _to_image_md_block(m.group(1).strip(), out_dir, base_dir),
        markdown,
    )
    markdown = _MATH_INLINE_RE.sub(
        lambda m: _to_image_md_inline(
            m.group(1).strip(), m.group(0), out_dir, base_dir
        ),
        markdown,
    )
    return markdown


def _to_image_md_block(latex: str, out_dir: Path, base_dir: Path | None) -> str:
    if not latex:
        return "$$$$"
    rendered = render_math_to_png(latex, inline=False, out_dir=out_dir)
    if rendered is None:
        return _block_latex_fallback(latex)
    # NO inyectar `\n\n` alrededor: el LLM a veces mete ``$$...$$`` en
    # medio de un list item (``  - $$x$$  - resto``); si agregamos blanks
    # se rompe la estructura del list y markdown_it interpreta líneas
    # adyacentes como setext heading, lo cual produce un h2 espurio que
    # revienta ``fitz.Story.set_toc`` con ``bad hierarchy level``. Si el
    # LLM ya separó el bloque con líneas en blanco propias, el replacement
    # mantiene el aire alrededor.
    return f"![]({_encode_image_url(rendered, base_dir)})"


def _to_image_md_inline(
    latex: str, original: str, out_dir: Path, base_dir: Path | None
) -> str:
    if not latex:
        return original
    if not _INLINE_BODY_HINT_RE.search(latex):
        return original
    rendered = render_math_to_png(latex, inline=True, out_dir=out_dir)
    if rendered is None:
        return _inline_latex_fallback(latex)
    return f"![]({_encode_image_url(rendered, base_dir)})"


def _encode_image_url(rendered: Path, base_dir: Path | None) -> str:
    """Devuelve una URL CommonMark-segura para la imagen.

    - Si ``base_dir`` se pasa: ruta relativa a ``base_dir`` en POSIX,
      URL-encoded (espacios, paréntesis, etc.). Esta forma es la que
      ``fitz.Story`` resuelve correctamente vía su ``Archive`` cuando
      ``Section.root`` apunta a ``base_dir``.
    - Si ``base_dir`` es ``None``: ruta absoluta POSIX URL-encoded.
    - Si la ruta no es subcarpeta de ``base_dir`` (caso raro): fallback a
      absoluta URL-encoded.
    """
    if base_dir is not None:
        try:
            rel = rendered.relative_to(base_dir)
            return urllib.parse.quote(rel.as_posix(), safe="/")
        except ValueError:
            pass
    return urllib.parse.quote(rendered.as_posix(), safe="/:")


def _block_latex_fallback(latex: str) -> str:
    """Fallback para fórmulas de bloque que matplotlib no soporta.

    Las matrices (``\\begin{pmatrix}``), entornos como ``align`` o paquetes
    no incluidos quedan ilegibles si los metemos como texto inline. Los
    devolvemos como un bloque de código fenced para que se vean bien
    formateados (monospace, multilínea) y al menos quede recuperable el
    LaTeX original.
    """
    return f"\n\n```latex\n{latex.strip()}\n```\n\n"


def _inline_latex_fallback(latex: str) -> str:
    """Fallback para fórmulas inline. Si tienen saltos de línea (matriz mal
    cerrada) las promovemos a bloque; si no, las dejamos en backticks."""
    if "\n" in latex or len(latex) > 80:
        return _block_latex_fallback(latex)
    return f"`{latex}`"


def _apply_outside_fences(markdown: str, transform) -> str:
    """Aplica ``transform`` a las regiones del Markdown fuera de fences ``\\`\\`\\```."""
    lines = markdown.splitlines(keepends=True)
    out: list[str] = []
    in_fence = False
    buffer: list[str] = []
    for line in lines:
        if line.lstrip().startswith("```"):
            if buffer:
                out.append(
                    transform("".join(buffer)) if not in_fence else "".join(buffer)
                )
                buffer = []
            out.append(line)
            in_fence = not in_fence
            continue
        buffer.append(line)
    if buffer:
        out.append(transform("".join(buffer)) if not in_fence else "".join(buffer))
    return "".join(out)
