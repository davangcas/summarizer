"""Saneador de fórmulas LaTeX dañadas por colisión de escapes JSON.

Los LLM locales suelen emitir ``\\text``, ``\\frac``, ``\\sqrt`` sin doblar
el backslash al rellenar campos string de JSON. El parser interpreta los pares
``\\t``, ``\\f``, ``\\b``, ``\\v``, ``\\r`` como caracteres de control, lo que
produce salidas como ``$10 \\text{m/s}^2$`` que renderizan ilegibles.

Este módulo reconstruye los comandos dañados aplicando dos pasadas:

1. Restauración global de caracteres de control sueltos a sus secuencias de
   escape de origen (TAB → ``\\t`` literal, etc.).
2. Limpieza dentro de spans matemáticos ``$...$`` y ``$$...$$`` para colapsar
   saltos de línea espurios y reinsertar el backslash de comandos huérfanos
   (``cdot``, ``frac``, ``alpha``, …).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TypeVar

from pydantic import BaseModel

_CTRL_TO_BACKSLASH = {
    "\t": "\\t",
    "\x0c": "\\f",
    "\x08": "\\b",
    "\x0b": "\\v",
    "\r": "\\r",
}

_ORPHAN_MATH_COMMANDS: tuple[str, ...] = (
    "text",
    "frac",
    "sqrt",
    "cdot",
    "times",
    "div",
    "left",
    "right",
    "to",
    "leftarrow",
    "rightarrow",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "varepsilon",
    "zeta",
    "eta",
    "theta",
    "vartheta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "varpi",
    "rho",
    "varrho",
    "sigma",
    "varsigma",
    "tau",
    "upsilon",
    "phi",
    "varphi",
    "chi",
    "psi",
    "omega",
    "Alpha",
    "Beta",
    "Gamma",
    "Delta",
    "Epsilon",
    "Zeta",
    "Eta",
    "Theta",
    "Iota",
    "Kappa",
    "Lambda",
    "Mu",
    "Nu",
    "Xi",
    "Pi",
    "Rho",
    "Sigma",
    "Tau",
    "Upsilon",
    "Phi",
    "Chi",
    "Psi",
    "Omega",
    "sum",
    "int",
    "prod",
    "lim",
    "log",
    "ln",
    "sin",
    "cos",
    "tan",
    "sec",
    "csc",
    "cot",
    "infty",
    "partial",
    "nabla",
    "forall",
    "exists",
    "notin",
    "subset",
    "supset",
    "subseteq",
    "supseteq",
    "cup",
    "cap",
    "leq",
    "geq",
    "neq",
    "approx",
    "equiv",
    "sim",
    "propto",
    "oplus",
    "otimes",
    "mathbb",
    "mathcal",
    "mathrm",
    "mathbf",
    "mathit",
    "vec",
    "hat",
    "bar",
    "tilde",
    "dot",
    "ddot",
    "overline",
    "underline",
    "binom",
    "choose",
    "big",
    "Big",
    "bigg",
    "Bigg",
    "quad",
    "qquad",
)

_MATH_BLOCK_RE = re.compile(r"\$\$([\s\S]+?)\$\$")
# Igual al regex de math_render: el body inline NO puede cruzar saltos de línea
# ni colisionar con un ``$$`` (display math). Sin esta restricción, un ``$``
# aislado en texto (p. ej. una mención de "5 dólares") se empareja con otro
# ``$`` varios párrafos más adelante creando un match gigante que consume
# secciones enteras, lo cual no sólo evita limpiar el span real sino que
# colapsa saltos de línea reales en espacios.
_MATH_INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)([^\$\n]+?)\$(?!\$)")

_NEWLINE_PREFIXED_COMMANDS = (
    "abla",
    "u",
    "otin",
    "eq",
    "eg",
)


def sanitize_math_text(text: str) -> str:
    """Repara LaTeX dañado por escapes JSON en una cadena.

    Si la cadena no contiene caracteres de control ni delimitadores
    matemáticos, se devuelve sin cambios.
    """
    if not text:
        return text
    if not _needs_sanitize(text):
        return text
    text = _restore_control_char_backslashes(text)
    text = _process_math_spans(text)
    return text


def sanitize_model(obj: object) -> object:
    """Aplica :func:`sanitize_math_text` recursivamente a todo string del valor.

    Soporta instancias Pydantic v2, listas, tuplas y dicts. Para BaseModel se
    devuelve una instancia nueva (vía ``model_validate``) sólo si algún campo
    string cambió, evitando trabajo innecesario.
    """
    return _recurse_sanitize(obj)


def _needs_sanitize(text: str) -> bool:
    if any(ctrl in text for ctrl in _CTRL_TO_BACKSLASH):
        return True
    return "$" in text


def _restore_control_char_backslashes(text: str) -> str:
    """Reescribe caracteres de control sueltos como su escape de origen.

    TAB → ``\\t`` literal (dos caracteres: backslash + ``t``), etc. El salto
    de línea (``\\n``) no se toca aquí: se decide caso por caso al limpiar
    los spans matemáticos.
    """
    for ctrl, replacement in _CTRL_TO_BACKSLASH.items():
        if ctrl in text:
            text = text.replace(ctrl, replacement)
    return text


def _process_math_spans(text: str) -> str:
    if "$" not in text:
        return text
    text = _MATH_BLOCK_RE.sub(
        lambda m: "$$" + _clean_math_body(m.group(1)) + "$$", text
    )
    text = _MATH_INLINE_RE.sub(lambda m: "$" + _clean_math_body(m.group(1)) + "$", text)
    return text


def _clean_math_body(body: str) -> str:
    """Limpia el cuerpo de un span matemático.

    Pasos:

    1. Restaura comandos huérfanos prefijados por salto de línea (NL
       consumido como escape) y colapsa los saltos restantes.
    2. Colapsa **over-escapes JSON** (``\\\\cmd`` con dos backslashes
       reales) a ``\\cmd`` con uno solo. matplotlib.mathtext trata ``\\``
       como salto de línea inline lo cual rompe el span; los LLM a veces
       producen ``\\\\\\\\text`` en JSON (4 backslashes) cuando bastaba
       ``\\\\text`` (2). Tras el parser JSON queda ``\\\\text`` (dos
       backslashes reales) en la cadena Python, que es lo que vemos aquí.
    3. Reinserta el backslash en comandos LaTeX comunes que perdieron su
       prefijo por completo (``ext`` → ``\\text``, etc.).
    """
    for suffix in _NEWLINE_PREFIXED_COMMANDS:
        body = re.sub(
            r"\n" + re.escape(suffix) + r"(?![A-Za-z])",
            r"\\n" + suffix,
            body,
        )
    body = body.replace("\n", " ")
    # Colapsa `\\<cmd>` (dos backslashes reales) → `\<cmd>` (uno) cuando
    # `<cmd>` es uno de los comandos conocidos. Restringido a la lista
    # blanca para no romper el uso legítimo de `\\` como salto de línea
    # dentro de matrices/aligned (que de todos modos caen al fallback de
    # code-fence cuando matplotlib no las soporta).
    for cmd in _ORPHAN_MATH_COMMANDS:
        body = re.sub(
            r"\\\\" + re.escape(cmd) + r"(?![A-Za-z])",
            r"\\" + cmd,
            body,
        )
    body = re.sub(r"(?<!\\)(?<!\w)ext\s*\{", r"\\text{", body)
    body = re.sub(r"(?<!\\)(?<!\w)rac\s*\{", r"\\frac{", body)
    body = re.sub(r"(?<!\\)(?<!\w)inom\s*\{", r"\\binom{", body)
    for cmd in _ORPHAN_MATH_COMMANDS:
        body = re.sub(
            r"(?<!\\)(?<!\w)" + re.escape(cmd) + r"(?![A-Za-z])",
            r"\\" + cmd,
            body,
        )
    return body


T = TypeVar("T", bound=BaseModel)


def _recurse_sanitize(value):
    if isinstance(value, str):
        return sanitize_math_text(value)
    if isinstance(value, BaseModel):
        return _sanitize_basemodel(value)
    if isinstance(value, list):
        return [_recurse_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_recurse_sanitize(v) for v in value)
    if isinstance(value, dict):
        return {k: _recurse_sanitize(v) for k, v in value.items()}
    return value


def _sanitize_basemodel(model: BaseModel) -> BaseModel:
    changed = False
    new_data: dict[str, object] = {}
    for name in _iter_field_names(model):
        original = getattr(model, name)
        cleaned = _recurse_sanitize(original)
        if cleaned is not original and cleaned != original:
            changed = True
        new_data[name] = cleaned
    if not changed:
        return model
    return model.__class__.model_validate(new_data)


def _iter_field_names(model: BaseModel) -> Iterable[str]:
    fields = getattr(model.__class__, "model_fields", None)
    if fields is not None:
        return list(fields.keys())
    return list(model.__dict__.keys())
