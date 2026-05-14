"""Conteo de tokens y troceado de texto (Hugging Face).

Optimizaciones:
- Cache LRU sobre :func:`count_tokens` para evitar re-tokenizar el mismo
  fragmento (esp. prefijos de prompt repetidos entre ventanas).
- :func:`chunk_text_by_tokens` y :func:`_split_oversized_piece` tokenizan
  cada pieza UNA vez y luego cortan a nivel de IDs, evitando el
  comportamiento cuadrático del enfoque "ir agregando palabras y
  retokenizar el acumulador".
- El ID del tokenizer ganador se persiste en ``.cache/tokenizer_id.txt``
  para acortar el arranque en frío en ejecuciones siguientes.
"""

from __future__ import annotations

import functools
import os
import re
import threading

from transformers import AutoTokenizer

from summarizer import paths
from summarizer.config import TOKENIZER_FALLBACKS

_tokenizer: AutoTokenizer | None = None
_tokenizer_lock = threading.Lock()

_TOKENIZER_CACHE_FILE = paths.PROJECT_ROOT / ".cache" / "tokenizer_id.txt"


def _read_cached_tokenizer_id() -> str | None:
    if not _TOKENIZER_CACHE_FILE.is_file():
        return None
    try:
        value = _TOKENIZER_CACHE_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _write_cached_tokenizer_id(model_id: str) -> None:
    try:
        _TOKENIZER_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _TOKENIZER_CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(model_id, encoding="utf-8")
        os.replace(tmp, _TOKENIZER_CACHE_FILE)
    except OSError:
        pass


def _candidate_tokenizer_ids() -> tuple[str, ...]:
    env_id = os.environ.get("GEMMA_TOKENIZER_ID", "").strip()
    if env_id:
        return (env_id,)
    cached = _read_cached_tokenizer_id()
    if cached:
        rest = tuple(c for c in TOKENIZER_FALLBACKS if c != cached)
        return (cached, *rest)
    return TOKENIZER_FALLBACKS


def get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    with _tokenizer_lock:
        if _tokenizer is None:
            candidates = _candidate_tokenizer_ids()
            last_err: Exception | None = None
            chosen: str | None = None
            for model_id in candidates:
                try:
                    _tokenizer = AutoTokenizer.from_pretrained(model_id)
                    _tokenizer.model_max_length = 10**9
                    chosen = model_id
                    if (
                        model_id == "gpt2"
                        and not os.environ.get("GEMMA_TOKENIZER_ID", "").strip()
                    ):
                        print(
                            "Tokenizer: using gpt2 for token counts (public). "
                            "Set GEMMA_TOKENIZER_ID to a Gemma repo if you use "
                            "huggingface-cli login."
                        )
                    break
                except Exception as ex:
                    last_err = ex
            if _tokenizer is None:
                assert last_err is not None
                raise last_err
            if chosen is not None and _read_cached_tokenizer_id() != chosen:
                _write_cached_tokenizer_id(chosen)
        return _tokenizer


@functools.lru_cache(maxsize=4096)
def _cached_count(text: str) -> int:
    tok = get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


def count_tokens(text: str) -> int:
    """Cuenta tokens del texto con cache LRU.

    Para textos muy cortos o muy largos esquivamos el cache para no
    saturarlo (textos largos ocupan demasiada memoria por entrada).
    """
    if not text:
        return 0
    if len(text) <= 32 or len(text) >= 65536:
        tok = get_tokenizer()
        return len(tok.encode(text, add_special_tokens=False))
    return _cached_count(text)


def _decode_ids(ids: list[int]) -> str:
    tok = get_tokenizer()
    return tok.decode(ids, skip_special_tokens=True)


def _split_oversized_piece(piece: str, max_tokens: int) -> list[str]:
    """Trocea una pieza que excede ``max_tokens``.

    Estrategia incremental:
    1. Si la pieza tiene varias líneas, trocea por líneas acumulando hasta
       el presupuesto. Para líneas que individualmente exceden, recursa.
    2. Si es una sola línea, tokeniza una vez y corta a nivel de IDs.
    """
    tok = get_tokenizer()
    ids = tok.encode(piece, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [piece]

    if "\n" in piece:
        out: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for line in piece.split("\n"):
            line_tokens = len(tok.encode(line, add_special_tokens=False))
            if line_tokens > max_tokens:
                if current:
                    out.append("\n".join(current))
                    current = []
                    current_tokens = 0
                out.extend(_split_oversized_piece(line, max_tokens))
                continue
            if current_tokens + line_tokens > max_tokens and current:
                out.append("\n".join(current))
                current = [line]
                current_tokens = line_tokens
            else:
                current.append(line)
                current_tokens += line_tokens
        if current:
            out.append("\n".join(current))
        return out

    chunks: list[str] = []
    for i in range(0, len(ids), max_tokens):
        chunk_ids = ids[i : i + max_tokens]
        chunks.append(_decode_ids(chunk_ids))
    return chunks


def chunk_text_by_tokens(body: str, max_content_tokens: int) -> list[str]:
    """Divide ``body`` en piezas que codifican como mucho ``max_content_tokens``.

    Itera por párrafos; cada párrafo se tokeniza una sola vez y los acumula
    midiendo IDs (sin retokenizar el acumulador).
    """
    body = body.strip()
    if not body:
        return []
    tok = get_tokenizer()
    total_ids = tok.encode(body, add_special_tokens=False)
    if len(total_ids) <= max_content_tokens:
        return [body]
    parts: list[str] = []
    paragraphs = re.split(r"\n\s*\n", body)
    current_paragraphs: list[str] = []
    current_tokens = 0
    join_ids = len(tok.encode("\n\n", add_special_tokens=False))
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        ptokens = len(tok.encode(p, add_special_tokens=False))
        if ptokens > max_content_tokens:
            if current_paragraphs:
                parts.append("\n\n".join(current_paragraphs))
                current_paragraphs = []
                current_tokens = 0
            parts.extend(_split_oversized_piece(p, max_content_tokens))
            continue
        extra = ptokens if not current_paragraphs else ptokens + join_ids
        if current_paragraphs and current_tokens + extra > max_content_tokens:
            parts.append("\n\n".join(current_paragraphs))
            current_paragraphs = [p]
            current_tokens = ptokens
        else:
            current_paragraphs.append(p)
            current_tokens += extra
    if current_paragraphs:
        parts.append("\n\n".join(current_paragraphs))
    return parts
