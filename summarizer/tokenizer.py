"""Conteo de tokens y troceado de texto (Hugging Face)."""

from __future__ import annotations

import os
import re
import threading

from transformers import AutoTokenizer

from summarizer.config import TOKENIZER_FALLBACKS

_tokenizer: AutoTokenizer | None = None
_tokenizer_lock = threading.Lock()


def get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    with _tokenizer_lock:
        if _tokenizer is None:
            env_id = os.environ.get("GEMMA_TOKENIZER_ID", "").strip()
            candidates: tuple[str, ...] = (env_id,) if env_id else TOKENIZER_FALLBACKS
            last_err: Exception | None = None
            for model_id in candidates:
                try:
                    _tokenizer = AutoTokenizer.from_pretrained(model_id)
                    _tokenizer.model_max_length = 10**9
                    if model_id == "gpt2" and not env_id:
                        print(
                            "Tokenizer: using gpt2 for token counts (public). "
                            "Set GEMMA_TOKENIZER_ID to a Gemma repo if you use huggingface-cli login."
                        )
                    break
                except Exception as ex:
                    last_err = ex
            if _tokenizer is None:
                assert last_err is not None
                raise last_err
        return _tokenizer


def count_tokens(text: str) -> int:
    tok = get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


def _split_oversized_piece(piece: str, max_tokens: int) -> list[str]:
    tok = get_tokenizer()
    if len(tok.encode(piece, add_special_tokens=False)) <= max_tokens:
        return [piece]
    words = piece.split()
    if len(words) == 1:
        step = max(256, max_tokens * 3)
        return [piece[i : i + step] for i in range(0, len(piece), step)]
    lines = piece.split("\n")
    if len(lines) > 1:
        out: list[str] = []
        for line in lines:
            out.extend(_split_oversized_piece(line, max_tokens))
        return out
    words = piece.split()
    if not words:
        return [piece[: max_tokens * 4]]
    chunks: list[str] = []
    current: list[str] = []
    for w in words:
        trial = (" ".join(current + [w])).strip()
        if len(tok.encode(trial, add_special_tokens=False)) <= max_tokens:
            current.append(w)
        else:
            if current:
                chunks.append(" ".join(current))
            current = [w]
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_text_by_tokens(body: str, max_content_tokens: int) -> list[str]:
    """Split body so each piece encodes to at most max_content_tokens (approximate via tokenizer)."""
    body = body.strip()
    if not body:
        return []
    tok = get_tokenizer()
    if len(tok.encode(body, add_special_tokens=False)) <= max_content_tokens:
        return [body]
    parts: list[str] = []
    paragraphs = re.split(r"\n\s*\n", body)
    current = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        trial = (current + "\n\n" + p).strip() if current else p
        if len(tok.encode(trial, add_special_tokens=False)) <= max_content_tokens:
            current = trial
        else:
            if current:
                parts.append(current)
                current = ""
            if len(tok.encode(p, add_special_tokens=False)) <= max_content_tokens:
                current = p
            else:
                parts.extend(_split_oversized_piece(p, max_content_tokens))
    if current:
        parts.append(current)
    return parts
