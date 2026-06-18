"""
pytest configuration – workspace-level fixes
=============================================
Works around a Windows issue where pytest fails to create its temp directory
when the Windows username contains spaces (e.g. "Abdul Salam").  Pytest tries
to create a directory like ``C:\\...\\Temp\\pytest-of-Abdul Salam`` but that
path triggers an Access Denied error on some Windows setups.

The fix: register a ``pytest_configure`` hook that redirects the base temp
directory to a repo-local ``.tmp/pytest`` folder **before** pytest tries to
create the system temp path.
"""

from __future__ import annotations

import hashlib
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest


class DeterministicSentenceTransformer:
    """Offline test double that mimics the subset of sentence-transformers used here."""

    _DIM = 8
    _TOPIC_GROUPS = (
        {"ai", "artificial", "intelligence", "machine", "learning", "ml", "neural", "deep"},
        {"python", "programming", "javascript", "web"},
        {"language", "natural", "text", "processing"},
        {"browser", "browsers"},
        {"data", "science"},
        {"technology"},
        {"models"},
        {"techniques"},
    )

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def get_sentence_embedding_dimension(self) -> int:
        return self._DIM

    def encode(
        self,
        sentences: str | list[str],
        *,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        **_: object,
    ) -> np.ndarray:
        single = isinstance(sentences, str)
        texts = [sentences] if single else sentences
        vectors = np.vstack([self._embed(text) for text in texts]).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
        if single:
            vectors = vectors[0]
        return vectors if convert_to_numpy else vectors.tolist()

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self._DIM, dtype=np.float32)
        for raw_token in text.lower().split():
            token = raw_token.strip(".,;:!?()[]{}\"'")
            if not token:
                continue
            vector[self._topic_bucket(token)] += 1.0
        if not vector.any():
            vector[self._stable_bucket(text)] = 1.0
        return vector

    @classmethod
    def _stable_bucket(cls, text: str) -> int:
        return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % cls._DIM

    @classmethod
    def _topic_bucket(cls, token: str) -> int:
        for bucket, tokens in enumerate(cls._TOPIC_GROUPS):
            if token in tokens:
                return bucket
        return cls._stable_bucket(token)


sentence_transformers: Any = types.ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = DeterministicSentenceTransformer
sys.modules.setdefault("sentence_transformers", sentence_transformers)

# Safe base directory (inside the repo, no spaces)
_SAFE_TMP = Path(__file__).parent / ".tmp" / "pytest"


def pytest_configure(config: pytest.Config) -> None:
    """
    Redirect ``tmp_path`` base to a safe repo-local dir when the default
    pytest-managed temp root would contain a space.

    Spaces in the path cause ``os.mkdir`` to fail with WinError 5 on some
    Windows systems (username with spaces → ``pytest-of-First Last``).
    """
    # Only redirect when not already overridden by the user
    if config.option.basetemp is not None:
        return

    # Detect whether the default tmp root would contain a space
    username = os.environ.get("USERNAME", os.environ.get("USER", ""))
    if " " not in username:
        return  # No problem on this machine

    # Create the safe directory and tell pytest to use it
    _SAFE_TMP.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(_SAFE_TMP)
