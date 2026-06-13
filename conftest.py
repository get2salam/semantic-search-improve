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

import os
from pathlib import Path

import numpy as np
import pytest

# Safe base directory (inside the repo, no spaces)
_SAFE_TMP = Path(__file__).parent / ".tmp" / "pytest"


class _DeterministicSentenceTransformer:
    """Tiny offline embedding model for tests.

    CI should not depend on downloading Hugging Face model files. The test
    suite only needs stable, topic-aware vectors, so this fake preserves the
    semantic-search behaviours the assertions cover while staying fully
    deterministic and network-free.
    """

    _DIM = 16
    _TOPIC_TERMS = {
        0: {"ai", "artificial", "intelligence", "machine", "learning", "ml", "model", "models"},
        1: {"deep", "neural", "network", "networks", "training"},
        2: {"python", "programming", "language", "code"},
        3: {"javascript", "web", "browser", "browsers", "frontend", "react", "vue", "frameworks"},
        4: {"natural", "processing", "text", "understanding"},
        5: {
            "pasta",
            "pizza",
            "italian",
            "tomato",
            "sauce",
            "dough",
            "cooking",
            "recipes",
            "cuisine",
        },
        6: {"data", "science", "techniques"},
        7: {"hello", "test", "document", "throwaway"},
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", *args, **kwargs) -> None:
        self.model_name = model_name

    def get_sentence_embedding_dimension(self) -> int:
        return self._DIM

    def encode(
        self,
        sentences,
        *,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        **kwargs,
    ):
        if isinstance(sentences, str):
            single = True
            items = [sentences]
        else:
            single = False
            items = list(sentences)

        vectors = np.vstack([self._embed(text) for text in items]).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
        if single:
            vectors = vectors[0]
        if convert_to_numpy:
            return vectors
        return vectors.tolist()

    @classmethod
    def _embed(cls, text: str) -> np.ndarray:
        vec = np.zeros(cls._DIM, dtype=np.float32)
        words = {word.strip(".,:;!?()[]{}\"'").lower() for word in text.split()}
        for dim, terms in cls._TOPIC_TERMS.items():
            vec[dim] = float(len(words & terms))
        # Add a small deterministic lexical signal so unrelated docs are not
        # all exact ties, without overpowering the topic dimensions.
        for word in words:
            vec[8 + (sum(ord(ch) for ch in word) % 8)] += 0.01
        if not np.any(vec):
            vec[-1] = 1.0
        return vec


def pytest_configure(config: pytest.Config) -> None:
    """
    Redirect ``tmp_path`` base to a safe repo-local dir when the default
    pytest-managed temp root would contain a space.

    Spaces in the path cause ``os.mkdir`` to fail with WinError 5 on some
    Windows systems (username with spaces → ``pytest-of-First Last``).
    """
    import semantic_search

    semantic_search.SentenceTransformer = _DeterministicSentenceTransformer
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
