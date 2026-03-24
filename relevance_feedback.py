"""
Relevance Feedback with Rocchio Algorithm
==========================================
Implements query refinement via the Rocchio algorithm to improve retrieval
quality using explicit or pseudo-relevance feedback signals.

The Rocchio algorithm adjusts a query embedding toward relevant documents
and away from non-relevant ones:

.. code-block:: text

    q' = alpha * q
       + beta  * (1/|R|) * Σ(r ∈ R)  r
       - gamma * (1/|N|) * Σ(n ∈ N)  n

where

* *q*  is the original query vector,
* *R*  is the set of relevant document vectors,
* *N*  is the set of non-relevant document vectors, and
* *alpha*, *beta*, *gamma* are Rocchio weighting parameters (typical
  defaults: 1.0 / 0.75 / 0.15).

The updated query *q'* is L2-normalised so it can be used directly with
cosine-similarity retrievers.

Architecture::

    RelevanceFeedbackStore ← user clicks / explicit labels
               ↓
    RocchioFeedback  ───► refined query embedding
               ↓
    FeedbackAwareDenseRetriever ───► improved ranked results

Usage::

    from relevance_feedback import (
        RocchioFeedback,
        RelevanceFeedbackStore,
        FeedbackAwareDenseRetriever,
    )

    store = RelevanceFeedbackStore()
    store.add_session("coffee health", relevant=["doc1", "doc3"], non_relevant=["doc7"])

    retriever = FeedbackAwareDenseRetriever()
    retriever.index(corpus)
    results = retriever.search_with_feedback(
        "coffee health benefits", store, top_k=10
    )

References:
    - Rocchio, J. (1971). Relevance feedback in information retrieval.
      In G. Salton (Ed.), *The SMART Retrieval System*. Prentice Hall.
    - Manning, Raghavan & Schütze (2008). *Introduction to Information
      Retrieval*. Cambridge University Press. Chapter 9.

Author: get2salam
License: MIT
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "RocchioFeedback",
    "FeedbackSession",
    "RelevanceFeedbackStore",
    "FeedbackAwareDenseRetriever",
]


# ---------------------------------------------------------------------------
# Rocchio core
# ---------------------------------------------------------------------------


class RocchioFeedback:
    """
    Rocchio relevance feedback for embedding-space query expansion.

    Parameters
    ----------
    alpha:
        Weight for the original query vector (default ``1.0``).
        Higher values preserve the original query direction.
    beta:
        Weight for the centroid of relevant document vectors (default
        ``0.75``).  Higher values pull the query toward relevant docs.
    gamma:
        Weight for the centroid of non-relevant document vectors (default
        ``0.15``).  Higher values push the query away from irrelevant docs.
        Set to ``0.0`` to disable negative feedback.
    normalize:
        If ``True`` (default), L2-normalise the refined vector so it stays
        on the unit hypersphere.  Required when using cosine similarity.

    Examples
    --------
    >>> import numpy as np
    >>> rf = RocchioFeedback(alpha=1.0, beta=0.75, gamma=0.15)
    >>> q = np.array([1.0, 0.0, 0.0])
    >>> relevant = np.array([[0.9, 0.4, 0.0], [0.8, 0.6, 0.0]])
    >>> refined = rf.refine(q, relevant_vectors=relevant)
    >>> refined.shape
    (3,)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.75,
        gamma: float = 0.15,
        normalize: bool = True,
    ) -> None:
        if alpha < 0 or beta < 0 or gamma < 0:
            raise ValueError("alpha, beta, and gamma must be non-negative.")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.normalize = normalize

    # ------------------------------------------------------------------

    def refine(
        self,
        query_vector: np.ndarray,
        relevant_vectors: np.ndarray | None = None,
        non_relevant_vectors: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the Rocchio-refined query vector.

        Parameters
        ----------
        query_vector:
            1-D numpy array representing the original query embedding.
        relevant_vectors:
            2-D array of shape ``(n_relevant, dim)`` with embeddings of
            documents marked as relevant.  May be ``None`` or empty.
        non_relevant_vectors:
            2-D array of shape ``(n_non_relevant, dim)`` with embeddings
            of documents marked as non-relevant.  May be ``None`` or empty.

        Returns
        -------
        np.ndarray
            Refined query vector of the same dimensionality as
            *query_vector*.  L2-normalised when ``self.normalize=True``.

        Raises
        ------
        ValueError
            If *query_vector* is not 1-D or its dimension does not match
            the provided document vectors.
        """
        q = np.asarray(query_vector, dtype=float)
        if q.ndim != 1:
            raise ValueError(f"query_vector must be 1-D, got shape {q.shape}")

        refined = self.alpha * q

        if relevant_vectors is not None and len(relevant_vectors) > 0:
            rv = np.asarray(relevant_vectors, dtype=float)
            if rv.ndim != 2 or rv.shape[1] != q.shape[0]:
                raise ValueError(
                    f"relevant_vectors shape {rv.shape} incompatible with "
                    f"query_vector dim {q.shape[0]}"
                )
            centroid_r = rv.mean(axis=0)
            refined += self.beta * centroid_r

        if non_relevant_vectors is not None and len(non_relevant_vectors) > 0:
            nv = np.asarray(non_relevant_vectors, dtype=float)
            if nv.ndim != 2 or nv.shape[1] != q.shape[0]:
                raise ValueError(
                    f"non_relevant_vectors shape {nv.shape} incompatible with "
                    f"query_vector dim {q.shape[0]}"
                )
            centroid_n = nv.mean(axis=0)
            refined -= self.gamma * centroid_n

        if self.normalize:
            norm = np.linalg.norm(refined)
            if norm > 0:
                refined = refined / norm

        return refined

    def pseudo_refine(
        self,
        query_vector: np.ndarray,
        top_k_vectors: np.ndarray,
        pseudo_relevant_k: int = 3,
    ) -> np.ndarray:
        """
        Pseudo-relevance feedback: assume the top-*k* retrieved docs are relevant.

        Parameters
        ----------
        query_vector:
            Original query embedding (1-D).
        top_k_vectors:
            Embeddings of the top-*k* retrieved documents, shape
            ``(k, dim)``.
        pseudo_relevant_k:
            Number of top documents to treat as pseudo-relevant.  Must be
            ``<= len(top_k_vectors)``.

        Returns
        -------
        np.ndarray
            Refined query vector.
        """
        k = min(pseudo_relevant_k, len(top_k_vectors))
        if k == 0:
            return np.asarray(query_vector, dtype=float)
        return self.refine(
            query_vector,
            relevant_vectors=np.asarray(top_k_vectors[:k], dtype=float),
        )

    def __repr__(self) -> str:
        return f"RocchioFeedback(alpha={self.alpha}, beta={self.beta}, gamma={self.gamma})"


# ---------------------------------------------------------------------------
# Feedback session
# ---------------------------------------------------------------------------


@dataclass
class FeedbackSession:
    """
    A single feedback interaction for a query.

    Attributes
    ----------
    query:
        The original user query.
    relevant:
        List of document strings (or IDs) marked as relevant.
    non_relevant:
        List of document strings (or IDs) marked as non-relevant.
    timestamp:
        Unix timestamp when the session was created.
    session_id:
        Optional identifier for grouping sessions.
    """

    query: str
    relevant: list[str] = field(default_factory=list)
    non_relevant: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    session_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "relevant": self.relevant,
            "non_relevant": self.non_relevant,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FeedbackSession:
        return cls(
            query=data["query"],
            relevant=data.get("relevant", []),
            non_relevant=data.get("non_relevant", []),
            timestamp=data.get("timestamp", 0.0),
            session_id=data.get("session_id"),
        )


# ---------------------------------------------------------------------------
# Feedback store
# ---------------------------------------------------------------------------


class RelevanceFeedbackStore:
    """
    Persistent store of user relevance feedback sessions.

    Maintains a history of feedback that can be replayed during retrieval
    to adapt query vectors at inference time.

    Parameters
    ----------
    max_sessions_per_query:
        Maximum feedback sessions to keep per query string.  Oldest
        sessions are evicted when the limit is exceeded.  Set to ``0``
        for unlimited.

    Examples
    --------
    >>> store = RelevanceFeedbackStore()
    >>> store.add_session("contract law", relevant=["doc1"], non_relevant=["doc4"])
    >>> store.get_feedback("contract law")
    (['doc1'], ['doc4'])
    """

    def __init__(self, max_sessions_per_query: int = 10) -> None:
        self.max_sessions_per_query = max_sessions_per_query
        self._sessions: dict[str, list[FeedbackSession]] = {}

    # ------------------------------------------------------------------

    def add_session(
        self,
        query: str,
        relevant: list[str] | None = None,
        non_relevant: list[str] | None = None,
        session_id: str | None = None,
    ) -> FeedbackSession:
        """
        Record a feedback session.

        Parameters
        ----------
        query:
            The query string this feedback applies to.
        relevant:
            Documents the user found relevant.
        non_relevant:
            Documents the user found irrelevant.
        session_id:
            Optional session identifier.

        Returns
        -------
        FeedbackSession
            The newly created session.
        """
        session = FeedbackSession(
            query=query,
            relevant=list(relevant or []),
            non_relevant=list(non_relevant or []),
            session_id=session_id,
        )

        if query not in self._sessions:
            self._sessions[query] = []

        self._sessions[query].append(session)

        # Evict oldest sessions beyond limit
        if self.max_sessions_per_query > 0:
            while len(self._sessions[query]) > self.max_sessions_per_query:
                self._sessions[query].pop(0)

        logger.debug(
            "Feedback recorded for query '%s': +%d / -%d",
            query,
            len(session.relevant),
            len(session.non_relevant),
        )
        return session

    def get_sessions(self, query: str) -> list[FeedbackSession]:
        """Return all feedback sessions for *query*."""
        return list(self._sessions.get(query, []))

    def get_feedback(
        self,
        query: str,
        last_n: int = 0,
    ) -> tuple[list[str], list[str]]:
        """
        Aggregate feedback across sessions for *query*.

        Parameters
        ----------
        query:
            Query string to look up.
        last_n:
            If > 0, only consider the most recent *n* sessions.

        Returns
        -------
        tuple[list[str], list[str]]
            ``(relevant_docs, non_relevant_docs)`` with duplicates removed.
            A document that appears in both is treated as relevant.
        """
        sessions = self._sessions.get(query, [])
        if last_n > 0:
            sessions = sessions[-last_n:]

        relevant: set[str] = set()
        non_relevant: set[str] = set()
        for s in sessions:
            relevant.update(s.relevant)
            non_relevant.update(s.non_relevant)

        # Docs confirmed relevant override any negative signal
        non_relevant -= relevant

        return sorted(relevant), sorted(non_relevant)

    def num_queries(self) -> int:
        """Return the number of distinct queries with feedback."""
        return len(self._sessions)

    def total_sessions(self) -> int:
        """Return the total number of feedback sessions."""
        return sum(len(v) for v in self._sessions.values())

    def clear(self, query: str | None = None) -> None:
        """
        Clear feedback.

        Parameters
        ----------
        query:
            If given, clear only sessions for that query.
            Otherwise, clear all feedback.
        """
        if query is not None:
            self._sessions.pop(query, None)
        else:
            self._sessions.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the store to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            query: [s.to_dict() for s in sessions] for query, sessions in self._sessions.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d feedback queries to %s", len(data), path)

    def load(self, path: str | Path) -> int:
        """
        Load sessions from a JSON file (merges with existing data).

        Returns
        -------
        int
            Number of sessions loaded.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Feedback store file not found: %s", path)
            return 0

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        loaded = 0
        for _query, session_list in data.items():
            for sd in session_list:
                self.add_session(
                    query=sd["query"],
                    relevant=sd.get("relevant", []),
                    non_relevant=sd.get("non_relevant", []),
                    session_id=sd.get("session_id"),
                )
                loaded += 1

        logger.info("Loaded %d feedback sessions from %s", loaded, path)
        return loaded

    def __len__(self) -> int:
        return self.total_sessions()

    def __repr__(self) -> str:
        return (
            f"RelevanceFeedbackStore("
            f"queries={self.num_queries()}, sessions={self.total_sessions()})"
        )


# ---------------------------------------------------------------------------
# Feedback-aware dense retriever
# ---------------------------------------------------------------------------


class FeedbackAwareDenseRetriever:
    """
    Dense retriever with Rocchio relevance feedback loop.

    Wraps a sentence-transformer model and applies the Rocchio algorithm
    to refine query embeddings using stored feedback before retrieval.

    Parameters
    ----------
    model_name:
        Sentence-transformer model name.
    rocchio:
        A :class:`RocchioFeedback` instance.  Defaults to standard
        parameters (alpha=1.0, beta=0.75, gamma=0.15).
    normalize:
        L2-normalise corpus embeddings.

    Examples
    --------
    >>> retriever = FeedbackAwareDenseRetriever()
    >>> retriever.index(["deep learning tutorial", "python basics", "neural nets"])
    >>> store = RelevanceFeedbackStore()
    >>> store.add_session("AI", relevant=["deep learning tutorial"])
    >>> results = retriever.search_with_feedback("AI", store, top_k=2)
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        rocchio: RocchioFeedback | None = None,
        normalize: bool = True,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.rocchio = rocchio or RocchioFeedback()
        self.normalize = normalize
        self.batch_size = batch_size

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._corpus: list[str] = []
        self._embeddings: np.ndarray | None = None

    # ------------------------------------------------------------------

    def index(self, documents: list[str]) -> None:
        """
        Encode and index *documents*.

        Replaces any previously indexed content.
        """
        self._corpus = list(documents)
        if documents:
            self._embeddings = self._model.encode(
                documents,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        else:
            self._embeddings = None

    # ------------------------------------------------------------------

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a query string to an embedding vector."""
        emb = self._model.encode(
            query,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return np.asarray(emb, dtype=float)

    def _embeddings_for_docs(self, doc_ids: list[str]) -> np.ndarray:
        """
        Retrieve stored embeddings for documents by their text content.

        Documents not found in the index are silently skipped.
        """
        if self._embeddings is None:
            return np.empty((0, 0))

        indices = []
        corpus_set = {doc: i for i, doc in enumerate(self._corpus)}
        for doc in doc_ids:
            if doc in corpus_set:
                indices.append(corpus_set[doc])

        if not indices:
            return np.empty((0, self._embeddings.shape[1]))

        return self._embeddings[indices]

    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Standard dense search (no feedback).

        Returns
        -------
        list[tuple[str, float]]
            ``(document, score)`` pairs sorted by descending cosine similarity.
        """
        if self._embeddings is None or not self._corpus:
            return []

        q_emb = self._encode_query(query).reshape(1, -1)
        scores = np.dot(self._embeddings, q_emb.T).flatten()
        k = min(top_k, len(self._corpus))
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self._corpus[i], float(scores[i])) for i in top_idx]

    def search_with_feedback(
        self,
        query: str,
        store: RelevanceFeedbackStore,
        top_k: int = 10,
        last_n_sessions: int = 0,
    ) -> list[tuple[str, float]]:
        """
        Dense search with Rocchio feedback applied.

        Looks up stored feedback for *query*, refines the query embedding
        using :class:`RocchioFeedback`, then performs retrieval.

        Parameters
        ----------
        query:
            User query string.
        store:
            Feedback store to look up relevance signals from.
        top_k:
            Number of results to return.
        last_n_sessions:
            If > 0, only use the most recent *n* feedback sessions.

        Returns
        -------
        list[tuple[str, float]]
            ``(document, score)`` pairs sorted by descending similarity
            with the refined query.
        """
        if self._embeddings is None or not self._corpus:
            return []

        q_emb = self._encode_query(query)

        relevant_docs, non_relevant_docs = store.get_feedback(query, last_n=last_n_sessions)

        rv = self._embeddings_for_docs(relevant_docs)
        nv = self._embeddings_for_docs(non_relevant_docs)

        refined_q = self.rocchio.refine(
            q_emb,
            relevant_vectors=rv if len(rv) > 0 else None,
            non_relevant_vectors=nv if len(nv) > 0 else None,
        )

        scores = np.dot(self._embeddings, refined_q.reshape(1, -1).T).flatten()
        k = min(top_k, len(self._corpus))
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self._corpus[i], float(scores[i])) for i in top_idx]

    def search_pseudo_feedback(
        self,
        query: str,
        top_k: int = 10,
        pseudo_k: int = 3,
    ) -> list[tuple[str, float]]:
        """
        Pseudo-relevance feedback: assume the top-*pseudo_k* initial results
        are relevant, then re-rank with the refined query.

        Parameters
        ----------
        query:
            User query string.
        top_k:
            Number of results to return in the final ranking.
        pseudo_k:
            Number of top documents to treat as pseudo-relevant for
            query expansion.

        Returns
        -------
        list[tuple[str, float]]
            ``(document, score)`` pairs.
        """
        if self._embeddings is None or not self._corpus:
            return []

        q_emb = self._encode_query(query)

        # First pass — retrieve initial candidates
        scores_first = np.dot(self._embeddings, q_emb.reshape(1, -1).T).flatten()
        k_candidates = min(max(pseudo_k, top_k), len(self._corpus))
        top_idx_first = np.argsort(scores_first)[::-1][:k_candidates]
        pseudo_embeddings = self._embeddings[top_idx_first[:pseudo_k]]

        # Rocchio expansion
        refined_q = self.rocchio.pseudo_refine(
            q_emb,
            top_k_vectors=pseudo_embeddings,
            pseudo_relevant_k=pseudo_k,
        )

        # Second pass — re-rank
        scores_second = np.dot(self._embeddings, refined_q.reshape(1, -1).T).flatten()
        k = min(top_k, len(self._corpus))
        top_idx = np.argsort(scores_second)[::-1][:k]
        return [(self._corpus[i], float(scores_second[i])) for i in top_idx]

    def __len__(self) -> int:
        return len(self._corpus)

    def __repr__(self) -> str:
        return (
            f"FeedbackAwareDenseRetriever("
            f"model='{self.model_name}', "
            f"rocchio={self.rocchio!r}, "
            f"docs={len(self)})"
        )
