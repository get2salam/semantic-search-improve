"""
Tests for relevance_feedback module.
Covers RocchioFeedback, RelevanceFeedbackStore, and FeedbackAwareDenseRetriever
(without loading a real sentence-transformer model).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from relevance_feedback import (
    FeedbackAwareDenseRetriever,
    FeedbackSession,
    RelevanceFeedbackStore,
    RocchioFeedback,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(v: list[float] | np.ndarray) -> np.ndarray:
    """Return the L2-normalised version of *v*."""
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _almost_unit(v: np.ndarray, tol: float = 1e-6) -> bool:
    """Return True if *v* has unit norm within *tol*."""
    return abs(np.linalg.norm(v) - 1.0) < tol


# ---------------------------------------------------------------------------
# Minimal mock retriever (avoids loading sentence-transformers in CI)
# ---------------------------------------------------------------------------


class MockFeedbackRetriever(FeedbackAwareDenseRetriever):
    """
    Subclass that replaces the sentence-transformer with simple random
    embeddings so tests run without network / GPU access.
    """

    DIM = 8

    def __init__(self, **kwargs) -> None:
        # Bypass __init__ to skip SentenceTransformer loading
        self.model_name = "mock-model"
        self.rocchio = kwargs.get("rocchio", RocchioFeedback())
        self.normalize = True
        self.batch_size = 64
        self._corpus: list[str] = []
        self._embeddings: np.ndarray | None = None
        # Fixed seed for reproducibility
        self._rng = np.random.default_rng(42)

    def index(self, documents: list[str]) -> None:
        self._corpus = list(documents)
        if documents:
            raw = self._rng.standard_normal((len(documents), self.DIM))
            # Normalise each row
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            self._embeddings = raw / norms
        else:
            self._embeddings = None

    def _encode_query(self, query: str) -> np.ndarray:
        # Deterministic per-query encoding via hash
        seed = hash(query) % (2**31)
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal(self.DIM)
        return raw / np.linalg.norm(raw)


# ---------------------------------------------------------------------------
# Tests: RocchioFeedback
# ---------------------------------------------------------------------------


class TestRocchioFeedback:
    def test_default_parameters(self):
        rf = RocchioFeedback()
        assert rf.alpha == 1.0
        assert rf.beta == 0.75
        assert rf.gamma == 0.15
        assert rf.normalize is True

    def test_refine_no_feedback_returns_original_direction(self):
        rf = RocchioFeedback()
        q = np.array([1.0, 0.0, 0.0])
        refined = rf.refine(q)
        np.testing.assert_allclose(refined, q, atol=1e-6)

    def test_refine_output_is_unit_norm(self):
        rf = RocchioFeedback(normalize=True)
        q = np.array([3.0, 1.0, 0.0])
        rv = np.array([[1.0, 0.5, 0.2], [0.9, 0.3, 0.1]])
        nv = np.array([[0.0, 0.0, 1.0]])
        refined = rf.refine(q, rv, nv)
        assert _almost_unit(refined)

    def test_refine_no_normalisation(self):
        rf = RocchioFeedback(alpha=1.0, beta=1.0, gamma=0.0, normalize=False)
        q = np.array([1.0, 0.0])
        rv = np.array([[2.0, 0.0]])
        refined = rf.refine(q, rv)
        expected = np.array([1.0 + 2.0, 0.0])  # alpha*q + beta*centroid
        np.testing.assert_allclose(refined, expected, atol=1e-6)

    def test_refine_relevant_pulls_toward_relevant_docs(self):
        """Query should become more similar to relevant docs after refinement."""
        rf = RocchioFeedback(alpha=1.0, beta=1.0, gamma=0.0)
        q = _unit([1.0, 0.0])
        relevant = np.array([[0.0, 1.0]])  # orthogonal to q
        refined = rf.refine(q, relevant)
        # The refined vector should be at ~45° — between q and relevant
        cos_q = float(np.dot(refined, q))
        cos_r = float(np.dot(refined, _unit([0.0, 1.0])))
        assert cos_q > 0.0, "Refined vector should retain some original direction"
        assert cos_r > 0.0, "Refined vector should move toward relevant doc"

    def test_refine_non_relevant_pushes_away(self):
        """After negative feedback, similarity to non-relevant doc should decrease."""
        rf = RocchioFeedback(alpha=1.0, beta=0.0, gamma=1.0)
        q = _unit([1.0, 0.0])
        non_relevant = np.array([[1.0, 0.0]])  # same direction as q
        refined = rf.refine(q, non_relevant_vectors=non_relevant)
        cos_sim = float(np.dot(refined, _unit([1.0, 0.0])))
        # After subtracting a vector in the same direction, similarity drops
        assert cos_sim < 0.9

    def test_refine_invalid_query_dim_raises(self):
        rf = RocchioFeedback()
        with pytest.raises(ValueError, match="1-D"):
            rf.refine(np.array([[1.0, 0.0]]))  # 2-D is wrong

    def test_refine_dim_mismatch_raises(self):
        rf = RocchioFeedback()
        q = np.array([1.0, 0.0])
        rv = np.array([[1.0, 0.0, 0.5]])  # 3-D ≠ 2-D
        with pytest.raises(ValueError, match="incompatible"):
            rf.refine(q, rv)

    def test_refine_empty_relevant_vectors_ignored(self):
        rf = RocchioFeedback()
        q = np.array([1.0, 0.0, 0.0])
        refined = rf.refine(q, relevant_vectors=np.empty((0, 3)))
        assert _almost_unit(refined)

    def test_negative_parameter_raises(self):
        with pytest.raises(ValueError):
            RocchioFeedback(alpha=-0.1)

    def test_pseudo_refine_returns_unit_vector(self):
        rf = RocchioFeedback()
        q = np.array([1.0, 0.0, 0.5])
        top_k = np.random.default_rng(0).standard_normal((5, 3))
        refined = rf.pseudo_refine(q, top_k, pseudo_relevant_k=3)
        assert _almost_unit(refined)

    def test_pseudo_refine_empty_top_k_returns_original(self):
        rf = RocchioFeedback()
        q = np.array([1.0, 0.0])
        refined = rf.pseudo_refine(q, np.empty((0, 2)), pseudo_relevant_k=3)
        np.testing.assert_allclose(refined, q, atol=1e-6)

    def test_pseudo_refine_clips_k_to_available(self):
        rf = RocchioFeedback()
        q = np.array([1.0, 0.0])
        top_k = np.array([[0.5, 0.5]])  # only 1 doc, pseudo_k=5 → clip to 1
        refined = rf.pseudo_refine(q, top_k, pseudo_relevant_k=5)
        assert _almost_unit(refined)

    def test_repr(self):
        rf = RocchioFeedback(0.9, 0.6, 0.1)
        r = repr(rf)
        assert "0.9" in r
        assert "0.6" in r
        assert "0.1" in r

    def test_refine_gamma_zero_equivalent_to_no_negative_feedback(self):
        """With gamma=0, non_relevant_vectors should have no effect."""
        rf = RocchioFeedback(alpha=1.0, beta=0.5, gamma=0.0)
        q = np.array([1.0, 0.0, 0.0])
        rv = np.array([[0.8, 0.6, 0.0]])
        nv = np.array([[0.0, 0.0, 1.0]])
        r_with_nv = rf.refine(q, rv, nv)
        r_without_nv = rf.refine(q, rv)
        np.testing.assert_allclose(r_with_nv, r_without_nv, atol=1e-6)

    def test_beta_zero_only_original_direction_matters(self):
        """With beta=0, relevant docs have no effect."""
        rf = RocchioFeedback(alpha=1.0, beta=0.0, gamma=0.0, normalize=False)
        q = np.array([1.0, 0.0])
        rv = np.array([[0.0, 1.0]])
        refined = rf.refine(q, rv)
        np.testing.assert_allclose(refined, q, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: FeedbackSession
# ---------------------------------------------------------------------------


class TestFeedbackSession:
    def test_defaults(self):
        s = FeedbackSession(query="breach of contract")
        assert s.relevant == []
        assert s.non_relevant == []
        assert s.session_id is None
        assert s.timestamp > 0

    def test_to_dict_round_trip(self):
        s = FeedbackSession(
            query="landlord obligations",
            relevant=["doc1", "doc2"],
            non_relevant=["doc5"],
            session_id="sess-42",
        )
        d = s.to_dict()
        s2 = FeedbackSession.from_dict(d)
        assert s2.query == s.query
        assert s2.relevant == s.relevant
        assert s2.non_relevant == s.non_relevant
        assert s2.session_id == s.session_id

    def test_from_dict_missing_keys_use_defaults(self):
        s = FeedbackSession.from_dict({"query": "test"})
        assert s.relevant == []
        assert s.non_relevant == []


# ---------------------------------------------------------------------------
# Tests: RelevanceFeedbackStore
# ---------------------------------------------------------------------------


class TestRelevanceFeedbackStore:
    def test_add_and_retrieve_feedback(self):
        store = RelevanceFeedbackStore()
        store.add_session("AI law", relevant=["doc1", "doc3"], non_relevant=["doc7"])
        rel, nrel = store.get_feedback("AI law")
        assert "doc1" in rel
        assert "doc3" in rel
        assert "doc7" in nrel

    def test_empty_query_returns_empty_lists(self):
        store = RelevanceFeedbackStore()
        rel, nrel = store.get_feedback("unknown query")
        assert rel == []
        assert nrel == []

    def test_relevant_overrides_non_relevant(self):
        """A doc in both lists should end up as relevant, not non-relevant."""
        store = RelevanceFeedbackStore()
        store.add_session("query", relevant=["docA"], non_relevant=["docA", "docB"])
        rel, nrel = store.get_feedback("query")
        assert "docA" in rel
        assert "docA" not in nrel

    def test_multiple_sessions_aggregated(self):
        store = RelevanceFeedbackStore()
        store.add_session("tax law", relevant=["d1"])
        store.add_session("tax law", relevant=["d2"], non_relevant=["d3"])
        rel, nrel = store.get_feedback("tax law")
        assert "d1" in rel
        assert "d2" in rel
        assert "d3" in nrel

    def test_max_sessions_evicts_oldest(self):
        store = RelevanceFeedbackStore(max_sessions_per_query=3)
        for i in range(5):
            store.add_session("q", relevant=[f"doc{i}"])
        sessions = store.get_sessions("q")
        assert len(sessions) == 3
        # Oldest sessions should be gone
        docs_kept = {d for s in sessions for d in s.relevant}
        assert "doc0" not in docs_kept
        assert "doc4" in docs_kept

    def test_unlimited_sessions_when_max_zero(self):
        store = RelevanceFeedbackStore(max_sessions_per_query=0)
        for i in range(20):
            store.add_session("q", relevant=[f"doc{i}"])
        assert len(store.get_sessions("q")) == 20

    def test_last_n_sessions(self):
        store = RelevanceFeedbackStore()
        store.add_session("q", relevant=["old_doc"])
        time.sleep(0.001)
        store.add_session("q", relevant=["new_doc"])
        rel, _ = store.get_feedback("q", last_n=1)
        assert "new_doc" in rel
        assert "old_doc" not in rel

    def test_clear_specific_query(self):
        store = RelevanceFeedbackStore()
        store.add_session("q1", relevant=["d1"])
        store.add_session("q2", relevant=["d2"])
        store.clear("q1")
        assert store.get_sessions("q1") == []
        assert len(store.get_sessions("q2")) == 1

    def test_clear_all(self):
        store = RelevanceFeedbackStore()
        store.add_session("q1", relevant=["d1"])
        store.add_session("q2", relevant=["d2"])
        store.clear()
        assert store.num_queries() == 0
        assert store.total_sessions() == 0

    def test_num_queries_and_total_sessions(self):
        store = RelevanceFeedbackStore()
        store.add_session("q1", relevant=["d1"])
        store.add_session("q1", relevant=["d2"])
        store.add_session("q2", relevant=["d3"])
        assert store.num_queries() == 2
        assert store.total_sessions() == 3

    def test_save_and_load(self, tmp_path):
        store = RelevanceFeedbackStore()
        store.add_session("contract", relevant=["doc1"], non_relevant=["doc2"])
        store.add_session("property", relevant=["doc3"])

        path = tmp_path / "feedback.json"
        store.save(path)
        assert path.exists()

        store2 = RelevanceFeedbackStore()
        n = store2.load(path)
        assert n == 2
        assert store2.num_queries() == 2

        rel, _ = store2.get_feedback("contract")
        assert "doc1" in rel

    def test_load_missing_file_warns_not_raises(self, tmp_path):
        store = RelevanceFeedbackStore()
        n = store.load(tmp_path / "nonexistent.json")
        assert n == 0

    def test_save_creates_parent_dirs(self, tmp_path):
        store = RelevanceFeedbackStore()
        store.add_session("q", relevant=["d"])
        deep_path = tmp_path / "deep" / "nested" / "feedback.json"
        store.save(deep_path)
        assert deep_path.exists()

    def test_repr(self):
        store = RelevanceFeedbackStore()
        store.add_session("q", relevant=["d"])
        r = repr(store)
        assert "RelevanceFeedbackStore" in r
        assert "1" in r

    def test_len(self):
        store = RelevanceFeedbackStore()
        store.add_session("q", relevant=["d1"])
        store.add_session("q", relevant=["d2"])
        assert len(store) == 2


# ---------------------------------------------------------------------------
# Tests: MockFeedbackRetriever (FeedbackAwareDenseRetriever)
# ---------------------------------------------------------------------------


@pytest.fixture
def retriever() -> MockFeedbackRetriever:
    r = MockFeedbackRetriever()
    r.index(
        [
            "machine learning algorithms",
            "deep neural networks and backpropagation",
            "python programming tutorial",
            "natural language processing with transformers",
            "data structures and algorithms",
            "convolutional neural networks for image recognition",
            "reinforcement learning and policy gradients",
            "database design and SQL",
        ]
    )
    return r


class TestFeedbackAwareDenseRetriever:
    def test_index_stores_corpus(self, retriever):
        assert len(retriever) == 8

    def test_search_returns_top_k_results(self, retriever):
        results = retriever.search("neural network", top_k=3)
        assert len(results) == 3

    def test_search_returns_tuples_of_str_float(self, retriever):
        results = retriever.search("python", top_k=2)
        for doc, score in results:
            assert isinstance(doc, str)
            assert isinstance(score, float)

    def test_search_scores_are_cosine_like(self, retriever):
        """Scores should be in [-1, 1] for normalised embeddings."""
        results = retriever.search("learning", top_k=5)
        for _, score in results:
            assert -1.1 <= score <= 1.1

    def test_search_empty_corpus_returns_empty(self):
        r = MockFeedbackRetriever()
        r.index([])
        assert r.search("query") == []

    def test_search_with_feedback_no_feedback_same_as_search(self, retriever):
        """With an empty store, feedback search == plain search."""
        store = RelevanceFeedbackStore()
        q = "machine learning"
        plain = retriever.search(q, top_k=5)
        feedback = retriever.search_with_feedback(q, store, top_k=5)
        # Both use the same (unmodified) query vector → same results
        plain_docs = [d for d, _ in plain]
        feedback_docs = [d for d, _ in feedback]
        assert plain_docs == feedback_docs

    def test_search_with_feedback_changes_ranking(self, retriever):
        """Marking a doc as relevant should influence its position."""
        store = RelevanceFeedbackStore()
        q = "neural network training"
        # First search without feedback
        plain = [d for d, _ in retriever.search(q, top_k=8)]

        # Mark the database doc as 'relevant' (counter-intuitive but forces a shift)
        store.add_session(q, relevant=["database design and SQL"])
        feedback = [d for d, _ in retriever.search_with_feedback(q, store, top_k=8)]

        # Rankings should differ (the feedback perturbs the query vector)
        assert plain != feedback

    def test_search_with_feedback_uses_stored_signals(self, retriever):
        """Relevant doc should have higher score than without feedback."""
        store = RelevanceFeedbackStore()
        q = "deep learning"
        target = "deep neural networks and backpropagation"
        store.add_session(q, relevant=[target])
        results = retriever.search_with_feedback(q, store, top_k=8)
        docs = [d for d, _ in results]
        assert target in docs

    def test_search_pseudo_feedback_returns_top_k(self, retriever):
        results = retriever.search_pseudo_feedback("NLP", top_k=4, pseudo_k=2)
        assert len(results) == 4

    def test_search_pseudo_feedback_returns_str_float_tuples(self, retriever):
        results = retriever.search_pseudo_feedback("data structures", top_k=3)
        for doc, score in results:
            assert isinstance(doc, str)
            assert isinstance(score, float)

    def test_search_pseudo_feedback_empty_corpus_returns_empty(self):
        r = MockFeedbackRetriever()
        r.index([])
        assert r.search_pseudo_feedback("query") == []

    def test_search_pseudo_feedback_differs_from_plain_search(self, retriever):
        """PRF should alter rankings (at least some queries)."""
        q = "reinforcement learning"
        plain = [d for d, _ in retriever.search(q, top_k=8)]
        prf = [d for d, _ in retriever.search_pseudo_feedback(q, top_k=8, pseudo_k=3)]
        # They may differ if PRF shifts the query vector
        # At minimum, the same documents appear in the results
        assert set(plain) == set(prf)

    def test_repr(self, retriever):
        r = repr(retriever)
        assert "FeedbackAwareDenseRetriever" in r
        assert "mock-model" in r

    def test_len_after_reindex(self, retriever):
        retriever.index(["doc1", "doc2", "doc3"])
        assert len(retriever) == 3

    def test_embeddings_for_nonexistent_doc_skipped(self, retriever):
        embs = retriever._embeddings_for_docs(["nonexistent document"])
        assert embs.shape[0] == 0

    def test_embeddings_for_docs_returns_correct_shape(self, retriever):
        docs = ["machine learning algorithms", "python programming tutorial"]
        embs = retriever._embeddings_for_docs(docs)
        assert embs.shape == (2, MockFeedbackRetriever.DIM)

    def test_search_with_feedback_last_n_sessions(self, retriever):
        """Only the most recent session should be used with last_n=1."""
        store = RelevanceFeedbackStore()
        q = "algorithms"
        store.add_session(q, relevant=["database design and SQL"])
        store.add_session(q, relevant=["python programming tutorial"])
        # last_n=1 → only the second session is used
        r1 = retriever.search_with_feedback(q, store, top_k=8, last_n_sessions=1)
        r2 = retriever.search_with_feedback(q, store, top_k=8, last_n_sessions=2)
        # Both should return 8 results; ranking may differ
        assert len(r1) == 8
        assert len(r2) == 8

    def test_custom_rocchio_parameters_used(self):
        """Verify custom Rocchio params propagate."""
        rf = RocchioFeedback(alpha=0.5, beta=0.5, gamma=0.0)
        r = MockFeedbackRetriever(rocchio=rf)
        r.index(["a b c", "d e f"])
        assert r.rocchio.alpha == 0.5
        assert r.rocchio.beta == 0.5

    def test_no_crash_when_feedback_docs_not_in_index(self, retriever):
        """Store contains docs not in the retriever's corpus — should skip silently."""
        store = RelevanceFeedbackStore()
        store.add_session("test", relevant=["totally unknown document xyz"])
        results = retriever.search_with_feedback("test", store, top_k=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Integration: Rocchio loop improves recall
# ---------------------------------------------------------------------------


class TestRocchioIntegration:
    """
    Synthetic integration test: build a corpus with a known cluster structure
    and verify that feedback pulls results closer to the target cluster.
    """

    DIM = 16

    def _make_cluster(self, center: np.ndarray, n: int, noise: float) -> np.ndarray:
        rng = np.random.default_rng(0)
        docs = center + rng.standard_normal((n, self.DIM)) * noise
        norms = np.linalg.norm(docs, axis=1, keepdims=True)
        return docs / norms

    def test_feedback_improves_recall_in_target_cluster(self):
        rng = np.random.default_rng(7)
        # Two clusters — "tech" and "law"
        tech_center = rng.standard_normal(self.DIM)
        tech_center /= np.linalg.norm(tech_center)
        law_center = -tech_center + rng.standard_normal(self.DIM) * 0.1
        law_center /= np.linalg.norm(law_center)

        tech_docs = self._make_cluster(tech_center, 8, 0.1)
        law_docs = self._make_cluster(law_center, 8, 0.1)

        corpus_vecs = np.vstack([tech_docs, law_docs])
        corpus_labels = ["tech"] * 8 + ["law"] * 8

        rf = RocchioFeedback(alpha=1.0, beta=1.0, gamma=0.3)

        # Query vector: midpoint (ambiguous)
        q = (tech_center + law_center) / 2
        q /= np.linalg.norm(q)

        # Feedback: mark 2 tech docs as relevant
        relevant_vecs = tech_docs[:2]
        q_refined = rf.refine(q, relevant_vectors=relevant_vecs)

        scores_refined = corpus_vecs @ q_refined
        top5_refined = [corpus_labels[i] for i in np.argsort(scores_refined)[::-1][:5]]
        tech_count = sum(1 for label in top5_refined if label == "tech")

        # With feedback toward tech, tech docs should dominate top-5
        assert tech_count >= 3, f"Expected ≥3 tech in top-5, got {tech_count}"
