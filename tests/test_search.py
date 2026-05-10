"""
Unit tests for Semantic Search Engine
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from semantic_search import SemanticSearchEngine


class TestSemanticSearchEngine:
    """Test suite for SemanticSearchEngine."""

    @pytest.fixture
    def engine(self):
        """Create a fresh engine instance."""
        return SemanticSearchEngine(use_faiss=False)  # Use numpy for faster tests

    @pytest.fixture
    def engine_with_docs(self, engine):
        """Create an engine with sample documents."""
        docs = [
            "Machine learning is a field of artificial intelligence",
            "Python is a programming language",
            "Deep learning uses neural networks",
            "JavaScript runs in web browsers",
            "Natural language processing analyzes text",
        ]
        engine.add_documents(docs)
        return engine

    def test_initialization(self, engine):
        """Test engine initializes correctly."""
        assert engine.model is not None
        assert engine.documents == []
        assert engine.embeddings is None
        assert len(engine) == 0

    def test_add_documents(self, engine):
        """Test adding documents."""
        docs = ["Hello world", "Test document"]
        engine.add_documents(docs)

        assert len(engine) == 2
        assert engine.documents == docs
        assert engine.embeddings is not None
        assert engine.embeddings.shape[0] == 2

    def test_add_documents_incremental(self, engine):
        """Test adding documents incrementally."""
        engine.add_documents(["First document"])
        assert len(engine) == 1

        engine.add_documents(["Second document", "Third document"])
        assert len(engine) == 3

    def test_add_empty_documents(self, engine):
        """Test adding empty document list."""
        engine.add_documents([])
        assert len(engine) == 0

    def test_search_basic(self, engine_with_docs):
        """Test basic search functionality."""
        results = engine_with_docs.search("AI and machine learning", top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert all(isinstance(r[0], str) and isinstance(r[1], float) for r in results)

    def test_search_relevance(self, engine_with_docs):
        """Test that search returns relevant results."""
        results = engine_with_docs.search("programming language", top_k=1)

        assert len(results) == 1
        doc, score = results[0]
        assert "Python" in doc or "JavaScript" in doc
        assert score > 0.5

    def test_search_top_k(self, engine_with_docs):
        """Test top_k parameter."""
        results = engine_with_docs.search("technology", top_k=3)
        assert len(results) == 3

        results = engine_with_docs.search("technology", top_k=10)
        assert len(results) == 5  # Only 5 documents exist

    def test_search_threshold(self, engine_with_docs):
        """Test similarity threshold filtering."""
        results = engine_with_docs.search("random unrelated query xyz", threshold=0.9)
        # High threshold should filter out low-relevance results
        assert all(score >= 0.9 for _, score in results)

    def test_search_empty_index(self, engine):
        """Test searching with no documents."""
        results = engine.search("test query")
        assert results == []

    def test_search_batch(self, engine_with_docs):
        """Test batch search."""
        queries = ["AI", "Python", "web"]
        results = engine_with_docs.search_batch(queries, top_k=2)

        assert len(results) == 3
        assert all(len(r) == 2 for r in results)

    def test_save_and_load(self, engine_with_docs):
        """Test saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_index"

            # Save
            engine_with_docs.save(save_path)
            assert (save_path / "documents.json").exists()
            assert (save_path / "embeddings.npy").exists()
            assert (save_path / "config.json").exists()

            # Load
            loaded = SemanticSearchEngine.load(save_path)
            assert len(loaded) == len(engine_with_docs)
            assert loaded.documents == engine_with_docs.documents
            assert loaded.model_name == engine_with_docs.model_name

    def test_clear(self, engine_with_docs):
        """Test clearing the engine."""
        assert len(engine_with_docs) > 0

        engine_with_docs.clear()

        assert len(engine_with_docs) == 0
        assert engine_with_docs.documents == []
        assert engine_with_docs.embeddings is None

    def test_repr(self, engine_with_docs):
        """Test string representation."""
        repr_str = repr(engine_with_docs)
        assert "SemanticSearchEngine" in repr_str
        assert "documents=5" in repr_str

    def test_search_rejects_empty_query(self, engine_with_docs):
        """Empty or whitespace-only queries raise ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            engine_with_docs.search("")
        with pytest.raises(ValueError, match="non-empty string"):
            engine_with_docs.search("   ")

    def test_search_rejects_invalid_top_k(self, engine_with_docs):
        """Non-positive or non-int top_k raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            engine_with_docs.search("test", top_k=0)
        with pytest.raises(ValueError, match="positive integer"):
            engine_with_docs.search("test", top_k=-3)
        with pytest.raises(ValueError, match="positive integer"):
            engine_with_docs.search("test", top_k=True)

    def test_search_rejects_invalid_threshold(self, engine_with_docs):
        """Out-of-range or non-numeric threshold raises ValueError."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            engine_with_docs.search("test", threshold=-0.1)
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            engine_with_docs.search("test", threshold=1.5)
        with pytest.raises(ValueError, match="number or None"):
            engine_with_docs.search("test", threshold="0.5")
        with pytest.raises(ValueError, match="number or None"):
            engine_with_docs.search("test", threshold=True)

    def test_search_accepts_boundary_threshold(self, engine_with_docs):
        """Threshold values 0.0 and 1.0 are accepted."""
        # Should not raise; 0.0 keeps everything, 1.0 likely filters all out.
        engine_with_docs.search("test", threshold=0.0)
        engine_with_docs.search("test", threshold=1.0)

    def test_add_documents_rejects_non_string(self, engine):
        """Non-string entries raise TypeError before any encoding work happens."""
        with pytest.raises(TypeError, match="must be strings"):
            engine.add_documents(["valid", 42, "also valid"])
        assert len(engine) == 0

    def test_add_documents_rejects_empty_string(self, engine):
        """Empty or whitespace-only entries raise ValueError."""
        with pytest.raises(ValueError, match="empty or whitespace"):
            engine.add_documents(["valid", "   "])
        assert len(engine) == 0

    def test_embeddings_normalized(self, engine):
        """Test that embeddings are L2-normalized."""
        engine.add_documents(["Test document"])

        embedding = engine.embeddings[0]
        norm = np.linalg.norm(embedding)

        # Should be approximately 1.0 (normalized)
        assert abs(norm - 1.0) < 0.01


class TestSemanticSearchEngineFAISS:
    """Test FAISS-specific functionality."""

    @pytest.fixture
    def engine_faiss(self):
        """Create engine with FAISS enabled."""
        return SemanticSearchEngine(use_faiss=True)

    def test_faiss_search(self, engine_faiss):
        """Test search with FAISS backend."""
        docs = ["Machine learning models", "Python programming", "Data science techniques"]
        engine_faiss.add_documents(docs)

        results = engine_faiss.search("ML and AI", top_k=2)
        assert len(results) == 2
        assert results[0][1] > results[1][1]  # Sorted by score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
