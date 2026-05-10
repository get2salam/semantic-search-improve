"""
Semantic Search Engine
======================
A lightweight semantic search engine using sentence embeddings and FAISS.

Author: get2salam
License: MIT
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearchEngine:
    """
    A semantic search engine that finds similar documents based on meaning.

    Uses sentence-transformers for embeddings and numpy for similarity search.
    Optionally uses FAISS for faster search on large datasets.

    Attributes:
        model_name (str): Name of the sentence-transformer model
        model (SentenceTransformer): The embedding model
        documents (List[str]): Stored documents
        embeddings (np.ndarray): Document embeddings matrix

    Example:
        >>> engine = SemanticSearchEngine()
        >>> engine.add_documents(["Hello world", "Machine learning is great"])
        >>> results = engine.search("AI and ML", top_k=1)
        >>> print(results[0][0])  # "Machine learning is great"
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        use_faiss: bool = True,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize the semantic search engine.

        Args:
            model_name: Name of the sentence-transformer model to use.
                        Default is 'all-MiniLM-L6-v2' (fast and efficient).
            use_faiss: Whether to use FAISS for similarity search.
                       Falls back to numpy if FAISS is not installed.
            normalize_embeddings: Whether to L2-normalize embeddings.
                                  Enables cosine similarity via dot product.
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.use_faiss = use_faiss

        # Load the embedding model
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded! Embedding dimension: {self.embedding_dim}")

        # Initialize storage
        self.documents: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.index = None

        # Try to import FAISS
        self._faiss = None
        if use_faiss:
            try:
                import faiss

                self._faiss = faiss
                print("FAISS enabled for fast similarity search")
            except ImportError:
                print("FAISS not installed, using numpy for search")
                self.use_faiss = False

    def add_documents(
        self, documents: list[str], batch_size: int = 64, show_progress: bool = True
    ) -> None:
        """
        Add documents to the search index.

        Args:
            documents: List of text documents to index
            batch_size: Batch size for encoding (affects memory usage)
            show_progress: Whether to show a progress bar

        Raises:
            TypeError: If any entry in ``documents`` is not a string.
            ValueError: If any entry is empty or whitespace-only.
        """
        if not documents:
            return

        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                raise TypeError(
                    f"All documents must be strings, got {type(doc).__name__} at index {i}"
                )
            if not doc.strip():
                raise ValueError(f"Document at index {i} is empty or whitespace-only")

        # Compute embeddings
        print(f"Encoding {len(documents)} documents...")
        new_embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        # Store documents
        self.documents.extend(documents)

        # Update embeddings matrix
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        # Rebuild index if using FAISS
        if self.use_faiss and self._faiss is not None:
            self._build_faiss_index()

        print(f"Indexed {len(self.documents)} documents total")

    def _build_faiss_index(self) -> None:
        """Build or rebuild the FAISS index."""
        if self._faiss is None or self.embeddings is None:
            return

        # Use Inner Product for normalized vectors (equivalent to cosine similarity)
        self.index = self._faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.embeddings.astype(np.float32))

    def search(
        self, query: str, top_k: int = 5, threshold: float | None = None
    ) -> list[tuple[str, float]]:
        """
        Search for documents similar to the query.

        Args:
            query: The search query text
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1). Results below this are filtered.

        Returns:
            List of (document, similarity_score) tuples, sorted by relevance

        Raises:
            ValueError: If ``query`` is empty/whitespace, ``top_k`` is not a
                positive int, or ``threshold`` is not ``None`` or in ``[0.0, 1.0]``.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, int | float):
                raise ValueError(
                    f"threshold must be a number or None, got {type(threshold).__name__}"
                )
            if not 0.0 <= float(threshold) <= 1.0:
                raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold!r}")

        if not self.documents:
            return []

        # Encode query
        query_embedding = self.model.encode(
            query, normalize_embeddings=self.normalize_embeddings, convert_to_numpy=True
        ).reshape(1, -1)

        # Search
        if self.use_faiss and self.index is not None:
            scores, indices = self.index.search(
                query_embedding.astype(np.float32), min(top_k, len(self.documents))
            )
            scores = scores[0]
            indices = indices[0]
        else:
            # Numpy fallback - compute all similarities
            scores = np.dot(self.embeddings, query_embedding.T).flatten()
            indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[indices]

        # Build results
        results = []
        for idx, score in zip(indices, scores, strict=False):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            if threshold is not None and score < threshold:
                continue
            results.append((self.documents[idx], float(score)))

        return results

    def search_batch(self, queries: list[str], top_k: int = 5) -> list[list[tuple[str, float]]]:
        """
        Search for multiple queries at once.

        Args:
            queries: List of search queries
            top_k: Number of results per query

        Returns:
            List of results for each query
        """
        return [self.search(q, top_k=top_k) for q in queries]

    def save(self, path: str | Path) -> None:
        """
        Save the search engine to disk.

        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save documents
        with open(path / "documents.json", "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

        # Save embeddings
        if self.embeddings is not None:
            np.save(path / "embeddings.npy", self.embeddings)

        # Save config
        config = {
            "model_name": self.model_name,
            "normalize_embeddings": self.normalize_embeddings,
            "embedding_dim": self.embedding_dim,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved search engine to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SemanticSearchEngine":
        """
        Load a search engine from disk.

        Args:
            path: Directory path to load from

        Returns:
            Loaded SemanticSearchEngine instance
        """
        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config = json.load(f)

        # Create instance
        engine = cls(
            model_name=config["model_name"],
            normalize_embeddings=config.get("normalize_embeddings", True),
        )

        # Load documents
        with open(path / "documents.json", encoding="utf-8") as f:
            engine.documents = json.load(f)

        # Load embeddings
        embeddings_path = path / "embeddings.npy"
        if embeddings_path.exists():
            engine.embeddings = np.load(embeddings_path)
            if engine.use_faiss:
                engine._build_faiss_index()

        print(f"Loaded {len(engine.documents)} documents from {path}")
        return engine

    def clear(self) -> None:
        """Clear all documents and embeddings."""
        self.documents = []
        self.embeddings = None
        self.index = None
        print("Search engine cleared")

    def __len__(self) -> int:
        """Return the number of indexed documents."""
        return len(self.documents)

    def __repr__(self) -> str:
        return (
            f"SemanticSearchEngine("
            f"model='{self.model_name}', "
            f"documents={len(self.documents)}, "
            f"faiss={self.use_faiss})"
        )


if __name__ == "__main__":
    # Quick demo
    engine = SemanticSearchEngine()

    # Sample documents
    docs = [
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language for data science",
        "Neural networks are inspired by biological neurons",
        "Deep learning requires large amounts of training data",
        "Natural language processing deals with text understanding",
        "Computer vision enables machines to interpret images",
        "Reinforcement learning trains agents through rewards",
        "Transfer learning reuses pre-trained models",
        "Transformers revolutionized NLP with attention mechanisms",
        "GPT and BERT are popular language models",
    ]

    engine.add_documents(docs)

    # Test search
    query = "AI and neural networks"
    print(f"\nSearch: '{query}'")
    print("-" * 50)

    for doc, score in engine.search(query, top_k=3):
        print(f"[{score:.3f}] {doc}")
