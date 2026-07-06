#!/usr/bin/env python3
"""
Hybrid Search Fusion Demo
=========================
Shows how Reciprocal Rank Fusion (RRF) blends a BM25 keyword ranking with
a second ranked list (e.g. dense embeddings) into one consensus ranking.

Runs instantly and fully offline: it only touches ``BM25Retriever`` and
``reciprocal_rank_fusion``, which are pure Python/numpy, so no
sentence-transformers model download is required.  In production,
``HybridSearchEngine`` (see hybrid_search.py) fuses BM25 against a real
``DenseRetriever`` the same way.

Usage (run from the repo root):
    python -m examples.hybrid_search_demo
"""

from hybrid_search import BM25Retriever, reciprocal_rank_fusion

CORPUS = [
    "machine learning algorithms improve with data",
    "deep learning uses neural networks for feature extraction",
    "python is a versatile programming language",
    "natural language processing handles text analysis",
    "gradient descent optimises neural network weights",
]

QUERY = "neural network training"

# Stand-in for DenseRetriever.search(QUERY) output. A real embedding model
# surfaces "machine learning algorithms improve with data" here because it
# is semantically about training/learning, even though it shares no
# keywords with the query -- BM25 alone would never find it.
DENSE_RANKING = [
    "machine learning algorithms improve with data",
    "deep learning uses neural networks for feature extraction",
    "gradient descent optimises neural network weights",
]


def main():
    bm25 = BM25Retriever()
    bm25.index(CORPUS)
    bm25_ranking = [doc for doc, _ in bm25.search(QUERY, top_k=2)]

    fused = reciprocal_rank_fusion([bm25_ranking, DENSE_RANKING], rrf_k=60)

    print(f"Query: {QUERY!r}\n")
    print("BM25-only ranking (keyword match):")
    for i, doc in enumerate(bm25_ranking, 1):
        print(f"  {i}. {doc}")

    print("\nDense-only ranking (semantic match, simulated):")
    for i, doc in enumerate(DENSE_RANKING, 1):
        print(f"  {i}. {doc}")

    print("\nFused ranking via Reciprocal Rank Fusion:")
    for i, (doc, score) in enumerate(fused, 1):
        print(f"  {i}. [{score:.6f}] {doc}")

    return fused


if __name__ == "__main__":
    main()
