"""
Guard test for examples/hybrid_search_demo.py.

Keeps the README-documented RRF fusion walkthrough honest: if the example
script's output or ranking behaviour ever drifts, this test fails instead
of the docs silently going stale.
"""

from __future__ import annotations

from examples.hybrid_search_demo import DENSE_RANKING, main


def test_demo_returns_full_fused_ranking():
    fused = main()
    fused_docs = [doc for doc, _ in fused]

    assert len(fused) == len(DENSE_RANKING)
    assert set(fused_docs) == set(DENSE_RANKING)


def test_demo_fused_scores_sorted_descending():
    fused = main()
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_demo_consensus_doc_beats_dense_only_top_hit():
    """A document ranked highly by BOTH BM25 and dense retrieval should
    outrank a document that only the dense side surfaced -- the whole
    point of fusing two retrieval signals instead of trusting one."""
    fused = main()
    fused_docs = [doc for doc, _ in fused]

    dense_only_top_hit = DENSE_RANKING[0]
    assert fused_docs.index(dense_only_top_hit) == len(fused_docs) - 1
