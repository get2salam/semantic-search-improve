"""
Unit tests for retrieval evaluation metrics.
"""

import math

import pytest

from evaluation import (
    EvalQuery,
    RetrievalEvaluator,
    average_precision,
    dcg_at_k,
    f1_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_third_position(self):
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1.0 / 3.0)

    def test_no_relevant_found(self):
        assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_empty_retrieved(self):
        assert reciprocal_rank([], {"a"}) == 0.0

    def test_multiple_relevant_uses_first(self):
        # Only the rank of the FIRST relevant doc matters.
        assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5


class TestAveragePrecision:
    def test_perfect_ranking(self):
        # All relevant docs at the top -> AP = 1.0
        assert average_precision(["a", "b", "c", "x"], {"a", "b", "c"}) == pytest.approx(1.0)

    def test_no_relevant_in_truth(self):
        assert average_precision(["a", "b"], set()) == 0.0

    def test_no_hits(self):
        assert average_precision(["x", "y"], {"a"}) == 0.0

    def test_mixed_ranking(self):
        # relevant={a,b}; retrieved=[a,x,b] -> (1/1 + 2/3) / 2
        ap = average_precision(["a", "x", "b"], {"a", "b"})
        assert ap == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


class TestPrecisionRecallAtK:
    def test_precision_at_k_basic(self):
        # 2 of top 3 are relevant -> 2/3
        assert precision_at_k(["a", "b", "x"], {"a", "b", "c"}, k=3) == pytest.approx(2.0 / 3.0)

    def test_precision_k_larger_than_results(self):
        # k=5 but only 2 retrieved: divides by k, not by len(retrieved)
        assert precision_at_k(["a", "b"], {"a", "b"}, k=5) == pytest.approx(2.0 / 5.0)

    def test_precision_k_zero_or_negative(self):
        assert precision_at_k(["a"], {"a"}, k=0) == 0.0
        assert precision_at_k(["a"], {"a"}, k=-1) == 0.0

    def test_recall_at_k_basic(self):
        # 2 of 3 relevant docs found in top 2 -> 2/3
        assert recall_at_k(["a", "b", "x"], {"a", "b", "c"}, k=2) == pytest.approx(2.0 / 3.0)

    def test_recall_no_relevant(self):
        assert recall_at_k(["a"], set(), k=5) == 0.0


class TestHitRateAtK:
    def test_relevant_in_top_k(self):
        assert hit_rate_at_k(["x", "a", "y"], {"a"}, k=3) == 1.0

    def test_relevant_outside_top_k(self):
        # "a" is at rank 4, outside top-3
        assert hit_rate_at_k(["x", "y", "z", "a"], {"a"}, k=3) == 0.0

    def test_no_relevant_in_truth(self):
        assert hit_rate_at_k(["a", "b"], set(), k=5) == 0.0

    def test_k_zero_or_negative(self):
        assert hit_rate_at_k(["a"], {"a"}, k=0) == 0.0
        assert hit_rate_at_k(["a"], {"a"}, k=-1) == 0.0

    def test_empty_retrieved(self):
        assert hit_rate_at_k([], {"a"}, k=5) == 0.0

    def test_multiple_relevant_still_returns_one(self):
        # Hit rate is binary regardless of how many relevant docs land in top-k
        assert hit_rate_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


class TestF1AtK:
    def test_perfect_precision_and_recall(self):
        # All relevant docs at top, k=2 -> P=1, R=1 -> F1=1
        assert f1_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_no_hits_returns_zero(self):
        assert f1_at_k(["x", "y"], {"a"}, k=2) == 0.0

    def test_harmonic_mean_formula(self):
        # retrieved=[a,x,b], relevant={a,b,c}, k=3
        # P@3 = 2/3, R@3 = 2/3 -> F1 = 2/3
        assert f1_at_k(["a", "x", "b"], {"a", "b", "c"}, k=3) == pytest.approx(2.0 / 3.0)

    def test_no_relevant_in_truth(self):
        assert f1_at_k(["a"], set(), k=3) == 0.0


class TestDCGandNDCG:
    def test_dcg_binary_relevance(self):
        # rel=[1,0,1] at positions 1,2,3 -> 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5
        q = EvalQuery(query="q", relevant_docs=["a", "c"])
        assert dcg_at_k(["a", "x", "c"], q, k=3) == pytest.approx(1.0 + 0.5)

    def test_ndcg_perfect_ranking_is_one(self):
        q = EvalQuery(query="q", relevant_docs=["a", "b"])
        assert ndcg_at_k(["a", "b", "x"], q, k=2) == pytest.approx(1.0)

    def test_ndcg_no_relevant_returns_zero(self):
        # No relevant docs in retrieved -> NDCG = 0.0
        q = EvalQuery(query="q", relevant_docs=["a"])
        assert ndcg_at_k(["x", "y"], q, k=2) == 0.0

    def test_ndcg_graded_relevance(self):
        # Graded: a=3, b=2, c=1. Retrieved [b,a,c] (k=3).
        # DCG = 2/log2(2) + 3/log2(3) + 1/log2(4)
        # IDCG (ideal=[3,2,1]) = 3/log2(2) + 2/log2(3) + 1/log2(4)
        q = EvalQuery(
            query="q",
            relevant_docs=["a", "b", "c"],
            relevance_grades={"a": 3, "b": 2, "c": 1},
        )
        dcg = 2.0 + 3.0 / math.log2(3) + 1.0 / 2.0
        idcg = 3.0 + 2.0 / math.log2(3) + 1.0 / 2.0
        assert ndcg_at_k(["b", "a", "c"], q, k=3) == pytest.approx(dcg / idcg)
        # And it must be strictly less than 1 since ranking is non-ideal.
        assert ndcg_at_k(["b", "a", "c"], q, k=3) < 1.0


class TestEvaluatorReportsHitRateAndF1:
    def test_aggregated_hit_rate_and_f1_are_reported(self):
        # Two queries: q1 finds its sole relevant doc at rank 1 (hit, perfect P/R),
        # q2 misses entirely (no hit, zero F1). Mean Hit@k should be 0.5.
        evaluator = RetrievalEvaluator()
        evaluator.add_queries(
            [
                EvalQuery(query="q1", relevant_docs=["a"]),
                EvalQuery(query="q2", relevant_docs=["z"]),
            ]
        )

        results = {"q1": ["a", "x", "y"], "q2": ["x", "y", "w"]}

        def search_fn(query: str, k: int) -> list[str]:
            return results[query][:k]

        report = evaluator.evaluate(search_fn, k_values=[1, 3])

        assert report.hit_rate[3] == pytest.approx(0.5)
        assert report.hit_rate[1] == pytest.approx(0.5)
        # q1: P=R=F1=1.0 at k=1; q2: all zero -> mean F1@1 = 0.5
        assert report.f1[1] == pytest.approx(0.5)
        # Round-trip via to_dict should expose the new fields
        d = report.to_dict()
        assert "hit_rate" in d and "f1" in d
        assert d["hit_rate"][3] == pytest.approx(0.5)
