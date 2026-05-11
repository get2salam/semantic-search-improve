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
    err_at_k,
    f1_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    r_precision,
    recall_at_k,
    reciprocal_rank,
    reciprocal_rank_at_k,
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


class TestReciprocalRankAtK:
    def test_first_position_within_cutoff(self):
        assert reciprocal_rank_at_k(["a", "b", "c"], {"a"}, k=3) == 1.0

    def test_relevant_outside_cutoff_returns_zero(self):
        # "a" is at rank 4, cutoff k=3 — must not contribute
        assert reciprocal_rank_at_k(["x", "y", "z", "a"], {"a"}, k=3) == 0.0

    def test_third_position_within_cutoff(self):
        assert reciprocal_rank_at_k(["x", "y", "a"], {"a"}, k=5) == pytest.approx(1.0 / 3.0)

    def test_k_zero_or_negative(self):
        assert reciprocal_rank_at_k(["a"], {"a"}, k=0) == 0.0
        assert reciprocal_rank_at_k(["a"], {"a"}, k=-1) == 0.0

    def test_matches_unbounded_when_k_large(self):
        retrieved = ["x", "y", "a", "b"]
        relevant = {"a", "b"}
        assert reciprocal_rank_at_k(retrieved, relevant, k=10) == reciprocal_rank(
            retrieved, relevant
        )


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


class TestRPrecision:
    def test_perfect_ranking_is_one(self):
        # |R|=2, top-2 are both relevant -> R-Precision = 1.0
        assert r_precision(["a", "b", "x", "y"], {"a", "b"}) == pytest.approx(1.0)

    def test_partial_hit(self):
        # |R|=3, top-3 contain 2 relevant -> 2/3
        assert r_precision(["a", "x", "b", "c"], {"a", "b", "c"}) == pytest.approx(2.0 / 3.0)

    def test_no_relevant_returns_zero(self):
        assert r_precision(["a", "b"], set()) == 0.0

    def test_relevant_below_cutoff_ignored(self):
        # |R|=2, but the second relevant doc is at rank 3 -> P@2 = 1/2
        assert r_precision(["a", "x", "b"], {"a", "b"}) == pytest.approx(0.5)


class TestERRAtK:
    def test_k_zero_or_negative(self):
        q = EvalQuery(query="q", relevant_docs=["a"])
        assert err_at_k(["a"], q, k=0) == 0.0
        assert err_at_k(["a"], q, k=-1) == 0.0

    def test_no_relevant_returns_zero(self):
        # All retrieved docs have grade 0 -> ERR is exactly 0.
        q = EvalQuery(query="q", relevant_docs=["a"])
        assert err_at_k(["x", "y", "z"], q, k=3) == 0.0

    def test_binary_relevance_rank_one(self):
        # Binary grade => max_grade=1, R(1) = 0.5. Doc at rank 1 -> 1*0.5/1
        q = EvalQuery(query="q", relevant_docs=["a"])
        assert err_at_k(["a", "x", "y"], q, k=3) == pytest.approx(0.5)

    def test_graded_perfect_top_rank_score(self):
        # Highest grade at rank 1: R = (2^3 - 1) / 2^3 = 7/8 -> ERR contribution = 7/8.
        q = EvalQuery(
            query="q",
            relevant_docs=["a", "b"],
            relevance_grades={"a": 3, "b": 2},
        )
        # Only rank 1 contributes meaningfully; rank 2 adds (1 - 7/8) * (3/8) / 2.
        r1 = 7.0 / 8.0
        r2 = 3.0 / 8.0
        expected = r1 + (1.0 - r1) * r2 / 2.0
        assert err_at_k(["a", "b"], q, k=2) == pytest.approx(expected)

    def test_rank_one_dominates_rank_two(self):
        # Swapping a relevant doc from rank 2 to rank 1 must increase ERR.
        q = EvalQuery(query="q", relevant_docs=["a"])
        worse = err_at_k(["x", "a"], q, k=2)
        better = err_at_k(["a", "x"], q, k=2)
        assert better > worse

    def test_reported_in_evaluator_aggregate(self):
        # End-to-end: ERR appears in EvalReport and to_dict, with the
        # expected mean across two queries.
        evaluator = RetrievalEvaluator()
        evaluator.add_queries(
            [
                EvalQuery(query="q1", relevant_docs=["a"]),  # hit at rank 1 -> 0.5
                EvalQuery(query="q2", relevant_docs=["z"]),  # miss -> 0.0
            ]
        )
        results = {"q1": ["a", "x"], "q2": ["x", "y"]}
        report = evaluator.evaluate(
            lambda query, k: results[query][:k], k_values=[1, 2]
        )
        assert report.err[1] == pytest.approx(0.25)  # mean of 0.5 and 0.0
        assert "err" in report.to_dict()


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
        assert "hit_rate" in d and "f1" in d and "mrr_at_k" in d
        assert d["hit_rate"][3] == pytest.approx(0.5)
        # q1 hits at rank 1 (RR@k=1.0), q2 misses (RR@k=0.0) -> mean = 0.5
        assert report.mrr_at_k[3] == pytest.approx(0.5)
        assert report.mrr_at_k[1] == pytest.approx(0.5)
        # q1 R-precision = 1.0 (|R|=1, rank-1 is relevant); q2 R-prec = 0.0
        assert report.r_precision == pytest.approx(0.5)
        assert d["r_precision"] == pytest.approx(0.5)

    def test_r_precision_fetches_enough_docs_when_relevant_exceeds_max_k(self):
        # |R|=5 but the user only asks for k_values=[1, 3]. The evaluator must
        # still fetch enough docs to compute R-precision correctly.
        evaluator = RetrievalEvaluator()
        evaluator.add_queries(
            [EvalQuery(query="q", relevant_docs=["a", "b", "c", "d", "e"])]
        )

        ranked = ["a", "x", "b", "y", "c"]  # 3 of top-5 relevant -> R-prec=0.6

        def search_fn(query: str, k: int) -> list[str]:
            return ranked[:k]

        report = evaluator.evaluate(search_fn, k_values=[1, 3])
        assert report.r_precision == pytest.approx(0.6)
