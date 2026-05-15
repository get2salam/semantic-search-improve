"""
Unit tests for retrieval evaluation metrics.
"""

import math

import pytest

from evaluation import (
    EvalQuery,
    RetrievalEvaluator,
    average_precision,
    average_precision_at_k,
    bpref,
    dcg_at_k,
    err_at_k,
    expected_search_length,
    f1_at_k,
    geometric_mean_average_precision,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    q_measure,
    r_precision,
    rbp_at_k,
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


class TestExpectedSearchLength:
    def test_first_position_is_zero(self):
        # First doc is relevant, n=1 -> no non-relevant examined
        assert expected_search_length(["a", "b", "c"], {"a"}, n=1) == 0.0

    def test_counts_non_relevant_above_first_hit(self):
        # First relevant at rank 3 -> 2 non-relevant before it
        assert expected_search_length(["x", "y", "a"], {"a"}, n=1) == 2.0

    def test_generalises_to_multiple_relevant(self):
        # Need n=2 relevant; relevant at ranks 2 and 4 -> 2 non-relevant by rank 4
        assert expected_search_length(["x", "a", "y", "b"], {"a", "b"}, n=2) == 2.0

    def test_partial_success_returns_total_non_relevant_seen(self):
        # Only 1 relevant in list, but caller asks for n=2 -> total non-rel = 2
        assert expected_search_length(["a", "x", "y"], {"a", "b"}, n=2) == 2.0

    def test_non_positive_n_returns_zero(self):
        assert expected_search_length(["a", "b"], {"a"}, n=0) == 0.0
        assert expected_search_length(["a", "b"], {"a"}, n=-1) == 0.0


class TestBpref:
    def test_no_relevant_returns_zero(self):
        assert bpref(["a", "b"], set(), {"a", "b"}) == 0.0

    def test_perfect_ranking_is_one(self):
        # All relevants ranked above all judged non-relevants -> Bpref = 1.
        assert bpref(["a", "b", "x", "y"], {"a", "b"}, {"x", "y"}) == pytest.approx(1.0)

    def test_unjudged_docs_are_ignored(self):
        # "u" is unjudged: it must NOT be counted as a non-rel penalty even
        # though it sits above the relevant doc. With no judged non-rels
        # above "a", Bpref = 1.0.
        assert bpref(["u", "a"], {"a"}, {"x"}) == pytest.approx(1.0)

    def test_penalty_uses_min_r_n(self):
        # R=1, N=2, one judged non-rel ranked above the single relevant doc
        # -> penalty = 1/min(1,2) = 1 -> Bpref = 0.
        assert bpref(["x", "a"], {"a"}, {"x", "y"}) == pytest.approx(0.0)

    def test_partial_penalty(self):
        # R=2, N=2. "a" at rank 1: 0 non-rels above -> 1.
        # "b" at rank 3: 1 non-rel above ("x") -> 1 - 1/2 = 0.5.
        # Bpref = (1 + 0.5) / 2 = 0.75.
        assert bpref(["a", "x", "b", "y"], {"a", "b"}, {"x", "y"}) == pytest.approx(0.75)

    def test_missing_relevant_doc_contributes_zero(self):
        # "b" never retrieved; only "a" contributes. R=2 -> divide by 2.
        # "a" at rank 1, no non-rels above -> 1. Bpref = 1/2 = 0.5.
        assert bpref(["a", "x"], {"a", "b"}, {"x"}) == pytest.approx(0.5)

    def test_reported_in_evaluator_aggregate(self):
        evaluator = RetrievalEvaluator()
        evaluator.add_queries(
            [
                EvalQuery(query="q1", relevant_docs=["a"]),  # rank-1 hit
                EvalQuery(query="q2", relevant_docs=["z"]),  # miss
            ]
        )
        results = {"q1": ["a", "x"], "q2": ["x", "y"]}
        report = evaluator.evaluate(lambda query, k: results[query][:k], k_values=[1, 2])
        # q1: relevant at rank 1, no non-rels above -> 1.0. q2: miss -> 0.
        assert report.bpref == pytest.approx(0.5)
        assert "bpref" in report.to_dict()


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
        report = evaluator.evaluate(lambda query, k: results[query][:k], k_values=[1, 2])
        assert report.err[1] == pytest.approx(0.25)  # mean of 0.5 and 0.0
        assert "err" in report.to_dict()


class TestRBPAtK:
    def test_k_zero_or_negative(self):
        q = EvalQuery(query="q", relevant_docs=["a"])
        assert rbp_at_k(["a"], q, k=0) == 0.0
        assert rbp_at_k(["a"], q, k=-1) == 0.0

    def test_invalid_persistence_returns_zero(self):
        q = EvalQuery(query="q", relevant_docs=["a"])
        assert rbp_at_k(["a"], q, k=3, persistence=1.0) == 0.0
        assert rbp_at_k(["a"], q, k=3, persistence=-0.1) == 0.0

    def test_no_relevant_returns_zero(self):
        q = EvalQuery(query="q", relevant_docs=["a"])
        assert rbp_at_k(["x", "y", "z"], q, k=3, persistence=0.8) == 0.0

    def test_binary_relevance_rank_one(self):
        # Single relevant doc at rank 1, p=0.5 -> RBP = (1-0.5)*1*1 = 0.5
        q = EvalQuery(query="q", relevant_docs=["a"])
        assert rbp_at_k(["a", "x"], q, k=2, persistence=0.5) == pytest.approx(0.5)

    def test_all_relevant_equals_one_minus_p_to_the_k(self):
        # If every doc in top-k is relevant, RBP@k collapses to
        # (1-p) * sum_{i=0..k-1} p^i = 1 - p^k.
        q = EvalQuery(query="q", relevant_docs=["a", "b", "c"])
        assert rbp_at_k(["a", "b", "c"], q, k=3, persistence=0.5) == pytest.approx(1.0 - 0.5**3)

    def test_deep_relevant_decays_with_low_persistence(self):
        # A single relevant doc at rank 3 contributes (1-p)*p^2. Lower p
        # makes deep ranks contribute less (the user gives up earlier).
        q = EvalQuery(query="q", relevant_docs=["c"])
        low = rbp_at_k(["x", "y", "c"], q, k=3, persistence=0.2)
        high = rbp_at_k(["x", "y", "c"], q, k=3, persistence=0.9)
        assert low == pytest.approx((1.0 - 0.2) * 0.2**2)
        assert high == pytest.approx((1.0 - 0.9) * 0.9**2)
        assert high > low

    def test_graded_normalises_by_max_grade(self):
        # max_grade=3, doc at rank 1 has grade 3 -> g_normalised=1.0;
        # p=0.5 -> RBP = (1-0.5) * 1.0 * 1 = 0.5
        q = EvalQuery(
            query="q",
            relevant_docs=["a"],
            relevance_grades={"a": 3},
        )
        assert rbp_at_k(["a"], q, k=1, persistence=0.5) == pytest.approx(0.5)

    def test_reported_in_evaluator_aggregate(self):
        evaluator = RetrievalEvaluator()
        evaluator.add_queries(
            [
                EvalQuery(query="q1", relevant_docs=["a"]),  # rank-1 hit
                EvalQuery(query="q2", relevant_docs=["z"]),  # miss
            ]
        )
        results = {"q1": ["a", "x"], "q2": ["x", "y"]}
        report = evaluator.evaluate(
            lambda query, k: results[query][:k],
            k_values=[1, 2],
            rbp_persistence=0.5,
        )
        # q1@1: (1-0.5)*1 = 0.5; q2@1: 0 -> mean = 0.25
        assert report.rbp[1] == pytest.approx(0.25)
        d = report.to_dict()
        assert "rbp" in d and d["rbp_persistence"] == 0.5


class TestAveragePrecisionAtK:
    def test_k_zero_or_negative(self):
        assert average_precision_at_k(["a"], {"a"}, k=0) == 0.0
        assert average_precision_at_k(["a"], {"a"}, k=-1) == 0.0

    def test_no_relevant_returns_zero(self):
        assert average_precision_at_k(["a", "b"], set(), k=5) == 0.0

    def test_perfect_top_k_matches_unbounded_ap(self):
        # All relevant docs at the top within the cutoff -> AP@k == AP == 1.0.
        retrieved = ["a", "b", "c", "x"]
        relevant = {"a", "b", "c"}
        assert average_precision_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)
        assert average_precision_at_k(retrieved, relevant, k=3) == pytest.approx(
            average_precision(retrieved, relevant)
        )

    def test_relevant_outside_cutoff_excluded(self):
        # relevant={a,b,c}; rank-3 c is excluded by k=2. Only a (rank 1) hits.
        # AP@2 = (1/3) * (P@1) = (1/3) * 1 = 1/3.
        assert average_precision_at_k(["a", "x", "c"], {"a", "b", "c"}, k=2) == pytest.approx(
            1.0 / 3.0
        )

    def test_normalisation_by_full_relevant_set(self):
        # TREC AP@k divides by |R|, not min(k, |R|). |R|=4, only "a" hit at rank 1
        # within k=2 -> AP@2 = (1/4) * 1 = 0.25.
        assert average_precision_at_k(["a", "x"], {"a", "b", "c", "d"}, k=2) == pytest.approx(0.25)

    def test_reported_in_evaluator_aggregate(self):
        evaluator = RetrievalEvaluator()
        evaluator.add_queries(
            [
                EvalQuery(query="q1", relevant_docs=["a"]),  # hit at rank 1 -> AP@1 = 1.0
                EvalQuery(query="q2", relevant_docs=["z"]),  # miss -> 0.0
            ]
        )
        results = {"q1": ["a", "x"], "q2": ["x", "y"]}
        report = evaluator.evaluate(lambda query, k: results[query][:k], k_values=[1, 2])
        # Mean AP@1 over the two queries.
        assert report.map_at_k[1] == pytest.approx(0.5)
        assert "map_at_k" in report.to_dict()


class TestGeometricMeanAveragePrecision:
    def test_empty_returns_zero(self):
        assert geometric_mean_average_precision([]) == 0.0

    def test_single_score_returns_itself(self):
        assert geometric_mean_average_precision([0.4]) == pytest.approx(0.4)

    def test_geometric_mean_formula(self):
        # GMAP([0.25, 1.0]) = sqrt(0.25) = 0.5; MAP would be 0.625.
        assert geometric_mean_average_precision([0.25, 1.0]) == pytest.approx(0.5)

    def test_more_sensitive_to_weak_queries_than_map(self):
        # One bad query (low AP) should pull GMAP below the arithmetic mean.
        scores = [0.01, 0.9, 0.9, 0.9]
        gmap = geometric_mean_average_precision(scores)
        arithmetic = sum(scores) / len(scores)
        assert gmap < arithmetic

    def test_zero_ap_floored_by_epsilon(self):
        # AP=0 would make log undefined; epsilon floors it. Result must be
        # strictly positive and below the epsilon floor's geometric pull.
        eps = 1e-5
        result = geometric_mean_average_precision([0.0, 1.0], epsilon=eps)
        # sqrt(eps * 1.0) = sqrt(eps)
        assert result == pytest.approx(math.sqrt(eps))

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError):
            geometric_mean_average_precision([0.5], epsilon=0.0)
        with pytest.raises(ValueError):
            geometric_mean_average_precision([0.5], epsilon=-1e-5)

    def test_reported_in_evaluator_aggregate(self):
        # q1 hits at rank 1 (AP=1.0); q2 misses (AP=0.0 -> floored to epsilon).
        # GMAP = sqrt(1.0 * 1e-5) = sqrt(1e-5), rounded to 4 dp -> 0.0032.
        evaluator = RetrievalEvaluator()
        evaluator.add_queries(
            [
                EvalQuery(query="q1", relevant_docs=["a"]),
                EvalQuery(query="q2", relevant_docs=["z"]),
            ]
        )
        results = {"q1": ["a", "x"], "q2": ["x", "y"]}
        report = evaluator.evaluate(lambda query, k: results[query][:k], k_values=[1, 2])
        assert report.gmap == pytest.approx(round(math.sqrt(1e-5), 4))
        assert "gmap" in report.to_dict()


class TestQMeasure:
    def test_no_relevant_returns_zero(self):
        q = EvalQuery(query="q", relevant_docs=[])
        assert q_measure(["a", "b"], q) == 0.0

    def test_negative_beta_returns_zero(self):
        q = EvalQuery(query="q", relevant_docs=["a"])
        assert q_measure(["a"], q, beta=-0.5) == 0.0

    def test_perfect_binary_ranking_is_one(self):
        # All relevant docs at the top with binary grades -> Q = 1.0.
        q = EvalQuery(query="q", relevant_docs=["a", "b"])
        assert q_measure(["a", "b", "x"], q, beta=1.0) == pytest.approx(1.0)

    def test_beta_zero_reduces_to_average_precision(self):
        # With beta=0, BR(i) collapses to count_rel(1..i)/i = P@i, so
        # Q-measure equals AP. Compare against the binary AP for
        # retrieved=[a,x,b], relevant={a,b}: AP = (1/1 + 2/3)/2.
        q = EvalQuery(query="q", relevant_docs=["a", "b"])
        expected_ap = (1.0 + 2.0 / 3.0) / 2.0
        assert q_measure(["a", "x", "b"], q, beta=0.0) == pytest.approx(expected_ap)

    def test_graded_imperfect_ranking(self):
        # Grades a=3, b=2; retrieved=[b,a] (swapped). Ideal CG at ranks
        # is [3, 5]. BR(1) = (1 + 1*2)/(1 + 1*3) = 3/4.
        # BR(2) = (2 + 1*5)/(2 + 1*5) = 1.0. Q = (0.75 + 1.0)/2 = 0.875.
        q = EvalQuery(
            query="q",
            relevant_docs=["a", "b"],
            relevance_grades={"a": 3, "b": 2},
        )
        assert q_measure(["b", "a"], q, beta=1.0) == pytest.approx(0.875)

    def test_rank_one_dominates_rank_two(self):
        # Moving a relevant doc earlier must not decrease Q-measure.
        q = EvalQuery(query="q", relevant_docs=["a"])
        assert q_measure(["a", "x"], q) >= q_measure(["x", "a"], q)


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

    def test_q_measure_reported_in_evaluator_aggregate(self):
        # q1: rank-1 hit on a graded query -> Q = 1.0.
        # q2: miss -> Q = 0.0. Mean Q-measure should be 0.5.
        evaluator = RetrievalEvaluator()
        evaluator.add_queries(
            [
                EvalQuery(query="q1", relevant_docs=["a"]),
                EvalQuery(query="q2", relevant_docs=["z"]),
            ]
        )
        results = {"q1": ["a", "x"], "q2": ["x", "y"]}
        report = evaluator.evaluate(lambda query, k: results[query][:k], k_values=[1, 2])
        assert report.q_measure == pytest.approx(0.5)
        d = report.to_dict()
        assert "q_measure" in d and d["q_measure_beta"] == pytest.approx(1.0)

    def test_r_precision_fetches_enough_docs_when_relevant_exceeds_max_k(self):
        # |R|=5 but the user only asks for k_values=[1, 3]. The evaluator must
        # still fetch enough docs to compute R-precision correctly.
        evaluator = RetrievalEvaluator()
        evaluator.add_queries([EvalQuery(query="q", relevant_docs=["a", "b", "c", "d", "e"])])

        ranked = ["a", "x", "b", "y", "c"]  # 3 of top-5 relevant -> R-prec=0.6

        def search_fn(query: str, k: int) -> list[str]:
            return ranked[:k]

        report = evaluator.evaluate(search_fn, k_values=[1, 3])
        assert report.r_precision == pytest.approx(0.6)
