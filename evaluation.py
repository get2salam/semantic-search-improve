"""
Retrieval Evaluation Metrics
==============================
Comprehensive evaluation suite for semantic search quality. Computes
standard IR metrics (MRR, MAP, NDCG, Precision@k, Recall@k) and
supports multi-model benchmarking with statistical significance testing.

Usage:
    from evaluation import RetrievalEvaluator, EvalQuery, ModelBenchmark

    evaluator = RetrievalEvaluator()
    evaluator.add_queries([
        EvalQuery(query="breach of contract", relevant_docs=["doc1", "doc3"]),
        EvalQuery(query="property law", relevant_docs=["doc2", "doc5"]),
    ])
    report = evaluator.evaluate(search_fn=my_search_function, k_values=[1, 3, 5, 10])
    report.print_summary()
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class EvalQuery:
    """A query with its known relevant documents."""

    query: str
    relevant_docs: list[str]
    relevance_grades: dict[str, int] | None = None  # doc_id -> grade (for NDCG)

    def get_grade(self, doc_id: str) -> int:
        """Return graded relevance (defaults to binary: 1 if relevant, 0 if not)."""
        if self.relevance_grades:
            return self.relevance_grades.get(doc_id, 0)
        return 1 if doc_id in self.relevant_docs else 0


@dataclass
class EvalResult:
    """Result for a single query evaluation."""

    query: str
    retrieved_docs: list[str]
    relevant_docs: list[str]
    reciprocal_rank: float
    average_precision: float
    ndcg: dict[int, float]  # k -> NDCG@k
    precision: dict[int, float]  # k -> P@k
    recall: dict[int, float]  # k -> R@k
    hit_rate: dict[int, float] = field(default_factory=dict)  # k -> HitRate@k
    f1: dict[int, float] = field(default_factory=dict)  # k -> F1@k
    mrr_at_k: dict[int, float] = field(default_factory=dict)  # k -> RR@k
    err: dict[int, float] = field(default_factory=dict)  # k -> ERR@k (graded)
    rbp: dict[int, float] = field(default_factory=dict)  # k -> RBP@k (user-model)
    ap_at_k: dict[int, float] = field(default_factory=dict)  # k -> AP@k (truncated AP)
    r_precision: float = 0.0  # P@|R|: precision at the size of the relevant set


@dataclass
class EvalReport:
    """Aggregated evaluation report across all queries."""

    num_queries: int
    k_values: list[int]
    mrr: float  # Mean Reciprocal Rank
    map_score: float  # Mean Average Precision
    ndcg: dict[int, float]  # Mean NDCG@k
    precision: dict[int, float]  # Mean Precision@k
    recall: dict[int, float]  # Mean Recall@k
    per_query: list[EvalResult]
    elapsed_seconds: float
    hit_rate: dict[int, float] = field(default_factory=dict)  # Mean HitRate@k
    f1: dict[int, float] = field(default_factory=dict)  # Mean F1@k
    mrr_at_k: dict[int, float] = field(default_factory=dict)  # Mean RR@k (MRR with cutoff)
    err: dict[int, float] = field(default_factory=dict)  # Mean ERR@k (graded)
    rbp: dict[int, float] = field(default_factory=dict)  # Mean RBP@k (user-model)
    rbp_persistence: float = 0.8  # p parameter used for the RBP computation
    map_at_k: dict[int, float] = field(default_factory=dict)  # Mean AP@k (i.e. MAP@k)
    r_precision: float = 0.0  # Mean R-precision (P@|R|) across queries
    model_name: str | None = None

    def to_dict(self) -> dict:
        return {
            "num_queries": self.num_queries,
            "k_values": self.k_values,
            "mrr": self.mrr,
            "map": self.map_score,
            "ndcg": self.ndcg,
            "precision": self.precision,
            "recall": self.recall,
            "hit_rate": self.hit_rate,
            "f1": self.f1,
            "mrr_at_k": self.mrr_at_k,
            "err": self.err,
            "rbp": self.rbp,
            "rbp_persistence": self.rbp_persistence,
            "map_at_k": self.map_at_k,
            "r_precision": self.r_precision,
            "elapsed_seconds": self.elapsed_seconds,
            "model_name": self.model_name,
        }

    def save(self, path: str | Path) -> None:
        """Persist report as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> None:
        """Print a formatted summary table to stdout."""
        name = self.model_name or "Model"
        print(f"\n{'=' * 60}")
        print(f"  Retrieval Evaluation Report — {name}")
        print(f"  Queries: {self.num_queries} | Time: {self.elapsed_seconds:.2f}s")
        print(f"{'=' * 60}")
        print(f"  MRR:  {self.mrr:.4f}")
        print(f"  MAP:  {self.map_score:.4f}")
        print(f"  R-Prec: {self.r_precision:.4f}")
        print()
        print(f"  {'k':>4}  {'NDCG@k':>8}  {'P@k':>8}  {'R@k':>8}  {'F1@k':>8}  {'Hit@k':>8}")
        print(f"  {'—' * 4}  {'—' * 8}  {'—' * 8}  {'—' * 8}  {'—' * 8}  {'—' * 8}")
        for k in self.k_values:
            ndcg = self.ndcg.get(k, 0)
            prec = self.precision.get(k, 0)
            rec = self.recall.get(k, 0)
            f1 = self.f1.get(k, 0)
            hit = self.hit_rate.get(k, 0)
            print(
                f"  {k:>4}  {ndcg:>8.4f}  {prec:>8.4f}  {rec:>8.4f}  {f1:>8.4f}  {hit:>8.4f}"
            )
        print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Metric Functions
# ---------------------------------------------------------------------------


def reciprocal_rank(retrieved: Sequence[str], relevant: set) -> float:
    """
    Compute Reciprocal Rank: 1/rank of the first relevant document.

    Returns 0.0 if no relevant document is found in the results.
    """
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def reciprocal_rank_at_k(retrieved: Sequence[str], relevant: set, k: int) -> float:
    """
    Reciprocal Rank with a rank cutoff: 1/rank of the first relevant document
    among the top-k results, or 0.0 if no relevant doc appears within the cutoff.

    The aggregate (mean over queries) is MRR@k, a standard RAG retrieval metric
    when only the top-k chunks reach the generator.
    """
    if k <= 0:
        return 0.0
    for i, doc in enumerate(retrieved[:k], 1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: set) -> float:
    """
    Compute Average Precision for a single query.

    AP = (1/|relevant|) * sum(Precision@k * rel(k))
    where rel(k) = 1 if doc at rank k is relevant.
    """
    if not relevant:
        return 0.0

    hits = 0
    sum_precision = 0.0

    for i, doc in enumerate(retrieved, 1):
        if doc in relevant:
            hits += 1
            sum_precision += hits / i

    return sum_precision / len(relevant) if relevant else 0.0


def average_precision_at_k(retrieved: Sequence[str], relevant: set, k: int) -> float:
    """
    Average Precision with a rank cutoff k (the TREC ``AP@k`` definition).

    Sums ``Precision@i`` over ranks i=1..k where rank i is relevant, then
    divides by the full size of the relevant set::

        AP@k = (1/|R|) * sum_{i=1..k} P@i * rel(i)

    The aggregate (mean over queries) is ``MAP@k``, the most common
    cutoff variant of MAP used in RAG and recommender evaluations where
    only the top-k results reach downstream stages. Returns 0.0 when
    either ``k <= 0`` or the relevant set is empty.
    """
    if k <= 0 or not relevant:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for i, doc in enumerate(retrieved[:k], 1):
        if doc in relevant:
            hits += 1
            sum_precision += hits / i
    return sum_precision / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set, k: int) -> float:
    """Precision@k: fraction of top-k results that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for d in top_k if d in relevant) / k


def recall_at_k(retrieved: Sequence[str], relevant: set, k: int) -> float:
    """Recall@k: fraction of relevant documents found in top-k."""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def hit_rate_at_k(retrieved: Sequence[str], relevant: set, k: int) -> float:
    """
    Hit Rate@k (a.k.a. Success@k): 1.0 if any relevant document appears in
    the top-k retrieved results, else 0.0.

    Commonly reported for RAG pipelines where surfacing at least one
    relevant chunk for the generator is sufficient.
    """
    if not relevant or k <= 0:
        return 0.0
    return 1.0 if any(d in relevant for d in retrieved[:k]) else 0.0


def f1_at_k(retrieved: Sequence[str], relevant: set, k: int) -> float:
    """
    F1@k: harmonic mean of Precision@k and Recall@k.

    Balances surface accuracy against coverage of the relevant set; returns
    0.0 when both precision and recall are zero.
    """
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def r_precision(retrieved: Sequence[str], relevant: set) -> float:
    """
    R-Precision: precision at rank R, where R = |relevant|.

    A standard TREC IR metric that adapts the cutoff to each query's
    own relevant-set size, balancing precision and recall without an
    arbitrary k. Returns 0.0 when no relevant docs are known.
    """
    r = len(relevant)
    if r == 0:
        return 0.0
    top_r = retrieved[:r]
    return sum(1 for d in top_r if d in relevant) / r


def err_at_k(
    retrieved: Sequence[str],
    query: EvalQuery,
    k: int,
    max_grade: int | None = None,
) -> float:
    """
    Expected Reciprocal Rank at k (Chapelle et al., 2009).

    Models a user who scans the top-k results until satisfied. Each doc has
    a grade-derived satisfaction probability R(g) = (2^g - 1) / 2^g_max::

        ERR@k = sum_{i=1..k} (1/i) * R(g_i) * prod_{j<i} (1 - R(g_j))

    Unlike NDCG, ERR penalises lower-ranked relevant docs even more
    heavily because the user is assumed to stop once satisfied, making it
    well-suited for top-result-quality scoring in RAG and ranking systems.
    Uses graded relevance via ``EvalQuery.relevance_grades`` when present;
    otherwise falls back to binary grades (1 if relevant, 0 otherwise).
    """
    if k <= 0:
        return 0.0
    if max_grade is None:
        max_grade = (
            max(query.relevance_grades.values(), default=1)
            if query.relevance_grades
            else 1
        )
    if max_grade <= 0:
        return 0.0

    denom = float(2**max_grade)
    err = 0.0
    survival = 1.0  # P(user not yet satisfied by previous ranks)
    for i, doc in enumerate(retrieved[:k], 1):
        g = query.get_grade(doc)
        r = (2**g - 1) / denom
        err += survival * r / i
        survival *= 1.0 - r
    return err


def rbp_at_k(
    retrieved: Sequence[str],
    query: EvalQuery,
    k: int,
    persistence: float = 0.8,
    max_grade: int | None = None,
) -> float:
    """
    Rank-Biased Precision at k (Moffat & Zobel, 2008).

    Models a user who examines rank ``i`` with probability ``p^(i-1)``,
    where ``p`` (``persistence``) is how likely the user is to continue
    past each result::

        RBP@k = (1 - p) * sum_{i=1..k} g_i * p^(i-1)

    ``g_i`` is the doc's grade normalised to ``[0, 1]`` by ``max_grade``
    (1 for binary relevance). Unlike NDCG, RBP has no per-query
    normalising denominator, so unjudged documents simply contribute
    zero — which makes the metric notably robust to incomplete
    relevance judgments. ``persistence`` must satisfy ``0 <= p < 1``.
    """
    if k <= 0 or not 0.0 <= persistence < 1.0:
        return 0.0
    if max_grade is None:
        max_grade = (
            max(query.relevance_grades.values(), default=1)
            if query.relevance_grades
            else 1
        )
    if max_grade <= 0:
        return 0.0

    total = 0.0
    weight = 1.0  # p^(i-1); starts at p^0 = 1
    for doc in retrieved[:k]:
        total += (query.get_grade(doc) / max_grade) * weight
        weight *= persistence
    return (1.0 - persistence) * total


def dcg_at_k(
    retrieved: Sequence[str],
    query: EvalQuery,
    k: int,
) -> float:
    """
    Discounted Cumulative Gain at k.

    DCG@k = sum(rel(i) / log2(i+1)) for i in 1..k
    Supports graded relevance via query.relevance_grades.
    """
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        rel = query.get_grade(doc)
        dcg += rel / math.log2(i + 2)  # i+2 because i is 0-indexed
    return dcg


def ndcg_at_k(
    retrieved: Sequence[str],
    query: EvalQuery,
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain at k.

    NDCG@k = DCG@k / IDCG@k where IDCG is the ideal DCG.
    """
    actual_dcg = dcg_at_k(retrieved, query, k)

    # Ideal ranking: sort by relevance grade descending
    if query.relevance_grades:
        ideal_grades = sorted(query.relevance_grades.values(), reverse=True)
    else:
        ideal_grades = [1] * len(query.relevant_docs)

    ideal_dcg = 0.0
    for i, grade in enumerate(ideal_grades[:k]):
        ideal_dcg += grade / math.log2(i + 2)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

# Type alias: search function takes (query, k) and returns list of doc IDs
SearchFn = Callable[[str, int], list[str]]


class RetrievalEvaluator:
    """
    Evaluate retrieval quality for semantic search.

    Example::

        evaluator = RetrievalEvaluator()
        evaluator.add_queries([
            EvalQuery("machine learning", ["doc1", "doc3"]),
        ])

        def search(query: str, k: int) -> List[str]:
            results = engine.search(query, top_k=k)
            return [doc_id for doc_id, _ in results]

        report = evaluator.evaluate(search, k_values=[1, 5, 10])
        report.print_summary()
    """

    def __init__(self):
        self._queries: list[EvalQuery] = []

    def add_queries(self, queries: Sequence[EvalQuery]) -> None:
        """Add evaluation queries."""
        self._queries.extend(queries)
        logger.info("Added %d eval queries (total: %d)", len(queries), len(self._queries))

    def load_queries_jsonl(self, path: str | Path) -> int:
        """
        Load evaluation queries from JSONL.

        Each line: {"query": "...", "relevant_docs": ["doc1", "doc2"], "relevance_grades": {...}}

        Returns:
            Number of queries loaded.
        """
        path = Path(path)
        loaded = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self._queries.append(
                    EvalQuery(
                        query=obj["query"],
                        relevant_docs=obj["relevant_docs"],
                        relevance_grades=obj.get("relevance_grades"),
                    )
                )
                loaded += 1
        logger.info("Loaded %d eval queries from %s", loaded, path)
        return loaded

    def evaluate(
        self,
        search_fn: SearchFn,
        k_values: list[int] | None = None,
        model_name: str | None = None,
        rbp_persistence: float = 0.8,
    ) -> EvalReport:
        """
        Run evaluation on all queries.

        Args:
            search_fn: A callable (query, k) -> list of doc IDs.
            k_values: List of k values to compute metrics at.
                      Defaults to [1, 3, 5, 10, 20].
            model_name: Optional model name for labelling the report.

        Returns:
            EvalReport with aggregated metrics.
        """
        if not self._queries:
            raise ValueError("No evaluation queries. Call add_queries() first.")

        k_values = k_values or [1, 3, 5, 10, 20]
        max_k = max(k_values)
        start = time.time()

        per_query_results: list[EvalResult] = []
        rr_scores = []
        ap_scores = []
        r_prec_scores: list[float] = []
        ndcg_scores: dict[int, list[float]] = {k: [] for k in k_values}
        prec_scores: dict[int, list[float]] = {k: [] for k in k_values}
        rec_scores: dict[int, list[float]] = {k: [] for k in k_values}
        hit_scores: dict[int, list[float]] = {k: [] for k in k_values}
        f1_scores: dict[int, list[float]] = {k: [] for k in k_values}
        rr_at_k_scores: dict[int, list[float]] = {k: [] for k in k_values}
        err_scores: dict[int, list[float]] = {k: [] for k in k_values}
        rbp_scores: dict[int, list[float]] = {k: [] for k in k_values}
        ap_at_k_scores: dict[int, list[float]] = {k: [] for k in k_values}

        for eq in self._queries:
            # Retrieve at least |R| docs so R-precision is well-defined even
            # when the relevant set is larger than max(k_values).
            fetch_k = max(max_k, len(eq.relevant_docs))
            retrieved = search_fn(eq.query, fetch_k)
            relevant = set(eq.relevant_docs)

            # Compute metrics
            rr = reciprocal_rank(retrieved, relevant)
            ap = average_precision(retrieved, relevant)
            rp = r_precision(retrieved, relevant)
            rr_scores.append(rr)
            ap_scores.append(ap)
            r_prec_scores.append(rp)

            q_ndcg = {}
            q_prec = {}
            q_rec = {}
            q_hit = {}
            q_f1 = {}
            q_rr_at_k = {}
            q_err = {}
            q_rbp = {}
            q_ap_at_k = {}
            for k in k_values:
                q_ndcg[k] = ndcg_at_k(retrieved, eq, k)
                q_prec[k] = precision_at_k(retrieved, relevant, k)
                q_rec[k] = recall_at_k(retrieved, relevant, k)
                q_hit[k] = hit_rate_at_k(retrieved, relevant, k)
                q_f1[k] = f1_at_k(retrieved, relevant, k)
                q_rr_at_k[k] = reciprocal_rank_at_k(retrieved, relevant, k)
                q_err[k] = err_at_k(retrieved, eq, k)
                q_rbp[k] = rbp_at_k(retrieved, eq, k, persistence=rbp_persistence)
                q_ap_at_k[k] = average_precision_at_k(retrieved, relevant, k)
                ndcg_scores[k].append(q_ndcg[k])
                prec_scores[k].append(q_prec[k])
                rec_scores[k].append(q_rec[k])
                hit_scores[k].append(q_hit[k])
                f1_scores[k].append(q_f1[k])
                rr_at_k_scores[k].append(q_rr_at_k[k])
                err_scores[k].append(q_err[k])
                rbp_scores[k].append(q_rbp[k])
                ap_at_k_scores[k].append(q_ap_at_k[k])

            per_query_results.append(
                EvalResult(
                    query=eq.query,
                    retrieved_docs=retrieved[:max_k],
                    relevant_docs=eq.relevant_docs,
                    reciprocal_rank=rr,
                    average_precision=ap,
                    ndcg=q_ndcg,
                    precision=q_prec,
                    recall=q_rec,
                    hit_rate=q_hit,
                    f1=q_f1,
                    mrr_at_k=q_rr_at_k,
                    err=q_err,
                    rbp=q_rbp,
                    ap_at_k=q_ap_at_k,
                    r_precision=rp,
                )
            )

        elapsed = time.time() - start

        report = EvalReport(
            num_queries=len(self._queries),
            k_values=k_values,
            mrr=round(float(np.mean(rr_scores)), 4),
            map_score=round(float(np.mean(ap_scores)), 4),
            ndcg={k: round(float(np.mean(v)), 4) for k, v in ndcg_scores.items()},
            precision={k: round(float(np.mean(v)), 4) for k, v in prec_scores.items()},
            recall={k: round(float(np.mean(v)), 4) for k, v in rec_scores.items()},
            hit_rate={k: round(float(np.mean(v)), 4) for k, v in hit_scores.items()},
            f1={k: round(float(np.mean(v)), 4) for k, v in f1_scores.items()},
            mrr_at_k={k: round(float(np.mean(v)), 4) for k, v in rr_at_k_scores.items()},
            err={k: round(float(np.mean(v)), 4) for k, v in err_scores.items()},
            rbp={k: round(float(np.mean(v)), 4) for k, v in rbp_scores.items()},
            rbp_persistence=rbp_persistence,
            map_at_k={k: round(float(np.mean(v)), 4) for k, v in ap_at_k_scores.items()},
            r_precision=round(float(np.mean(r_prec_scores)), 4),
            per_query=per_query_results,
            elapsed_seconds=round(elapsed, 2),
            model_name=model_name,
        )

        logger.info("Evaluation complete: MRR=%.4f, MAP=%.4f", report.mrr, report.map_score)
        return report


# ---------------------------------------------------------------------------
# Model Benchmark
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Comparison of multiple models on the same evaluation set."""

    models: list[str]
    reports: dict[str, dict]  # model_name -> EvalReport.to_dict()
    ranking: list[tuple[str, float]]  # sorted by MRR descending
    best_model: str
    elapsed_seconds: float

    def print_comparison(self) -> None:
        """Print a side-by-side comparison table."""
        print(f"\n{'=' * 72}")
        print("  Model Benchmark Comparison")
        print(f"{'=' * 72}")
        print(f"  {'Model':<30} {'MRR':>8} {'MAP':>8} {'NDCG@5':>8} {'R@10':>8}")
        print(f"  {'—' * 30} {'—' * 8} {'—' * 8} {'—' * 8} {'—' * 8}")

        for model_name, mrr in self.ranking:
            r = self.reports[model_name]
            map_s = r.get("map", 0)
            ndcg5 = r.get("ndcg", {}).get("5", r.get("ndcg", {}).get(5, 0))
            r10 = r.get("recall", {}).get("10", r.get("recall", {}).get(10, 0))
            marker = " 🏆" if model_name == self.best_model else ""
            print(f"  {model_name:<30} {mrr:>8.4f} {map_s:>8.4f} {ndcg5:>8.4f} {r10:>8.4f}{marker}")

        print(f"\n  Best model: {self.best_model}")
        print(f"  Total benchmark time: {self.elapsed_seconds:.1f}s")
        print(f"{'=' * 72}\n")

    def save(self, path: str | Path) -> None:
        """Save benchmark results."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "models": self.models,
                    "ranking": self.ranking,
                    "best_model": self.best_model,
                    "reports": self.reports,
                    "elapsed_seconds": self.elapsed_seconds,
                },
                f,
                indent=2,
            )


class ModelBenchmark:
    """
    Compare multiple sentence-transformer models on the same evaluation set.

    Example::

        benchmark = ModelBenchmark(
            models=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
            queries=[EvalQuery("q1", ["d1"])],
            corpus=["d1: some document", "d2: another document"],
        )
        result = benchmark.run()
        result.print_comparison()
    """

    def __init__(
        self,
        models: list[str],
        queries: list[EvalQuery],
        corpus: list[str],
        corpus_ids: list[str] | None = None,
        k_values: list[int] | None = None,
    ):
        self.models = models
        self.queries = queries
        self.corpus = corpus
        self.corpus_ids = corpus_ids or [f"doc_{i}" for i in range(len(corpus))]
        self.k_values = k_values or [1, 3, 5, 10, 20]

    def run(self) -> BenchmarkResult:
        """
        Run the benchmark: encode corpus with each model, evaluate, compare.

        Returns:
            BenchmarkResult with ranking and per-model reports.
        """
        from sentence_transformers import SentenceTransformer

        start = time.time()
        reports: dict[str, dict] = {}

        for model_name in self.models:
            logger.info("Benchmarking model: %s", model_name)

            model = SentenceTransformer(model_name)
            corpus_embeddings = model.encode(
                self.corpus,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            def search_fn(
                query: str,
                k: int,
                _model: SentenceTransformer = model,
                _corpus_embeddings: np.ndarray = corpus_embeddings,
            ) -> list[str]:
                q_emb = _model.encode(
                    query,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).reshape(1, -1)
                scores = np.dot(_corpus_embeddings, q_emb.T).flatten()
                top_indices = np.argsort(scores)[::-1][:k]
                return [self.corpus_ids[i] for i in top_indices]

            evaluator = RetrievalEvaluator()
            evaluator.add_queries(self.queries)
            report = evaluator.evaluate(search_fn, k_values=self.k_values, model_name=model_name)
            reports[model_name] = report.to_dict()

        elapsed = time.time() - start

        # Rank by MRR
        ranking = sorted(
            [(name, r["mrr"]) for name, r in reports.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        return BenchmarkResult(
            models=self.models,
            reports=reports,
            ranking=ranking,
            best_model=ranking[0][0] if ranking else "",
            elapsed_seconds=round(elapsed, 1),
        )


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--queries", required=True, help="JSONL file with eval queries")
    parser.add_argument(
        "--corpus",
        required=True,
        help="JSONL file with corpus documents (one per line: {id, text})",
    )
    parser.add_argument(
        "--models", nargs="+", default=["all-MiniLM-L6-v2"], help="Models to benchmark"
    )
    parser.add_argument(
        "--k", nargs="+", type=int, default=[1, 3, 5, 10], help="k values for metrics"
    )
    parser.add_argument("--output", default="eval_report.json", help="Output report path")
    args = parser.parse_args()

    # Load corpus
    corpus_docs = []
    corpus_ids = []
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            corpus_ids.append(obj["id"])
            corpus_docs.append(obj["text"])

    # Load queries
    evaluator = RetrievalEvaluator()
    evaluator.load_queries_jsonl(args.queries)

    if len(args.models) > 1:
        # Multi-model benchmark
        benchmark = ModelBenchmark(
            models=args.models,
            queries=evaluator._queries,
            corpus=corpus_docs,
            corpus_ids=corpus_ids,
            k_values=args.k,
        )
        result = benchmark.run()
        result.print_comparison()
        result.save(args.output)
    else:
        # Single-model evaluation
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.models[0])
        embs = model.encode(corpus_docs, normalize_embeddings=True, convert_to_numpy=True)

        def search(query: str, k: int) -> list[str]:
            q = model.encode(query, normalize_embeddings=True, convert_to_numpy=True).reshape(1, -1)
            scores = np.dot(embs, q.T).flatten()
            top = np.argsort(scores)[::-1][:k]
            return [corpus_ids[i] for i in top]

        report = evaluator.evaluate(search, k_values=args.k, model_name=args.models[0])
        report.print_summary()
        report.save(args.output)
