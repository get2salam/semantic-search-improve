"""
Agent Workflow Evaluation
=========================
Evaluation helpers for agentic search workflows. Measures task
completion, search efficiency, and step overhead for agents that
use retrieval as a tool in multi-step reasoning loops.

Usage:
    from agent_eval import AgentWorkflowEvaluator, AgentTrace, AgentAction

    trace = AgentTrace(
        task_id="q1",
        goal="Find documents about contract law",
        actions=[
            AgentAction(step=1, action_type="search", query="contract law",
                        retrieved_docs=["doc1", "doc2"]),
            AgentAction(step=2, action_type="search", query="breach of contract",
                        retrieved_docs=["doc3"]),
            AgentAction(step=3, action_type="answer", retrieved_docs=["doc1", "doc3"]),
        ],
        success=True,
        relevant_docs=["doc1", "doc3"],
    )

    evaluator = AgentWorkflowEvaluator()
    evaluator.add_trace(trace)
    report = evaluator.evaluate()
    report.print_summary()
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Failure Mode Taxonomy
# ---------------------------------------------------------------------------


class FailureMode(str, Enum):
    """Primary reason an agent task underperformed or succeeded efficiently."""

    SUCCESS = "success"
    INEFFICIENT_PATH = "inefficient_path"  # succeeded but too many steps or redundant queries
    NOISY_RESULTS = "noisy_results"  # succeeded but precision below threshold (high noise)
    LOW_RECALL = "low_recall"  # failed; found some relevant docs but below threshold
    NO_RESULTS = "no_results"  # failed; retrieved no relevant docs at all


def classify_failure_mode(
    result: AgentTaskResult,
    *,
    overhead_threshold: float = 2.0,
    precision_threshold: float = 0.3,
) -> FailureMode:
    """Return the primary failure mode for an evaluated agent task.

    Priority (checked in order):
    1. NO_RESULTS – failed with zero recall (agent found nothing relevant).
    2. LOW_RECALL – failed with positive but incomplete recall.
    3. NOISY_RESULTS – succeeded but precision is below *precision_threshold*,
       meaning the agent retrieved many irrelevant docs alongside the right ones.
       Skipped when the task has no ground-truth relevant docs.
    4. INEFFICIENT_PATH – succeeded but step overhead exceeded *overhead_threshold*
       or redundant queries were issued.
    5. SUCCESS – completed efficiently with adequate recall and precision.
    """
    if not result.success:
        return FailureMode.NO_RESULTS if result.recall_at_final == 0.0 else FailureMode.LOW_RECALL
    if result.recall_at_final > 0 and result.precision_at_final < precision_threshold:
        return FailureMode.NOISY_RESULTS
    if result.step_overhead > overhead_threshold or result.redundant_queries > 0:
        return FailureMode.INEFFICIENT_PATH
    return FailureMode.SUCCESS


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class AgentAction:
    """A single step in an agent's execution trace."""

    step: int
    action_type: str  # "search", "select", "answer", "clarify", "escalate"
    query: str | None = None
    retrieved_docs: list[str] = field(default_factory=list)
    tool_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTrace:
    """Full execution trace for a single agent task."""

    task_id: str
    goal: str
    actions: list[AgentAction]
    success: bool
    relevant_docs: list[str]  # ground-truth relevant docs for this task
    min_steps_required: int = 1  # theoretical minimum for an oracle agent


@dataclass
class AgentTaskResult:
    """Evaluation result for a single agent task."""

    task_id: str
    success: bool
    total_steps: int
    search_steps: int
    docs_retrieved: list[str]
    recall_at_final: float  # recall of all retrieved docs vs relevant_docs
    precision_at_final: float  # precision of all retrieved docs vs relevant_docs
    f1_at_final: float  # harmonic mean of precision_at_final and recall_at_final
    step_overhead: float  # total_steps / min_steps_required
    queries_issued: list[str]
    tool_diversity: float  # unique tool names / total tool-bearing actions (0 if none)
    redundant_queries: int  # queries issued more than once within this trace
    failure_mode: FailureMode  # primary reason the task underperformed or succeeded


@dataclass
class AgentWorkflowReport:
    """Aggregated evaluation across all agent traces."""

    num_tasks: int
    task_success_rate: float
    mean_steps: float
    mean_search_steps: float
    mean_recall: float
    mean_precision: float  # average precision across tasks
    mean_f1: float  # average F1 across tasks
    mean_step_overhead: float
    mean_tool_diversity: float  # average tool diversity across tasks
    mean_redundant_queries: float  # average redundant queries per task
    per_task: list[AgentTaskResult]

    def failure_mode_distribution(self) -> dict[str, int]:
        """Return counts of each failure mode across all evaluated tasks."""
        counts: Counter[str] = Counter(r.failure_mode.value for r in self.per_task)
        return dict(counts)

    def failure_mode_rates(self) -> dict[str, float]:
        """Return the share of tasks in each failure mode (fractions summing to 1.0).

        Useful for comparing failure-mode breakdowns across runs of different
        sizes, where raw counts would not be directly comparable.
        """
        if not self.per_task:
            return {}
        total = len(self.per_task)
        return {mode: count / total for mode, count in self.failure_mode_distribution().items()}

    def tasks_with_failure_mode(self, mode: FailureMode) -> list[AgentTaskResult]:
        """Return all per-task results whose primary failure mode matches *mode*.

        Handy for drilling into a specific failure class (e.g. inspecting every
        NOISY_RESULTS task) without re-scanning per_task by hand.
        """
        return [r for r in self.per_task if r.failure_mode == mode]

    def print_summary(self) -> None:
        dist = self.failure_mode_distribution()
        dist_str = ", ".join(f"{k}={v}" for k, v in sorted(dist.items()))
        lines = [
            "Agent Workflow Evaluation",
            "=" * 40,
            f"Tasks evaluated    : {self.num_tasks}",
            f"Success rate       : {self.task_success_rate:.3f}",
            f"Mean steps         : {self.mean_steps:.2f}",
            f"Mean search steps  : {self.mean_search_steps:.2f}",
            f"Mean recall        : {self.mean_recall:.3f}",
            f"Mean precision     : {self.mean_precision:.3f}",
            f"Mean F1            : {self.mean_f1:.3f}",
            f"Mean step overhead : {self.mean_step_overhead:.2f}x",
            f"Mean tool diversity: {self.mean_tool_diversity:.3f}",
            f"Mean redundant qrys: {self.mean_redundant_queries:.2f}",
            f"Failure modes      : {dist_str}",
        ]
        print("\n".join(lines))


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class AgentWorkflowEvaluator:
    """Evaluates multi-step agentic search workflows.

    Aggregates per-task metrics over a collection of AgentTrace objects
    and returns an AgentWorkflowReport with summary statistics.
    """

    def __init__(self) -> None:
        self._traces: list[AgentTrace] = []

    def add_trace(self, trace: AgentTrace) -> None:
        """Register a single agent trace for evaluation."""
        self._traces.append(trace)

    def add_traces(self, traces: list[AgentTrace]) -> None:
        """Register multiple agent traces at once."""
        self._traces.extend(traces)

    def _evaluate_trace(self, trace: AgentTrace) -> AgentTaskResult:
        search_actions = [a for a in trace.actions if a.action_type == "search"]

        # Collect all retrieved docs across all actions, preserving first-seen order.
        seen: set[str] = set()
        unique_retrieved: list[str] = []
        for action in trace.actions:
            for doc in action.retrieved_docs:
                if doc not in seen:
                    seen.add(doc)
                    unique_retrieved.append(doc)

        relevant_set = set(trace.relevant_docs)
        hits = sum(1 for d in unique_retrieved if d in relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 0.0
        precision = hits / len(unique_retrieved) if unique_retrieved else 0.0
        pr_sum = precision + recall
        f1 = 2 * precision * recall / pr_sum if pr_sum > 0 else 0.0

        total_steps = len(trace.actions)
        step_overhead = total_steps / max(trace.min_steps_required, 1)

        # Tool diversity: unique tool names / total tool-bearing actions.
        # High diversity means the agent uses a broad mix of tools; low diversity
        # signals over-reliance on a single tool.
        tool_names = [a.tool_name for a in trace.actions if a.tool_name]
        tool_diversity = len(set(tool_names)) / len(tool_names) if tool_names else 0.0

        # Redundant queries: number of distinct queries issued more than once.
        # Repeated queries waste tool calls without new information.
        query_counts = Counter(a.query for a in search_actions if a.query)
        redundant_queries = sum(1 for cnt in query_counts.values() if cnt > 1)

        result = AgentTaskResult(
            task_id=trace.task_id,
            success=trace.success,
            total_steps=total_steps,
            search_steps=len(search_actions),
            docs_retrieved=unique_retrieved,
            recall_at_final=recall,
            precision_at_final=precision,
            f1_at_final=f1,
            step_overhead=step_overhead,
            queries_issued=[a.query for a in search_actions if a.query],
            tool_diversity=tool_diversity,
            redundant_queries=redundant_queries,
            failure_mode=FailureMode.SUCCESS,  # placeholder; classified below
        )
        result.failure_mode = classify_failure_mode(result)
        return result

    def evaluate(self) -> AgentWorkflowReport:
        """Evaluate all registered traces and return an aggregated report."""
        if not self._traces:
            raise ValueError("No traces to evaluate. Call add_trace() first.")

        results = [self._evaluate_trace(t) for t in self._traces]
        n = len(results)

        return AgentWorkflowReport(
            num_tasks=n,
            task_success_rate=sum(r.success for r in results) / n,
            mean_steps=sum(r.total_steps for r in results) / n,
            mean_search_steps=sum(r.search_steps for r in results) / n,
            mean_recall=sum(r.recall_at_final for r in results) / n,
            mean_precision=sum(r.precision_at_final for r in results) / n,
            mean_f1=sum(r.f1_at_final for r in results) / n,
            mean_step_overhead=sum(r.step_overhead for r in results) / n,
            mean_tool_diversity=sum(r.tool_diversity for r in results) / n,
            mean_redundant_queries=sum(r.redundant_queries for r in results) / n,
            per_task=results,
        )
