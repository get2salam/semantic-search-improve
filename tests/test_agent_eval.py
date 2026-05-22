"""Unit tests for agent workflow evaluation helpers."""

import pytest

from agent_eval import (
    AgentAction,
    AgentTrace,
    AgentWorkflowEvaluator,
)


def _trace(task_id, actions, success, relevant_docs, min_steps=1):
    return AgentTrace(
        task_id=task_id,
        goal=f"Goal for {task_id}",
        actions=actions,
        success=success,
        relevant_docs=relevant_docs,
        min_steps_required=min_steps,
    )


# ---------------------------------------------------------------------------
# Per-task result metrics
# ---------------------------------------------------------------------------


class TestAgentTaskResult:
    def test_perfect_recall(self):
        trace = _trace(
            "t1",
            [
                AgentAction(step=1, action_type="search", query="q", retrieved_docs=["a", "b"]),
                AgentAction(step=2, action_type="answer"),
            ],
            success=True,
            relevant_docs=["a", "b"],
        )
        ev = AgentWorkflowEvaluator()
        ev.add_trace(trace)
        assert ev.evaluate().per_task[0].recall_at_final == pytest.approx(1.0)

    def test_partial_recall(self):
        trace = _trace(
            "t2",
            [
                AgentAction(step=1, action_type="search", query="q", retrieved_docs=["a"]),
                AgentAction(step=2, action_type="answer"),
            ],
            success=False,
            relevant_docs=["a", "b", "c"],
        )
        ev = AgentWorkflowEvaluator()
        ev.add_trace(trace)
        assert ev.evaluate().per_task[0].recall_at_final == pytest.approx(1 / 3)

    def test_zero_recall(self):
        trace = _trace(
            "t3",
            [AgentAction(step=1, action_type="search", query="q", retrieved_docs=["x"])],
            success=False,
            relevant_docs=["a", "b"],
        )
        ev = AgentWorkflowEvaluator()
        ev.add_trace(trace)
        assert ev.evaluate().per_task[0].recall_at_final == 0.0

    def test_step_overhead(self):
        trace = _trace(
            "t4",
            [
                AgentAction(step=1, action_type="search", query="q1"),
                AgentAction(step=2, action_type="search", query="q2"),
                AgentAction(step=3, action_type="answer"),
            ],
            success=True,
            relevant_docs=["a"],
            min_steps=1,
        )
        ev = AgentWorkflowEvaluator()
        ev.add_trace(trace)
        assert ev.evaluate().per_task[0].step_overhead == pytest.approx(3.0)

    def test_deduplicates_docs_across_steps(self):
        trace = _trace(
            "t5",
            [
                AgentAction(step=1, action_type="search", query="q1", retrieved_docs=["a", "b"]),
                AgentAction(step=2, action_type="search", query="q2", retrieved_docs=["b", "c"]),
                AgentAction(step=3, action_type="answer"),
            ],
            success=True,
            relevant_docs=["a", "b", "c"],
        )
        ev = AgentWorkflowEvaluator()
        ev.add_trace(trace)
        result = ev.evaluate().per_task[0]
        assert len(result.docs_retrieved) == 3
        assert result.recall_at_final == pytest.approx(1.0)

    def test_queries_collected(self):
        trace = _trace(
            "t6",
            [
                AgentAction(step=1, action_type="search", query="first"),
                AgentAction(step=2, action_type="search", query="second"),
                AgentAction(step=3, action_type="answer"),
            ],
            success=True,
            relevant_docs=["a"],
        )
        ev = AgentWorkflowEvaluator()
        ev.add_trace(trace)
        result = ev.evaluate().per_task[0]
        assert result.queries_issued == ["first", "second"]
        assert result.search_steps == 2

    def test_no_relevant_docs_gives_zero_recall(self):
        trace = _trace(
            "t7",
            [AgentAction(step=1, action_type="answer")],
            success=True,
            relevant_docs=[],
        )
        ev = AgentWorkflowEvaluator()
        ev.add_trace(trace)
        assert ev.evaluate().per_task[0].recall_at_final == 0.0


# ---------------------------------------------------------------------------
# Aggregated report
# ---------------------------------------------------------------------------


class TestAgentWorkflowReport:
    def _two_trace_evaluator(self):
        traces = [
            _trace(
                "success",
                [
                    AgentAction(step=1, action_type="search", query="q", retrieved_docs=["a"]),
                    AgentAction(step=2, action_type="answer"),
                ],
                success=True,
                relevant_docs=["a"],
                min_steps=2,
            ),
            _trace(
                "failure",
                [
                    AgentAction(step=1, action_type="search", query="q", retrieved_docs=[]),
                    AgentAction(step=2, action_type="answer"),
                ],
                success=False,
                relevant_docs=["b"],
                min_steps=2,
            ),
        ]
        ev = AgentWorkflowEvaluator()
        ev.add_traces(traces)
        return ev

    def test_success_rate(self):
        assert self._two_trace_evaluator().evaluate().task_success_rate == pytest.approx(0.5)

    def test_mean_steps(self):
        assert self._two_trace_evaluator().evaluate().mean_steps == pytest.approx(2.0)

    def test_mean_recall(self):
        assert self._two_trace_evaluator().evaluate().mean_recall == pytest.approx(0.5)

    def test_num_tasks(self):
        assert self._two_trace_evaluator().evaluate().num_tasks == 2

    def test_empty_evaluator_raises(self):
        with pytest.raises(ValueError, match="No traces"):
            AgentWorkflowEvaluator().evaluate()

    def test_print_summary_runs(self, capsys):
        report = self._two_trace_evaluator().evaluate()
        report.print_summary()
        out = capsys.readouterr().out
        assert "Success rate" in out
        assert "Mean recall" in out
