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
        assert "tool diversity" in out
        assert "redundant" in out


# ---------------------------------------------------------------------------
# Tool diversity and redundant query metrics
# ---------------------------------------------------------------------------


class TestToolDiversityAndRedundancy:
    def test_tool_diversity_all_same_tool(self):
        trace = _trace(
            "td1",
            [
                AgentAction(step=1, action_type="search", query="q1", tool_name="vector_search"),
                AgentAction(step=2, action_type="search", query="q2", tool_name="vector_search"),
                AgentAction(step=3, action_type="answer", tool_name="vector_search"),
            ],
            success=True,
            relevant_docs=["a"],
        )
        ev = AgentWorkflowEvaluator()
        ev.add_trace(trace)
        # 1 unique tool / 3 tool-bearing actions = 1/3
        assert ev.evaluate().per_task[0].tool_diversity == pytest.approx(1 / 3)

    def test_tool_diversity_all_different_tools(self):
        trace = _trace(
            "td2",
            [
                AgentAction(step=1, action_type="search", query="q1", tool_name="vector_search"),
                AgentAction(step=2, action_type="search", query="q2", tool_name="keyword_search"),
                AgentAction(step=3, action_type="answer", tool_name="synthesizer"),
            ],
            success=True,
            relevant_docs=["a"],
        )
        ev = AgentWorkflowEvaluator()
        ev.add_trace(trace)
        # 3 unique tools / 3 tool-bearing actions = 1.0
        assert ev.evaluate().per_task[0].tool_diversity == pytest.approx(1.0)

    def test_tool_diversity_no_tool_names_gives_zero(self):
        trace = _trace(
            "td3",
            [AgentAction(step=1, action_type="search", query="q")],
            success=True,
            relevant_docs=["a"],
        )
        ev = AgentWorkflowEvaluator()
        ev.add_trace(trace)
        assert ev.evaluate().per_task[0].tool_diversity == 0.0

    def test_redundant_queries_detected(self):
        trace = _trace(
            "rq1",
            [
                AgentAction(step=1, action_type="search", query="contract law"),
                AgentAction(step=2, action_type="search", query="breach of contract"),
                AgentAction(step=3, action_type="search", query="contract law"),  # duplicate
                AgentAction(step=4, action_type="answer"),
            ],
            success=True,
            relevant_docs=["a"],
        )
        ev = AgentWorkflowEvaluator()
        ev.add_trace(trace)
        assert ev.evaluate().per_task[0].redundant_queries == 1

    def test_no_redundant_queries(self):
        trace = _trace(
            "rq2",
            [
                AgentAction(step=1, action_type="search", query="alpha"),
                AgentAction(step=2, action_type="search", query="beta"),
                AgentAction(step=3, action_type="answer"),
            ],
            success=True,
            relevant_docs=["a"],
        )
        ev = AgentWorkflowEvaluator()
        ev.add_trace(trace)
        assert ev.evaluate().per_task[0].redundant_queries == 0

    def test_report_aggregates_redundant_queries(self):
        traces = [
            _trace(
                "rqa",
                [
                    AgentAction(step=1, action_type="search", query="q"),
                    AgentAction(step=2, action_type="search", query="q"),  # duplicate
                ],
                success=True,
                relevant_docs=["a"],
            ),
            _trace(
                "rqb",
                [AgentAction(step=1, action_type="search", query="unique")],
                success=True,
                relevant_docs=["a"],
            ),
        ]
        ev = AgentWorkflowEvaluator()
        ev.add_traces(traces)
        # task rqa: 1 redundant; task rqb: 0 redundant → mean = 0.5
        assert ev.evaluate().mean_redundant_queries == pytest.approx(0.5)

    def test_report_aggregates_tool_diversity(self):
        traces = [
            _trace(
                "tda",
                [
                    AgentAction(step=1, action_type="search", query="q", tool_name="search"),
                    AgentAction(step=2, action_type="answer", tool_name="search"),
                ],
                success=True,
                relevant_docs=["a"],
            ),
            _trace(
                "tdb",
                [
                    AgentAction(step=1, action_type="search", query="q", tool_name="search"),
                    AgentAction(step=2, action_type="answer", tool_name="synth"),
                ],
                success=True,
                relevant_docs=["a"],
            ),
        ]
        ev = AgentWorkflowEvaluator()
        ev.add_traces(traces)
        # tda: 1/2 = 0.5; tdb: 2/2 = 1.0 → mean = 0.75
        assert ev.evaluate().mean_tool_diversity == pytest.approx(0.75)
