"""Unit tests for agent workflow evaluation helpers."""

import pytest

from agent_eval import (
    AgentAction,
    AgentTrace,
    AgentWorkflowEvaluator,
    FailureMode,
    classify_failure_mode,
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

    def test_report_aggregates_precision_and_f1(self):
        traces = [
            _trace(
                "pf1",
                [AgentAction(step=1, action_type="search", query="q", retrieved_docs=["a", "b"])],
                success=True,
                relevant_docs=["a", "b"],  # precision=1.0, recall=1.0, f1=1.0
            ),
            _trace(
                "pf2",
                [AgentAction(step=1, action_type="search", query="q", retrieved_docs=["a", "x"])],
                success=True,
                relevant_docs=["a", "b"],  # precision=0.5, recall=0.5, f1=0.5
            ),
        ]
        ev = AgentWorkflowEvaluator()
        ev.add_traces(traces)
        report = ev.evaluate()
        assert report.mean_precision == pytest.approx(0.75)
        assert report.mean_f1 == pytest.approx(0.75)

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


# ---------------------------------------------------------------------------
# Precision and F1 at final step
# ---------------------------------------------------------------------------


class TestPrecisionAndF1:
    def test_precision_perfect(self):
        trace = _trace(
            "p1",
            [AgentAction(step=1, action_type="search", query="q", retrieved_docs=["a", "b"])],
            success=True,
            relevant_docs=["a", "b"],
        )
        ev = AgentWorkflowEvaluator()
        ev.add_trace(trace)
        result = ev.evaluate().per_task[0]
        assert result.precision_at_final == pytest.approx(1.0)
        assert result.f1_at_final == pytest.approx(1.0)

    def test_precision_with_noise(self):
        # agent retrieved 2 relevant + 2 irrelevant docs
        trace = _trace(
            "p2",
            [
                AgentAction(
                    step=1, action_type="search", query="q", retrieved_docs=["a", "b", "x", "y"]
                )
            ],
            success=True,
            relevant_docs=["a", "b"],
        )
        ev = AgentWorkflowEvaluator()
        ev.add_trace(trace)
        result = ev.evaluate().per_task[0]
        assert result.precision_at_final == pytest.approx(0.5)  # 2/4
        assert result.recall_at_final == pytest.approx(1.0)  # 2/2
        assert result.f1_at_final == pytest.approx(2 / 3)  # harmonic mean(0.5, 1.0)

    def test_precision_zero_retrieved_gives_zero(self):
        trace = _trace(
            "p3",
            [AgentAction(step=1, action_type="answer")],
            success=False,
            relevant_docs=["a"],
        )
        ev = AgentWorkflowEvaluator()
        ev.add_trace(trace)
        result = ev.evaluate().per_task[0]
        assert result.precision_at_final == 0.0
        assert result.f1_at_final == 0.0

    def test_f1_symmetric(self):
        # precision = 1/3, recall = 1/2 → F1 = 2*(1/3)*(1/2) / (1/3+1/2) = 2/5
        trace = _trace(
            "p4",
            [AgentAction(step=1, action_type="search", query="q", retrieved_docs=["a", "x", "y"])],
            success=True,
            relevant_docs=["a", "b"],
        )
        ev = AgentWorkflowEvaluator()
        ev.add_trace(trace)
        result = ev.evaluate().per_task[0]
        assert result.precision_at_final == pytest.approx(1 / 3)
        assert result.recall_at_final == pytest.approx(1 / 2)
        assert result.f1_at_final == pytest.approx(2 / 5)


# ---------------------------------------------------------------------------
# Failure mode taxonomy
# ---------------------------------------------------------------------------


class TestFailureModeTaxonomy:
    def _eval(self, task_id, actions, success, relevant_docs, min_steps=1):
        ev = AgentWorkflowEvaluator()
        ev.add_trace(_trace(task_id, actions, success, relevant_docs, min_steps))
        return ev.evaluate().per_task[0]

    def test_clean_success_mode(self):
        result = self._eval(
            "fm1",
            [AgentAction(step=1, action_type="search", query="q", retrieved_docs=["a"])],
            success=True,
            relevant_docs=["a"],
        )
        assert result.failure_mode == FailureMode.SUCCESS

    def test_no_results_mode(self):
        result = self._eval(
            "fm2",
            [AgentAction(step=1, action_type="search", query="q", retrieved_docs=["x"])],
            success=False,
            relevant_docs=["a", "b"],
        )
        assert result.failure_mode == FailureMode.NO_RESULTS

    def test_low_recall_mode(self):
        # recall = 1/4 < 0.5 threshold, task failed
        result = self._eval(
            "fm3",
            [AgentAction(step=1, action_type="search", query="q", retrieved_docs=["a"])],
            success=False,
            relevant_docs=["a", "b", "c", "d"],
        )
        assert result.failure_mode == FailureMode.LOW_RECALL

    def test_inefficient_path_due_to_overhead(self):
        # success=True but 6 steps vs min_steps=1 → overhead=6 > threshold=2
        actions = [AgentAction(step=i, action_type="search", query=f"q{i}") for i in range(1, 7)]
        result = self._eval("fm4", actions, success=True, relevant_docs=[], min_steps=1)
        assert result.failure_mode == FailureMode.INEFFICIENT_PATH

    def test_inefficient_path_due_to_redundant_queries(self):
        result = self._eval(
            "fm5",
            [
                AgentAction(step=1, action_type="search", query="dup", retrieved_docs=["a"]),
                AgentAction(step=2, action_type="search", query="dup", retrieved_docs=["a"]),
                AgentAction(step=3, action_type="answer"),
            ],
            success=True,
            relevant_docs=["a"],
            min_steps=1,
        )
        assert result.failure_mode == FailureMode.INEFFICIENT_PATH

    def test_failure_mode_stored_on_task_result(self):
        result = self._eval(
            "fm6",
            [AgentAction(step=1, action_type="answer")],
            success=False,
            relevant_docs=["a"],
        )
        assert isinstance(result.failure_mode, FailureMode)

    def test_failure_mode_distribution(self):
        traces = [
            _trace(
                "d1",
                [AgentAction(step=1, action_type="search", query="q", retrieved_docs=["a"])],
                True,
                ["a"],
            ),
            _trace(
                "d2",
                [AgentAction(step=1, action_type="search", query="q", retrieved_docs=["x"])],
                False,
                ["a"],
            ),
            _trace(
                "d3",
                [AgentAction(step=1, action_type="search", query="q", retrieved_docs=["x"])],
                False,
                ["a"],
            ),
        ]
        ev = AgentWorkflowEvaluator()
        ev.add_traces(traces)
        report = ev.evaluate()
        dist = report.failure_mode_distribution()
        assert dist["success"] == 1
        assert dist["no_results"] == 2

        rates = report.failure_mode_rates()
        assert rates["success"] == pytest.approx(1 / 3)
        assert rates["no_results"] == pytest.approx(2 / 3)
        assert sum(rates.values()) == pytest.approx(1.0)

        no_results_tasks = report.tasks_with_failure_mode(FailureMode.NO_RESULTS)
        assert {r.task_id for r in no_results_tasks} == {"d2", "d3"}
        assert report.tasks_with_failure_mode(FailureMode.SUCCESS)[0].task_id == "d1"
        assert report.tasks_with_failure_mode(FailureMode.LOW_RECALL) == []

    def test_noisy_results_mode(self):
        # success=True, retrieved 1 relevant + 4 irrelevant → precision=0.2 < 0.3 threshold
        result = self._eval(
            "fm_noise",
            [
                AgentAction(
                    step=1,
                    action_type="search",
                    query="q",
                    retrieved_docs=["a", "x", "y", "z", "w"],
                ),
            ],
            success=True,
            relevant_docs=["a"],
        )
        assert result.failure_mode == FailureMode.NOISY_RESULTS

    def test_noisy_results_skipped_when_no_relevant_docs(self):
        # success=True, no ground-truth relevant docs — precision is 0 but not "noisy"
        result = self._eval(
            "fm_noise_skip",
            [AgentAction(step=1, action_type="search", query="q", retrieved_docs=["x"])],
            success=True,
            relevant_docs=[],
        )
        assert result.failure_mode == FailureMode.SUCCESS

    def test_noisy_results_takes_priority_over_inefficient_path(self):
        # Both noisy (precision=0.25) and inefficient (3 steps vs min=1) → noisy wins
        result = self._eval(
            "fm_priority",
            [
                AgentAction(
                    step=1, action_type="search", query="q1", retrieved_docs=["a", "x", "y", "z"]
                ),
                AgentAction(step=2, action_type="search", query="q2"),
                AgentAction(step=3, action_type="answer"),
            ],
            success=True,
            relevant_docs=["a"],
            min_steps=1,
        )
        assert result.failure_mode == FailureMode.NOISY_RESULTS

    def test_classify_failure_mode_custom_precision_threshold(self):
        # precision = 0.5 (1 relevant + 1 noise)
        result = self._eval(
            "fm_pt",
            [AgentAction(step=1, action_type="search", query="q", retrieved_docs=["a", "x"])],
            success=True,
            relevant_docs=["a"],
        )
        # 0.5 is acceptable at default threshold (0.3)
        assert classify_failure_mode(result, precision_threshold=0.3) == FailureMode.SUCCESS
        # 0.5 is noisy at a stricter threshold (0.8)
        assert classify_failure_mode(result, precision_threshold=0.8) == FailureMode.NOISY_RESULTS

    def test_classify_failure_mode_custom_overhead_threshold(self):
        # success=True, 3 steps, min_steps=2 → overhead=1.5
        result = self._eval(
            "fm7",
            [
                AgentAction(step=1, action_type="search", query="q1", retrieved_docs=["a"]),
                AgentAction(step=2, action_type="search", query="q2"),
                AgentAction(step=3, action_type="answer"),
            ],
            success=True,
            relevant_docs=["a"],
            min_steps=2,
        )
        # overhead=1.5 is efficient at default threshold (2.0)
        assert classify_failure_mode(result, overhead_threshold=2.0) == FailureMode.SUCCESS
        # overhead=1.5 is inefficient at a stricter threshold (1.0)
        assert classify_failure_mode(result, overhead_threshold=1.0) == FailureMode.INEFFICIENT_PATH

    def test_print_summary_includes_failure_modes(self, capsys):
        ev = AgentWorkflowEvaluator()
        ev.add_trace(
            _trace(
                "ps1",
                [AgentAction(step=1, action_type="search", query="q", retrieved_docs=["a"])],
                True,
                ["a"],
            )
        )
        ev.evaluate().print_summary()
        out = capsys.readouterr().out
        assert "Failure modes" in out
        assert "success" in out
