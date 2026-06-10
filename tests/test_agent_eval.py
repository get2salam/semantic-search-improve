"""Unit tests for agent workflow evaluation helpers."""

import pytest

from agent_eval import (
    AgentAction,
    AgentTrace,
    AgentWorkflowEvaluator,
    FailureMode,
    classify_failure_mode,
    compare_reports,
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

    def test_evaluator_respects_custom_precision_threshold(self):
        # precision = 0.5 (1 relevant + 1 noise); default threshold 0.3 → SUCCESS,
        # stricter threshold 0.8 → NOISY_RESULTS. The evaluator must pass the
        # configured threshold through to classify_failure_mode.
        actions = [
            AgentAction(step=1, action_type="search", query="q", retrieved_docs=["a", "x"]),
        ]
        relevant = ["a"]

        lenient = AgentWorkflowEvaluator(precision_threshold=0.3)
        lenient.add_trace(_trace("cfg_p", actions, success=True, relevant_docs=relevant))
        assert lenient.evaluate().per_task[0].failure_mode == FailureMode.SUCCESS

        strict = AgentWorkflowEvaluator(precision_threshold=0.8)
        strict.add_trace(_trace("cfg_p", actions, success=True, relevant_docs=relevant))
        assert strict.evaluate().per_task[0].failure_mode == FailureMode.NOISY_RESULTS

    def test_evaluator_respects_custom_overhead_threshold(self):
        # 3 steps vs min_steps=2 → overhead=1.5; lenient default 2.0 → SUCCESS,
        # strict threshold 1.0 → INEFFICIENT_PATH.
        actions = [
            AgentAction(step=1, action_type="search", query="q1", retrieved_docs=["a"]),
            AgentAction(step=2, action_type="search", query="q2"),
            AgentAction(step=3, action_type="answer"),
        ]

        lenient = AgentWorkflowEvaluator(overhead_threshold=2.0)
        lenient.add_trace(_trace("cfg_o", actions, True, ["a"], min_steps=2))
        assert lenient.evaluate().per_task[0].failure_mode == FailureMode.SUCCESS

        strict = AgentWorkflowEvaluator(overhead_threshold=1.0)
        strict.add_trace(_trace("cfg_o", actions, True, ["a"], min_steps=2))
        assert strict.evaluate().per_task[0].failure_mode == FailureMode.INEFFICIENT_PATH

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


# ---------------------------------------------------------------------------
# Clean success rate
# ---------------------------------------------------------------------------


class TestCleanSuccessRate:
    def test_clean_success_rate_excludes_noisy_and_inefficient(self):
        # Three success=True traces: one clean SUCCESS, one NOISY_RESULTS,
        # one INEFFICIENT_PATH (redundant query). Raw success rate is 1.0,
        # but clean_success_rate is 1/3.
        traces = [
            _trace(
                "clean",
                [AgentAction(step=1, action_type="search", query="q", retrieved_docs=["a"])],
                success=True,
                relevant_docs=["a"],
            ),
            _trace(
                "noisy",
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
            ),
            _trace(
                "inefficient",
                [
                    AgentAction(step=1, action_type="search", query="dup", retrieved_docs=["a"]),
                    AgentAction(step=2, action_type="search", query="dup", retrieved_docs=["a"]),
                ],
                success=True,
                relevant_docs=["a"],
            ),
        ]
        ev = AgentWorkflowEvaluator()
        ev.add_traces(traces)
        report = ev.evaluate()
        assert report.task_success_rate == pytest.approx(1.0)
        assert report.clean_success_rate == pytest.approx(1 / 3)

    def test_clean_success_rate_in_print_summary(self, capsys):
        ev = AgentWorkflowEvaluator()
        ev.add_trace(
            _trace(
                "cs1",
                [AgentAction(step=1, action_type="search", query="q", retrieved_docs=["a"])],
                True,
                ["a"],
            )
        )
        ev.evaluate().print_summary()
        assert "Clean success rate" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Report comparison (baseline → candidate regression detection)
# ---------------------------------------------------------------------------


def _report(traces):
    ev = AgentWorkflowEvaluator()
    ev.add_traces(traces)
    return ev.evaluate()


def _clean_trace(task_id="t"):
    return _trace(
        task_id,
        [AgentAction(step=1, action_type="search", query="q", retrieved_docs=["a"])],
        success=True,
        relevant_docs=["a"],
    )


def _noisy_trace(task_id="t"):
    return _trace(
        task_id,
        [
            AgentAction(
                step=1,
                action_type="search",
                query="q",
                retrieved_docs=["a", "x", "y", "z", "w"],
            )
        ],
        success=True,
        relevant_docs=["a"],
    )


def _no_results_trace(task_id="t"):
    return _trace(
        task_id,
        [AgentAction(step=1, action_type="search", query="q", retrieved_docs=["x"])],
        success=False,
        relevant_docs=["a"],
    )


class TestCompareReports:
    def test_identical_reports_have_no_regressions(self):
        report = _report([_clean_trace("t1"), _clean_trace("t2")])
        diff = compare_reports(report, report)
        assert diff.improvements == []
        assert diff.regressions == []
        assert not diff.has_regression()

    def test_higher_is_better_metric_improvement(self):
        # baseline: 1 success + 1 failure → success_rate 0.5
        # candidate: 2 successes → success_rate 1.0
        baseline = _report([_clean_trace("a"), _no_results_trace("b")])
        candidate = _report([_clean_trace("a"), _clean_trace("b")])
        diff = compare_reports(baseline, candidate)

        success_metric = next(m for m in diff.metrics if m.name == "task_success_rate")
        assert success_metric.delta == pytest.approx(0.5)
        assert success_metric.is_improvement
        assert not success_metric.is_regression
        assert success_metric in diff.improvements

    def test_higher_is_better_metric_regression(self):
        # Drop from all-success baseline to half-success candidate
        baseline = _report([_clean_trace("a"), _clean_trace("b")])
        candidate = _report([_clean_trace("a"), _no_results_trace("b")])
        diff = compare_reports(baseline, candidate)

        recall = next(m for m in diff.metrics if m.name == "mean_recall")
        assert recall.delta == pytest.approx(-0.5)
        assert recall.is_regression
        assert not recall.is_improvement
        assert diff.has_regression()

    def test_lower_is_better_metric_improvement(self):
        # Candidate uses fewer steps than baseline → mean_steps drop is an improvement
        long_trace = _trace(
            "long",
            [
                AgentAction(step=1, action_type="search", query="q1", retrieved_docs=["a"]),
                AgentAction(step=2, action_type="search", query="q2"),
                AgentAction(step=3, action_type="answer"),
            ],
            success=True,
            relevant_docs=["a"],
        )
        baseline = _report([long_trace])
        candidate = _report([_clean_trace("short")])
        diff = compare_reports(baseline, candidate)

        steps = next(m for m in diff.metrics if m.name == "mean_steps")
        assert steps.delta == pytest.approx(-2.0)
        assert steps.is_improvement
        assert not steps.is_regression

    def test_lower_is_better_metric_regression(self):
        # Candidate has redundant query, baseline does not
        baseline = _report([_clean_trace("a")])
        candidate = _report(
            [
                _trace(
                    "dup",
                    [
                        AgentAction(step=1, action_type="search", query="q", retrieved_docs=["a"]),
                        AgentAction(step=2, action_type="search", query="q", retrieved_docs=["a"]),
                    ],
                    success=True,
                    relevant_docs=["a"],
                )
            ]
        )
        diff = compare_reports(baseline, candidate)

        redundant = next(m for m in diff.metrics if m.name == "mean_redundant_queries")
        assert redundant.delta == pytest.approx(1.0)
        assert redundant.is_regression
        assert diff.has_regression()

    def test_tolerance_filters_noise(self):
        # Build two reports with identical metrics — both rely on the same
        # AgentTrace, so deltas should be exactly zero. Negative tolerance
        # is rejected; non-negative tolerances treat exact zeros as no-change.
        report = _report([_clean_trace("t1")])
        diff = compare_reports(report, report, tolerance=0.01)
        # All deltas are exactly zero; nothing should be flagged.
        assert diff.improvements == []
        assert diff.regressions == []

    def test_negative_tolerance_rejected(self):
        report = _report([_clean_trace("t1")])
        with pytest.raises(ValueError, match="tolerance"):
            compare_reports(report, report, tolerance=-0.1)

    def test_failure_mode_rate_deltas(self):
        # baseline: 1 clean success + 1 no_results
        # candidate: 2 noisy
        baseline = _report([_clean_trace("a"), _no_results_trace("b")])
        candidate = _report([_noisy_trace("a"), _noisy_trace("b")])
        diff = compare_reports(baseline, candidate)

        # baseline rates: success=0.5, no_results=0.5
        # candidate rates: noisy_results=1.0
        assert diff.failure_mode_rate_deltas["success"] == pytest.approx(-0.5)
        assert diff.failure_mode_rate_deltas["no_results"] == pytest.approx(-0.5)
        assert diff.failure_mode_rate_deltas["noisy_results"] == pytest.approx(1.0)

    def test_metric_delta_zero_is_neither_improvement_nor_regression(self):
        report = _report([_clean_trace("t1")])
        diff = compare_reports(report, report)
        for m in diff.metrics:
            assert m.delta == 0
            assert not m.is_improvement
            assert not m.is_regression

    def test_print_summary_runs(self, capsys):
        baseline = _report([_clean_trace("a"), _no_results_trace("b")])
        candidate = _report([_clean_trace("a"), _clean_trace("b")])
        compare_reports(baseline, candidate).print_summary()
        out = capsys.readouterr().out
        assert "Agent Workflow Comparison" in out
        assert "task_success_rate" in out
        assert "Regressions" in out
        assert "Improvements" in out

    def test_compares_all_aggregate_metrics(self):
        # Every aggregate metric on AgentWorkflowReport that has a defined
        # direction should appear in the comparison so callers can rely on
        # full coverage when gating eval pipelines.
        report = _report([_clean_trace("t1")])
        diff = compare_reports(report, report)
        names = {m.name for m in diff.metrics}
        expected = {
            "task_success_rate",
            "clean_success_rate",
            "mean_recall",
            "mean_precision",
            "mean_f1",
            "mean_tool_diversity",
            "mean_steps",
            "mean_search_steps",
            "mean_step_overhead",
            "mean_redundant_queries",
        }
        assert names == expected
