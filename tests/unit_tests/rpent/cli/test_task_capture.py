# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task capture outcomes at runner cleanup boundaries."""

from __future__ import annotations

import argparse
import json
import logging
from types import SimpleNamespace

import pytest

from rpent.cli.task_capture import end_api_task, start_api_task
from rpent.planner.base import PlannerResult
from rpent.robots.robot_spec import RunConfig
from rpent.session import EnvState, Payload


@pytest.mark.parametrize(
    "exception,cancelled_reason,expected_status,expected_reason",
    [
        (None, None, "failed", "timeout"),
        (KeyboardInterrupt(), None, "cancelled", "operator"),
        (RuntimeError("cleanup failed"), None, "failed", "execution_error"),
        (None, "task_replaced", "cancelled", "task_replaced"),
    ],
)
def test_task_end_keeps_partial_records_and_actual_stop_cause(
    tmp_path, exception, cancelled_reason, expected_status, expected_reason
):
    state = EnvState(tmp_path)
    args = argparse.Namespace(planner="api", robot_name="libero", model="test", seed=42)
    config = RunConfig("test", tmp_path, {}, {"suite": "libero_object", "task": 3})
    assert (
        start_api_task(SimpleNamespace(state=state), args, config, "Move the cup.")
        is state
    )
    planner_id = state.register_planner(
        backend="api",
        status="completed",
        request=Payload("test", {"messages": ["Move the cup."]}),
    )
    args.seed = 99
    result = PlannerResult(
        error="planner timed out", stop_reason="timeout", stats={"turns_used": 1}
    )
    end_api_task(
        state,
        result,
        finish_result=None,
        environment_success=False,
        exception=exception,
        cancelled_reason=cancelled_reason,
    )
    task = state.get_task()
    assert (task.status, task.stop_reason) == (expected_status, expected_reason)
    assert task.config.data["arguments"]["seed"] == 42
    assert task.config.data["run"]["output_dir"] == str(tmp_path)
    assert task.result.data["environment_success"] is False
    assert task.ended_at >= task.started_at
    assert state.get_planner(planner_id).status == "completed"
    root = tmp_path / "trajectories" / task.task_id
    saved_task = json.loads((root / "task.json").read_text())["task"]
    assert (saved_task["status"], saved_task["stop_reason"]) == (
        expected_status,
        expected_reason,
    )
    assert saved_task["result"]["data"] == task.result.data
    assert json.loads((root / "planners.jsonl").read_text())["planner_id"] == planner_id
    with pytest.raises(RuntimeError, match="not running"):
        state.update_planner(planner_id, status="failed")


def test_successful_operator_verdict_is_separate_from_planner_error(tmp_path):
    state = EnvState(tmp_path)
    state.start_task(instruction="Task", config=Payload("test", {}))
    result = PlannerResult(error="interrupted model", stop_reason="planner_error")
    end_api_task(
        state,
        result,
        finish_result={"operator_finished": True, "status": "success"},
        environment_success=True,
        exception=None,
    )
    task = state.get_task()
    assert task.status == "completed" and task.stop_reason == "operator_finished"
    assert task.result.data["error"] == "interrupted model"
    root = tmp_path / "trajectories" / task.task_id
    assert json.loads((root / "task.json").read_text())["task"]["status"] == "completed"


def test_save_failure_is_logged_without_replacing_execution_error(
    tmp_path, monkeypatch, caplog
):
    state = EnvState(tmp_path)
    state.start_task(instruction="Task", config=Payload("test", {}))
    monkeypatch.setattr(logging.getLogger("rpent"), "propagate", True)

    def fail_save():
        raise OSError("disk full")

    monkeypatch.setattr(state, "save_planner_trajectory", fail_save)
    error = RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed") as caught:
        try:
            raise error
        finally:
            end_api_task(
                state,
                None,
                finish_result=None,
                environment_success=None,
                exception=error,
            )
    assert caught.value is error
    assert state.get_task().status == "failed"
    assert state.get_task().result.data["error"] == "RuntimeError: model failed"
    assert "Failed to save planner trajectory" in caplog.text
    assert "disk full" in caplog.text
