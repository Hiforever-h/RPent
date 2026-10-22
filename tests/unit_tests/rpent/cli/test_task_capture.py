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
import os
from types import SimpleNamespace

import numpy as np
import pytest

from rpent.cli.task_capture import end_api_task, start_api_task
from rpent.planner.base import PlannerResult
from rpent.robots.robot_spec import RunConfig
from rpent.session import EnvState, MissingInfo, Payload


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
    tool_id = state.register_tool_call(
        planner_id=planner_id, name="move", arguments=Payload("json", {})
    )
    observation_id = state.register_observation(payload=Payload("test.obs", [1]))
    action_id = state.register_action(
        tool_call_id=tool_id,
        source="vla",
        action=Payload("test.action", [0.5]),
        before_observation_id=None,
        after_observation_id=observation_id,
        missing={"before_observation_id": MissingInfo("capture_failed", "RPC failed.")},
    )
    prediction_id = state.register_prediction(
        tool_call_id=tool_id,
        input_observation_id=None,
        instruction="Move the cup.",
        actions=Payload("test.actions", [[0.5]]),
        action_ids=[action_id],
        missing={"input_observation_id": MissingInfo("capture_failed", "RPC failed.")},
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
    saved_action = json.loads((root / "actions.jsonl").read_text())
    assert saved_action["action_id"] == action_id
    assert saved_action["tool_call_id"] == tool_id
    assert saved_action["before_observation"] is None
    assert (
        saved_action["missing"]["before_observation_id"]["reason"] == "capture_failed"
    )
    assert saved_action["after_observation"]["payload"]["data"] == [1]
    prediction = json.loads((root / "predictions.jsonl").read_text())
    assert prediction["prediction_id"] == prediction_id
    assert (
        prediction["planner_id"] == planner_id and prediction["tool_call_id"] == tool_id
    )
    assert prediction["action_ids"] == [action_id]
    assert prediction["input_observation"] is None
    assert prediction["missing"]["input_observation_id"]["reason"] == "capture_failed"
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


@pytest.mark.parametrize(
    "attachment", ["planner", "action", "observation", "prediction"]
)
def test_task_end_saves_partial_trajectory_after_attachment_failure(
    tmp_path, monkeypatch, attachment
):
    state = EnvState(tmp_path)
    state.start_task(instruction="Move", config=Payload("test", {}))
    planner_id = state.register_planner(
        backend="api",
        request=Payload(
            "test.request",
            {"messages": [{"image/~": b"image", "text": "Move", "optional": None}]},
            completeness="partial",
            missing={"/metadata": MissingInfo("not_exposed", "No metadata.")},
        ),
    )
    tool_id = state.register_tool_call(
        planner_id=planner_id, name="move", arguments=Payload("json", {})
    )
    states = np.ones(8, np.float32)
    observation_id = state.register_observation(
        payload=Payload("test.obs", {"states": states}, completeness="complete")
    )
    actions = [
        state.register_action(
            tool_call_id=tool_id,
            source="vla",
            action=Payload("test.action", np.array([index]), completeness="complete"),
            before_observation_id=observation_id if index else None,
            after_observation_id=None if index else observation_id,
        )
        for index in range(2)
    ]
    prediction_actions = np.array([[1]])
    state.register_prediction(
        tool_call_id=tool_id,
        input_observation_id=observation_id,
        instruction="Move.",
        actions=Payload("test.actions", prediction_actions, completeness="complete"),
        action_ids=actions[1:],
    )
    failed_data = {
        "planner": b"image",
        "action": np.array([0]),
        "observation": states,
        "prediction": prediction_actions,
    }[attachment]
    replace = os.replace

    def fail_attachment(source, destination):
        if destination.parent.name == "artifacts":
            data = (
                source.read_bytes()
                if destination.suffix == ".bin"
                else np.load(source, allow_pickle=False)
            )
            if np.array_equal(data, failed_data):
                raise OSError("attachment write failed")
        replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr("rpent.session.base.os.replace", fail_attachment)
        end_api_task(
            state,
            None,
            finish_result=None,
            environment_success=False,
            exception=RuntimeError("model failed"),
        )

    root = tmp_path / "trajectories" / state.get_task().task_id
    task = json.loads((root / "task.json").read_text())["task"]
    request = json.loads((root / "planners.jsonl").read_text())["request"]
    rows = [
        json.loads(line) for line in (root / "actions.jsonl").read_text().splitlines()
    ]
    assert (task["status"], task["stop_reason"]) == ("failed", "execution_error")
    assert task["result"]["data"]["error"] == "RuntimeError: model failed"
    assert [row["action_id"] for row in rows] == actions
    assert rows[0]["after_observation"] == rows[1]["before_observation"]
    prediction = json.loads((root / "predictions.jsonl").read_text())
    assert prediction["action_ids"] == actions[1:]
    assert prediction["input_observation"] == rows[1]["before_observation"]
    prediction_payload = prediction["actions"]
    action = rows[0]["action"]
    observation = rows[0]["after_observation"]["payload"]
    payload, pointer, absent = {
        "planner": (
            request,
            "/messages/0/image~1~0",
            request["data"]["messages"][0]["image/~"],
        ),
        "action": (action, "", action["data"]),
        "observation": (observation, "/states", observation["data"]["states"]),
        "prediction": (prediction_payload, "", prediction_payload["data"]),
    }[attachment]
    assert absent is None
    assert payload["completeness"] == "partial"
    assert payload["missing"][pointer] == {
        "reason": "capture_failed",
        "detail": "OSError: attachment write failed",
    }
    assert request["missing"]["/metadata"]["reason"] == "not_exposed"
    assert "/messages/0/optional" not in request["missing"]
    assert rows[1]["action"]["completeness"] == "complete"
    np.testing.assert_array_equal(
        np.load(root / rows[1]["action"]["data"]["path"], allow_pickle=False),
        [1],
    )
    assert not list(root.rglob(".*.tmp*"))

    state.save_planner_trajectory()
    restored = json.loads((root / "planners.jsonl").read_text())["request"]
    assert set(restored["missing"]) == {"/metadata"}
    for line in (root / "actions.jsonl").read_text().splitlines():
        row = json.loads(line)
        observation = row["before_observation"] or row["after_observation"]
        for restored in (row["action"], observation["payload"]):
            assert restored["completeness"] == "complete"
            assert restored["missing"] == {}
    restored = json.loads((root / "predictions.jsonl").read_text())["actions"]
    assert restored["completeness"] == "complete" and restored["missing"] == {}
    np.testing.assert_array_equal(
        np.load(root / restored["data"]["path"], allow_pickle=False), prediction_actions
    )


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
