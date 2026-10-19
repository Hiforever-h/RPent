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

"""Task ownership, record identity, and independent model/action trajectories."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from rpent.session import DataRef, EnvState, MissingInfo, Payload


@pytest.fixture
def state(tmp_path: Path) -> EnvState:
    state = EnvState(tmp_path)
    state.start_task(instruction="Move the cup.", config=Payload("test.config", {}))
    return state


@pytest.fixture
def tool_id(state: EnvState) -> str:
    planner_id = state.register_planner(
        backend="api",
        status="completed",
        response=Payload("test.response", {"tool_calls": ["move"]}),
    )
    return state.register_tool_call(
        planner_id=planner_id, name="move", arguments=Payload("json", {})
    )


@pytest.mark.parametrize(
    "status,reason,result",
    [
        ("completed", "request_limit", {"environment_success": False}),
        ("failed", "timeout", {"error": "Planner timed out."}),
        ("cancelled", "operator", None),
    ],
)
def test_task_end_preserves_collected_data_and_independent_call_statuses(
    state: EnvState, tool_id: str, status: str, reason: str, result: dict | None
) -> None:
    observation_id = state.register_observation(
        control_step=0, capture_sequence=0, payload=Payload("test.observation", [0.0])
    )
    action_id = state.register_action(
        tool_call_id=tool_id,
        source="vla",
        action_format_id="joint",
        control_step=0,
        action=Payload("test.action", np.array([0.5], dtype=np.float32)),
        before_observation_id=observation_id,
        after_observation_id=None,
        missing={
            "after_observation_id": MissingInfo("capture_failed", "Camera timed out.")
        },
    )
    start = state.get_task()
    outcome = Payload("test.result", result) if result is not None else None

    state.end_task(status=status, stop_reason=reason, result=outcome)

    task = state.get_task()
    assert task.task_id == start.task_id
    assert task.status == status and task.stop_reason == reason
    assert task.started_at == start.started_at <= task.ended_at
    assert task.started_at.utcoffset().total_seconds() == 0
    assert task.result == outcome
    assert state.get_tool_call(tool_id).status == "requested"
    assert state.planner_records()[0].status == "completed"
    action = state.get_action(action_id)
    assert action.before_observation_id == observation_id
    assert action.after_observation_id is None
    assert action.missing["after_observation_id"].reason == "capture_failed"
    assert state.get_observation(observation_id).payload.data == [0.0]
    np.testing.assert_array_equal(action.action.data, [0.5])


def test_task_is_explicit_and_cannot_be_overwritten(tmp_path: Path) -> None:
    state = EnvState(tmp_path)
    assert state.get_task() is None
    assert state.planner_records() == state.action_records() == []
    assert state.save("recipe.json", [], step=None) == "recipe.json"
    with state.record_step(state={"ready": True}):
        pass
    assert state.get_task() is None
    with pytest.raises(RuntimeError, match="not running"):
        state.register_planner(backend="api")
    with pytest.raises(RuntimeError, match="not running"):
        state.end_task(status="completed", stop_reason="finish")

    task_id = state.start_task(instruction="Move.", config=Payload("json", {}))
    with pytest.raises(RuntimeError, match="already been started"):
        state.start_task(instruction="Replace.", config=Payload("json", {}))
    with pytest.raises(ValueError, match="invalid task end status"):
        state.end_task(status="running", stop_reason="finish")
    assert state.get_task().status == "running"

    state.end_task(status="completed", stop_reason="finish")
    with pytest.raises(RuntimeError, match="not running"):
        state.end_task(status="failed", stop_reason="error")
    with pytest.raises(RuntimeError, match="already been started"):
        state.start_task(instruction="Replace.", config=Payload("json", {}))
    assert state.get_task().task_id == task_id
    assert state.get_task().status == "completed"
    assert state.get_task().instruction == "Move."
    assert state.get().state == {"ready": True}

    other = EnvState(tmp_path / "other")
    assert other.start_task(instruction="Move.", config=Payload("json", {})) != task_id


def test_ended_task_rejects_new_records(state: EnvState, tool_id: str) -> None:
    planner_id = state.get_tool_call(tool_id).planner_id
    state.end_task(status="cancelled", stop_reason="operator")

    registrations = [
        lambda: state.register_planner(backend="api"),
        lambda: state.register_tool_call(
            planner_id=planner_id, name="move", arguments=Payload("json", {})
        ),
        lambda: state.register_observation(
            control_step=0, capture_sequence=0, payload=Payload("json", {})
        ),
        lambda: state.register_action(
            tool_call_id=tool_id,
            source="direct",
            action_format_id="joint",
            control_step=0,
            action=Payload("json", [0.5]),
            before_observation_id=None,
            after_observation_id=None,
        ),
    ]
    for register in registrations:
        with pytest.raises(RuntimeError, match="not running"):
            register()
    assert len(state.planner_records()) == 1
    assert len(state.get_planner(planner_id).tool_calls) == 1
    assert state.action_records() == []


def test_model_and_action_trajectories_share_explicit_links(state: EnvState) -> None:
    first = state.register_planner(backend="api", status="completed")
    second = state.register_planner(backend="api", status="completed")
    observations = [
        state.register_observation(
            control_step=step,
            capture_sequence=sequence,
            payload=Payload("test.observation", {"position": step}),
        )
        for sequence, step in enumerate([0, 1, 1, 2])
    ]
    first_tool = state.register_tool_call(
        planner_id=first,
        name="move",
        arguments=Payload("json", {}),
        status="completed",
        before_observation_id=observations[0],
        after_observation_id=observations[2],
    )
    second_tool = state.register_tool_call(
        planner_id=second,
        name="move",
        arguments=Payload("json", {}),
        status="failed",
        before_observation_id=observations[2],
        after_observation_id=observations[3],
    )
    read_tool = state.register_tool_call(
        planner_id=first,
        name="observe",
        arguments=Payload("json", {}),
        status="completed",
    )
    action_ids = []
    for tool, step, before, after, source in [
        (second_tool, 1, observations[2], observations[3], "tool_computed"),
        (first_tool, 0, observations[0], observations[1], "vla"),
        (second_tool, None, observations[3], None, "direct"),
    ]:
        action_ids.append(
            state.register_action(
                tool_call_id=tool,
                source=source,
                action_format_id="joint",
                control_step=step,
                action=Payload("test.action", [0.5]),
                before_observation_id=before,
                after_observation_id=after,
                missing=(
                    {
                        "control_step": MissingInfo("not_exposed", "Position unknown."),
                        "after_observation_id": MissingInfo(
                            "capture_failed", "Camera failed."
                        ),
                    }
                    if step is None
                    else {}
                ),
            )
        )

    planners = state.planner_records()
    assert [record.planner_id for record in planners] == [first, second]
    assert [call.tool_call_id for call in planners[0].tool_calls] == [
        first_tool,
        read_tool,
    ]
    assert [call.tool_call_id for call in planners[1].tool_calls] == [second_tool]
    assert state.get_tool_call(read_tool).planner_id == first
    actions = state.action_records()
    assert [record.action_id for record in actions] == [
        action_ids[1],
        action_ids[0],
        action_ids[2],
    ]
    assert [record.control_step for record in actions] == [0, 1, None]
    assert [record.source for record in actions] == ["vla", "tool_computed", "direct"]
    assert actions[1].after_observation_id == actions[2].before_observation_id
    assert (
        actions[0].after_observation_id
        != state.get_tool_call(first_tool).after_observation_id
    )
    assert observations[1] != observations[2]
    assert (
        state.get_observation(observations[1]).control_step
        == state.get_observation(observations[2]).control_step
    )
    assert state.get_observation(observations[2]).capture_sequence == 2
    assert state.get_tool_call(actions[1].tool_call_id).planner_id == second
    assert actions[2].missing["control_step"].reason == "not_exposed"


def test_registration_and_queries_isolate_nested_payloads(tmp_path: Path) -> None:
    state = EnvState(tmp_path)
    config = Payload("test.config", {"model": {"max_tokens": 1024}})
    state.start_task(instruction="Move.", config=config)
    request = Payload(
        "test.request", {"messages": [{"role": "user", "content": "Move."}]}
    )
    source_ids = {"pydantic_ai.run_id": "run_1"}
    planner = state.register_planner(
        backend="api", request=request, source_ids=source_ids
    )
    raw_result = Payload("test.result", {"done": True})
    tool = state.register_tool_call(
        planner_id=planner,
        name="move",
        arguments=Payload("json", {}),
        raw_result=raw_result,
    )
    values = np.array([0.1, 0.2], dtype=np.float32)
    obs = state.register_observation(
        control_step=0, capture_sequence=0, payload=Payload("test.obs", values)
    )
    action = state.register_action(
        tool_call_id=tool,
        source="vla",
        action_format_id="joint",
        control_step=0,
        action=Payload("test.action", {"joints": values}),
        before_observation_id=obs,
        after_observation_id=None,
        result=Payload("test.result", DataRef("tool_call", tool, "/raw_result/data")),
    )

    config.data["model"]["max_tokens"] = 1
    request.data["messages"].clear()
    source_ids.clear()
    raw_result.data["done"] = False
    values[:] = 99
    state.get_task().config.data.clear()
    state.get_planner(planner).request.data.clear()
    state.get_tool_call(tool).raw_result.data.clear()
    state.get_observation(obs).payload.data[:] = 88
    state.get_action(action).action.data["joints"][:] = 77
    state.planner_records()[0].tool_calls.clear()
    state.action_records()[0].before_observation_id = None

    assert state.get_task().config.data == {"model": {"max_tokens": 1024}}
    assert state.get_planner(planner).request.data["messages"][0]["content"] == "Move."
    assert state.get_planner(planner).source_ids == {"pydantic_ai.run_id": "run_1"}
    assert state.get_planner(planner).tool_calls[0].raw_result.data == {"done": True}
    np.testing.assert_array_equal(
        state.get_observation(obs).payload.data, np.array([0.1, 0.2], dtype=np.float32)
    )
    stored = state.get_action(action)
    np.testing.assert_array_equal(
        stored.action.data["joints"], state.get_observation(obs).payload.data
    )
    assert stored.before_observation_id == obs
    assert stored.result.data == DataRef("tool_call", tool, "/raw_result/data")
    outcome = Payload("test.result", {"environment_success": False})
    state.end_task(status="completed", stop_reason="finish", result=outcome)
    outcome.data.clear()
    state.get_task().result.data.clear()
    assert state.get_task().result.data == {"environment_success": False}


def test_interleaved_registration_keeps_ids_and_parent_links(state: EnvState) -> None:
    planners = [state.register_planner(backend="api") for _ in range(2)]

    def register(index: int) -> tuple[str, str, str]:
        tool = state.register_tool_call(
            planner_id=planners[index % 2],
            name="move",
            arguments=Payload("json", {"index": index}),
        )
        observation = state.register_observation(
            control_step=index,
            capture_sequence=index,
            payload=Payload("test.obs", index),
        )
        action = state.register_action(
            tool_call_id=tool,
            source="direct",
            action_format_id="joint",
            control_step=index,
            action=Payload("test.action", [index]),
            before_observation_id=observation,
            after_observation_id=None,
        )
        return tool, observation, action

    with ThreadPoolExecutor(max_workers=8) as pool:
        identities = list(pool.map(register, range(32)))

    assert len({identity for group in identities for identity in group}) == 96
    assert [action.control_step for action in state.action_records()] == list(range(32))
    for index, (tool, observation, action) in enumerate(identities):
        assert state.get_tool_call(tool).planner_id == planners[index % 2]
        assert state.get_tool_call(tool).arguments.data == {"index": index}
        assert state.get_action(action).tool_call_id == tool
        assert state.get_action(action).before_observation_id == observation
        assert state.get_observation(observation).payload.data == index
    for index, planner in enumerate(state.planner_records()):
        assert sorted(
            call.arguments.data["index"] for call in planner.tool_calls
        ) == list(range(index, 32, 2))


def test_legacy_reset_and_rollback_preserve_task_records(
    state: EnvState, tool_id: str
) -> None:
    task = state.get_task()
    original_planner = state.get_tool_call(tool_id).planner_id
    with state.record_step(state={"initial": True}):
        state.save("old.bin", b"initial")
    observation = state.register_observation(
        control_step=0, capture_sequence=0, payload=Payload("test.obs", [0])
    )
    with pytest.raises(RuntimeError, match="failed to record step"):
        with state.record_step(state={"partial": True}) as step:
            state.save("partial.bin", b"partial")
            action = state.register_action(
                tool_call_id=tool_id,
                source="direct",
                action_format_id="joint",
                control_step=0,
                action=Payload("test.action", [1]),
                before_observation_id=observation,
                after_observation_id=None,
            )
            raise ValueError("capture failed after execution")

    assert not state.exists("partial.bin", step=step)
    state.reset()
    assert state.records() == [] and state.latest_step is None
    assert state.load("old.bin", step=0) == b"initial"
    assert state.get_task() == task
    assert state.get_action(action).before_observation_id == observation
    assert state.get_tool_call(tool_id).planner_id == original_planner
    assert state.get_observation(observation).payload.data == [0]
    assert state.register_planner(backend="api") != original_planner
    assert (
        state.register_tool_call(
            planner_id=original_planner, name="move", arguments=Payload("json", {})
        )
        != tool_id
    )
    assert (
        state.register_observation(
            control_step=0, capture_sequence=1, payload=Payload("test.obs", [0])
        )
        != observation
    )
    assert (
        state.register_action(
            tool_call_id=tool_id,
            source="direct",
            action_format_id="joint",
            control_step=1,
            action=Payload("test.action", [2]),
            before_observation_id=observation,
            after_observation_id=None,
        )
        != action
    )
    assert len(state.action_records()) == 2
    assert len(state.get_planner(original_planner).tool_calls) == 2


def test_unknown_parents_do_not_publish_child_records(state: EnvState) -> None:
    with pytest.raises(KeyError, match="unknown-planner"):
        state.register_tool_call(
            planner_id="unknown-planner", name="move", arguments=Payload("json", {})
        )
    with pytest.raises(KeyError, match="unknown-tool"):
        state.register_action(
            tool_call_id="unknown-tool",
            source="direct",
            action_format_id="joint",
            control_step=0,
            action=Payload("json", [1]),
            before_observation_id=None,
            after_observation_id=None,
        )
    assert state.planner_records() == []
    assert state.action_records() == []
    for get in [
        state.get_planner,
        state.get_tool_call,
        state.get_observation,
        state.get_action,
    ]:
        with pytest.raises(KeyError):
            get("unknown")
