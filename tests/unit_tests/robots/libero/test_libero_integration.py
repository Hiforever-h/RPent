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

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from robots.libero import toolkit as libero_toolkit
from robots.libero.env_server import LiberoEnvFacade
from robots.libero.tools import LiberoPrimitives
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.planner.api_loop import ApiAgentLoop
from rpent.robots.components.action_spec import validate_action
from rpent.session import EnvState, Payload
from rpent.session.capture import CapturingEnvClient, ToolCallCapture
from rpent.tools import ToolResult
from rpent.tools.toolkit import ToolCancelled
from rpent.utils.rpc import RpcError


def _obs(value: int) -> dict:
    return {
        "main_images": np.full((256, 256, 3), value, np.uint8),
        "wrist_images": np.full((256, 256, 3), value + 1, np.uint8),
        "states": np.full(8, value, np.float32),
        "task_descriptions": "put the bowl on the plate",
    }


class _Env:
    action_specs = {"default": {"low": [-1.0] * 7, "high": [1.0] * 7}}

    def __init__(self):
        self.terminated = False
        self.truncated = False
        self.return_all_frames = False
        self.chunk_return_all_frames = None
        self.last_obs = _obs(0)

    def reset(self):
        self.last_obs = _obs(0)
        return self.last_obs, {}

    def _step(self, action):
        del action
        return _obs(1), 0, False, False, {}

    def validate_action(self, values):
        return validate_action(values, self.action_specs["default"])

    def _chunk_step(self, actions, *, return_all_frames=None):
        observations = [_obs(index + 2) for index in range(len(actions))]
        result = observations if return_all_frames else observations[-1]
        terminated = np.zeros(len(actions), np.bool_)
        terminated[-1] = True
        self.terminated = True
        return (
            result,
            np.zeros(len(actions)),
            terminated,
            np.zeros(len(actions)),
            [{} for _ in actions],
        )

    def step(self, action, *, capture=False):
        before = self.last_obs
        result = self._step(action)
        self.last_obs = result[0]
        if capture:
            return LiberoEnvFacade._capture_result(
                result, before, [np.asarray(action)], [result]
            )
        return result

    def chunk_step(self, actions, *, return_all_frames=None, capture=False):
        self.chunk_return_all_frames = return_all_frames
        before = self.last_obs
        all_frames = (
            self.return_all_frames if return_all_frames is None else return_all_frames
        )
        result = self._chunk_step(actions, return_all_frames=all_frames or capture)
        obs, rewards, term, trunc, infos = result
        self.last_obs = obs[-1] if all_frames or capture else obs
        if not capture:
            return result
        native_result = (obs if all_frames else obs[-1], rewards, term, trunc, infos)
        return LiberoEnvFacade._capture_result(
            native_result,
            before,
            np.asarray(actions),
            zip(obs, rewards, term, trunc, infos),
        )


class _Model:
    def predict(self, observation, *, options):
        assert observation["task_descriptions"] == "pick up the bowl"
        assert options["mode"] == "eval"
        return np.ones((2, 7), np.float32)


def _primitives(env, molmo_client=None):
    return LiberoPrimitives(
        env=env,
        model=_Model(),
        sam3_client=SimpleNamespace(),
        check_cancelled=lambda: None,
        molmo_client=molmo_client,
    )


@pytest.mark.parametrize(
    ("mode", "record_trajectory"),
    [("evaluation", True), ("exploration", True), ("evaluation", False)],
)
def test_actions_follow_api_tool_calls(tmp_path, monkeypatch, mode, record_trajectory):
    monkeypatch.setattr("rpent.tools.toolkit.substitute", lambda value: value)
    monkeypatch.setattr("rpent.planner.api_loop.substitute", lambda value: value)
    monkeypatch.setattr(
        libero_toolkit.libero_tools,
        "dump_state",
        lambda *args, **kwargs: SimpleNamespace(step_idx=0, terminated=False),
    )
    monkeypatch.setattr(
        libero_toolkit.LiberoToolkit,
        "get_env_state",
        lambda self, *, command, result, elapsed_s: ToolResult(data=result),
    )
    env = _Env()
    events = NullDashboardEventSink()
    toolkit = libero_toolkit.LiberoToolkit(
        runtime_kwargs={
            "env": env,
            "model": _Model(),
            "sam3_client": SimpleNamespace(),
        },
        dashboard_events=events,
        memory=MemoryManager(tmp_path / "memory"),
        mode=mode,
        enable_direct_action=True,
        state_output_dir=tmp_path / "state",
    )
    if record_trajectory:
        toolkit.state.start_task(
            instruction="release, use VLA, then move",
            config=Payload(format="test", data={}),
        )
    executed = []

    def step(action):
        executed.append(action)
        result = (
            _obs(len(executed)),
            np.float32(0.5),
            np.bool_(False),
            np.bool_(False),
            {
                "step": np.int64(len(executed)),
                "episode": {"episode_len": np.array([len(executed)], np.int32)},
            },
        )
        return result

    monkeypatch.setattr(env, "_step", step)
    chunk_calls = []

    def chunk_step(actions, *, return_all_frames=None):
        chunk_calls.append(return_all_frames)
        returned = [step(action) for action in actions]
        observations, rewards, terminated, truncated, infos = zip(*returned)
        return (
            list(observations) if return_all_frames else observations[-1],
            np.asarray(rewards),
            np.asarray(terminated),
            np.asarray(truncated),
            list(infos),
        )

    monkeypatch.setattr(env, "_chunk_step", chunk_step)
    direct = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0]
    calls = iter(
        [
            ("release", {"max_steps": 2}),
            ("pi0_doubled", {"prompt": "pick up the bowl", "max_chunks": 2}),
            ("execute_action", {"values": direct}),
            ("finish", {"status": "success", "summary": "done"}),
        ]
    )

    async def stream(messages, info):
        name, arguments = next(calls)
        yield {0: DeltaToolCall(name=name, json_args=json.dumps(arguments))}

    # Manual execution must not borrow an API call's identity.
    toolkit.execute_tool("release", {"max_steps": 1})
    result = ApiAgentLoop(
        model=FunctionModel(stream_function=stream), dashboard_events=events
    ).solve(
        system_prompt="Use the robot toolkit.",
        user_message="release, use VLA, then move",
        toolkit=toolkit,
        max_turns=4,
    )
    toolkit.resume_calls()
    toolkit.execute_tool("release", {"max_steps": 1})

    assert result.error is None
    assert result.finish_result["status"] == "success"
    assert len(executed) == 9
    assert chunk_calls == [True, True]
    actions = toolkit.state.action_records()
    if not record_trajectory:
        assert toolkit.state.get_task() is None
        assert toolkit.state.planner_records() == []
        assert actions == []
        assert toolkit.state.prediction_records() == []
        assert not (tmp_path / "state" / "trajectories").exists()
        return
    toolkit.state.end_task(status="completed", stop_reason="finish")
    root = toolkit.state.save_planner_trajectory()
    saved = [
        json.loads(line) for line in (root / "actions.jsonl").read_text().splitlines()
    ]
    saved_predictions = [
        json.loads(line)
        for line in (root / "predictions.jsonl").read_text().splitlines()
    ]
    if mode == "exploration":
        assert actions == []
        assert saved == []
        assert toolkit.state.prediction_records() == []
        assert saved_predictions == []
        return

    assert [row["action_id"] for row in saved] == [
        action.action_id for action in actions
    ]
    assert [row["source"] for row in saved] == [action.source for action in actions]
    image = saved[-1]["after_observation"]["payload"]["data"]["main_images"]
    np.testing.assert_array_equal(
        np.load(root / image["path"], allow_pickle=False), _obs(8)["main_images"]
    )
    assert [action.source for action in actions] == [
        "tool_computed",
        "tool_computed",
        "vla",
        "vla",
        "vla",
        "vla",
        "direct",
    ]
    assert actions[0].tool_call_id == actions[1].tool_call_id
    assert actions[1].tool_call_id != actions[2].tool_call_id
    for previous, current in zip(actions, actions[1:]):
        assert previous.after_observation_id == current.before_observation_id
    for row, action, executed_action in zip(saved, actions, executed[1:-1]):
        np.testing.assert_array_equal(action.action.data, executed_action)
        np.testing.assert_array_equal(row["action"]["data"]["value"], executed_action)
    assert saved[-1]["result"]["data"]["info"]["episode"]["episode_len"]["value"] == [8]

    tool_calls = [
        toolkit.state.get_tool_call(group[0].tool_call_id)
        for group in (actions[:2], actions[2:6], actions[6:])
    ]
    assert [call.name for call in tool_calls] == [
        "release",
        "pi0_doubled",
        "execute_action",
    ]
    assert len({call.planner_id for call in tool_calls}) == 3
    for call, group in zip(tool_calls, (actions[:2], actions[2:6], actions[6:])):
        assert {action.tool_call_id for action in group} == {call.tool_call_id}
        assert call.before_observation_id == group[0].before_observation_id
        assert call.after_observation_id == group[-1].after_observation_id
        assert call.status == "completed"
        assert "before_observation_id" not in call.missing
        assert "after_observation_id" not in call.missing
        planner = toolkit.state.get_planner(call.planner_id)
        assert call.tool_call_id in {item.tool_call_id for item in planner.tool_calls}

    vla_call = tool_calls[1]
    assert vla_call.effective_instruction == "pick up the bowl"
    assert vla_call.model_config.data == {"predict_options": {"mode": "eval"}}
    assert vla_call.model_config.completeness == "partial"
    assert vla_call.model_config.missing["/server_config"].reason == "not_exposed"
    assert "effective_instruction" not in vla_call.missing
    assert "model_config" not in vla_call.missing

    assert len(saved_predictions) == 2
    for row, group in zip(saved_predictions, (actions[2:4], actions[4:6])):
        assert row["planner_id"] == vla_call.planner_id
        assert row["tool_call_id"] == vla_call.tool_call_id
        assert row["instruction"] == "pick up the bowl"
        assert row["action_ids"] == [action.action_id for action in group]
        assert row["input_observation"]["observation_id"] == row["input_observation_id"]
        assert row["input_observation_id"] != group[0].before_observation_id
        assert (
            row["input_observation"]["payload"]["data"]["task_descriptions"]
            == "pick up the bowl"
        )
        assert "actions" not in row


@pytest.fixture
def captured_primitives(tmp_path):
    state = EnvState(tmp_path)
    state.start_task(instruction="release", config=Payload(format="test", data={}))
    planner_id = state.register_planner(backend="api")
    tool_call_id = state.register_tool_call(
        planner_id=planner_id,
        name="release",
        arguments=Payload(format="json", data={"max_steps": 3}),
    )
    capture = ToolCallCapture(state, tool_call_id)
    capture.start()
    primitives = _primitives(CapturingEnvClient(_Env(), state))
    primitives.reset()
    return primitives, capture


@pytest.mark.parametrize("failure", ["step", "cancel", "frame"])
def test_action_capture_keeps_confirmed_prefix(
    captured_primitives, monkeypatch, failure
):
    primitives, capture = captured_primitives
    if failure == "step":
        monkeypatch.setattr(
            primitives.env.wrapped,
            "_step",
            Mock(side_effect=[(_obs(1), 0, False, False, {}), RuntimeError("step")]),
        )
    elif failure == "cancel":
        monkeypatch.setattr(
            primitives,
            "_check_cancelled",
            Mock(side_effect=[None, ToolCancelled("cancel")]),
        )
    else:
        primitives.start_recording()
        monkeypatch.setattr(
            primitives, "record_frame", Mock(side_effect=RuntimeError("frame"))
        )

    with capture.bind(), pytest.raises((RuntimeError, ToolCancelled), match=failure):
        primitives.release(max_steps=3)

    actions = capture.state.action_records()
    assert len(actions) == 1
    call = capture.state.get_tool_call(capture.tool_call_id)
    assert call.before_observation_id == actions[0].before_observation_id
    if failure == "step":
        assert call.after_observation_id is None
        assert call.missing["after_observation_id"].reason == "capture_failed"
    else:
        assert call.after_observation_id == actions[0].after_observation_id
        assert "after_observation_id" not in call.missing
    assert capture.state.get_observation(actions[0].after_observation_id)


@pytest.mark.parametrize("terminal", ["terminated", "truncated"])
def test_action_capture_preserves_terminal_flags(
    captured_primitives, monkeypatch, terminal
):
    primitives, capture = captured_primitives

    def step(action):
        setattr(primitives.env.wrapped, terminal, True)
        return (
            _obs(1),
            np.float32(1),
            np.bool_(primitives.env.terminated),
            np.bool_(primitives.env.truncated),
            {},
        )

    monkeypatch.setattr(primitives.env.wrapped, "_step", step)
    with capture.bind():
        primitives.release(max_steps=3)

    actions = capture.state.action_records()
    assert len(actions) == 1
    assert isinstance(actions[0].result.data[terminal], np.bool_)
    assert actions[0].result.data[terminal]


@pytest.mark.parametrize("frame_failure", [False, True])
def test_vla_capture_keeps_all_confirmed_actions(
    captured_primitives, monkeypatch, frame_failure, tmp_path
):
    primitives, capture = captured_primitives
    initial = primitives._last_obs
    predict = primitives.model.predict

    def predict_separately(observation, *, options):
        prediction = capture.state.prediction_records()[-1]
        captured = capture.state.get_observation(prediction.input_observation_id)
        root = tmp_path / "trajectories" / capture.state.get_task().task_id
        np.testing.assert_array_equal(
            np.load(root / captured.payload.data["states"].path), initial["states"]
        )
        assert observation is not initial
        assert initial["task_descriptions"] == "put the bowl on the plate"
        assert "extra_view_images" not in initial
        return predict(observation, options=options)

    monkeypatch.setattr(primitives.model, "predict", predict_separately)
    if frame_failure:
        primitives.start_recording()
        monkeypatch.setattr(
            primitives, "record_frame", Mock(side_effect=RuntimeError("frame"))
        )
    with capture.bind():
        if frame_failure:
            with pytest.raises(RuntimeError, match="frame"):
                primitives._vlm_chunk("pick up the bowl")
        else:
            primitives._vlm_chunk("pick up the bowl")

    assert primitives.env.chunk_return_all_frames is (True if frame_failure else None)
    actions = capture.state.action_records()
    assert [action.source for action in actions] == ["vla", "vla"]
    assert actions[0].after_observation_id == actions[1].before_observation_id
    assert not actions[0].result.data["terminated"]
    assert actions[1].result.data["terminated"]
    assert primitives._last_obs["task_descriptions"] == initial["task_descriptions"]
    call = capture.state.get_tool_call(capture.tool_call_id)
    assert call.before_observation_id == actions[0].before_observation_id
    assert call.after_observation_id == actions[-1].after_observation_id
    observation = capture.state.get_observation(call.after_observation_id)
    np.testing.assert_array_equal(observation.payload.data["states"], _obs(3)["states"])
    (prediction,) = capture.state.prediction_records()
    assert prediction.input_observation_id != actions[0].before_observation_id
    assert prediction.action_ids == [action.action_id for action in actions]
    np.testing.assert_array_equal(
        np.stack(
            [capture.state.get_action(key).action.data for key in prediction.action_ids]
        ),
        np.ones((2, 7), np.float32),
    )


def test_predictions_keep_each_calls_context_and_confirmed_actions(
    captured_primitives, monkeypatch, tmp_path
):
    primitives, capture = captured_primitives
    step_count = 3
    predicted = np.linspace(-1, 1, 35, dtype=np.float32).reshape(5, 7)
    chunks = (predicted, -predicted)
    instructions = ("move toward the bowl", "lift the bowl")
    flags = {name: np.zeros(step_count, bool) for name in ("terminated", "truncated")}
    flags["terminated"][2] = True
    predict = Mock(side_effect=chunks)
    monkeypatch.setattr(primitives.model, "predict", predict)
    monkeypatch.setattr(
        primitives.env.wrapped,
        "_chunk_step",
        Mock(
            side_effect=[
                (
                    [
                        _obs(call * step_count + index + 1)
                        for index in range(step_count)
                    ],
                    np.zeros(step_count),
                    flags["terminated"] if call else np.zeros(step_count, bool),
                    flags["truncated"] if call else np.zeros(step_count, bool),
                    [{} for _ in range(step_count)],
                )
                for call in range(2)
            ]
        ),
    )
    with capture.bind():
        for instruction in instructions:
            primitives._vlm_chunk(instruction)

    predictions = capture.state.prediction_records()
    actions = capture.state.action_records()
    assert len(predictions) == 2 and len(actions) == 2 * step_count
    for index, prediction in enumerate(predictions):
        group = actions[index * step_count : (index + 1) * step_count]
        request = predict.call_args_list[index].args[0]
        assert (
            prediction.instruction
            == request["task_descriptions"]
            == instructions[index]
        )
        assert prediction.input_observation_id != group[0].before_observation_id
        observation = capture.state.get_observation(prediction.input_observation_id)
        np.testing.assert_array_equal(
            np.load(
                tmp_path
                / "trajectories"
                / capture.state.get_task().task_id
                / observation.payload.data["states"].path
            ),
            request["states"],
        )
        assert prediction.action_ids == [action.action_id for action in group]
        np.testing.assert_array_equal(
            np.stack(
                [
                    capture.state.get_action(key).action.data
                    for key in prediction.action_ids
                ]
            ),
            chunks[index][:step_count],
        )


@pytest.mark.parametrize("failure", ["predict", "cancel", "chunk"])
def test_unexecuted_vla_actions_are_not_recorded(
    captured_primitives, monkeypatch, failure
):
    primitives, capture = captured_primitives
    with capture.bind():
        primitives._vlm_chunk("pick up the bowl")
    confirmed = capture.state.action_records()
    initial = primitives._last_obs
    chunk_step = Mock(side_effect=RuntimeError("chunk"))
    monkeypatch.setattr(primitives.env.wrapped, "_chunk_step", chunk_step)
    if failure == "predict":
        monkeypatch.setattr(
            primitives.model, "predict", Mock(side_effect=RuntimeError("predict"))
        )
    elif failure == "cancel":
        monkeypatch.setattr(
            primitives,
            "_check_cancelled",
            Mock(side_effect=[None, ToolCancelled("cancel")]),
        )
    with capture.bind(), pytest.raises((RuntimeError, ToolCancelled), match=failure):
        primitives._vlm_chunk("pick up the bowl")

    assert [a.action_id for a in capture.state.action_records()] == [
        a.action_id for a in confirmed
    ]
    previous, failed = capture.state.prediction_records()
    assert previous.action_ids == [a.action_id for a in confirmed]
    assert failed.action_ids == []
    assert primitives._last_obs is initial
    assert initial["task_descriptions"] == "put the bowl on the plate"
    assert "extra_view_images" not in initial
    assert chunk_step.call_count == (failure == "chunk")


@pytest.mark.parametrize("capture_actions", [False, True])
def test_no_capture_context_keeps_fast_chunk_path(capture_actions, tmp_path):
    env = _Env()
    primitives = _primitives(
        CapturingEnvClient(env, EnvState(tmp_path)) if capture_actions else env
    )
    primitives.reset()
    primitives._vlm_chunk("pick up the bowl")
    assert env.chunk_return_all_frames is None


@pytest.mark.parametrize("failure", [None, "stop", "save"])
def test_close_saves_video_and_handles_video_errors(tmp_path, monkeypatch, failure):
    toolkit = libero_toolkit.LiberoToolkit.__new__(libero_toolkit.LiberoToolkit)
    from rpent.memory import MemoryManager
    from rpent.tools import Toolkit

    Toolkit.__init__(
        toolkit, dashboard_events=None, memory=MemoryManager(tmp_path / "memory")
    )
    frames = [_obs(0)["main_images"]]
    stop = Mock(return_value=frames)
    save = Mock()
    if failure is not None:
        {"stop": stop, "save": save}[failure].side_effect = RuntimeError(failure)
    toolkit._primitives = SimpleNamespace(stop_recording=stop)
    toolkit._state = SimpleNamespace(save=save)
    logger = Mock()
    monkeypatch.setattr(libero_toolkit, "logger", logger)

    toolkit.close()

    stop.assert_called_once_with()
    if failure == "stop":
        save.assert_not_called()
    else:
        save.assert_called_once_with("episode.mp4", frames, step=None, fps=20)
    assert logger.warning.call_count == (failure is not None)


@pytest.mark.parametrize("after_reset", [False, True])
def test_recipe_exports_each_segment_once_and_keeps_repeated_calls(
    tmp_path, after_reset
):
    import json

    from robots.libero.tools import write_recipe_from_states
    from rpent.session import EnvState

    state = EnvState(tmp_path)
    command = {"action": "segment", "prompt": "cup", "camera": "agentview"}
    artifact = {
        "mode": "text",
        "prompt": "cup",
        "camera": "agentview",
        "segment_index": 0,
    }
    if after_reset:
        with state.record_step(state={}, command={"action": "move_to"}, result={}):
            state.save("segment_00.json", artifact)
    with state.record_step(
        state={}, command={"action": "reset"} if after_reset else None, result={}
    ):
        pass
    for index in range(2):
        state.save("segment_00.json", artifact)
        with state.record_step(
            state={}, command=command, result={}, terminated=index == 1
        ):
            pass
    name = write_recipe_from_states(state, "test", output_dir=tmp_path)
    commands = [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
    assert commands == [command, command]


class _LiberoBackend:
    def __init__(self):
        self.executed = []
        self.wrap_count = 0
        self.reset_count = 0
        self.fail_at = None

    def _observation(self):
        return _obs(len(self.executed))

    def _wrap_obs(self, observations):
        self.wrap_count += 1
        return {
            key: np.stack([obs[key] for obs in observations])
            if isinstance(value, np.ndarray)
            else [obs[key] for obs in observations]
            for key, value in observations[0].items()
        }

    def reset(self):
        self.reset_count += 1
        self.executed.clear()
        self.current_raw_obs = [self._observation()]
        return self._wrap_obs(self.current_raw_obs), {}

    def step(self, actions):
        assert actions.shape == (1, 7)
        self.executed.append(actions[0].copy())
        self.current_raw_obs = [self._observation()]
        if len(self.executed) == self.fail_at:
            raise RuntimeError("native step failed")
        return (
            self._wrap_obs(self.current_raw_obs),
            np.array([0.5], dtype=np.float32),
            np.array([len(self.executed) == 3]),
            np.array([False]),
            {"episode": {"episode_len": np.array([len(self.executed)])}},
        )

    def chunk_step(self, actions):
        steps = [self.step(actions[:, index]) for index in range(actions.shape[1])]
        return (
            [step[0] for step in steps],
            *(np.stack([step[index] for step in steps], axis=1) for index in (1, 2, 3)),
            [step[4] for step in steps],
        )


@pytest.mark.parametrize(
    ("transport", "all_frames", "record"),
    [
        ("socket", False, True),
        ("http", True, True),
        ("socket", True, False),
        ("http", False, False),
    ],
)
def test_libero_client_facade_capture_and_raw_trajectory_roundtrip(
    transport, make_server_and_client, tmp_path, all_frames, record
):
    from robots.libero.env_client import LiberoEnvClient
    from robots.libero.env_server import LiberoEnvFacade
    from rpent.data.reader import iter_trajectories, read_array
    from rpent.session import EnvState
    from rpent.session.capture import CapturingEnvClient, ToolCallCapture

    backend = _LiberoBackend()
    state = EnvState(tmp_path)
    capture = None
    if record:
        state.start_task(
            instruction="put the bowl on the plate",
            config=Payload(
                "test.config", {"arguments": {"robot_name": "libero", "seed": 0}}
            ),
        )
        tool_id = state.register_tool_call(
            planner_id=state.register_planner(backend="api"),
            name="pi0_pick",
            arguments=Payload("json", {}),
        )
        capture = ToolCallCapture(state, tool_id)
    commands = np.linspace(-1, 1, 35, dtype=np.float32).reshape(5, 7)
    with make_server_and_client(LiberoEnvFacade(backend, meta={}), transport) as rpc:
        client = LiberoEnvClient(rpc, expected_meta={})
        env = CapturingEnvClient(client, state)
        with capture.bind() if capture else nullcontext():
            env.step(np.zeros(7, dtype=np.float32))
            policy_obs = {**env.last_obs, "task_descriptions": "pick up the bowl"}
            with env.prediction(
                policy_obs, "pick up the bowl", format="libero.observation"
            ):
                result = env.chunk_step(commands, return_all_frames=all_frames)
        assert client.terminated and not client.truncated
        np.testing.assert_array_equal(result[2], [False, True, False, False, False])
        np.testing.assert_array_equal(client.last_obs["states"], np.full(8, 6))
        assert isinstance(result[0], list) is all_frames
    assert backend.reset_count == 1
    assert backend.wrap_count == 7 + (2 if record else 0)
    np.testing.assert_array_equal(backend.executed[1:], commands)
    if not record:
        assert state.action_records() == state.prediction_records() == []
        assert not (tmp_path / "trajectories").exists()
        return
    state.end_task(status="completed", stop_reason="finish")
    root = state.save_planner_trajectory()
    (trajectory,) = list(iter_trajectories(root))
    actions = trajectory["actions"]
    assert len(actions) == 6
    assert [action["source"] for action in actions] == ["tool_computed", *(["vla"] * 5)]
    (prediction,) = trajectory["predictions"]
    assert prediction["action_ids"] == [action["action_id"] for action in actions[1:]]
    assert "actions" not in prediction
    assert (
        prediction["input_observation"]["payload"]["data"]["task_descriptions"]
        == "pick up the bowl"
    )
    for index, (action, command) in enumerate(
        zip(actions, backend.executed, strict=True)
    ):
        np.testing.assert_array_equal(
            read_array(root, action["action"]["data"]), command
        )
        for side, value in (("before", index), ("after", index + 1)):
            data = action[f"{side}_observation"]["payload"]["data"]
            np.testing.assert_array_equal(
                read_array(root, data["states"]), np.full(8, value)
            )
            np.testing.assert_array_equal(
                read_array(root, data["main_images"]), np.full((256, 256, 3), value)
            )


def test_libero_failed_chunk_discards_its_actions_and_refreshes_next_before_observation(
    transport, make_server_and_client, tmp_path
):
    from robots.libero.env_client import LiberoEnvClient
    from robots.libero.env_server import LiberoEnvFacade
    from rpent.session import EnvState
    from rpent.session.capture import CapturingEnvClient, ToolCallCapture

    state = EnvState(tmp_path)
    state.start_task(instruction="move", config=Payload("test.config", {}))
    tool_id = state.register_tool_call(
        planner_id=state.register_planner(backend="api"),
        name="move",
        arguments=Payload("json", {}),
    )
    backend = _LiberoBackend()
    with make_server_and_client(LiberoEnvFacade(backend, meta={}), transport) as rpc:
        env = CapturingEnvClient(LiberoEnvClient(rpc, expected_meta={}), state)
        with ToolCallCapture(state, tool_id).bind():
            env.step(np.zeros(7))
            backend.fail_at = 3
            with pytest.raises(RpcError, match="native step failed"):
                with env.prediction(env.last_obs, "move", format="libero.observation"):
                    env.chunk_step(np.ones((4, 7)))
            assert len(state.action_records()) == 1
            assert state.prediction_records()[0].action_ids == []
            backend.fail_at = None
            env.step(np.zeros(7))
    first, second = state.action_records()
    assert second.before_observation_id != first.after_observation_id
    before = state.get_observation(second.before_observation_id).payload
    np.testing.assert_array_equal(before.data["states"], np.full(8, 3))
