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

from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import pytest

from robots.robocasa.env_client import RoboCasaEnvClient
from robots.robocasa.env_server import DEFAULT_CAMS, RoboCasaEnvFacade
from robots.robocasa.primitives import RoboCasaPrimitives
from rpent.dashboard.events import NullDashboardEventSink
from rpent.data.reader import iter_trajectories, read_array
from rpent.memory import MemoryManager
from rpent.robots.components.action_spec import box_action_spec
from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.session import EnvState, Payload
from rpent.session.capture import CapturingEnvClient, ToolCallCapture
from rpent.tools import Toolkit, ToolResult, iter_tools
from rpent.tools.direct_action import direct_action_tool
from rpent.utils.rpc import RpcError


class _Backend:
    action_dim = 12

    def __init__(self):
        self.executed = []
        self.reset_count = 0
        self.observation_reads = 0
        self.fail_at = None
        self.obs = {
            key: np.zeros(size)
            for key, size in (
                ("robot0_eef_pos", 3),
                ("robot0_eef_quat", 4),
                ("robot0_gripper_qpos", 2),
                ("robot0_base_pos", 3),
                ("robot0_base_quat", 4),
                ("robot0_base_to_eef_pos", 3),
                ("robot0_base_to_eef_quat", 4),
            )
        }

    def reset(self):
        self.reset_count += 1
        self.executed.clear()
        for value in self.obs.values():
            value.fill(0)
        return self.obs

    def _get_observations(self):
        self.observation_reads += 1
        return self.obs

    def step(self, action):
        self.executed.append(action.copy())
        count = len(self.executed)
        for value in self.obs.values():
            value.fill(count)
        if count == self.fail_at:
            raise RuntimeError("native step failed")
        return self.obs, float(count), False, {"count": count}

    def _check_success(self):
        return False

    def get_ep_meta(self):
        return {"lang": "Open the left drawer."}

    def render_camera(self, camera_name, height, width, depth=False):
        assert not depth
        rows = np.arange(height, dtype=np.uint8) + 10 * len(self.executed)
        return np.broadcast_to(rows[:, None, None], (height, width, 3)).copy()


class _VLA:
    def __init__(self):
        self.inputs = []

    def get_modality_config(self):
        return {"video_delta_indices": [-2, 0], "hist_maxlen": 3}

    def predict(self, obs, options):
        self.inputs.append((deepcopy(obs), deepcopy(options)))
        return {
            "action.end_effector_position": np.full((1, 4, 3), 0.25),
            "action.end_effector_rotation": np.full((1, 4, 3), 0.1),
            "action.gripper_close": np.array([[[0.2], [0.8], [0.4], [0.6]]]),
            "action.base_motion": np.tile([0.8, -0.8, 0.2, -0.2], (1, 4, 1)),
            "action.control_mode": np.full((1, 4, 1), 0.2),
        }


@pytest.fixture
def facade(monkeypatch):
    for name in (
        "RLDX_RESET_SEED",
        "RLDX_VIDEO_DIR",
        "RLDX_MAX_CHUNKS",
        "RLDX_ACTION_STEPS_PER_CHUNK",
        "RLDX_SETTLE_PATIENCE",
    ):
        monkeypatch.delenv(name, raising=False)
    facade = RoboCasaEnvFacade.__new__(RoboCasaEnvFacade)
    facade.env = _Backend()
    facade.cameras = DEFAULT_CAMS
    facade.camera_h = facade.camera_w = 4
    facade._meta = {"camera_h": 4, "camera_w": 4}
    facade.render_camera = facade.env.render_camera
    facade.grasp_contact = lambda: (False, None)
    facade.reassemble_env_action = lambda action: action
    facade.get_action_spec = lambda: {
        "default": box_action_spec(-np.ones(12), np.ones(12), "test action")
    }
    BaseEnvFacade.__init__(facade)
    return facade


def _primitives(env, tmp_path):
    primitives = RoboCasaPrimitives(env, tmp_path, 4, _VLA(), allow_reset=False)
    skill = primitives._rldx
    skill.VLA_OBS_RES = 4
    # Stand in for the optional RoboCasa converter; keep the skill's conversion call.
    skill._unmap = lambda fields: np.concatenate(
        [
            np.where(value >= 0.5, 1.0, -1.0)
            if key in ("action.gripper_close", "action.control_mode")
            else value
            for key, value in fields.items()
        ]
    )
    return primitives


@pytest.mark.parametrize("mode", ["capture", "disabled", "unwrapped"])
def test_robocasa_tools_capture_executed_actions_and_rldx_history(
    facade, transport, make_server_and_client, tmp_path, monkeypatch, mode
):
    state = EnvState(tmp_path)
    recording = mode == "capture"
    planner = None
    if recording:
        state.start_task(
            instruction="Open the left drawer.", config=Payload("test", {})
        )
        planner = state.register_planner(backend="api")
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    toolkit = Toolkit(
        state=state,
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    monkeypatch.setattr(
        toolkit, "get_env_state", lambda **kw: ToolResult(data=kw["result"])
    )

    def invoke(name, arguments):
        capture = None
        if recording:
            tool_id = state.register_tool_call(
                planner_id=planner, name=name, arguments=Payload("json", arguments)
            )
            capture = ToolCallCapture(state, tool_id)
        with capture.bind() if capture else nullcontext():
            result = toolkit.execute_tool(name, arguments).data
        assert "error" not in result
        return result

    with make_server_and_client(facade, transport) as rpc:
        client = RoboCasaEnvClient(rpc, expected_meta=facade._meta)
        env = client if mode == "unwrapped" else CapturingEnvClient(client, state)
        primitives = _primitives(env, tmp_path)
        assert primitives._rldx.env is env
        toolkit.add_tools(iter_tools(primitives))
        toolkit.add_tool(
            direct_action_tool(env.action_specs, primitives.execute_action)
        )
        invoke("set_gripper", {"steps": 1})
        result = invoke(
            "rldx_arm",
            {
                "prompt": "Open it",
                "max_chunks": 2,
                "n_action_steps": 2,
                "base_clip": 0.1,
                "settle_patience": 99,
            },
        )
        assert result["steps_applied"] == 4
        assert result["effective_prompt"] == "Open the left drawer."
        invoke("release", {"steps": 1})
        invoke("execute_action", {"values": [0.2] * 12})
        np.testing.assert_array_equal(client.last_obs["robot0_eef_pos"], [7] * 3)
        assert "images" not in client.last_obs
    assert facade.env.reset_count == 2
    assert len(facade.env.executed) == 7
    assert facade.env.observation_reads == (7 if recording else 0)
    np.testing.assert_allclose(facade.env.executed[1][7:11], [0.1, -0.1, 0.1, -0.1])
    assert [a[6] for a in facade.env.executed[1:5]] == [-1, 1, -1, 1]
    if not recording:
        assert state.action_records() == state.prediction_records() == []
        assert not (tmp_path / "trajectories").exists()
        return

    state.end_task(status="completed", stop_reason="finish")
    root = state.save_planner_trajectory()
    (trajectory,) = list(iter_trajectories(root))
    actions = trajectory["actions"]
    assert [a["source"] for a in actions] == [
        "tool_computed",
        "vla",
        "vla",
        "vla",
        "vla",
        "tool_computed",
        "direct",
    ]
    for index, action in enumerate(actions):
        assert action["action"]["format"] == "robocasa.action"
        native = read_array(root, action["action"]["data"])
        assert native.dtype == np.float64
        np.testing.assert_array_equal(native, facade.env.executed[index])
        assert action["result"]["data"] == {
            "reward": float(index + 1),
            "done": False,
            "info": {"count": index + 1},
        }
        for side, count in (("before", index), ("after", index + 1)):
            payload = action[f"{side}_observation"]["payload"]
            assert payload["format"] == "robocasa.observation"
            np.testing.assert_array_equal(
                read_array(root, payload["data"]["robot0_eef_pos"]), [count] * 3
            )
            for camera in DEFAULT_CAMS:
                image = read_array(root, payload["data"]["images"][camera])
                np.testing.assert_array_equal(
                    image[:, 0, 0], np.arange(4)[::-1] + count * 10
                )
        if index:
            assert (
                action["before_observation_id"]
                == actions[index - 1]["after_observation_id"]
            )
    predictions = trajectory["predictions"]
    assert [p["action_ids"] for p in predictions] == [["A2", "A3"], ["A4", "A5"]]
    for prediction, (expected, options) in zip(
        predictions, primitives._rldx._vla_client.inputs, strict=True
    ):
        assert prediction["instruction"] == "Open the left drawer."
        payload = prediction["input_observation"]["payload"]
        assert payload["format"] == "rldx.observation"
        for key, value in expected.items():
            actual = payload["data"][key]
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(read_array(root, actual), value)
            else:
                assert actual == value
        assert "actions" not in prediction
    first, second = primitives._rldx._vla_client.inputs
    assert first[1] == {"reset_memory": [True]}
    assert second[1] == {"reset_memory": [False]}
    history = second[0]["video.robot0_agentview_left"]
    assert history.shape == (1, 2, 4, 4, 3)
    assert history[0, 0, 0, 0, 0] != history[0, 1, 0, 0, 0]


def test_robocasa_failed_step_keeps_prior_steps_and_ends_prediction_scope(
    facade, transport, make_server_and_client, tmp_path
):
    state = EnvState(tmp_path)
    state.start_task(instruction="Open drawer", config=Payload("test", {}))
    tool_id = state.register_tool_call(
        planner_id=state.register_planner(backend="api"),
        name="rldx_skill",
        arguments=Payload("json", {}),
    )
    capture = ToolCallCapture(state, tool_id)
    with make_server_and_client(facade, transport) as rpc:
        env = CapturingEnvClient(
            RoboCasaEnvClient(rpc, expected_meta=facade._meta), state
        )
        primitives = _primitives(env, tmp_path)
        facade.env.fail_at = 2
        with capture.bind():
            with pytest.raises(RpcError, match="native step failed"):
                primitives._rldx.run("Open drawer", max_chunks=1, n_action_steps=3)
            assert capture.prediction_id is None
            assert len(state.action_records()) == 1
            assert state.prediction_records()[0].action_ids == ["A1"]
            env.step(np.zeros(12))
    first, last = state.action_records()
    assert first.source == "vla" and last.source == "tool_computed"
    assert last.before_observation_id != first.after_observation_id
    before = state.get_observation(last.before_observation_id)
    np.testing.assert_array_equal(before.payload.data["robot0_eef_pos"], [2] * 3)
