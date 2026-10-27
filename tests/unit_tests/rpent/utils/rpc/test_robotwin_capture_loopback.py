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

"""RoboTwin execution snapshots survive real RPC and trajectory storage."""

import importlib
import sys
from contextlib import nullcontext
from copy import deepcopy
from threading import Lock
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from robots.robotwin.env_client import RoboTwinEnvClient
from robots.robotwin.primitives import RoboTwinPrimitives
from rpent.dashboard.events import NullDashboardEventSink
from rpent.data.reader import iter_trajectories, read_array
from rpent.memory import MemoryManager
from rpent.session import EnvState, Payload
from rpent.session.capture import CapturingEnvClient, ToolCallCapture
from rpent.tools import Toolkit, ToolResult, tool
from rpent.tools.direct_action import direct_action_tool
from rpent.utils.rpc import RpcError

CAMERAS = {
    "head": "head_camera",
    "left_wrist": "left_camera",
    "right_wrist": "right_camera",
}


class _Task:
    def __init__(self):
        self.take_action_cnt = 0
        self.step_lim = 200
        self.ep_num = 7
        self.eval_success = False
        self.success_at = None
        self.fail_at = None
        self.executed = []
        self.observation_reads = 0
        self.reset_count = 0
        self.lock = Lock()
        self.obs = {
            "observation": {
                name: {
                    "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                    "intrinsic_cv": np.eye(3),
                }
                for name in CAMERAS.values()
            }
        }
        self.robot = SimpleNamespace()
        for arm in ("left", "right"):
            for kind in ("arm_jointState", "arm_real_jointState"):
                setattr(
                    self.robot,
                    f"get_{arm}_{kind}",
                    lambda: [self.take_action_cnt] * 6 + [0.5],
                )
            for kind in ("ee_pose", "tcp_pose"):
                setattr(
                    self.robot,
                    f"get_{arm}_{kind}",
                    lambda: [self.take_action_cnt, 0, 0, 1, 0, 0, 0],
                )
            setattr(self.robot, f"get_{arm}_gripper_val", lambda: 0.5)

    def get_instruction(self):
        return "native instruction"

    def get_obs(self):
        assert self.lock.locked()
        self.observation_reads += 1
        return self.obs

    def take_action(self, action, *, action_type):
        assert self.lock.locked()
        self.executed.append((action.copy(), action_type))
        self.take_action_cnt += 1
        for camera in self.obs["observation"].values():
            camera["rgb"].fill(self.take_action_cnt)
        self.eval_success = self.take_action_cnt == self.success_at
        if self.take_action_cnt == self.fail_at:
            raise RuntimeError("native action failed")


@pytest.fixture
def facade(monkeypatch):
    pytest.importorskip("torch")
    for name, attributes in (
        ("rlinf.envs.sim.robotwin.robotwin_env", {"RoboTwinEnv": object}),
        ("robotwin.assets", {"validate_root": lambda *a: None}),
        ("robotwin.config", {"load_task_config": lambda *a: {}}),
    ):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            stub = ModuleType(name)
            stub.__dict__.update(attributes)
            monkeypatch.setitem(sys.modules, name, stub)
    for name in ("robots.robotwin.rlinf_env", "robots.robotwin.env_server"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module("robots.robotwin.env_server")
    task = _Task()
    env = object.__new__(module.RoboTwinAgentEnv)
    slot = SimpleNamespace(task=task, lock=task.lock)
    env.venv = SimpleNamespace(envs=[slot])
    env.record_metrics = False
    env.center_crop = False
    env.center_and_crop = lambda image, **kw: np.array(image)
    env._extract_obs_image = lambda obs: obs
    env._get_camera = lambda *a: None

    def get_obs():
        assert not task.lock.locked()
        with task.lock:
            native = task.get_obs()
            return {
                "main_images": native["observation"]["head_camera"]["rgb"][None].copy()
            }

    def reset(**kwargs):
        task.reset_count += 1
        observation = get_obs()
        with task.lock:
            info = {
                "robot_state": env._robot_state(slot),
                "episode_status": env._episode_status(slot),
            }
        return observation, info

    env.venv.get_obs = get_obs
    env.reset = reset
    yield module.RoboTwinEnvFacade(
        env, metadata={"seed": 7, "execution": {"chunk_step_all_frames": True}}
    )
    sys.modules.pop("robots.robotwin.env_server", None)
    sys.modules.pop("robots.robotwin.rlinf_env", None)


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("all_frames", [False, True])
@pytest.mark.parametrize("stop", ["success", "budget"])
def test_chunk_returns_only_executed_steps_with_distinct_observations(
    facade, make_server_and_client, transport, capture, all_frames, stop
):
    task = facade._env.venv.envs[0].task
    if stop == "success":
        task.success_at = 3
    else:
        task.step_lim = 3
    commands = np.arange(80, dtype=np.float64).reshape(5, 16)
    with make_server_and_client(facade, transport) as rpc:
        client = RoboTwinEnvClient(rpc, expected_meta=facade.get_env_meta())
        first = client.step(np.zeros(14), capture=capture)
        if capture:
            assert first["steps"][0]["action"]["format"] == "robotwin.action.qpos"
            np.testing.assert_array_equal(
                first["steps"][0]["action"]["data"], np.zeros(14)
            )
        reads = task.observation_reads
        response = client.chunk_step(
            commands, action_type="ee", capture=capture, return_all_frames=all_frames
        )
        result = response["result"] if capture else response
        obs, rewards, terminated, truncated, info = result
        assert info["executed_actions"] == 2
        assert info["requested_actions"] == 5
        assert "capture" not in info
        np.testing.assert_array_equal(rewards, [0, float(stop == "success")])
        np.testing.assert_array_equal(terminated, [False, stop == "success"])
        np.testing.assert_array_equal(truncated, [False, stop == "budget"])
        assert client.terminated is (stop == "success")
        assert client.truncated is (stop == "budget")
        np.testing.assert_array_equal(
            client.last_obs["main_images"], np.full((2, 2, 3), 3)
        )
        np.testing.assert_array_equal(
            client.last_info["robot_state"]["qpos_target14"],
            [3] * 6 + [0.5] + [3] * 6 + [0.5],
        )
        assert task.observation_reads - reads == 1 + int(capture) + (
            2 if capture or all_frames else 0
        )
        if all_frames:
            assert len(obs["frames"]) == 2
            for frame, count in zip(obs["frames"], (2, 3), strict=True):
                np.testing.assert_array_equal(frame, np.full((2, 2, 3), count))
    assert task.reset_count == 1
    assert len(task.executed) == 3
    if not capture:
        return
    assert len(response["steps"]) == 2
    observations = [response["before_observation"]]
    for index, step in enumerate(response["steps"]):
        assert step["action"]["format"] == "robotwin.action.ee"
        np.testing.assert_array_equal(step["action"]["data"], commands[index])
        assert (
            step["result"]["data"]["info"]["episode_status"]["take_action_cnt"]
            == index + 2
        )
        observations.append(step["after_observation"])
    for count, observation in enumerate(observations, 1):
        data = observation["data"]
        assert data["robot_state"]["left_eef_pose"][0] == count
        for camera in data["observation"].values():
            np.testing.assert_array_equal(camera["rgb"], np.full((2, 2, 3), count))


@pytest.mark.parametrize("recording", [False, True])
def test_lingbot_predictions_and_action_sources_reach_disk(
    facade, make_server_and_client, transport, tmp_path, recording
):
    state = EnvState(tmp_path)
    task = facade._env.venv.envs[0].task
    task.success_at = 104
    inputs = []
    commands = np.arange(53 * 16, dtype=np.float64).reshape(53, 16) / 1000

    def infer(observation):
        inputs.append(deepcopy(observation))
        return commands

    if recording:
        state.start_task(instruction="native instruction", config=Payload("json", {}))
        planner = state.register_planner(backend="api")
    with make_server_and_client(facade, transport) as rpc:
        client = RoboTwinEnvClient(rpc, expected_meta=facade.get_env_meta())
        env = CapturingEnvClient(client, state)
        primitives = RoboTwinPrimitives(
            env=env,
            model=SimpleNamespace(infer=infer),
            seed=7,
            check_cancelled=lambda: None,
        )
        primitives.start_recording()
        toolkit = Toolkit(
            state=state,
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "memory"),
        )
        toolkit.get_env_state = lambda **kw: ToolResult(data={})

        @tool
        def computed():
            env.step(np.zeros(14))
            return ToolResult(data={})

        toolkit.add_tool(computed)
        toolkit.add_tool(primitives.lingbot_act)
        toolkit.add_tool(
            direct_action_tool(env.action_specs, primitives.execute_action)
        )
        for name, arguments in (
            ("execute_action", {"values": [0.0] * 14, "action_type": "qpos"}),
            ("lingbot_act", {"chunks": 1, "prompt": "ignored instruction"}),
            ("computed", {}),
            ("lingbot_act", {"chunks": 2}),
        ):
            if recording:
                tool_id = state.register_tool_call(
                    planner_id=planner, name=name, arguments=Payload("json", arguments)
                )
            with ToolCallCapture(state, tool_id).bind() if recording else nullcontext():
                result = toolkit.execute_tool(name, arguments)
            assert not result.is_error, result.data
        assert client.terminated
        assert len(task.executed) == 104
        assert len(inputs) == 3
        assert primitives.policy_actions == 102
        assert primitives.recorded_frame_count() == 103
    assert task.reset_count == 1
    if not recording:
        assert state.action_records() == state.prediction_records() == []
        assert not (tmp_path / "trajectories").exists()
        return
    state.end_task(status="completed", stop_reason="finish")
    root = state.save_planner_trajectory()
    (trajectory,) = list(iter_trajectories(root))
    actions, predictions = trajectory["actions"], trajectory["predictions"]
    assert [a["source"] for a in actions] == [
        "direct",
        *(["vla"] * 50),
        "tool_computed",
        *(["vla"] * 52),
    ]
    for prediction, start, length, expected in zip(
        predictions, (2, 53, 103), (50, 50, 2), inputs, strict=True
    ):
        assert prediction["action_ids"] == [
            f"A{i}" for i in range(start, start + length)
        ]
        assert (
            prediction["instruction"]
            == expected["task_language"]
            == "native instruction"
        )
        payload = prediction["input_observation"]["payload"]
        assert payload["format"] == "lingbot.observation"
        for name, camera in expected["views"].items():
            np.testing.assert_array_equal(
                read_array(root, payload["data"]["views"][name]["rgb"]), camera["rgb"]
            )
        np.testing.assert_array_equal(
            read_array(root, payload["data"]["robot_state"]["left_eef_pose"]),
            expected["robot_state"]["left_eef_pose"],
        )
    assert predictions[1]["tool_call_id"] == predictions[2]["tool_call_id"]
    for index, (action, (native, action_type)) in enumerate(
        zip(actions, task.executed, strict=True)
    ):
        assert action["action"]["format"] == f"robotwin.action.{action_type}"
        np.testing.assert_array_equal(
            read_array(root, action["action"]["data"]), native
        )
        if index:
            assert (
                action["before_observation_id"]
                == actions[index - 1]["after_observation_id"]
            )
        for side, count in (("before", index), ("after", index + 1)):
            data = action[f"{side}_observation"]["payload"]["data"]
            np.testing.assert_array_equal(
                read_array(root, data["robot_state"]["left_eef_pose"]),
                [count, 0, 0, 1, 0, 0, 0],
            )
            for camera in data["observation"].values():
                np.testing.assert_array_equal(
                    read_array(root, camera["rgb"]), np.full((2, 2, 3), count)
                )


def test_failed_chunk_discards_its_steps_but_keeps_prior_calls(
    facade, make_server_and_client, transport, tmp_path
):
    task = facade._env.venv.envs[0].task
    task.fail_at = 3
    state = EnvState(tmp_path)
    state.start_task(instruction="native instruction", config=Payload("json", {}))
    planner = state.register_planner(backend="api")
    tool_id = state.register_tool_call(
        planner_id=planner, name="move", arguments=Payload("json", {})
    )
    with make_server_and_client(facade, transport) as rpc:
        client = RoboTwinEnvClient(rpc, expected_meta=facade.get_env_meta())
        env = CapturingEnvClient(client, state)
        with ToolCallCapture(state, tool_id).bind():
            env.step(np.zeros(14))
            with env.prediction({}, "native instruction", format="lingbot.observation"):
                with pytest.raises(RpcError, match="native action failed"):
                    env.chunk_step(np.zeros((5, 16)), action_type="ee")
    assert len(task.executed) == 3
    assert len(state.action_records()) == 1
    assert state.prediction_records()[0].action_ids == []
