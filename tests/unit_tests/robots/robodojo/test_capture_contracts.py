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

"""Executed RoboDojo actions and prediction inputs reach the trajectory store."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robodojo.env_client import RoboDojoEnvClient
from robots.robodojo.env_facade import RoboDojoEnvFacade
from robots.robodojo.toolkit import RoboDojoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.data.reader import iter_trajectories, read_array
from rpent.memory import MemoryManager
from rpent.session import Payload
from rpent.session.capture import ToolCallCapture


class _Scene:
    def __init__(self):
        self.take_action_cnt = [0]
        self.step_lim = 99
        self.executed = []
        self.frames = []
        self.reset_count = 0
        self.obs = {"state": {}, "vision": {}, "score": 100}
        for arm in ("left", "right"):
            self.obs["state"].update(
                {
                    f"{arm}_ee_pose": np.array([0, 0, 1.0, 1, 0, 0, 0]),
                    f"{arm}_ee_joint_state": np.array([0.2]),
                    f"{arm}_arm_joint_state": np.zeros(6),
                }
            )
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            self.obs["vision"][camera] = {
                "color": np.zeros((2, 2, 3), dtype=np.uint8),
                "depth": np.ones((2, 2)),
                "intrinsic_matrix": np.eye(3),
                "extrinsic_matrix": np.eye(4),
            }
        self.obs_manager = SimpleNamespace(
            desc_manager=SimpleNamespace(get_one_description=lambda: ["pick bottle"])
        )

    def reset(self, **kwargs):
        self.reset_count += 1

    def get_obs(self, env_idx=0):
        return self.obs

    def is_success(self, env_idx=0):
        return False

    def apply_action(self, action, action_type, *, eval_fair):
        self.executed.append((deepcopy(action), action_type))
        self.take_action_cnt[0] += 1
        count = self.take_action_cnt[0]
        z = [1, 1, 0.9, 0.96, 0.96][count - 1]
        for arm in ("left", "right"):
            self.obs["state"][f"{arm}_ee_pose"][2] = z
        for camera in self.obs["vision"].values():
            camera["color"].fill(count)


@pytest.fixture
def scene(make_agent, agent_module, monkeypatch):
    scene = _Scene()
    scene.agent = make_agent(
        scene,
        meta={"mode": "dev", "layout": 0},
        slot=SimpleNamespace(env=scene, apply_action=scene.apply_action),
        recorder=SimpleNamespace(record=lambda obs: scene.frames.append(deepcopy(obs))),
    )
    scene.agent.venv.reset = scene.reset
    monkeypatch.setattr(agent_module._SafetyMonitor, "check", lambda self, env: [])
    monkeypatch.setattr(agent_module, "_reward_details", lambda *a: {"reward": 0.5})
    monkeypatch.setattr(
        agent_module,
        "_status",
        lambda env, monitor: {
            "step": env.take_action_cnt[0],
            "step_limit": env.step_lim,
        },
    )
    return scene


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("eval_fair", [False, True])
@pytest.mark.parametrize("action_type", ["joint", "ee"])
def test_native_capture_preserves_execution_observations_and_video(
    scene, capture, eval_fair, action_type
):
    scene.agent.eval_fair = eval_fair
    facade = RoboDojoEnvFacade(scene.agent)
    command = (
        np.linspace(0, 1, 14)
        if action_type == "joint"
        else {
            f"{arm}_ee_pose": np.array([0, 0, 0.9, 1, 0, 0, 0])
            for arm in ("left", "right")
        }
    )
    response = facade.step(command, capture=capture)
    result = response["result"] if capture else response
    obs, reward, done, info = result
    assert reward == (0.0 if eval_fair else 0.5)
    assert done is False
    assert info["status"] == {"step": 1, "step_limit": 99}
    assert ("score" in obs) is not eval_fair
    assert len(scene.frames) == len(scene.executed) == 1
    if not capture:
        return
    (step,) = response["steps"]
    native_action, native_type = scene.executed[0]
    assert native_type == action_type
    assert step["action"]["format"] == f"robodojo.action.{action_type}"
    for key, value in native_action.items():
        np.testing.assert_array_equal(step["action"]["data"][key], value)
    assert step["result"]["data"] == {"reward": reward, "done": done, "info": info}
    for payload, value in (
        (response["before_observation"], 0),
        (step["after_observation"], 1),
    ):
        data = payload["data"]
        assert payload["format"] == "robodojo.observation"
        assert data["instruction"] == "pick bottle"
        assert ("score" in data) is not eval_fair
        for camera in data["vision"].values():
            np.testing.assert_array_equal(camera["color"], np.full((2, 2, 3), value))
            np.testing.assert_array_equal(camera["intrinsic_matrix"], np.eye(3))


@pytest.mark.parametrize("recording", [False, True])
def test_toolkit_saves_only_executed_predictions_and_restores_source(
    scene, tmp_path, monkeypatch, recording
):
    from rpent.utils import logging

    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    facade = RoboDojoEnvFacade(scene.agent)

    def call(method, *, args=(), kwargs=None, **options):
        return deepcopy(facade._rpc[method](*args, **(kwargs or {})))

    client = RoboDojoEnvClient(
        SimpleNamespace(call=call), expected_meta=scene.agent.get_env_meta()
    )
    inputs = []
    commands = np.linspace(0, 1, 42).reshape(3, 14)

    def predict(obs):
        inputs.append(deepcopy(obs))
        return commands

    toolkit = RoboDojoToolkit(
        primitives_kwargs={
            "env": client,
            "vla_client": SimpleNamespace(predict=predict),
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    state = toolkit.state
    if recording:
        state.start_task(instruction="pick bottle", config=Payload("json", {}))
        planner = state.register_planner(backend="api")
    for name, arguments in (
        ("pi0_pick", {}),
        ("set_gripper", {"arm": "left", "gripper": 1.0}),
    ):
        if recording:
            tool = state.register_tool_call(
                planner_id=planner, name=name, arguments=Payload("json", arguments)
            )
        with ToolCallCapture(state, tool).bind() if recording else nullcontext():
            result = toolkit.execute_tool(name, arguments)
        assert not result.is_error, result.data
    assert scene.reset_count == 1
    assert len(inputs) == 2
    assert len(scene.executed) == 5
    assert len(scene.frames) == 12
    assert [action_type for _, action_type in scene.executed] == ["joint"] * 4 + ["ee"]
    if not recording:
        assert state.action_records() == state.prediction_records() == []
        assert not (tmp_path / "trajectories").exists()
        return
    state.end_task(status="completed", stop_reason="finish")
    root = state.save_planner_trajectory()
    (trajectory,) = list(iter_trajectories(root))
    actions, predictions = trajectory["actions"], trajectory["predictions"]
    assert [a["source"] for a in actions] == ["vla"] * 4 + ["tool_computed"]
    assert [p["action_ids"] for p in predictions] == [["A1", "A2", "A3"], ["A4"]]
    for index, action in enumerate(actions):
        native, action_type = scene.executed[index]
        assert action["action"]["format"] == f"robodojo.action.{action_type}"
        for key, value in native.items():
            saved = action["action"]["data"][key]
            np.testing.assert_array_equal(
                read_array(root, saved) if isinstance(value, np.ndarray) else saved,
                value,
            )
        if index:
            assert (
                action["before_observation_id"]
                == actions[index - 1]["after_observation_id"]
            )
        for side, count in (("before", index), ("after", index + 1)):
            obs = action[f"{side}_observation"]["payload"]["data"]
            for camera in obs["vision"].values():
                np.testing.assert_array_equal(
                    read_array(root, camera["color"]), np.full((2, 2, 3), count)
                )
    for prediction, expected in zip(predictions, inputs, strict=True):
        assert prediction["instruction"] == expected["instruction"] == "pick bottle"
        obs = prediction["input_observation"]["payload"]["data"]
        for key, value in expected["state"].items():
            np.testing.assert_array_equal(read_array(root, obs["state"][key]), value)
        for name, camera in expected["vision"].items():
            for key, value in camera.items():
                np.testing.assert_array_equal(
                    read_array(root, obs["vision"][name][key]), value
                )
