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

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import pytest

from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.session.records import Payload
from rpent.utils.rpc import RpcError


class _LiberoBackend:
    def __init__(self):
        self.executed = []
        self.wrap_count = 0
        self.reset_count = 0
        self.fail_at = None

    def _observation(self):
        return {
            "states": np.full(8, len(self.executed), dtype=np.float32),
            "main_images": np.full((256, 256, 3), len(self.executed), dtype=np.uint8),
            "wrist_images": np.full(
                (256, 256, 3), len(self.executed) + 1, dtype=np.uint8
            ),
            "task_descriptions": "put the bowl on the plate",
        }

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


@pytest.mark.parametrize("all_frames", [False, True])
@pytest.mark.parametrize("record", [False, True])
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


class _Env(BaseEnvFacade):
    def __init__(self, tuple_size=5):
        self.tuple_size = tuple_size
        super().__init__()

    def get_env_meta(self):
        return {}

    def _observation(self):
        return {
            "state": np.array([self.position], dtype=np.float32),
            "image": np.full((2, 3, 3), self.count, dtype=np.uint8),
        }

    def reset(self):
        self.position = 0.0
        self.count = 0
        return self._observation()

    def step(self, action, *, capture=False):
        before = self._observation() if capture else None
        if action[0] < 0:
            raise ValueError("environment step failed")
        native_action = np.clip(action, 0, 1).astype(np.float32)
        self.position += float(native_action[0])
        self.count += 1
        obs = self._observation()
        reward = self.position
        done = self.count == 2
        info = {"count": self.count}
        result = (obs, reward, done, info)
        if self.tuple_size == 5:
            result = (obs, reward, done, False, info)
        if not capture:
            return result
        return {
            "result": result,
            "before_observation": asdict(
                Payload("test.observation", before, completeness="complete")
            ),
            "steps": [
                {
                    "action": asdict(
                        Payload("test.action", native_action, completeness="complete")
                    ),
                    "after_observation": asdict(
                        Payload("test.observation", obs, completeness="complete")
                    ),
                    "result": asdict(
                        Payload(
                            "test.result",
                            {"reward": reward, "done": done, "info": info},
                            completeness="complete",
                        )
                    ),
                }
            ],
        }

    def chunk_step(self, actions, *, return_all_frames=False, capture=False):
        results = []
        steps = []
        before = None
        for action in actions:
            response = self.step(action, capture=capture)
            result = response["result"] if capture else response
            if capture:
                if before is None:
                    before = response["before_observation"]
                steps.extend(response["steps"])
            results.append(result)
            if result[2]:
                break
        obs = [result[0] for result in results]
        result = (
            obs if return_all_frames else obs[-1],
            np.array([result[1] for result in results]),
            np.array([result[2] for result in results]),
            {"executed_actions": len(results)},
        )
        if self.tuple_size == 5:
            result = (*result[:3], np.zeros(len(results), dtype=bool), result[-1])
        if not capture:
            return result
        return {"result": result, "before_observation": before, "steps": steps}


@pytest.mark.parametrize("tuple_size", [4, 5])
@pytest.mark.parametrize(
    ("method", "all_frames"),
    [("step", False), ("chunk_step", False), ("chunk_step", True)],
)
def test_capture_preserves_results_and_executed_steps(
    transport, make_server_and_client, tuple_size, method, all_frames
):
    facade = _Env(tuple_size)
    actions = np.array([[0.5], [2.0], [0.75]], dtype=np.float64)
    action = actions[0] if method == "step" else actions
    kwargs = {} if method == "step" else {"return_all_frames": all_frames}
    with make_server_and_client(facade, transport) as rpc:
        env = BaseEnvClient(rpc, expected_meta={})
        original = getattr(env, method)(action, **kwargs)
        original_obs = env.last_obs
        env.reset()
        captured = getattr(env, method)(action, capture=True, **kwargs)

    np.testing.assert_equal(captured["result"], original)
    np.testing.assert_equal(env.last_obs, original_obs)
    before = captured["before_observation"]
    assert before["format"] == "test.observation"
    np.testing.assert_array_equal(before["data"]["state"], [0.0])
    steps = captured["steps"]
    assert len(steps) == (1 if method == "step" else 2)
    for index, step in enumerate(steps):
        assert step["action"]["format"] == "test.action"
        assert step["action"]["completeness"] == "complete"
        assert step["action"]["data"].dtype == np.float32
        np.testing.assert_array_equal(
            step["action"]["data"], [0.5 if index == 0 else 1]
        )
        after = step["after_observation"]
        assert after["format"] == "test.observation"
        assert after["data"]["state"].dtype == np.float32
        np.testing.assert_array_equal(
            after["data"]["state"], [0.5 if index == 0 else 1.5]
        )
        assert after["data"]["image"].dtype == np.uint8
        np.testing.assert_array_equal(
            after["data"]["image"], np.full((2, 3, 3), index + 1, dtype=np.uint8)
        )
        assert step["result"]["format"] == "test.result"
        assert step["result"]["data"]["info"] == {"count": index + 1}


def test_failed_chunk_raises_without_updating_cached_observation(
    transport, make_server_and_client
):
    facade = _Env()
    with make_server_and_client(facade, transport) as rpc:
        env = BaseEnvClient(rpc, expected_meta={})
        before = env.last_obs
        with pytest.raises(RpcError, match="environment step failed"):
            env.chunk_step(np.array([[0.5], [-1.0]]), capture=True)
        assert facade.count == 1
        assert env.last_obs is before


def test_disabled_capture_supports_existing_facade_signatures(
    transport, make_server_and_client
):
    class ExistingEnv(_Env):
        def step(self, action):
            return super().step(action)

        def chunk_step(self, actions, *, return_all_frames=False):
            results = [self.step(action) for action in actions]
            return results[-1]

    with make_server_and_client(ExistingEnv(), transport) as rpc:
        env = BaseEnvClient(rpc, expected_meta={})
        assert env.step(np.array([0.5]), capture=False)[1] == 0.5
        assert env.chunk_step(np.array([[0.5]]))[1] == 1.0
