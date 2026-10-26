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

import numpy as np
import pytest

from rpent.data.reader import iter_trajectories
from rpent.session import EnvState, MissingInfo, Payload


@pytest.fixture
def trajectory(tmp_path, request):
    vla_steps = getattr(request, "param", 6)
    state = EnvState(tmp_path)
    state.start_task(
        instruction="Put the bowl on the stove",
        config=Payload(
            "rpent.config",
            {
                "arguments": {
                    "robot_name": "libero",
                    "suite": "libero_goal",
                    "task": 1,
                    "seed": 0,
                }
            },
        ),
    )
    planner = state.register_planner(backend="api")
    tools = [
        state.register_tool_call(
            planner_id=planner, name=name, arguments=Payload("json", {})
        )
        for name in ("move_to", "pi0_pick")
    ]
    image = state.save_trajectory_artifact(np.zeros((256, 256, 3), dtype=np.uint8))
    observations = [
        state.register_observation(
            payload=Payload(
                "libero.observation",
                {
                    "main_images": image,
                    "wrist_images": image,
                    "states": np.full(8, index, dtype=np.float32),
                    "task_descriptions": "Put the bowl on the stove",
                },
                completeness="complete",
                inline_array_paths=("/states",),
            )
        )
        for index in range(vla_steps + 3)
    ]
    values = np.arange((vla_steps + 2) * 7, dtype=np.float32).reshape(-1, 7) / 100
    actions = [
        state.register_action(
            tool_call_id=tools[1] if 1 <= index <= vla_steps else tools[0],
            source="vla" if 1 <= index <= vla_steps else "tool_computed",
            action=Payload(
                "libero.action",
                value,
                completeness="complete",
                inline_array_paths=("",),
            ),
            before_observation_id=observations[index],
            after_observation_id=observations[index + 1],
            result=Payload(
                "libero.step_result",
                {"terminated": index == vla_steps + 1, "truncated": False},
                completeness="complete",
            ),
            missing={"executed_at": MissingInfo("not_exposed", "No execution clock")},
        )
        for index, value in enumerate(values)
    ]
    for start in range(1, vla_steps + 1, 3):
        stop = min(start + 3, vla_steps + 1)
        state.register_prediction(
            tool_call_id=tools[1],
            input_observation_id=observations[start],
            instruction="pick up the bowl",
            actions=Payload(
                "libero.action_chunk",
                values[start:stop],
                completeness="complete",
                inline_array_paths=("",),
            ),
            action_ids=actions[start:stop],
        )
    state.end_task(
        status="completed",
        stop_reason="finish",
        result=Payload("libero.result", {"environment_success": True}),
    )
    return next(iter_trajectories(state.save_planner_trajectory()))
