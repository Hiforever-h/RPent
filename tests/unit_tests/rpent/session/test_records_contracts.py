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

"""Constructed trajectories exercising the record data contracts, without I/O."""

from datetime import datetime, timezone

import numpy as np
import pytest

from rpent.session import (
    ActionField,
    ActionFormat,
    ActionRecord,
    ArtifactRef,
    DataRef,
    MissingInfo,
    ObservationRecord,
    Payload,
    PlannerRecord,
    ToolCallRecord,
)

_TIME = datetime(2026, 10, 19, tzinfo=timezone.utc)


def test_model_context_keeps_actual_feedback_and_selected_observation_field() -> None:
    observation = ObservationRecord(
        observation_id="O1",
        control_step=0,
        capture_sequence=0,
        payload=Payload(
            format="test.observation",
            data={
                "front": ArtifactRef(
                    "images/front.png", "bytes", media_type="image/png"
                ),
                "wrist": b"unused wrist image",
                "joints": np.array([0.1, 0.2], dtype=np.float32),
            },
        ),
    )
    image_ref = DataRef("observation", "O1", "/payload/data/front")
    call = ToolCallRecord(
        tool_call_id="T1",
        planner_id="P1",
        origin="model",
        name="observe",
        arguments=Payload(format="json", data="{}"),
        status="completed",
        source_ids={"pydantic_ai.tool_call_id": "call_1"},
        raw_result=Payload(
            format="rpent.ToolResult",
            data={"data": {"objects": ["cup", "plate"]}, "images": [image_ref]},
        ),
        feedback=Payload(
            format="pydantic_ai.ToolReturn",
            data={"return_value": ["A cup is visible.", image_ref]},
        ),
        after_observation_id="O1",
        missing={
            "before_observation_id": MissingInfo(
                "not_recorded", "Only captured after the read."
            ),
            "effective_instruction": MissingInfo(
                "not_applicable", "Read-only observation."
            ),
        },
    )
    metadata_call = ToolCallRecord(
        tool_call_id="T2",
        planner_id="P1",
        origin="model",
        name="read_metadata",
        arguments=Payload("json", {"key": "task"}),
        status="completed",
        source_ids={"pydantic_ai.tool_call_id": "call_2"},
        raw_result=Payload("json", {"task": "Move the cup.", "seed": 1}),
        feedback=Payload("pydantic_ai.ToolReturn", {"return_value": "Move the cup."}),
    )
    response = {
        "kind": "response",
        "parts": [
            {
                "part_kind": "thinking",
                "content": "Inspect the scene.",
                "signature": "sig",
            },
            {
                "part_kind": "tool-call",
                "tool_name": "observe",
                "args": "{}",
                "tool_call_id": "call_1",
            },
            {
                "part_kind": "tool-call",
                "tool_name": "read_metadata",
                "args": {"key": "task"},
                "tool_call_id": "call_2",
            },
        ],
    }
    request = Payload(
        format="pydantic_ai.ModelRequestContext",
        format_version="2.53.0",
        completeness="complete",
        data={
            "messages": [
                {
                    "kind": "request",
                    "parts": [{"part_kind": "user-prompt", "content": "Move the cup."}],
                },
                response,
                {
                    "kind": "request",
                    "parts": [
                        {
                            "part_kind": "tool-return",
                            "tool_name": "observe",
                            "tool_call_id": "call_1",
                            "content": ["A cup is visible.", image_ref],
                        },
                        {
                            "part_kind": "tool-return",
                            "tool_name": "read_metadata",
                            "tool_call_id": "call_2",
                            "content": "Move the cup.",
                        },
                    ],
                },
            ],
            "model_settings": {"max_tokens": 1024},
            "model_request_parameters": {
                "function_tools": [
                    {"name": "move_to", "parameters_json_schema": {"type": "object"}}
                ]
            },
        },
    )
    first = PlannerRecord(
        planner_id="P1",
        backend="api",
        status="completed",
        response=Payload("pydantic_ai.ModelResponse", response),
        tool_calls=[call, metadata_call],
        ended_at=_TIME,
        missing={
            "request": MissingInfo("not_recorded", "Capture started after the request.")
        },
    )
    second = PlannerRecord(
        planner_id="P2",
        backend="api",
        status="open",
        request=request,
        missing={"response": MissingInfo("pending", "Waiting for a response.")},
    )

    context = second.request.data
    assert [message["kind"] for message in context["messages"]] == [
        "request",
        "response",
        "request",
    ]
    assert context["messages"][1]["parts"][0]["signature"] == "sig"
    feedback_parts = context["messages"][2]["parts"]
    returned = feedback_parts[0]["content"]
    assert returned == ["A cup is visible.", image_ref]
    assert returned != call.raw_result.data
    assert (
        image_ref.record_id == observation.observation_id == call.after_observation_id
    )
    assert image_ref.path == "/payload/data/front"
    assert context["model_settings"] == {"max_tokens": 1024}
    assert context["model_request_parameters"]["function_tools"][0]["name"] == "move_to"
    declared_calls = first.response.data["parts"][1:]
    assert [part["tool_call_id"] for part in declared_calls] == [
        tool.source_ids["pydantic_ai.tool_call_id"] for tool in first.tool_calls
    ]
    assert [part["tool_call_id"] for part in feedback_parts] == ["call_1", "call_2"]
    assert all(tool.planner_id == first.planner_id for tool in first.tool_calls)
    assert declared_calls[0]["args"] == "{}"
    assert feedback_parts[1]["content"] == metadata_call.feedback.data["return_value"]
    assert first.request is None and first.missing["request"].reason == "not_recorded"
    assert second.response is None and second.tool_calls == []


def test_partial_model_response_preserves_fragments_without_inventing_context() -> None:
    record = PlannerRecord(
        planner_id="P1",
        backend="test_sdk",
        status="cancelled",
        source_ids={"test_sdk.model_call_id": "model_1"},
        response=Payload(
            "test_sdk.ModelResponseStream",
            [
                {"sequence": 1, "delta": "Looking"},
                {"sequence": 3, "delta": " at the scene"},
            ],
            completeness="partial",
        ),
        missing={
            "request": MissingInfo(
                "not_exposed", "SDK does not expose the model request."
            ),
            "response": MissingInfo(
                "not_recorded", "Interrupted before the complete model response."
            ),
        },
    )

    assert [part["delta"] for part in record.response.data] == [
        "Looking",
        " at the scene",
    ]
    assert [part["sequence"] for part in record.response.data] == [1, 3]
    assert record.response.completeness == "partial"
    assert record.request is None
    assert record.missing["request"].reason == "not_exposed"
    assert record.missing["response"].reason == "not_recorded"


def test_completed_model_call_retains_independent_tool_execution_states() -> None:
    response = Payload(
        "pydantic_ai.ModelResponse",
        {
            "parts": [
                {
                    "part_kind": "tool-call",
                    "tool_call_id": f"call_{index}",
                    "tool_name": "move_to",
                    "args": {"target": target},
                }
                for index, target in enumerate(["cup", "plate", "bowl"], start=1)
            ]
        },
    )
    record = PlannerRecord(
        planner_id="P1",
        backend="api",
        status="completed",
        response=response,
        ended_at=_TIME,
        tool_calls=[
            ToolCallRecord(
                tool_call_id=f"T{index}",
                planner_id="P1",
                origin="model",
                name=part["tool_name"],
                arguments=Payload("json", part["args"]),
                status="requested",
                source_ids={"pydantic_ai.tool_call_id": part["tool_call_id"]},
            )
            for index, part in enumerate(response.data["parts"], start=1)
        ],
        missing={"request": MissingInfo("not_recorded", "Request was not captured.")},
    )

    first, second, third = record.tool_calls
    second.status = "failed"
    second.raw_result = Payload("rpent.ToolResult", {"error": "Target unreachable."})
    first.status = "completed"
    first.raw_result = Payload("rpent.ToolResult", {"done": True})
    third.status = "not_executed"
    third.missing["raw_result"] = MissingInfo(
        "not_applicable", "Execution stopped before this call."
    )

    assert record.status == "completed" and record.ended_at == _TIME
    assert record.response is response
    assert [call.tool_call_id for call in record.tool_calls] == ["T1", "T2", "T3"]
    assert [call.status for call in record.tool_calls] == [
        "completed",
        "failed",
        "not_executed",
    ]
    assert [part["args"] for part in record.response.data["parts"]] == [
        {"target": "cup"},
        {"target": "plate"},
        {"target": "bowl"},
    ]
    assert third.raw_result is None


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_unsuccessful_model_call_can_retain_input_without_a_response(
    status: str,
) -> None:
    record = PlannerRecord(
        planner_id="P1",
        backend="api",
        status=status,
        request=Payload("pydantic_ai.ModelRequestContext", {"messages": []}),
        missing={
            "response": MissingInfo("not_recorded", "Request ended without a response.")
        },
    )

    assert record.request.data == {"messages": []}
    assert record.response is None
    assert record.missing["response"].reason == "not_recorded"


def test_program_calls_and_unlinked_model_calls_keep_distinct_missing_reasons() -> None:
    calls = []
    for tool_call_id, origin, reason in [
        ("T1", "program", "not_applicable"),
        ("T2", "model", "not_exposed"),
    ]:
        call = ToolCallRecord(
            tool_call_id=tool_call_id,
            planner_id=None,
            origin=origin,
            name="read_metadata",
            arguments=Payload("json", {}),
            status="completed",
            raw_result=Payload("json", None, completeness="complete"),
            feedback=Payload(
                "json", DataRef("tool_call", tool_call_id, "/raw_result/data")
            ),
            missing={
                "planner_id": MissingInfo(
                    reason, "No issuing model record is available."
                ),
                "effective_instruction": MissingInfo(
                    "not_applicable", "Reads metadata."
                ),
                "before_observation_id": MissingInfo(
                    "not_applicable", "No environment access."
                ),
                "after_observation_id": MissingInfo(
                    "not_applicable", "No environment access."
                ),
            },
        )
        calls.append(call)

    program, model = calls
    assert program.planner_id is None and model.planner_id is None
    assert program.missing["planner_id"].reason != model.missing["planner_id"].reason
    assert program.raw_result is not None and program.raw_result.data is None
    assert "raw_result" not in program.missing
    assert program.feedback.data == DataRef("tool_call", "T1", "/raw_result/data")


def test_action_boundaries_allow_shared_recaptured_and_unavailable_observations() -> (
    None
):
    observations = {
        key: ObservationRecord(
            observation_id=key,
            control_step=step,
            capture_sequence=sequence,
            payload=Payload(
                "test.observation", {"joints": np.array([step], dtype=np.float32)}
            ),
        )
        for sequence, (key, step) in enumerate(
            [("O0", 0), ("O1", 1), ("O2", 2), ("O3", 2)]
        )
    }
    call = ToolCallRecord(
        tool_call_id="T1",
        planner_id="P1",
        origin="model",
        name="move",
        arguments=Payload("json", {"instruction": "Move the cup."}),
        status="failed",
        effective_instruction="Move the cup.",
        before_observation_id="O0",
        after_observation_id="O3",
        missing={
            "raw_result": MissingInfo(
                "capture_failed", "Execution layer disconnected."
            ),
            "feedback": MissingInfo("not_recorded", "No feedback was delivered."),
        },
    )
    actions = [
        ActionRecord(
            action_id="A1",
            tool_call_id="T1",
            source="vla",
            action_format_id="native-1",
            control_step=0,
            action=Payload("test.native_action", np.array([0.25], dtype=np.float32)),
            before_observation_id="O0",
            after_observation_id="O1",
            result=Payload("test.step_result", {"done": False}),
        ),
        ActionRecord(
            action_id="A2",
            tool_call_id="T1",
            source="tool_computed",
            action_format_id="native-1",
            control_step=1,
            action=Payload("test.native_action", np.array([-0.5], dtype=np.float32)),
            before_observation_id="O1",
            after_observation_id="O2",
            result=Payload("test.step_result", {"done": False}),
        ),
        ActionRecord(
            action_id="A3",
            tool_call_id="T2",
            source="direct",
            action_format_id="native-1",
            control_step=2,
            action=Payload("test.native_action", np.array([0.0], dtype=np.float32)),
            before_observation_id="O3",
            after_observation_id=None,
            missing={
                "after_observation_id": MissingInfo(
                    "capture_failed", "Camera timed out after execution."
                ),
                "result": MissingInfo(
                    "not_exposed", "Only execution acknowledgement is available."
                ),
            },
        ),
    ]

    assert call.before_observation_id == actions[0].before_observation_id
    assert actions[0].after_observation_id == actions[1].before_observation_id
    assert call.after_observation_id != actions[1].after_observation_id
    assert observations["O2"].control_step == observations["O3"].control_step
    assert observations["O2"].capture_sequence < observations["O3"].capture_sequence
    assert [action.source for action in actions] == ["vla", "tool_computed", "direct"]
    assert actions[2].after_observation_id is None
    assert actions[2].missing["after_observation_id"].reason == "capture_failed"
    assert actions[0].action.data.dtype == np.dtype("float32")
    assert actions[0].result.data == {"done": False}


def test_dictionary_action_format_retains_native_layout_and_unknown_semantics() -> None:
    action_format = ActionFormat(
        action_format_id="joint-dict",
        representation="dict",
        fields=[
            ActionField(
                name="arm",
                path="/arm/qpos",
                shape=(2,),
                dtype="float32",
                control_mode="joint_position",
                unit="radian",
                missing={
                    "coordinate_frame": MissingInfo(
                        "not_applicable", "Joint coordinates."
                    )
                },
            )
        ],
        native_spec=Payload(
            "test.action_spec", {"description": "Native joint dictionary."}
        ),
        missing={
            "control_period_s": MissingInfo(
                "not_exposed", "Environment does not provide a period."
            )
        },
    )
    action = ActionRecord(
        action_id="A1",
        tool_call_id="T1",
        source="vla",
        action_format_id=action_format.action_format_id,
        control_step=None,
        action=Payload(
            "test.native_joint_action",
            {
                "arm": {"qpos": np.array([0.1, 0.2], dtype=np.float32)},
                "gripper": np.float32(0.5),
            },
        ),
        before_observation_id=None,
        after_observation_id=None,
        result=Payload(
            "test.step_result", {"reward": 1.0, "done": True, "info": {"success": True}}
        ),
        missing={
            "control_step": MissingInfo(
                "not_exposed", "Execution result does not include a control step."
            ),
            "before_observation_id": MissingInfo(
                "not_recorded", "Boundary capture was disabled."
            ),
            "after_observation_id": MissingInfo(
                "not_recorded", "Boundary capture was disabled."
            ),
        },
    )

    assert action_format.control_period_s is None
    assert action_format.fields[0].coordinate_frame is None
    assert action_format.fields[0].path == "/arm/qpos"
    assert action.action.data["arm"]["qpos"].shape == action_format.fields[0].shape
    assert action.action.data["arm"]["qpos"].dtype == np.dtype(
        action_format.fields[0].dtype
    )
    assert isinstance(action.action.data["gripper"], np.float32)
    assert set(action.result.data) == {"reward", "done", "info"}
    assert action.control_step is None
    assert action.missing["control_step"].reason == "not_exposed"
