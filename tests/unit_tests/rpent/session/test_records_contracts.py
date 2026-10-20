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

"""Native payload ownership and independent record defaults."""

import numpy as np
import pytest

from rpent.session import (
    ActionRecord,
    ArtifactRef,
    DataRef,
    MissingInfo,
    ObservationRecord,
    Payload,
    PlannerRecord,
    ToolCallRecord,
)


def test_payload_keeps_native_data_without_copying_or_coercing():
    values = np.array([0.1, 0.2], dtype=np.float32)
    data = {
        "values": values,
        "image": bytes([0, 255]),
        "references": [
            ArtifactRef("artifacts/image.bin", "bytes"),
            DataRef("observation", "O1", "/payload/data/image"),
        ],
    }
    payload = Payload("test.native", data)

    assert payload.data is data
    assert payload.data["values"] is values
    values[0] = 0.5
    data["references"].clear()
    assert payload.data["values"][0] == 0.5
    assert payload.data["references"] == []


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Payload("json", {}),
        lambda: PlannerRecord("P1", "api", "open"),
        lambda: ToolCallRecord(
            "T1", "P1", "model", "observe", Payload("json", {}), "requested"
        ),
        lambda: ObservationRecord("O1", Payload("test.observation", {})),
        lambda: ActionRecord(
            "A1", "T1", "vla", Payload("test.action", [0.0]), None, None
        ),
    ],
    ids=["payload", "planner", "tool_call", "observation", "action"],
)
def test_missing_reasons_are_not_shared_between_records(factory):
    first, second = factory(), factory()
    first.missing["result"] = MissingInfo("pending", "Waiting for a result.")
    assert second.missing == {}


def test_call_collections_are_not_shared_between_records():
    first, second = [PlannerRecord(f"P{index}", "api", "open") for index in range(2)]
    calls = [
        ToolCallRecord(
            tool_call_id=f"T{index}",
            planner_id=first.planner_id,
            origin="model",
            name="observe",
            arguments=Payload("json", {}),
            status="requested",
        )
        for index in range(2)
    ]
    first.tool_calls.append(calls[0])
    first.source_ids["provider.response_id"] = "response-1"
    calls[0].source_ids["pydantic_ai.tool_call_id"] = "call-1"

    assert second.tool_calls == []
    assert second.source_ids == {}
    assert calls[1].source_ids == {}
