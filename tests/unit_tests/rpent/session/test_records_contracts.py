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

"""Native payload ownership."""

import numpy as np

from rpent.session import ArtifactRef, DataRef, Payload


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
