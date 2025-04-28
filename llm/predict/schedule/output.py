# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from schedule.request import Request


@dataclass
class NewRequestData:

    req_id: int
    prompt_token_ids: list[int]
    prompt: Optional[str]
    block_ids: list[int]
    exceed_blocks: list[int]
    num_computed_tokens: int

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: list[int],
    ) -> NewRequestData:

        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            prompt=request.prompt,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
        )


@dataclass
class SchedulerOutput:

    scheduled_new_reqs: list[Request]
    scheduled_running_reqs: list[Request]
    req_block_ids: dict[int, tuple[list[int]]]
    num_scheduled_tokens: dict[int, int]


@dataclass
class ModelRunnerOutput:

    req_ids: list[int]
    # req_id -> index
    req_id_to_index: dict[int, int]

    next_tokens: list[int]
    stop_flags: list[bool]
