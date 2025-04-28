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

import enum
from typing import Optional, Union


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    RUNNING_PREFILL = enum.auto()
    RUNNING_WAITING_FIRST_TOKEN = enum.auto()
    RUNNING_DECODER = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED


class Request:
    def __init__(
        self,
        request_id: int,
        query_id: int,
        prompt: Optional[str],
        prompt_token_ids: list[int],
        repeat_num: int,
        max_length: int,
        min_length: int,
        eos_token_id: Optional[int],
        arrival_time: float,
    ) -> None:
        self.request_id = request_id
        self.query_id = query_id
        self.repeat_num = repeat_num
        self.status = RequestStatus.WAITING
        self.max_length = max_length
        self.min_length = min_length
        self.eos_token_id = eos_token_id
        self.arrival_time = arrival_time

        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)

        # output token list
        self.output_token_ids: list[int] = []
        self.all_token_ids: list[int] = self.prompt_token_ids.copy()
        self.num_computed_tokens = 0

    def append_output_token_ids(
        self,
        token_ids: Union[int, list[int]],
    ) -> None:
        if isinstance(token_ids, int):
            self.output_token_ids.append(token_ids)
            # self.all_token_ids.append(token_ids)
        else:
            self.output_token_ids.extend(token_ids)
            # self.all_token_ids.extend(token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.all_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)
