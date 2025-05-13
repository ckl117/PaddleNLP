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

from collections import deque

import paddle
from schedule.config import CacheConfig, SchedulerConfig
from schedule.kv_cache_manager import KVCacheManager
from schedule.output import ModelRunnerOutput, SchedulerOutput
from schedule.request import Request, RequestStatus


class Scheduler:
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.kv_cache_manager = KVCacheManager(self.cache_config)

        self.max_batch_size = scheduler_config.max_batch_size
        # req_id -> Request
        self.requests: dict[int, Request] = {}
        # Priority queues for requests.
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.swapped: list[Request] = []

    def init_kv_cache(
        self,
    ) -> paddle.Tensor:
        kv_cache_tensor = self.kv_cache_manager.initialize_kv_caches()
        return kv_cache_tensor

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)
        self.requests[request.request_id] = request

    def schedule(self) -> SchedulerOutput:

        scheduled_new_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []
        req_block_ids: dict[int, tuple[list[int]]] = {}
        num_scheduled_tokens: dict[int, int] = {}
        scheduled_req_ids: set[int] = set()

        req_index = 0
        while req_index < len(self.running):
            if len(scheduled_req_ids) >= self.max_batch_size:
                break
            request = self.running[req_index]
            num_new_tokens = 1
            while True:
                decoder_block_tuple = self.kv_cache_manager.allocate_decoder(request, num_new_tokens)
                if decoder_block_tuple is None:
                    preempted_req = self.running.pop()
                    self.kv_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.output_token_ids = []
                    self.waiting.appendleft(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert decoder_block_tuple is not None

            # Schedule the request.
            request.status = RequestStatus.RUNNING_DECODER
            scheduled_running_reqs.append(request)
            num_scheduled_tokens[request.request_id] = 1
            scheduled_req_ids.add(request.request_id)

            req_block_ids[request.request_id] = decoder_block_tuple
            req_index += 1

        if not preempted_reqs:
            while self.waiting:
                if len(scheduled_req_ids) >= self.max_batch_size:
                    break

                request = self.waiting[0]

                block_ids_tuple = None
                block_ids_tuple = self.kv_cache_manager.allocate_prefill(request)

                # can't schedule
                if block_ids_tuple is None:
                    break

                self.waiting.popleft()
                req_index += 1
                self.running.append(request)
                request.status = RequestStatus.RUNNING_PREFILL
                scheduled_new_reqs.append(request)
                num_scheduled_tokens[request.request_id] = request.num_prompt_tokens
                scheduled_req_ids.add(request.request_id)
                req_block_ids[request.request_id] = block_ids_tuple

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_running_reqs=scheduled_running_reqs,
            req_block_ids=req_block_ids,
            num_scheduled_tokens=num_scheduled_tokens,
        )
        return scheduler_output

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ):
        req_id_to_index = model_runner_output.req_id_to_index
        next_tokens = model_runner_output.next_tokens
        stop_flags = model_runner_output.stop_flags
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        new_running = []
        for idx, request in enumerate(self.running):
            req_id = request.request_id
            num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_tokens_scheduled == 0:
                new_running.append(request)
                continue
            req_index = req_id_to_index[req_id]
            next_token = next_tokens[req_index]
            stop_flag = stop_flags[req_index]
            request.output_token_ids.append(next_token)
            if stop_flag:
                self.kv_cache_manager.free(request, stop=True)
                continue
            new_running.append(request)
        self.running = new_running

    def has_unfinished_requests(
        self,
    ):
        return len(self.waiting) + len(self.running) + len(self.swapped) > 0
