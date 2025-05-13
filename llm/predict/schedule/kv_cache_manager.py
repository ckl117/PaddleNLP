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

import os
from collections import deque
from typing import Optional

import paddle
from schedule.config import CacheConfig
from schedule.request import Request


class KVCacheManager:
    def __init__(self, config: CacheConfig):
        self.config = config
        self.dtype = paddle.bfloat16
        self.block_size = self.config.block_size

        self.cache_k_shapes = self.config.cache_k_shapes
        self.cache_v_shapes = self.config.cache_v_shapes

        self.gpu_memory_utilization = self.config.gpu_memory_utilization

        self.single_cache_bytes = self.get_single_cache_bytes()
        self.max_num_blocks = self.get_max_num_blocks()
        self.num_tokens = self.max_num_blocks * self.block_size

        self.free_block_deque: deque[int] = deque(list(range(self.max_num_blocks)))

        self.request_prefill_block_cache: dict[int, tuple[list[int]]] = {}
        self.request_prefill_block_cache_ref_cnt: dict[int, int] = {}

        self.request_decoder_block_cache: dict[str, list[int]] = {}

    @property
    def free_block_num(self) -> int:
        return len(self.free_block_deque)

    def initialize_kv_caches(
        self,
    ):
        cache_kvs = []
        if self.cache_k_shapes and self.cache_v_shapes:
            for cache_k_shape, cache_v_shape in zip(self.cache_k_shapes, self.cache_v_shapes):
                cache_k_shape[0] = self.max_num_blocks
                cache_v_shape[0] = self.max_num_blocks
                cache_kvs.append(paddle.zeros(cache_k_shape, dtype=self.dtype))
                cache_kvs.append(paddle.zeros(cache_v_shape, dtype=self.dtype))
        else:
            # for mla's absorption
            assert self.cache_v_shapes is None
            for cache_k_shape in self.cache_k_shapes:
                cache_k_shape[1] = self.max_num_blocks
                cache_kvs.append(paddle.zeros(cache_k_shape, dtype=self.dtype))
        return cache_kvs

    def get_max_num_blocks(
        self,
    ):
        return self.determine_available_memory() // self.single_cache_bytes

    def get_single_cache_bytes(
        self,
    ):
        total_bytes = 1
        for k_shape in self.cache_k_shapes:
            current_layer_num = 1
            for i in k_shape[1:]:
                current_layer_num *= i
            total_bytes += current_layer_num

        if self.cache_v_shapes is not None:
            for v_shape in self.cache_v_shapes:
                current_layer_num = 1
                for i in v_shape[1:]:
                    current_layer_num *= i
                total_bytes += current_layer_num

        if self.dtype == paddle.bfloat16 or self.dtype == paddle.float16:
            total_bytes *= 2
        return total_bytes

    def determine_available_memory(
        self,
    ) -> int:

        device_id = int(os.getenv("FLAGS_selected_gpus"))
        print(f"current_device = {device_id}")

        # import pynvml
        # pynvml.nvmlInit()
        # handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # total_gpu_memory = meminfo.total
        # used_gpu_memory = meminfo.used
        # pynvml.nvmlShutdown()

        prop = paddle.device.cuda.get_device_properties()
        total_gpu_memory = prop.total_memory

        max_memory_reserved = paddle.device.cuda.max_memory_reserved()
        # paddle_max_memory_allocated = paddle.device.cuda.max_memory_allocated()

        available_kv_cache_memory = total_gpu_memory * self.gpu_memory_utilization - max_memory_reserved

        return int(available_kv_cache_memory)

    def allocate_prefill(self, request: Request) -> Optional[tuple[list[int]]]:
        """
        返回该request的prefill阶段的block以及tail_block
        """
        query_id = request.query_id
        prefill_cache_ref_cnt = self.request_prefill_block_cache_ref_cnt.get(query_id, None)
        if prefill_cache_ref_cnt:
            self.request_prefill_block_cache_ref_cnt[query_id] += 1
            return self.request_prefill_block_cache[query_id]

        # check block_num <= free_block_num
        num_prompt_tokens = request.num_prompt_tokens
        prefill_block_num = num_prompt_tokens // self.block_size
        tail_block_num = request.repeat_num
        if prefill_block_num + tail_block_num > self.free_block_num:
            return None

        # allocate
        prefill_block_ids: list = []
        excess_block_ids: list = []
        for _ in range(prefill_block_num):
            prefill_block_ids.append(self.free_block_deque.popleft())
        for _ in range(tail_block_num):
            excess_block_ids.append(self.free_block_deque.popleft())

        self.request_prefill_block_cache[query_id] = (prefill_block_ids, excess_block_ids)
        self.request_prefill_block_cache_ref_cnt[query_id] = 1

        return (prefill_block_ids, excess_block_ids)

    def allocate_decoder(
        self,
        request: Request,
        num_tokens: int = 1,
    ) -> Optional[tuple[list[int]]]:
        request_id = request.request_id
        query_id = request.query_id
        repeat_num = request.repeat_num

        prefill_ref_cnt = self.request_prefill_block_cache_ref_cnt.get(query_id, None)
        assert prefill_ref_cnt is not None
        assert prefill_ref_cnt > 0

        # check block_num owned by request satisfy infer
        allocated_block_num = (request.num_output_tokens - 1 + self.block_size - 1) // self.block_size
        new_block_num = (request.num_output_tokens - 1 + num_tokens + self.block_size - 1) // self.block_size

        need_block_num = new_block_num - allocated_block_num
        # print(f'need_block_num = {need_block_num}')

        if self.free_block_num < need_block_num:
            return None
        prefill_block_ids, excess_block_ids = self.request_prefill_block_cache[query_id]
        decoder_block_ids = []
        decoder_block_ids.extend(prefill_block_ids)
        decoder_block_ids.append(excess_block_ids[request_id - query_id * repeat_num])
        cache_decoder_block_ids = self.request_decoder_block_cache.get(request_id, [])
        decoder_block_ids.extend(cache_decoder_block_ids)
        if need_block_num > 0:
            for _ in range(need_block_num):
                new_block_ids = self.free_block_deque.popleft()
                if self.request_decoder_block_cache.get(request_id, None) is None:
                    self.request_decoder_block_cache[request_id] = []
                self.request_decoder_block_cache[request_id].append(new_block_ids)
                decoder_block_ids.append(new_block_ids)
        return (decoder_block_ids, None)

    def has_cached_prefill(
        self,
        request: Request,
    ) -> bool:
        query_id = request.query_id
        prefill_cache_ref_cnt = self.request_prefill_block_cache_ref_cnt.get(query_id, 0)
        if prefill_cache_ref_cnt:
            return True
        return False

    def free(self, request: Request, stop=False) -> None:
        # decoder blocks
        blocks = self.request_decoder_block_cache.pop(request.request_id, [])
        self.free_block_deque.extend(blocks)

        # prefill blocks
        query_id = request.query_id
        prefill_cache_ref_cnt = self.request_prefill_block_cache_ref_cnt.get(query_id, 0)
        if prefill_cache_ref_cnt > 0:
            self.request_prefill_block_cache_ref_cnt[query_id] -= 1
            if self.request_prefill_block_cache_ref_cnt[query_id] == 0:
                prefill_blocks, tail_blocks = self.request_prefill_block_cache[query_id]
                self.free_block_deque.extend(prefill_blocks)
                self.free_block_deque.extend(tail_blocks)
                self.request_prefill_block_cache.pop(query_id, [])
                self.request_prefill_block_cache_ref_cnt.pop(query_id, [])
                if not stop:
                    print(f"Free preempted_req:{query_id}, output_token_num={request.num_output_tokens}")
