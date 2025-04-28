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

import copy
from dataclasses import dataclass

import paddle


class CacheConfig:
    def __init__(
        self,
        dtype=paddle.bfloat16,
        gpu_memory_utilization=0.8,
        block_size=64,
        cache_k_shapes=None,
        cache_v_shapes=None,
    ):
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.cache_k_shapes = []
        self.cache_v_shapes = []
        if cache_k_shapes is None:
            self.cache_k_shapes = [
                [
                    1,  # max_block_num
                    8 // 4,  # num_key_value_heads // tensor_parallel_degree
                    64,  # block_size
                    5120 // 40,  # hidden_size // num_attention_heads
                ]
                for i in range(64)
            ]
        else:
            self.cache_k_shapes = copy.deepcopy(cache_k_shapes)
        if cache_v_shapes is None:
            self.cache_v_shapes = copy.deepcopy(self.cache_k_shapes)
        else:
            self.cache_v_shapes = copy.deepcopy(cache_v_shapes)
        self.block_size = block_size


@dataclass
class SchedulerConfig:

    # Maximum number of sequences to be processed in a single iteration.
    max_batch_size: int = 128

    # Maximum length of a sequence (including prompt and generated text).
    total_max_length: int = 8192
