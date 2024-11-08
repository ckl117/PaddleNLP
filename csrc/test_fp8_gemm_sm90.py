# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddlenlp_ops import cutlass_fp8_fp8_half_gemm_sm90_fused


def fp16_gemm_sm90(A, B, bias=None, scale=1.0, output_scale=1.0, act="identity"):
    # A = A.cast('float16')
    # B = B.cast('float16')
    if bias is not None:
        bias = bias.cast("float16")
    fp16_gemm_res = cutlass_fp8_fp8_half_gemm_sm90_fused(
        A, B, bias=bias, transpose_x=False, transpose_y=True, scale=scale, output_dtype="float16", act=act
    )
    # fp16_gemm_res = fp16_gemm_res.cast("float32") * output_scale
    # fp16_gemm_res = fp16_gemm_res.cast("float8_e4m3fn").cast("float32")
    return fp16_gemm_res


def gemm(m, n, k):
    paddle.seed(1)
    A = paddle.rand(shape=[m, k]).cast("float8_e4m3fn")
    B = paddle.rand(shape=[n, k]).cast("float8_e4m3fn")
    bias = paddle.rand(shape=[n]).cast("float32").cast("float8_e4m3fn").cast("float32")
    # bias = None
    act = "identity"
    # act = "relu"
    # act = "gelu"

    scale = 1.0
    output_scale = 1.0

    fp16_gemm_sm90(A, B, bias=bias, scale=scale, output_scale=output_scale, act=act)

    return None


if __name__ == "__main__":
    m_max = 32
    # qwen2-7B
    ns = [4608, 3584, 3584]
    ks = [3584, 3584, 18944]
    for m in range(32, m_max + 32, 32):
        for idx in range(len(ns)):
            n = ns[idx]
            k = ks[idx]
            gemm(m, n, k)
        paddle.device.cuda.empty_cache()
