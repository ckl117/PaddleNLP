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

import unittest

import numpy as np
import paddle
from paddlenlp_ops import group_quant

paddle.seed(2023)


class GroupQuantTest(unittest.TestCase):
    def native_per_token_group_quant_fp8(self, x, group_size, quant_max_bound, quant_min_bound):
        eps = 0.000001
        x_ = x.reshape([x.numel() // group_size, group_size])
        amax = x_.abs().max(axis=-1, keepdim=True).clip(min=eps).to(paddle.float32)
        x_s = amax / quant_max_bound
        x_q = (x_ / x_s).clip(min=quant_min_bound, max=quant_max_bound).to(paddle.float8_e4m3fn)
        x_q = x_q.reshape(x.shape)
        s_shape = x.shape[:-1] + [x.shape[-1] // group_size]
        x_s = x_s.reshape(s_shape)
        return x_q, x_s

    def test_group_quant_fp32(self):
        M = 32
        K = 1024
        x = np.random.rand(M, K).astype(np.float32)

        x_tensor = paddle.to_tensor(x)
        x_q_ref, x_s_ref = self.native_per_token_group_quant_fp8(
            x_tensor, group_size=128, quant_max_bound=448.0, quant_min_bound=-448.0
        )
        x_s_ref = x_s_ref.transpose([1, 0])

        x_q, x_s = group_quant(
            x_tensor, group_size=128, transpose_scale=True, quant_max_bound=448.0, quant_min_bound=-448.0
        )

        x_q_ref = x_q_ref.cast(paddle.float32)
        x_q = x_q.cast(paddle.float32)
        np.testing.assert_allclose(x_q.numpy(), x_q_ref.numpy(), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(x_s.numpy(), x_s_ref.numpy(), rtol=1e-3, atol=1e-3)

    def test_group_quant_fp16(self):
        M = 32
        K = 1024
        x = np.random.rand(M, K).astype(np.float32)

        x_tensor = paddle.to_tensor(x).cast(paddle.float16)
        x_q_ref, x_s_ref = self.native_per_token_group_quant_fp8(
            x_tensor, group_size=128, quant_max_bound=448.0, quant_min_bound=-448.0
        )
        x_s_ref = x_s_ref.transpose([1, 0])

        x_q, x_s = group_quant(
            x_tensor, group_size=128, transpose_scale=True, quant_max_bound=448.0, quant_min_bound=-448.0
        )

        x_q_ref = x_q_ref.cast(paddle.float32)
        x_q = x_q.cast(paddle.float32)
        np.testing.assert_allclose(x_q.numpy(), x_q_ref.numpy(), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(x_s.numpy(), x_s_ref.numpy(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
