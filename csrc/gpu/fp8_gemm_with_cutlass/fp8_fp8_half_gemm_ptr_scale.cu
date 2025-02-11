// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include "fp8_common.h"

#include "cutlass/cutlass.h"

#include "cutlass_kernels/epilogue/scaled_mm_epilogues_c3x.hpp"
#include "fp8_gemm_fused/c3x/scaled_mm_sm90_fp8_dispatch.cuh"


namespace pd {

void cutlass_scaled_mm_sm90_fp8(paddle::Tensor& out, paddle::Tensor const& a,
                                paddle::Tensor const& b,
                                paddle::Tensor const& a_scales,
                                paddle::Tensor const& b_scales,
                                paddle::optional<paddle::Tensor> const& bias) {
  if (bias) {
    PD_CHECK(bias->dtype() == out.dtype(),
                "currently bias dtype must match output dtype ", out.dtype());
    return cutlass_scaled_mm_sm90_fp8_epilogue<c3x::ScaledEpilogueBias>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm90_fp8_epilogue<c3x::ScaledEpilogue>(
        out, a, b, a_scales, b_scales);
  }
}
}

std::vector<paddle::Tensor> cutlass_fp8_fp8_half_gemm_ptr_scale(
    const paddle::Tensor& x,
    const paddle::Tensor& y,
    const paddle::optional<paddle::Tensor>& x_scale,
    const paddle::optional<paddle::Tensor>& y_scale,
    const paddle::optional<paddle::Tensor>& bias,
    bool trans_x,
    bool trans_y,
    float scale,  
    std::string output_dtype,
    std::string activation_type) {
  paddle::Tensor out;

  int rank = x.dims().size();
  int M = 0;
  int K = 0;
  int N = 0;
  int ldd = 0;
  
  int lda = x.dims()[rank - 1];
  int ldb = y.dims()[rank - 1];

  if (!trans_x) {
    M = x.dims()[rank - 2];
    K = x.dims()[rank - 1];

  } else {
    M = x.dims()[rank - 1];
    K = x.dims()[rank - 2];
  }
  if (!trans_y) {
    N = y.dims()[rank - 1];
    ldd = y.dims()[rank - 1];
  } else {
    N = y.dims()[rank - 2];
    ldd = y.dims()[rank - 2];
  }

  std::vector<int64_t> out_shape = x.shape();
  out_shape[rank - 1] = N;
  out_shape[rank - 2] = M;

  if (output_dtype == "bfloat16") {
    out = paddle::empty(out_shape, paddle::DataType::BFLOAT16, x.place());
  } else if (output_dtype == "float16") {
    out = paddle::empty(out_shape, paddle::DataType::FLOAT16, x.place());
  } else {
    PADDLE_THROW(phi::errors::Fatal(
        "fp8_fp8_half_gemm_fused only support bfloat16 and float16 output"));
  }

  pd::cutlass_scaled_mm_sm90_fp8(out, x, y, x_scale.get(), y_scale.get(), bias);
  return {out};
}

std::vector<std::vector<int64_t>> CutlassFp8Fp8HalfGemmPtrScaleFusedInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y_shape,
    const paddle::optional<std::vector<int64_t>>&  x_scale_shape,
    const paddle::optional<std::vector<int64_t>>&  y_scale_shape,
    const paddle::optional<std::vector<int64_t>>&  bias_shape,
    bool trans_x,
    bool trans_y){
  PADDLE_ENFORCE_EQ(x_shape.size(),
                    y_shape.size(),
                    phi::errors::InvalidArgument(
                      "The rank of input X and Y should be equal, but received X's rank is %d, Y's rank is %d.",
                      x_shape.size(),
                      y_shape.size()));
      
  int rank = x_shape.size();
  int M = 0;
  int N = 0;

  if (!trans_x) {
    M = x_shape[rank - 2];
  } else {
    M = x_shape[rank - 1];
  }
  if (!trans_y) {
    N = y_shape[rank - 1];
  } else {
    N = y_shape[rank - 2];
  }
  std::vector<int64_t> out_shape = x_shape;
  out_shape[rank - 1] = N;
  out_shape[rank - 2] = M;
  return {out_shape};
}

std::vector<paddle::DataType> CutlassFp8Fp8HalfGemmPtrScaleFusedInferDtype(
    const paddle::DataType& x_type,
    const paddle::DataType& y_type,
    const paddle::optional<paddle::DataType>& x_scale_type,
    const paddle::optional<paddle::DataType>& y_scale_type,
    const paddle::optional<paddle::DataType>& bias_type,
    bool trans_x,
    bool trans_y,
    float scale,
    std::string output_dtype) {
    paddle::DataType data_type;
    if (output_dtype == "bfloat16")
        data_type = paddle::DataType::BFLOAT16;
    else if (output_dtype ==  "float16")
        data_type = paddle::DataType::FLOAT16;
    else 
        PD_THROW(
                "fp8_fp8_half_gemm_fused only support bfloat16 and float16 output");
    return {data_type};
}

PD_BUILD_OP(cutlass_fp8_fp8_half_gemm_ptr_scale_fused)
    .Inputs({"x", "y", paddle::Optional("x_scale"), paddle::Optional("y_scale"), paddle::Optional("bias")})
    .Attrs({"transpose_x: bool",
            "transpose_y: bool",
            "scale: float",
            "output_dtype: std::string",
            "act: std::string"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(cutlass_fp8_fp8_half_gemm_ptr_scale))
    .SetInferShapeFn(PD_INFER_SHAPE(CutlassFp8Fp8HalfGemmPtrScaleFusedInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(CutlassFp8Fp8HalfGemmPtrScaleFusedInferDtype));
