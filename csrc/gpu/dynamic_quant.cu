// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "helper.h"
#include<string.h>

#ifdef PADDLE_WITH_HIP
constexpr int32_t WARP_SIZE = 64; 
constexpr int32_t HALF_WARP = 32; 
#else
constexpr int32_t WARP_SIZE = 32; 
constexpr int32_t HALF_WARP = 16; 
#endif

template <typename T>
__inline__ __device__ T WarpReduceAbsMax(T val, unsigned lane_mask) {
  #pragma unroll
  for (int mask = HALF_WARP; mask > 0; mask >>= 1){
#ifdef PADDLE_WITH_HIP
    val = max(val, static_cast<T>(__shfl_xor(static_cast<float>(val), mask, WARP_SIZE)));
#else
    val = max(val, __shfl_xor_sync(lane_mask, val, mask, WARP_SIZE));
#endif
  }
  return val;
}

template <typename InType, typename OutType>
__global__ void DynamicQuantKernel(const InType* input,
                                   const int64_t numel,
                                   const float quant_max_bound,
                                   OutType* output,
                                   float* out_scale_data) {

  int32_t lane_id = threadIdx.x % WARP_SIZE;
  int32_t warp_id = threadIdx.x / WARP_SIZE;

  __shared__ float smem[128/WARP_SIZE];

  const int block_idx = blockIdx.x;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) return;

  float in_val = static_cast<float>(input[idx]);
  float max_val = abs(in_val);
  
  max_val = WarpReduceAbsMax(max_val, 0xffffffff);

  if (lane_id == 0) {
    smem[warp_id] = max_val;
  }
  __syncthreads();
  max_val = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? smem[threadIdx.x] : 0.0f;
  max_val = WarpReduceAbsMax(max_val, 0xffffffff);
  
  if (threadIdx.x == 0) {
    smem[0] = max_val / quant_max_bound;
  }
  float scale = smem[0];
  out_scale_data[block_idx] = scale;
  output[idx] = static_cast<OutType>(in_val / scale);
}

template <paddle::DataType InType>
std::vector<paddle::Tensor> LaunchDynamicQuant(const paddle::Tensor& x,
                                                const int block_size,
                                                const float quant_max_bound) {
    
    typedef PDTraits<InType> in_traits;
    typedef typename in_traits::DataType InDataType;
    typedef typename in_traits::data_t in_data_t;

    paddle::Tensor out;
    paddle::Tensor scale_out;
    auto place = x.place();
    cudaStream_t stream = x.stream();

    int rank = x.dims().size();
    
    std::vector<int64_t> out_shape = x.shape();
    std::vector<int64_t> scale_shape = x.shape();
    if(!(fabs(quant_max_bound - 448.0f) < 0.000001)){
        PD_THROW("Only supported float8_e4m3fn quantization, please set quant_max_bound=448.");
    }

    out = paddle::empty(out_shape, paddle::DataType::FLOAT8_E4M3FN, place);

    // per tensor quant
    if(block_size <= 0){
        scale_out = paddle::empty({1}, paddle::DataType::FLOAT32, place);
    }else{
        scale_shape[rank - 1] = scale_shape[rank - 1] / block_size;
        scale_out = paddle::empty(scale_shape, paddle::DataType::FLOAT32, place);
    }

    int64_t numel = x.numel();
    int64_t block_per_grid = (numel + block_size - 1) / block_size;

    typedef PDTraits<paddle::DataType::FLOAT8_E4M3FN> out_traits;
    typedef typename out_traits::DataType OutDataType;
    typedef typename out_traits::data_t out_data_t;

    DynamicQuantKernel<InDataType, OutDataType><<<block_per_grid, block_size, 0, stream>>>(reinterpret_cast<const InDataType*>(x.data<in_data_t>()),
                            numel,
                            quant_max_bound,
                            reinterpret_cast<OutDataType*>(out.data<out_data_t>()),
                            reinterpret_cast<float*>(scale_out.data<float>()));
    return {out, scale_out};
    
}


std::vector<paddle::Tensor> DynamicQuant(const paddle::Tensor& x,
                                        const int block_size,
                                        const float quant_max_bound) {
    
    if(x.dtype() == paddle::DataType::FLOAT32){
        return LaunchDynamicQuant<paddle::DataType::FLOAT32>(x, block_size, quant_max_bound);
    }else if(x.dtype() == paddle::DataType::FLOAT16){
        return LaunchDynamicQuant<paddle::DataType::FLOAT16>(x, block_size, quant_max_bound);
    }else if(x.dtype() == paddle::DataType::BFLOAT16){
        return LaunchDynamicQuant<paddle::DataType::BFLOAT16>(x, block_size, quant_max_bound);
    }else{
        PD_THROW("Unsupported data type.");
    }
}

std::vector<std::vector<int64_t>> DynamicQuantInferShape(const std::vector<int64_t>& input_shape, const int block_size, const float quant_max_bound) {
    std::vector<int64_t> scale_shape = input_shape;
    int rank = input_shape.size();
    // per tensor quant
    if(block_size <= 0){
        return {input_shape, {1}};
    }
    // per block quant: 1xblock_size
    scale_shape[rank - 1] = input_shape[rank - 1] / block_size;
    return {input_shape, scale_shape};
}

std::vector<paddle::DataType> DynamicQuantInferDtype(const paddle::DataType& input_dtype, const int block_size, const float quant_max_bound) {
    
    if(!(fabs(quant_max_bound - 448.0f) < 0.000001)){
        PD_THROW("Only supported attr of quant_max_bound in ['448.0'].");
    }
    return {paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::FLOAT32};
}

PD_BUILD_OP(dynamic_quant)
    .Inputs({"x"})
    .Outputs({"output", "scale"})
    .Attrs({"block_size: int",
            "quant_max_bound: float"})
    .SetKernelFn(PD_KERNEL(DynamicQuant))
    .SetInferShapeFn(PD_INFER_SHAPE(DynamicQuantInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DynamicQuantInferDtype));