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

template <typename InType, typename OutType, int GroupSize>
__global__ void GroupQuantKernel(const InType* input,
                                   const int64_t numel,
                                   const float quant_max_bound,
                                   const float quant_min_bound,
                                   OutType* output,
                                   float* out_scale_data) {

  int32_t lane_id = threadIdx.x % WARP_SIZE;
  int32_t warp_id = threadIdx.x / WARP_SIZE;

  __shared__ float smem[GroupSize / WARP_SIZE];

  const int block_idx = blockIdx.x;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) return;

  float in_val = static_cast<float>(input[idx]);
  float abs_max_val = abs(in_val);
  
  abs_max_val = WarpReduceAbsMax(abs_max_val, 0xffffffff);

  if (lane_id == 0) {
    smem[warp_id] = abs_max_val;
  }
  __syncthreads();
  abs_max_val = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? smem[threadIdx.x] : 0.0f;
  abs_max_val = WarpReduceAbsMax(abs_max_val, 0xffffffff);
  
  if (threadIdx.x == 0) {
    smem[0] = max(abs_max_val, 0.000001) / quant_max_bound;
  }
  __syncthreads();
  float scale = smem[0];
  out_scale_data[block_idx] = scale;
  float quant_value = in_val / scale;
  quant_value = quant_value > quant_max_bound ? quant_max_bound : quant_value;
  quant_value = quant_value < quant_min_bound ? quant_min_bound : quant_value;
  output[idx] = static_cast<OutType>(quant_value);
}

template <paddle::DataType InType, paddle::DataType OutType>
std::vector<paddle::Tensor> LaunchGroupQuantKernel(const paddle::Tensor& x,
                                                   const int group_size,
                                                   const float quant_max_bound,
                                                   const float quant_min_bound) {
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

    out = paddle::empty(out_shape, OutType, place);

    scale_shape[rank - 1] = scale_shape[rank - 1] / group_size;
    scale_out = paddle::empty(scale_shape, paddle::DataType::FLOAT32, place);

    int64_t numel = x.numel();
    int64_t block_per_grid = (numel + group_size - 1) / group_size;

    typedef PDTraits<OutType> out_traits;
    typedef typename out_traits::DataType OutDataType;
    typedef typename out_traits::data_t out_data_t;
    
    if(group_size == 128){
        GroupQuantKernel<InDataType, OutDataType, 128><<<block_per_grid, group_size, 0, stream>>>(reinterpret_cast<const InDataType*>(x.data<in_data_t>()),
                            numel,
                            quant_max_bound,
                            quant_min_bound,
                            reinterpret_cast<OutDataType*>(out.data<out_data_t>()),
                            reinterpret_cast<float*>(scale_out.data<float>()));
    }else{
        PD_THROW("group_quant's group_size only support 128.");
    }
    
    return {out, scale_out};
}
template <paddle::DataType InType>
std::vector<paddle::Tensor> LaunchGroupQuant(const paddle::Tensor& x,
                                             const int group_size,
                                             const float quant_max_bound,
                                             const float quant_min_bound) {

    if(fabs(quant_max_bound - 448.0f) < 0.000001){
        LaunchGroupQuantKernel<InType, paddle::DataType::FLOAT8_E4M3FN>(x, group_size, quant_max_bound, quant_min_bound);
    }else{
        PD_THROW("Only supported float8_e4m3fn quantization, please set quant_max_bound=448, quant_min_bound=-448.");
    }
    
}


std::vector<paddle::Tensor> GroupQuant(const paddle::Tensor& x,
                                        const int group_size,
                                        const float quant_max_bound,
                                        const float quant_min_bound) {
    if(x.dtype() == paddle::DataType::FLOAT32){
        return LaunchGroupQuant<paddle::DataType::FLOAT32>(x, group_size, quant_max_bound, quant_min_bound);
    }else if(x.dtype() == paddle::DataType::FLOAT16){
        return LaunchGroupQuant<paddle::DataType::FLOAT16>(x, group_size, quant_max_bound, quant_min_bound);
    }else if(x.dtype() == paddle::DataType::BFLOAT16){
        return LaunchGroupQuant<paddle::DataType::BFLOAT16>(x, group_size, quant_max_bound, quant_min_bound);
    }else{
        PD_THROW("Unsupported data type.");
    }
}

std::vector<std::vector<int64_t>> GroupQuantInferShape(const std::vector<int64_t>& input_shape, const int group_size, const float quant_max_bound) {
    std::vector<int64_t> scale_shape = input_shape;
    int rank = input_shape.size();
    // per tensor quant
    if(group_size <= 0){
        return {input_shape, {1}};
    }
    // per block quant: 1xgroup_size
    scale_shape[rank - 1] = input_shape[rank - 1] / group_size;
    return {input_shape, scale_shape};
}

std::vector<paddle::DataType> GroupQuantInferDtype(const paddle::DataType& input_dtype, const int group_size, const float quant_max_bound) {
    
    if(!(fabs(quant_max_bound - 448.0f) < 0.000001)){
        PD_THROW("Only supported attr of quant_max_bound in ['448.0'].");
    }
    return {paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::FLOAT32};
}

PD_BUILD_OP(group_quant)
    .Inputs({"x"})
    .Outputs({"output", "scale"})
    .Attrs({"group_size: int",
            "quant_max_bound: float",
            "quant_min_bound: float"})
    .SetKernelFn(PD_KERNEL(GroupQuant))
    .SetInferShapeFn(PD_INFER_SHAPE(GroupQuantInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GroupQuantInferDtype));