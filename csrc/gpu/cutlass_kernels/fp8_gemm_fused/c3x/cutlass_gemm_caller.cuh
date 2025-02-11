#pragma once

// clang-format will break include orders
// clang-format off

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass_kernels/common.hpp"
// clang-format on

#include <climits>
#include <iostream>
inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

namespace pd::c3x {

static inline cute::Shape<int, int, int, int> get_problem_shape(
    paddle::Tensor const& a, paddle::Tensor const& b) {
  
//   int32_t batch_size = 1;
//   int batch_count = 1;
//   for (size_t i = 0; i < rank - 2; ++i) {
//     batch_count *= a.dims()[i];
//   }
  int rank = a.dims().size();
  int32_t m = a.dims()[rank - 2], n = b.dims()[rank - 2], k = a.dims()[rank - 1];
  return {m, n, k, 1};
}

template <typename GemmKernel>
bool cutlass_gemm_caller(const phi::GPUPlace &place,
                         cudaStream_t stream,
                         cute::Shape<int, int, int, int> prob_shape,
                         typename GemmKernel::MainloopArguments mainloop_args,
                         typename GemmKernel::EpilogueArguments epilogue_args) {
  typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                      prob_shape, mainloop_args, epilogue_args};

  // Launch the CUTLASS GEMM kernel.
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  cutlass::Status status = gemm_op.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::cout << "Gemm::can_implement() failed. " << cutlassGetStatusString(status) << std::endl;
    return false;
  }
  size_t workspace_size = GemmOp::get_workspace_size(args);
  phi::Allocator* allocator = paddle::GetAllocator(place);
  auto workspace = allocator->Allocate(workspace_size);

  status = gemm_op(args, workspace->ptr(), stream);
  if (status != cutlass::Status::kSuccess) {
    std::cout << "Gemm::run() failed." << cutlassGetStatusString(status) << std::endl;
    return false;
  }
  return true;
}

template <typename Gemm, typename... EpilogueArgs>
void cutlass_gemm_caller(paddle::Tensor& out, paddle::Tensor const& a,
                         paddle::Tensor const& b,
                         EpilogueArgs&&... epilogue_params) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;
  using GemmKernel = typename Gemm::GemmKernel;
  
  int rank = a.dims().size();
  int64_t lda = a.dims()[rank - 1];
  int64_t ldb = b.dims()[rank - 1];
  int64_t ldc = out.dims()[rank - 1];

  using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideC = typename Gemm::StrideC;

  StrideA a_stride{lda, cute::Int<1>{}, 0};
  StrideB b_stride{ldb, cute::Int<1>{}, 0};
  StrideC c_stride{ldc, cute::Int<1>{}, cute::Int<0>{}};

  typename GemmKernel::ProblemShape prob_shape = get_problem_shape(a, b);

  auto a_ptr = static_cast<ElementAB*>(const_cast<void*>(a.data()));
  auto b_ptr = static_cast<ElementAB*>(const_cast<void*>(b.data()));
  typename GemmKernel::MainloopArguments mainloop_args{a_ptr, a_stride, b_ptr,
                                                       b_stride};

  auto c_ptr = static_cast<ElementD*>(out.data());
  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(
          std::forward<EpilogueArgs>(epilogue_params)...),
      c_ptr, c_stride, c_ptr, c_stride};

  cutlass_gemm_caller<GemmKernel>(a.place(), a.stream(), prob_shape, mainloop_args,
                                  epilogue_args);
}

}  // namespace pd::c3x