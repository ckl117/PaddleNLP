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

#pragma once

#include "fp8_gemm_fused/fuse_gemm_noact_template_3x.h"


template<>
bool dispatch_fuse_gemm_noact_sm90<phi::dtype::float8_e4m3fn, phi::dtype::float16,
                                 cute::Shape<cute::_128, cute::_128, cute::_128>,
                                 cute::Shape<cute::_2, cute::_1, cute::_1>,
                                 cutlass::gemm::collective::KernelScheduleAuto,
                                 cutlass::epilogue::collective::EpilogueScheduleAuto
                                 >(GemmEpilogueAllParams);

template<>
bool dispatch_fuse_gemm_noact_sm90<phi::dtype::float8_e4m3fn, phi::dtype::float16,
                                 cute::Shape<cute::_64, cute::_128, cute::_128>,
                                 cute::Shape<cute::_2, cute::_1, cute::_1>,
                                 cutlass::gemm::collective::KernelScheduleAuto,
                                 cutlass::epilogue::collective::EpilogueScheduleAuto
                                 >(GemmEpilogueAllParams);

template<>
bool dispatch_fuse_gemm_noact_sm90<phi::dtype::float8_e4m3fn, phi::dtype::float16,
                                 cute::Shape<cute::_64, cute::_64, cute::_128>,
                                 cute::Shape<cute::_1, cute::_8, cute::_1>,
                                 cutlass::gemm::collective::KernelScheduleAuto,
                                 cutlass::epilogue::collective::EpilogueScheduleAuto
                                 >(GemmEpilogueAllParams);

template<>
bool dispatch_fuse_gemm_noact_sm90<phi::dtype::float8_e4m3fn, phi::dtype::float16,
                                 cute::Shape<cute::_128, cute::_128, cute::_128>,
                                 cute::Shape<cute::_2, cute::_1, cute::_1>,
                                 cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum,
                                 cutlass::epilogue::TmaWarpSpecialized
                                 >(GemmEpilogueAllParams);

template<>
bool dispatch_fuse_gemm_noact_sm90<phi::dtype::float8_e4m3fn, phi::dtype::float16,
                                 cute::Shape<cute::_64, cute::_128, cute::_128>,
                                 cute::Shape<cute::_2, cute::_1, cute::_1>,
                                 cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum,
                                 cutlass::epilogue::TmaWarpSpecialized
                                 >(GemmEpilogueAllParams);

template<>
bool dispatch_fuse_gemm_noact_sm90<phi::dtype::float8_e4m3fn, phi::dtype::float16,
                                 cute::Shape<cute::_64, cute::_64, cute::_128>,
                                 cute::Shape<cute::_1, cute::_8, cute::_1>,
                                 cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum,
                                 cutlass::epilogue::TmaWarpSpecialized
                                 >(GemmEpilogueAllParams);