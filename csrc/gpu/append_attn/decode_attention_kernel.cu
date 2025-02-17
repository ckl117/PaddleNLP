#include "decode_attention_func.cuh"

#define CHECK(call)                                                             \
do                                                                              \
{                                                                               \
    const cudaError_t error_code = call;                                        \
    if (error_code != cudaSuccess)                                              \
    {                                                                           \
        printf("CUDA Error:\n");                                                \
        printf("     File:      %s\n", __FILE__);                               \
        printf("     Line       %d:\n", __LINE__);                              \
        printf("     Error code:%d\n", error_code);                             \
        printf("     Error text:%s\n", cudaGetErrorString(error_code));         \
        exit(1);                                                                \
    }                                                                           \
}while(0)
// #define DEBUG_DEC_ATTN
template <typename T, typename OutT, int vec_size, uint32_t bdy, uint32_t HEAD_DIM>
__global__ void merge_varlen_multi_chunks_v2_kernel(const T * __restrict__ multi_out, // [bsz, num_chunks, num_heads, head_dim]
                                                    const T * __restrict__ multi_m, // [bsz, num_chunks, num_heads]
                                                    const T * __restrict__ multi_d, // [bsz, num_chunks, num_heads]
                                                    const int * __restrict__ seq_lens_q,
                                                    const int * __restrict__ seq_lens_kv,
                                                    const int * __restrict__ cum_offsets,
                                                    const T * __restrict__ shift_bias, // [q_num_heads * HEAD_DIM]
                                                    const T * __restrict__ smooth_weight, // [q_num_heads * HEAD_DIM]
                                                    OutT * __restrict__ out, // [token_num, num_heads, head_dim]
                                                    const float in_scale,
                                                    const int num_chunks,
                                                    const int chunk_size,
                                                    const int max_seq_len,
                                                    const int num_heads,
                                                    const int head_dim) {
  const int vid = threadIdx.x, ty = threadIdx.y;
  const int qid = blockIdx.x, hid = blockIdx.y;
  const int seq_len_q = seq_lens_q[qid];
  if (seq_len_q == 0) return;
  int seq_len_kv = seq_lens_kv[qid];
  if (seq_len_kv == 0) return;
  seq_len_kv += seq_len_q;
  const int num_chunks_this_seq = div_up(seq_len_kv, chunk_size);
  if (num_chunks_this_seq == 1) {
    return;
  }
  __shared__ T smem[bdy * HEAD_DIM];
  __shared__ T md_smem[bdy * 2];

  const int start_token_ids = qid * max_seq_len - __ldg(&cum_offsets[qid]);
  using LoadT = AlignedVector<T, vec_size>;
  LoadT load_vec;
  LoadT res_vec;
  if constexpr (std::is_same<T, half>::value) {
#pragma unroll
    for (int i = 0; i < vec_size / 2; ++i) {
      *((half2*)(&res_vec) + i) = make_half2(0, 0);
    }
  } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
#pragma unroll
    for (int i = 0; i < vec_size / 2; ++i) {
      *((nv_bfloat162*)(&res_vec) + i) = make_bfloat162(0, 0);
    }
  }
  T m;
  T d = 1.f;
  if constexpr (std::is_same<T, half>::value) {
    m = __float2half(-5e4f);
  } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
    m = __float2bfloat16(-3.38953e38f);
  }
  // merge per ty
#pragma unroll 2
  for (int i = ty; i < num_chunks_this_seq; i += bdy) {
    uint32_t offset = (qid * num_chunks + i) * num_heads + hid;
    T m_prev = m;
    T d_prev = d;
    const T m_now = multi_m[offset];
    const T d_now = multi_d[offset];
    m = m_prev > m_now ? m_prev : m_now;
    offset = (qid * num_chunks * num_heads + i * num_heads + hid) * head_dim + vid * vec_size;
    Load<T, vec_size>(&multi_out[offset], &load_vec);
    const T scale1 = hexp(m_prev - m), scale2 = hexp(m_now - m);
    // const T scale1 = __expf(m_prev - m), scale2 = __expf(m_now - m);
    d = d * scale1 + d_now * scale2;
#pragma once
    for (int j = 0; j < vec_size; j++) {
      res_vec[j] = res_vec[j] * scale1 + load_vec[j] * scale2;
    }
  }
  // store ty res
  Store<T, vec_size>(res_vec, &smem[ty * head_dim + vid * vec_size]);
  md_smem[2 * ty] = m;
  md_smem[2 * ty + 1] = d;
  __syncthreads();

  // merge bdy
  softmax_state_t<vec_size, T> st{};
#pragma once
  for (int i = 0; i < bdy; i++) {
    Load<T, vec_size>(&smem[i * head_dim + vid * vec_size], &load_vec);
    const T m_tmp = md_smem[2 * i], d_tmp = md_smem[2 * i + 1];
    st.merge(load_vec, m_tmp, d_tmp);
  }
  st.normalize();

  AlignedVector<OutT, vec_size> out_vec;
  if (in_scale > 0) {
    const uint32_t shift_smooth_offset = hid * head_dim + vid * vec_size;
    AlignedVector<T, vec_size> shift_bias_vec;
    AlignedVector<T, vec_size> smooth_weight_vec;
    
    Load<T, vec_size>(shift_bias + shift_smooth_offset, &shift_bias_vec);
    Load<T, vec_size>(smooth_weight + shift_smooth_offset, &smooth_weight_vec);
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      float quant_value  = 127.0f * static_cast<float>((st.o[i] + shift_bias_vec[i]) * smooth_weight_vec[i]) * in_scale;
      quant_value = rintf(quant_value);
      quant_value = quant_value > 127.0f ? 127.0f : quant_value;
      quant_value = quant_value < -127.0f ? -127.0f : quant_value;
      out_vec[i] = static_cast<OutT>(quant_value);
    }
  } else {
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      out_vec[i] = static_cast<OutT>(st.o[i]);
    }
  }
  Store<OutT, vec_size>(out_vec, &out[(start_token_ids * num_heads + hid) * head_dim + vid * vec_size]);
}

template <typename T, typename OutT, int vec_size>
__global__ void merge_varlen_multi_chunks_kernel(const T * __restrict__ multi_out, // [bsz, num_chunks, num_heads, head_dim]
                                                const T * __restrict__ multi_m, // [bsz, num_chunks, num_heads]
                                                const T * __restrict__ multi_d, // [bsz, num_chunks, num_heads]
                                                const int * __restrict__ seq_lens_q,
                                                const int * __restrict__ seq_lens_kv,
                                                const int * __restrict__ cum_offsets,
                                                const T * __restrict__ shift_bias, // [q_num_heads * HEAD_DIM]
                                                const T * __restrict__ smooth_weight, // [q_num_heads * HEAD_DIM]
                                                OutT * __restrict__ out, // [token_num, num_heads, head_dim]
                                                const float in_scale,
                                                const int num_chunks,
                                                const int chunk_size,
                                                const int max_seq_len,
                                                const int num_heads,
                                                const int head_dim) {
  const int vid = threadIdx.x, hid = threadIdx.y;
  const int qid = blockIdx.x;
  const int seq_len_q = seq_lens_q[qid];
  if (seq_len_q == 0) return;
  int seq_len_kv = seq_lens_kv[qid];
  if (seq_len_kv == 0) return;
  seq_len_kv += seq_len_q;
  const int num_chunks_this_seq = div_up(seq_len_kv, chunk_size);
  if (num_chunks_this_seq == 1) {
    return;
  }
  const int start_token_ids = qid * max_seq_len - __ldg(&cum_offsets[qid]);
  using LoadT = AlignedVector<T, vec_size>;
  LoadT load_vec;
  LoadT res_vec;
  if constexpr (std::is_same<T, half>::value) {
#pragma unroll
    for (int i = 0; i < vec_size / 2; ++i) {
      *((half2*)(&res_vec) + i) = make_half2(0, 0);
    }
  } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
#pragma unroll
    for (int i = 0; i < vec_size / 2; ++i) {
      *((nv_bfloat162*)(&res_vec) + i) = make_bfloat162(0, 0);
    }
  }
  T m;
  T d = 1.f;
  if constexpr (std::is_same<T, half>::value) {
    m = __float2half(-5e4f);
  } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
    m = __float2bfloat16(-3.38953e38f);
  }
#pragma unroll 2
  for (int i = 0; i < num_chunks_this_seq; ++i) {
    uint32_t offset = (qid * num_chunks + i) * num_heads + hid;
    T m_prev = m;
    T d_prev = d;
    const T m_now = multi_m[offset];
    const T d_now = multi_d[offset];
    m = m_prev > m_now ? m_prev : m_now;
    offset = (qid * num_chunks * num_heads + i * num_heads + hid) * head_dim + vid * vec_size;
    Load<T, vec_size>(&multi_out[offset], &load_vec);
    const T scale1 = hexp(m_prev - m), scale2 = hexp(m_now - m);
    // const T scale1 = __expf(m_prev - m), scale2 = __expf(m_now - m);
    d = d * scale1 + d_now * scale2;
#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      res_vec[j] = res_vec[j] * scale1 + load_vec[j] * scale2;
    }
  }
#pragma unroll 
  for (int j = 0; j < vec_size; j++) {
    res_vec[j] /= d;
  }

  AlignedVector<OutT, vec_size> out_vec;
  if (in_scale > 0) {
    const uint32_t shift_smooth_offset = hid * head_dim + vid * vec_size;
    AlignedVector<T, vec_size> shift_bias_vec;
    AlignedVector<T, vec_size> smooth_weight_vec;
    Load<T, vec_size>(shift_bias + shift_smooth_offset, &shift_bias_vec);
    Load<T, vec_size>(smooth_weight + shift_smooth_offset, &smooth_weight_vec);
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      float quant_value  = 127.0f * static_cast<float>((res_vec[i] + shift_bias_vec[i]) * smooth_weight_vec[i]) * in_scale;
      quant_value = rintf(quant_value);
      quant_value = quant_value > 127.0f ? 127.0f : quant_value;
      quant_value = quant_value < -127.0f ? -127.0f : quant_value;
      out_vec[i] = static_cast<OutT>(quant_value);
    }
  } else {
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      out_vec[i] = static_cast<OutT>(res_vec[i]);
    }
  }
  Store<OutT, vec_size>(out_vec, &out[(start_token_ids * num_heads + hid) * head_dim + vid * vec_size]);
}

template <bool partition_kv, typename T, typename OutT, typename CacheT, uint32_t NUM_STAGES, uint32_t DEAL_EACH_TIME, uint32_t GROUP_SIZE, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_V, 
          uint32_t BLOCK_SIZE, uint32_t VEC_SIZE, uint32_t CACHE_VEC_SIZE, uint32_t bdx, uint32_t bdy, uint32_t bdxc, PosEncMode pos_enc_mode, CacheType cache_type>
__global__ void multi_query_decode_attention_kernel(T * __restrict__ q, // [token_num, num_heads, head_dim]
                                                    CacheT * __restrict__ cache_k, // [max_block_num, num_heads, block_size, head_dim]
                                                    CacheT * __restrict__ cache_v,
                                                    const T * __restrict__ shift_bias, // [q_num_heads * HEAD_DIM]
                                                    const T * __restrict__ smooth_weight, // [q_num_heads * HEAD_DIM]
                                                    const int * __restrict__ seq_lens_q,
                                                    const int * __restrict__ seq_lens_kv,
                                                    const int * __restrict__ cum_offsets,
                                                    const int * __restrict__ block_table, // [bsz, block_num_per_seq]
                                                    const int max_seq_len,
                                                    const int max_dec_len,
                                                    const int max_block_num_per_seq,
                                                    const float scale,
                                                    const float in_scale,
                                                    const uint32_t chunk_size,
                                                    T * __restrict__ tmp_workspace, // [batch_size, num_chunks, num_heads, head_dim]
                                                    T * __restrict__ tmp_m, // [batch_size, num_chunks, num_heads]
                                                    T * __restrict__ tmp_d, // [batch_size, num_chunks, num_heads]
                                                    OutT * __restrict__ out) {
  const uint32_t bidx = blockIdx.x, kv_head_idx = blockIdx.z;
  const uint32_t bid = bidx / GROUP_SIZE, gid = bidx % GROUP_SIZE;
  const uint32_t tidx = threadIdx.x, tidy = threadIdx.y, tidz = threadIdx.z;
  const uint32_t vid = tidy * bdx + tidx;
  constexpr uint32_t num_vec_per_head_qk = HEAD_DIM_QK / VEC_SIZE;
  constexpr uint32_t num_vec_per_head_v = HEAD_DIM_QK / VEC_SIZE;
  if (vid >= num_vec_per_head_qk) return;
  
  const uint32_t kv_bid = vid / bdxc, kv_vid = vid % bdxc;
  const uint32_t q_head_idx = kv_head_idx * GROUP_SIZE + gid;
  const uint32_t kv_num_heads = gridDim.z;
  const uint32_t q_num_heads = kv_num_heads * GROUP_SIZE;
  
  constexpr uint32_t HALF_VEC_SIZE = VEC_SIZE / 2;
  constexpr uint32_t HALF_HEAD_DIM_QK = HEAD_DIM_QK / 2;
  constexpr uint32_t HALF_HEAD_DIM_V = HEAD_DIM_V / 2;
  const int *block_table_now = block_table + bid * max_block_num_per_seq;
  
  const uint32_t num_chunks = gridDim.y;
  const uint32_t chunk_id = blockIdx.y;
  const uint32_t q_len = seq_lens_q[bid];
  if (q_len <= 0) {
    return;
  }
  uint32_t kv_len = seq_lens_kv[bid]; // !!!!!!!!
  if (kv_len <= 0) {
     return;
  }
  kv_len += q_len;
  const uint32_t num_chunk_this_seq = div_up(kv_len, chunk_size);
  // const uint32_t q_start_idx = bid * max_seq_len - __ldg(&cum_offsets[bid]);
  const uint32_t q_start_idx = bid;
  const uint32_t q_write_idx = bid * max_seq_len - __ldg(&cum_offsets[bid]);
  if (chunk_id >= num_chunk_this_seq) {
    return;
  }

  const uint32_t chunk_start = partition_kv ? chunk_id * chunk_size : 0;
  const uint32_t chunk_end = partition_kv ? min(kv_len, chunk_start + chunk_size) : kv_len;
  const uint32_t chunk_len = chunk_end - chunk_start;
#ifdef DEBUG_DEC_ATTN
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    printf("chunk_id: %d, q_len: %d, kv_len: %d, chunk_start: %d, chunk_end: %d, chunk_len: %d, chunk_size: %d\n", 
           (int)chunk_id, (int)q_len, (int)kv_len, (int)chunk_start, (int)chunk_end, (int)chunk_len, (int)chunk_size);
  }
  __syncthreads();
#endif

  extern __shared__ uint8_t smem[];
#ifdef DEBUG_DEC_ATTN
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      printf("q_now start\n");
    }
    __syncthreads();
#endif
  const T *q_now = q + (q_start_idx * q_num_heads + kv_head_idx * GROUP_SIZE + gid) * HEAD_DIM_QK;
#ifdef DEBUG_DEC_ATTN
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    printf("q_smem start\n");
  }
  __syncthreads();
#endif
  T *q_smem = reinterpret_cast<T*>(smem); // [HEAD_DIM_QK * sizeof(T)]

  using QVec = AlignedVector<T, VEC_SIZE>;
  QVec q_vec;
  QVec freq_vec;
  const uint32_t q_offset = (q_start_idx * q_num_heads + q_head_idx) * HEAD_DIM_QK;

  Load<T, VEC_SIZE>(q_smem + vid * VEC_SIZE, &q_vec);

#pragma unroll
  for (int i = 0; i < VEC_SIZE; i++) {
    q_vec[i] *= scale;
  }

#ifdef DEBUG_DEC_ATTN
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      printf("qk_tmp end\n");
    }
    __syncthreads();
#endif
#pragma unroll
  for(int local_id = vid; local_id < HEAD_DIM_QK; local_id += num_vec_per_head_qk) {
    q_smem[local_id] = q_now[local_id];
  }
  __syncthreads();

  CacheT *kv_smem = reinterpret_cast<CacheT*>(smem + HEAD_DIM_QK * sizeof(T)); // [NUM_STAGES * DEAL_EACH_TIME * HEAD_DIM_QK]
#ifdef DEBUG_DEC_ATTN
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    printf("md_smem start\n");
  }
  __syncthreads();
#endif
  T *qk_tmp = reinterpret_cast<T*>(smem);
  // T *md_smem = reinterpret_cast<T*>(smem + HEAD_DIM_QK * sizeof(T) + NUM_STAGES * DEAL_EACH_TIME * sizeof(CacheT));

  uint32_t stage_idx = 0;  
  constexpr int loop_times = DEAL_EACH_TIME ;
#ifdef DEBUG_DEC_ATTN
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    printf("load start\n");
  }
  __syncthreads();
#endif
#pragma unroll
  for (int i = 0; i < NUM_STAGES; ++i) {
#pragma unroll
    for (int j = 0; j < loop_times; ++j) {
      const uint32_t k_seq_offset = i * DEAL_EACH_TIME + j + kv_bid;
      const uint32_t k_seq_id = chunk_start + k_seq_offset;
      produce_kv<SharedMemFillMode::kNoFill, cache_type, HEAD_DIM_QK, HEAD_DIM_V, VEC_SIZE, HALF_VEC_SIZE, BLOCK_SIZE, bdxc, CACHE_VEC_SIZE>(
        kv_smem,
        cache_k,
        block_table_now,
        k_seq_id,
        k_seq_offset,
        kv_head_idx,
        kv_num_heads,
        kv_vid,
        chunk_start,
        chunk_end
      );
    }
    commit_group();
#ifdef DEBUG_DEC_ATTN
    wait_group<0>();
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      printf("load_k_done\n");
    }
    __syncthreads();
#endif
//     if (kv_vid < num_vec_per_head_v) {
// #pragma unroll
//       for (int j = 0; j < loop_times; ++j) {
//         const uint32_t v_seq_offset = i * DEAL_EACH_TIME + j + kv_bid;
//         const uint32_t v_seq_id = chunk_start + v_seq_offset;
        
//         produce_v<SharedMemFillMode::kFillZero, cache_type, HEAD_DIM_QK, HEAD_DIM_V, VEC_SIZE, HALF_VEC_SIZE, BLOCK_SIZE, bdxc, CACHE_VEC_SIZE>(
//           v_smem,
//           cache_v,
//           block_table_now,
//           v_seq_id,
//           v_seq_offset,
//           kv_head_idx,
//           kv_num_heads,
//           kv_vid,
//           chunk_start,
//           chunk_end
//         );
//       }
//       commit_group();
//     }
    stage_idx = (stage_idx + 1) % NUM_STAGES;
  }

  softmax_state_t<VEC_SIZE, T> st;
  T s[DEAL_EACH_TIME];
  
  const uint32_t num_iters = div_up(chunk_len, DEAL_EACH_TIME);
  for (int iter = 0; iter < num_iters; ++iter) {
    wait_group<NUM_STAGES - 1>();
    __syncthreads();
    // compute qk
    compute_qk<VEC_SIZE, HALF_VEC_SIZE, bdx, bdy, DEAL_EACH_TIME, HEAD_DIM_QK, HALF_HEAD_DIM_QK, pos_enc_mode, cache_type>(
      kv_smem,
      q_vec,
      freq_vec,
      chunk_start + iter * DEAL_EACH_TIME,
      stage_idx,
      iter * DEAL_EACH_TIME,
      chunk_len,
      vid,
      qk_tmp,
      s,
      st
    );
    __syncthreads();
#ifdef DEBUG_DEC_ATTN
    wait_group<0>();
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      printf("compute_qk_done\n");
    }
    __syncthreads();
#endif
    // compute sv
    if (kv_vid < num_vec_per_head_v) {
      compute_sv<VEC_SIZE, HALF_VEC_SIZE, DEAL_EACH_TIME, HEAD_DIM_V, HALF_HEAD_DIM_V, cache_type>(
        s,
        kv_smem,
        stage_idx,
        st
      );
      __syncthreads();
    }
#ifdef DEBUG_DEC_ATTN
    wait_group<0>();
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      printf("compute_sv done\n");
    }
    __syncthreads();
#endif

#pragma unroll
    for (int j = 0; j < loop_times; ++j) {
      const uint32_t k_seq_offset = j + kv_bid;
      produce_kv<SharedMemFillMode::kNoFill, cache_type, HEAD_DIM_QK, HEAD_DIM_V, VEC_SIZE, HALF_VEC_SIZE, BLOCK_SIZE, bdxc, CACHE_VEC_SIZE>(
        kv_smem,
        cache_k,
        block_table_now,
        chunk_start + k_seq_offset + (iter + NUM_STAGES) * DEAL_EACH_TIME,
        stage_idx * DEAL_EACH_TIME + k_seq_offset,
        kv_head_idx,
        kv_num_heads,
        kv_vid,
        chunk_start,
        chunk_end
      );
    }
    commit_group();
    // load v next tile
// #pragma unroll
      // for (int j = 0; j < loop_times; ++j) {
      //   const uint32_t v_seq_offset = j + kv_bid;;
      //   produce_v<SharedMemFillMode::kFillZero, cache_type, HEAD_DIM_QK, HEAD_DIM_V, VEC_SIZE, HALF_VEC_SIZE, BLOCK_SIZE, bdxc, CACHE_VEC_SIZE>(
      //     v_smem,
      //     cache_v,
      //     block_table_now,
      //     chunk_start + v_seq_offset + (iter + NUM_STAGES) * DEAL_EACH_TIME,
      //     stage_idx * DEAL_EACH_TIME + v_seq_offset,
      //     kv_head_idx,
      //     kv_num_heads,
      //     kv_vid,
      //     chunk_start,
      //     chunk_end
      //   );
      // }
      // commit_group();
    stage_idx = (stage_idx + 1) % NUM_STAGES;
#ifdef DEBUG_DEC_ATTN
    wait_group<0>();
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      printf("iter:%d, num_iters:%d, compute_sv done\n", iter, num_iters);
    }
    __syncthreads();
#endif
  }
  wait_group<0>();
  __syncthreads();

  // merge bdz
  // merge_res_per_block<VEC_SIZE, HEAD_DIM_V, bdy, bdz>(
  //   st,
  //   reinterpret_cast<T*>(smem), 
  //   md_smem
  // );
#ifdef DEBUG_DEC_ATTN
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      printf(" iter end ");
    }
    __syncthreads();
#endif
  // normize if not partition_kv
  if (kv_vid < num_vec_per_head_v) {
    if (!partition_kv || num_chunk_this_seq == 1) {
      st.normalize();
    }
#ifdef DEBUG_DEC_ATTN
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      printf(" normalize end ");
    }
    __syncthreads();
#endif
    if (partition_kv && num_chunk_this_seq > 1) {
      const uint32_t head_idx = (bid * num_chunks + chunk_id) * q_num_heads + q_head_idx;
      Store<T, VEC_SIZE>(st.o, tmp_workspace + head_idx * HEAD_DIM_V + vid * VEC_SIZE);

      tmp_m[head_idx] = st.m;
      tmp_d[head_idx] = st.d;
#ifdef DEBUG_DEC_ATTN
      __syncthreads();
      if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf("partition_kv store end ");
      }
      __syncthreads();
#endif
    } else {
      AlignedVector<OutT, VEC_SIZE> out_vec;
      if (in_scale > 0) {
        const uint32_t shift_smooth_offset = q_head_idx * HEAD_DIM_V + vid * VEC_SIZE;
        AlignedVector<T, VEC_SIZE> shift_bias_vec;
        AlignedVector<T, VEC_SIZE> smooth_weight_vec;
        
        Load<T, VEC_SIZE>(shift_bias + shift_smooth_offset, &shift_bias_vec);
        Load<T, VEC_SIZE>(smooth_weight + shift_smooth_offset, &smooth_weight_vec);
  #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          float quant_value  = 127.0f * static_cast<float>((st.o[i] + shift_bias_vec[i]) * smooth_weight_vec[i]) * in_scale;

          quant_value = rintf(quant_value);
          quant_value = quant_value > 127.0f ? 127.0f : quant_value;
          quant_value = quant_value < -127.0f ? -127.0f : quant_value;
          out_vec[i] = static_cast<OutT>(quant_value);
        }
      } else {
        for (int i = 0; i < VEC_SIZE; ++i) {
          out_vec[i] = static_cast<OutT>(st.o[i]);
        }
      }
#ifdef DEBUG_DEC_ATTN
      __syncthreads();
      if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf(" no partition stroe start ");
      }
      __syncthreads();
#endif
      Store<OutT, VEC_SIZE>(out_vec, out + (q_write_idx * q_num_heads + q_head_idx) * HEAD_DIM_V + vid * VEC_SIZE);
#ifdef DEBUG_DEC_ATTN
      __syncthreads();
      if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf(" no partition stroe end ");
      }
      __syncthreads();
#endif
    }
  }
#ifdef DEBUG_DEC_ATTN
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    printf(" kernel end ");
  }
  __syncthreads();
#endif
}


template <typename T, uint32_t GROUP_SIZE, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_V, uint32_t BLOCK_SIZE, bool CAUSAL, uint32_t NUM_STAGE, CacheType cache_type, uint32_t cache_bytes, uint32_t DEAL_EACH_TIME, uint32_t NUM_THREADS, PosEncMode pos_enc_mode = PosEncMode::kNonePos>
void MultiQueryDecoderAttention(
  const AppendAttnMetaData& meta_data,
  cudaStream_t &stream,
  const paddle::Tensor &q,
  const paddle::Tensor &cache_k, // [max_block_num, num_kv_heads, block_size, head_dim]
  const paddle::Tensor &cache_v, // [num_kv_heads, head_dim]
  const paddle::optional<paddle::Tensor>& attn_mask,
  const paddle::optional<paddle::Tensor>& shift_bias,
  const paddle::optional<paddle::Tensor>& smooth_weight,
  const paddle::Tensor &seq_lens_q,
  const paddle::Tensor &seq_lens_kv,
  const paddle::Tensor &padding_offsets,
  const paddle::Tensor &cum_offsets,
  const paddle::Tensor &block_table,
  const int max_seq_len,
  const int max_dec_len,
  const float rope_scale,
  const float rope_theta,
  const float softmax_scale,
  const float in_scale,
  paddle::Tensor *out) {
  using NV_TYPE = typename cascade_attn_type_traits<T>::type;
  using CACHE_TYPE = typename cache_type_traits<T, cache_type>::type;
  using NV_CACHE_TYPE = typename cascade_attn_type_traits<CACHE_TYPE>::type;
  static_assert((cache_type == CacheType::CacheT && std::is_same<NV_CACHE_TYPE, NV_TYPE>::value) || 
                ((cache_type == CacheType::CacheInt4CwZp || cache_type == CacheType::CacheInt8Hw) && std::is_same<NV_CACHE_TYPE, uint8_t>::value));


  auto num_heads = meta_data.q_num_heads;
  auto kv_num_heads = meta_data.kv_num_heads;
  auto token_num = meta_data.token_nums;
  auto bsz = meta_data.batch_size;
  auto max_block_num_per_seq = meta_data.max_blocks_per_seq;
  // const float scale = 1.f / sqrt(HEAD_DIM);
  // std::cout << "bsz: " << bsz << ", token_num: " << token_num << ", num_heads: " << num_heads << ", kv_num_heads: " << kv_num_heads << ", group_size: " << GROUP_SIZE << std::endl;
  // dev_ctx.template Alloc<T>(out);

  constexpr int num_stages = NUM_STAGE;

  constexpr int vec_size = 16 / sizeof(T); // 8 16 32
  constexpr int cache_vec_size = 128 / cache_bytes; // 8 16 32
  constexpr int blockxc = HEAD_DIM_QK / cache_vec_size;
  constexpr int num_vec_per_head = HEAD_DIM_QK / vec_size;
  constexpr int blockx = num_vec_per_head < 32 ? num_vec_per_head : 32;
  // constexpr int blockx = 32;

  constexpr int blocky = (num_vec_per_head + blockx - 1) / blockx;
  const int gridx = GROUP_SIZE * bsz;
  // static_assert(blockx <= 32);
  
  constexpr int num_threads = blockx * blocky;
  // std::cout << "blockx: " << blockx << ", blocky: " << blocky << ", << ", blockxc: " << blockxc << ", DEAL_EACH_TIME: " << DEAL_EACH_TIME << std::endl;
  
  auto splitkv_kernel = multi_query_decode_attention_kernel<true, NV_TYPE, NV_TYPE, NV_CACHE_TYPE, num_stages, DEAL_EACH_TIME, GROUP_SIZE, HEAD_DIM_QK, HEAD_DIM_V,
                                                                        BLOCK_SIZE, vec_size, cache_vec_size, blockx, blocky, blockxc, pos_enc_mode, cache_type>;
  uint32_t cache_smem_bytes = 0;
  
  const T *shift_bias_ptr = shift_bias ? shift_bias.get().data<T>() : nullptr;
  const T *smooth_weight_ptr = smooth_weight ? smooth_weight.get().data<T>() : nullptr;
  cache_smem_bytes = num_stages * DEAL_EACH_TIME * HEAD_DIM_QK * sizeof(CACHE_TYPE);
  
  const uint32_t chunk_size = get_max_partition_size(bsz);
  const int num_chunks = div_up(max_dec_len, chunk_size);
  // size_t smem_size = blocky * sizeof(T) * 2 + div_up(max_block_num_per_seq, 4) * 4 * sizeof(int)
  //                    + cache_smem_bytes + blocky * HEAD_DIM * sizeof(T);
  size_t smem_size = cache_smem_bytes + HEAD_DIM_QK * sizeof(T);
  // size_t smem_size = max(size_t(cache_smem_bytes), HEAD_DIM_QK * sizeof(T)) + blocky * sizeof(T) * 2 + blocky * sizeof(T) * 2 + blocky * sizeof(T) * 2;

  // std::cout << "smem_size: " << div_up(smem_size, 1024) << "KB";
  
  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(
      splitkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }
  const int dev_id = 0;
  int sm_count;
  int act_blocks_per_sm;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &act_blocks_per_sm, splitkv_kernel, num_threads, smem_size);
  assert(act_blocks_per_sm > 1);
  
  const int num_blocks_per_wave = sm_count * act_blocks_per_sm;
  const int num_blocks_need = gridx * num_chunks * kv_num_heads;
  const int max_num_chunks = div_up(num_blocks_per_wave, num_blocks_need);
  const float ratio = static_cast<float>(num_blocks_need) / static_cast<float>(num_blocks_per_wave);

  
  // std::cout << "num_blocks_per_wave: " << num_blocks_per_wave << ", num_blocks_need: " << num_blocks_need << ", max_num_chunks: " << max_num_chunks << ", ratio: " << ratio << std::endl;
  // std::cout << "num_chunks: " << num_chunks;
  
  dim3 grids(gridx, num_chunks, kv_num_heads);
  dim3 blocks(blockx, blocky);
  // std::cout << "grids: " << grids.x << ", " << grids.y << ", " << grids.z
  //   << " blocks: " << blocks.x << ", " << blocks.y
  //   << " smem_size: " << div_up(smem_size, 1024) << "KB" << " HEAD_DIM_QK:" << HEAD_DIM_QK << " HEAD_DIM_V:" << HEAD_DIM_V
  //   << " num_blocks_need: " << num_blocks_need << ", num_blocks_per_wave: " << num_blocks_per_wave << std::endl;
  if (num_chunks <= 1) {
    // std::cout << "not split kv";
    auto no_splitkv_kernel = multi_query_decode_attention_kernel<false, NV_TYPE, NV_TYPE, NV_CACHE_TYPE, num_stages, DEAL_EACH_TIME, GROUP_SIZE, HEAD_DIM_QK, HEAD_DIM_V, BLOCK_SIZE, vec_size, 
                                                                             cache_vec_size, blockx, blocky, blockxc, pos_enc_mode, cache_type>;
    if (smem_size >= 48 * 1024) {
      cudaFuncSetAttribute(
        no_splitkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    no_splitkv_kernel<<<grids, blocks, smem_size, stream>>>(
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(q.data<T>())),
      reinterpret_cast<NV_CACHE_TYPE*>(const_cast<CACHE_TYPE*>(cache_k.data<CACHE_TYPE>())),
      reinterpret_cast<NV_CACHE_TYPE*>(const_cast<CACHE_TYPE*>(cache_v.data<CACHE_TYPE>())),
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(shift_bias_ptr)),
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(smooth_weight_ptr)),
      seq_lens_q.data<int>(),
      seq_lens_kv.data<int>(),
      cum_offsets.data<int>(),
      block_table.data<int>(),
      max_seq_len,
      max_dec_len,
      max_block_num_per_seq,
      softmax_scale,
      in_scale,
      chunk_size,
      nullptr,
      nullptr,
      nullptr,
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(out->data<T>()))
    );
  } else {
    // std::cout << "split kv";
    auto *allocator = paddle::GetAllocator(q.place());
    phi::Allocator::AllocationPtr tmp_workspace, tmp_m, tmp_d;
    tmp_workspace = allocator->Allocate(
        phi::SizeOf(q.dtype()) *
        static_cast<size_t>(bsz * num_chunks * num_heads * HEAD_DIM_V));
    tmp_m = allocator->Allocate(
        phi::SizeOf(q.dtype()) *
        static_cast<size_t>(bsz * num_chunks * num_heads));
    tmp_d = allocator->Allocate(
        phi::SizeOf(q.dtype()) *
        static_cast<size_t>(bsz * num_chunks * num_heads));

    splitkv_kernel<<<grids, blocks, smem_size, stream>>>(
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(q.data<T>())),
      reinterpret_cast<NV_CACHE_TYPE*>(const_cast<CACHE_TYPE*>(cache_k.data<CACHE_TYPE>())),
      reinterpret_cast<NV_CACHE_TYPE*>(const_cast<CACHE_TYPE*>(cache_v.data<CACHE_TYPE>())),
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(shift_bias_ptr)),
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(smooth_weight_ptr)),
      seq_lens_q.data<int>(),
      seq_lens_kv.data<int>(),
      cum_offsets.data<int>(),
      block_table.data<int>(),
      max_seq_len,
      max_dec_len,
      max_block_num_per_seq,
      softmax_scale,
      in_scale,
      chunk_size,
      reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
      reinterpret_cast<NV_TYPE*>(tmp_m->ptr()),
      reinterpret_cast<NV_TYPE*>(tmp_d->ptr()),
      reinterpret_cast<NV_TYPE*>(const_cast<T*>(out->data<T>()))
    );
    // CHECK(cudaGetLastError());  // 捕捉同步前的最后一个错误。
    // CHECK(cudaDeviceSynchronize());
    // merge
    if (num_chunks <= bsz) {
      dim3 grids_merge(bsz);
      dim3 blocks_merge(blockx, num_heads);
      assert(blockx * num_heads <= 1024);
      merge_varlen_multi_chunks_kernel<NV_TYPE, NV_TYPE, vec_size><<<grids_merge, blocks_merge, 0, stream>>>(
        reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
        reinterpret_cast<NV_TYPE*>(tmp_m->ptr()),
        reinterpret_cast<NV_TYPE*>(tmp_d->ptr()),
        seq_lens_q.data<int>(),
        seq_lens_kv.data<int>(),
        cum_offsets.data<int>(),
        reinterpret_cast<NV_TYPE*>(const_cast<T*>(shift_bias_ptr)),
        reinterpret_cast<NV_TYPE*>(const_cast<T*>(smooth_weight_ptr)),
        reinterpret_cast<NV_TYPE*>(const_cast<T*>(out->data<T>())),
        in_scale,
        num_chunks,
        chunk_size,
        max_seq_len,
        num_heads,
        HEAD_DIM_V
      );
    } else {
      constexpr int bdy = 256 / blockx;
      dim3 grids_merge(bsz, num_heads);
      dim3 blocks_merge(blockx, bdy);
      merge_varlen_multi_chunks_v2_kernel<NV_TYPE, NV_TYPE, vec_size, bdy, HEAD_DIM_V><<<grids_merge, blocks_merge, 0, stream>>>(
        reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
        reinterpret_cast<NV_TYPE*>(tmp_m->ptr()),
        reinterpret_cast<NV_TYPE*>(tmp_d->ptr()),
        seq_lens_q.data<int>(),
        seq_lens_kv.data<int>(),
        cum_offsets.data<int>(),
        reinterpret_cast<NV_TYPE*>(const_cast<T*>(shift_bias_ptr)),
        reinterpret_cast<NV_TYPE*>(const_cast<T*>(smooth_weight_ptr)),
        reinterpret_cast<NV_TYPE*>(const_cast<T*>(out->data<T>())),
        in_scale,
        num_chunks,
        chunk_size,
        max_seq_len,
        num_heads,
        HEAD_DIM_V
      );
    }
  }
  // CHECK(cudaGetLastError());  // 捕捉同步前的最后一个错误。
  // CHECK(cudaDeviceSynchronize());
}

template <typename T>
void DecodeMLAAttentionKernel(
  const AppendAttnMetaData& meta_data,
  const paddle::Tensor &q, // [token_num, num_heads, head_dim]
  const paddle::Tensor &cache_k,
  const paddle::Tensor &cache_v,
  const paddle::optional<paddle::Tensor>& attn_mask,
  const paddle::optional<paddle::Tensor>& shift_bias,
  const paddle::optional<paddle::Tensor>& smooth_weight,
  const paddle::Tensor &seq_lens_q, // q_seq_len is 1
  const paddle::Tensor &seq_lens_kv,
  const paddle::Tensor &padding_offsets,
  const paddle::Tensor &cum_offsets,
  const paddle::Tensor &block_table,
  int max_seq_len,
  int max_dec_len,
  float softmax_scale,
  float in_scale,
  bool causal,
  cudaStream_t &stream,
  paddle::Tensor *out) { 
  // std::cout << "start: " << std::endl;
  const auto token_num = meta_data.token_nums;
  const auto block_size = meta_data.block_size;
  const auto bsz = meta_data.batch_size;
  const auto num_heads = meta_data.q_num_heads;
  const auto group_size = meta_data.q_num_heads / meta_data.kv_num_heads;
  const auto head_dim_qk = meta_data.head_dims;
  const auto head_dim_v = meta_data.head_dims_v;
  const float rope_scale = 0.0;
  const float rope_theta = 0.0;
  const uint32_t deal_each_time = get_cascade_attention_deal_each_time();
  const uint32_t num_stage = get_cascade_attention_num_stages();
  const uint32_t num_threads = get_cascade_attention_num_threads();
  // const uint32_t deal_each_time = 1;
  // const uint32_t num_stage = 2;
  // const uint32_t num_threads = 1;
  uint32_t cache_type = 0;
  // if (cache_k_scale) {
  //   if (cache_k_zp) {
  //     cache_type = 2;
  //   } else {
  //     cache_type = 1;
  //   }
  // }
  std::cout << "cache_type: " << cache_type << ", group_size: " << group_size << std::endl;
  // std::cout << "cache_type: " << cache_type << ", group_size: " << group_size << ", deal_each_time: " << deal_each_time << ", num_stage: " << num_stage;
  DISPATCH_CAUSAL(causal, CAUSAL,
    {DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE,
      {DISPATCH_HEAD_DIM(head_dim_qk, HEAD_DIM_QK,  
        {DISPATCH_HEAD_DIM(head_dim_v, HEAD_DIM_V, 
          {DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, 
            {DISPATCH_NUM_STAGE(num_stage, NUM_STAGE,
              {DISPATCH_DEAL_EACH_TIME(deal_each_time, DEAL_EACH_TIME,
                {DISPATCH_NUM_THREADS(num_threads, NUM_THREADS, 
                  {MultiQueryDecoderAttention<T, GROUP_SIZE, HEAD_DIM_QK, HEAD_DIM_V, BLOCK_SIZE, CAUSAL, NUM_STAGE, CacheType::CacheT, 16, DEAL_EACH_TIME, NUM_THREADS>(
                  meta_data, stream, q, cache_k, cache_v, attn_mask, shift_bias, smooth_weight, seq_lens_q, seq_lens_kv, padding_offsets, cum_offsets, 
                  block_table, max_seq_len, max_dec_len, rope_scale, rope_theta, softmax_scale, in_scale, out);})})})})})})})});
}

template void DecodeMLAAttentionKernel<paddle::bfloat16>(
  const AppendAttnMetaData& meta_data,
  const paddle::Tensor &q, // [token_num, num_heads, head_dim]
  const paddle::Tensor &cache_k,
  const paddle::Tensor &cache_v,
  const paddle::optional<paddle::Tensor>& attn_mask,
  const paddle::optional<paddle::Tensor>& shift_bias,
  const paddle::optional<paddle::Tensor>& smooth_weight,
  const paddle::Tensor &seq_lens_q, // q_seq_len is 1
  const paddle::Tensor &seq_lens_kv,
  const paddle::Tensor &padding_offsets,
  const paddle::Tensor &cum_offsets,
  const paddle::Tensor &block_table,
  int max_seq_len,
  int max_dec_len,
  float softmax_scale,
  float in_scale,
  bool causal,
  cudaStream_t &stream,
  paddle::Tensor *out);

template void DecodeMLAAttentionKernel<paddle::float16>(
  const AppendAttnMetaData& meta_data,
  const paddle::Tensor &q, // [token_num, num_heads, head_dim]
  const paddle::Tensor &cache_k,
  const paddle::Tensor &cache_v,
  const paddle::optional<paddle::Tensor>& attn_mask,
  const paddle::optional<paddle::Tensor>& shift_bias,
  const paddle::optional<paddle::Tensor>& smooth_weight,
  const paddle::Tensor &seq_lens_q, // q_seq_len is 1
  const paddle::Tensor &seq_lens_kv,
  const paddle::Tensor &padding_offsets,
  const paddle::Tensor &cum_offsets,
  const paddle::Tensor &block_table,
  int max_seq_len,
  int max_dec_len,
 float softmax_scale,
  float in_scale,
  bool causal,
  cudaStream_t &stream,
  paddle::Tensor *out);