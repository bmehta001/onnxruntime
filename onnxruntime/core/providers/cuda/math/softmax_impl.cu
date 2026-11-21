/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Modifications Copyright (c) Microsoft. */

// The code below is mostly copied from Pytorch PersistentSoftmax.cuh

#include "core/providers/cuda/cu_inc/common.cuh"
#include "softmax_warpwise_impl.cuh"
#include "softmax_blockwise_impl.cuh"
#include "softmax.h"

#include <limits>

namespace onnxruntime {
namespace cuda {

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
Status dispatch_warpwise_softmax_forward(SoftmaxComputeStreamT ort_stream, output_t* dst, const input_t* src, int softmax_elements,
                                         int softmax_elements_stride, int batch_count) {
  auto stream = ort_stream;
  if (softmax_elements == 0) {
    return Status::OK();
  } else {
    int log2_elements = log2_ceil(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
    int warp_size = (next_power_of_two < GPU_WARP_SIZE_HOST) ? next_power_of_two : GPU_WARP_SIZE_HOST;
    int threads_per_block, shared_memory_size;
    // there are 2 options to save one row of the input matrix: register or shared memory
    // when the number of elements is small, we use register; otherwise, we use shared memory;
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
    if (log2_elements <= 10) {
      // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
      // use 128 threads per block to maximimize gpu utilization
      threads_per_block = 128;
      shared_memory_size = 0;
    } else {
      // setting the number of threads per block to 32 will make index offset calculations easier,
      // under this setting, the cuda block number will be equal to batch size.
      threads_per_block = 32;
      // use shared memory to contain one row of elements
      // TODO: one more optimization can be done here: we actually not need to save next_power_of_two elements, we can just save the valid elements
      shared_memory_size = next_power_of_two * sizeof(input_t);
    }
    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);
    // Launch code would be more elegant if C++ supported FOR CONSTEXPR
    switch (log2_elements) {
#define LAUNCH_KERNEL(kernel_name, log2_elements_value)                      \
  kernel_name<input_t, output_t, acc_t, log2_elements_value, is_log_softmax> \
      <<<blocks, threads, shared_memory_size, stream>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);

#define CASE_LOG2_ELEMENTS(log2_elements_value)                                   \
  case log2_elements_value: {                                                     \
    if constexpr (log2_elements_value <= 10) {                                    \
      LAUNCH_KERNEL(softmax_warp_forward, log2_elements_value)                    \
    } else {                                                                      \
      LAUNCH_KERNEL(softmax_warp_forward_resource_efficient, log2_elements_value) \
    }                                                                             \
  } break

      CASE_LOG2_ELEMENTS(0);
      CASE_LOG2_ELEMENTS(1);
      CASE_LOG2_ELEMENTS(2);
      CASE_LOG2_ELEMENTS(3);
      CASE_LOG2_ELEMENTS(4);
      CASE_LOG2_ELEMENTS(5);
      CASE_LOG2_ELEMENTS(6);
      CASE_LOG2_ELEMENTS(7);
      CASE_LOG2_ELEMENTS(8);
      CASE_LOG2_ELEMENTS(9);
      CASE_LOG2_ELEMENTS(10);
      CASE_LOG2_ELEMENTS(11);  // start to use softmax_warp_forward_resource_efficient instead of softmax_warp_forward for better performance
#undef LAUNCH_KERNEL
#undef CASE_LOG2_ELEMENTS
    }  // switch
  }  // else
  return CUDA_CALL(cudaGetLastError());
}

#define SPECIALIZED_WRAPWISE_SOFTMAX_IMPL(input_t, output_t, acc_t)                                                    \
  template Status dispatch_warpwise_softmax_forward<input_t, output_t, acc_t, false>(SoftmaxComputeStreamT ort_stream, \
                                                                                     output_t * dst,                   \
                                                                                     const input_t* src,               \
                                                                                     int softmax_elements,             \
                                                                                     int softmax_elements_stride,      \
                                                                                     int batch_count);                 \
  template Status dispatch_warpwise_softmax_forward<input_t, output_t, acc_t, true>(SoftmaxComputeStreamT ort_stream,  \
                                                                                    output_t * dst,                    \
                                                                                    const input_t* src,                \
                                                                                    int softmax_elements,              \
                                                                                    int softmax_elements_stride,       \
                                                                                    int batch_count);

SPECIALIZED_WRAPWISE_SOFTMAX_IMPL(float, float, float)
SPECIALIZED_WRAPWISE_SOFTMAX_IMPL(half, half, float)
SPECIALIZED_WRAPWISE_SOFTMAX_IMPL(half, float, float)
SPECIALIZED_WRAPWISE_SOFTMAX_IMPL(double, double, double)
SPECIALIZED_WRAPWISE_SOFTMAX_IMPL(BFloat16, BFloat16, float)

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
Status dispatch_blockwise_softmax_forward(SoftmaxComputeStreamT ort_stream, output_t* output, const input_t* input, int softmax_elements,
                                          int input_stride, int output_stride, int batch_count) {
  auto stream = ort_stream;
  dim3 grid(batch_count);
  constexpr int ILP = sizeof(float4) / sizeof(input_t);
  dim3 block = SoftMax_getBlockSize(ILP, softmax_elements);
  if (is_log_softmax) {
    softmax_block_forward<ILP, input_t, acc_t, output_t, LogSoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(acc_t), stream>>>(output, const_cast<input_t*>(input),
                                                           softmax_elements, input_stride, output_stride);
  } else {
    softmax_block_forward<ILP, input_t, acc_t, output_t, SoftMaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(acc_t), stream>>>(output, const_cast<input_t*>(input),
                                                           softmax_elements, input_stride, output_stride);
  }
  return CUDA_CALL(cudaGetLastError());
}

#define SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(input_t, output_t, acc_t)                                 \
  template Status dispatch_blockwise_softmax_forward<input_t, output_t, acc_t, false>(               \
      SoftmaxComputeStreamT ort_stream, output_t * output, const input_t* src, int softmax_elements, \
      int input_stride, int output_stride, int batch_count);                                         \
  template Status dispatch_blockwise_softmax_forward<input_t, output_t, acc_t, true>(                \
      SoftmaxComputeStreamT ort_stream, output_t * output, const input_t* src, int softmax_elements, \
      int input_stride, int output_stride, int batch_count);

SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(float, float, float)
SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(half, half, float)
SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(half, float, float)
SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(double, double, double)
SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(BFloat16, BFloat16, float)

namespace {

// Merges two online-softmax partials (running max m, running sum s of exp(x - m)).
// The running max starts at lowest() rather than -inf so that merging two empty partials yields 0, not NaN.
template <typename acc_t>
__device__ __forceinline__ void OnlineSoftmaxMerge(acc_t& m, acc_t& s, acc_t other_m, acc_t other_s) {
  const acc_t new_m = m > other_m ? m : other_m;
  s = s * std::exp(m - new_m) + other_s * std::exp(other_m - new_m);
  m = new_m;
}

// Each block handles blockDim.x adjacent inner columns of one or more outer slices. threadIdx.x walks the
// contiguous inner dim (coalesced loads) and threadIdx.y strides over the softmax axis. Partials are combined
// with a shared-memory tree over threadIdx.y, so blockDim.y must be a power of two.
template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
__global__ void softmax_strided_forward(output_t* dst, const input_t* src, int64_t outer, int64_t axis_dim,
                                        int64_t inner) {
  extern __shared__ unsigned char softmax_strided_smem[];
  acc_t* smem_m = reinterpret_cast<acc_t*>(softmax_strided_smem);
  acc_t* smem_s = smem_m + blockDim.x * blockDim.y;

  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool active = col < inner;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  for (int64_t o = blockIdx.y; o < outer; o += gridDim.y) {
    const int64_t base = o * axis_dim * inner + col;

    acc_t m = std::numeric_limits<acc_t>::lowest();
    acc_t s = acc_t(0);
    if (active) {
#pragma unroll 4
      for (int64_t d = threadIdx.y; d < axis_dim; d += blockDim.y) {
        const acc_t x = static_cast<acc_t>(src[base + d * inner]);
        if (x > m) {
          s = s * std::exp(m - x) + acc_t(1);
          m = x;
        } else {
          s += std::exp(x - m);
        }
      }
    }

    smem_m[tid] = m;
    smem_s[tid] = s;
    __syncthreads();
    for (unsigned int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
      if (threadIdx.y < stride) {
        const int other = tid + stride * blockDim.x;
        OnlineSoftmaxMerge(m, s, smem_m[other], smem_s[other]);
        smem_m[tid] = m;
        smem_s[tid] = s;
      }
      __syncthreads();
    }
    m = smem_m[threadIdx.x];
    s = smem_s[threadIdx.x];
    // The next outer iteration overwrites shared memory.
    __syncthreads();

    if (active) {
      if constexpr (is_log_softmax) {
        const acc_t shift = m + std::log(s);
#pragma unroll 4
        for (int64_t d = threadIdx.y; d < axis_dim; d += blockDim.y) {
          const int64_t idx = base + d * inner;
          dst[idx] = static_cast<output_t>(static_cast<acc_t>(src[idx]) - shift);
        }
      } else {
        const acc_t inv_s = acc_t(1) / s;
#pragma unroll 4
        for (int64_t d = threadIdx.y; d < axis_dim; d += blockDim.y) {
          const int64_t idx = base + d * inner;
          dst[idx] = static_cast<output_t>(std::exp(static_cast<acc_t>(src[idx]) - m) * inv_s);
        }
      }
    }
  }
}

int NextPowerOfTwo(int64_t v) {
  int p = 1;
  while (p < v && p < (1 << 30)) p <<= 1;
  return p;
}

}  // namespace

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
Status dispatch_strided_softmax_forward(SoftmaxComputeStreamT stream, output_t* dst, const input_t* src,
                                        int64_t outer, int64_t axis_dim, int64_t inner) {
  if (outer == 0 || axis_dim == 0 || inner == 0) {
    return Status::OK();
  }

  constexpr int kMaxThreadsPerBlock = 512;
  constexpr int64_t kMaxGridDimY = 65535;
  const int block_x = std::min(GPU_WARP_SIZE_HOST, NextPowerOfTwo(inner));
  const int block_y = std::min(kMaxThreadsPerBlock / block_x, NextPowerOfTwo(axis_dim));
  const int64_t grid_x = (inner + block_x - 1) / block_x;
  ORT_RETURN_IF(grid_x > std::numeric_limits<int>::max(), "Softmax inner dimension is too large: ", inner);

  const dim3 block(block_x, block_y);
  const dim3 grid(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(std::min(outer, kMaxGridDimY)));
  const size_t smem_bytes = 2 * static_cast<size_t>(block_x) * block_y * sizeof(acc_t);
  softmax_strided_forward<input_t, output_t, acc_t, is_log_softmax>
      <<<grid, block, smem_bytes, stream>>>(dst, src, outer, axis_dim, inner);
  return CUDA_CALL(cudaGetLastError());
}

#define SPECIALIZED_STRIDED_SOFTMAX_IMPL(input_t, output_t, acc_t)                     \
  template Status dispatch_strided_softmax_forward<input_t, output_t, acc_t, false>(   \
      SoftmaxComputeStreamT stream, output_t * dst, const input_t* src, int64_t outer, \
      int64_t axis_dim, int64_t inner);                                                \
  template Status dispatch_strided_softmax_forward<input_t, output_t, acc_t, true>(    \
      SoftmaxComputeStreamT stream, output_t * dst, const input_t* src, int64_t outer, \
      int64_t axis_dim, int64_t inner);

SPECIALIZED_STRIDED_SOFTMAX_IMPL(float, float, float)
SPECIALIZED_STRIDED_SOFTMAX_IMPL(half, half, float)
SPECIALIZED_STRIDED_SOFTMAX_IMPL(double, double, double)
SPECIALIZED_STRIDED_SOFTMAX_IMPL(BFloat16, BFloat16, float)
}  // namespace cuda
}  // namespace onnxruntime
