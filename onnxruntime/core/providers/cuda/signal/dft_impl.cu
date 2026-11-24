// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/signal/dft_impl.h"

#include <algorithm>

namespace onnxruntime {
namespace cuda {

namespace {

constexpr int kThreadsPerBlock = 256;

inline int GridSize(int64_t total) {
  // Grid-stride loops cover any remainder.
  constexpr int64_t kMaxBlocks = 65535LL * 16;
  return static_cast<int>(std::min<int64_t>((total + kThreadsPerBlock - 1) / kThreadsPerBlock, kMaxBlocks));
}

template <typename T>
__global__ void DftPrepareRowsKernel(const T* __restrict__ x, bool complex_input, int64_t num_samples, int64_t inner,
                                     int64_t dft_length, int64_t irfft_conj_end, int64_t total, T* __restrict__ a) {
  const int64_t components = complex_input ? 2 : 1;
  const int64_t used_samples = num_samples < dft_length ? num_samples : dft_length;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = idx / dft_length;
    const int64_t j = idx - row * dft_length;
    const int64_t o = row / inner;
    const int64_t i = row - o * inner;

    int64_t n = j;
    bool conjugate = false;
    const int64_t mirrored = dft_length - j;
    if (j > 0 && mirrored < irfft_conj_end) {
      n = mirrored;
      conjugate = true;
    }

    T re = T(0);
    T im = T(0);
    if (n < used_samples) {
      const int64_t src = ((o * num_samples + n) * inner + i) * components;
      re = x[src];
      if (complex_input) {
        im = x[src + 1];
      }
    }
    a[2 * idx] = re;
    a[2 * idx + 1] = conjugate ? -im : im;
  }
}

template <typename T>
__global__ void DftFinalizeRowsKernel(const T* a, int64_t dft_length, int64_t inner, int64_t output_length,
                                      bool real_output, T scale, int64_t total, T* y) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    // idx enumerates Y in [outer, output_length, inner] order.
    const int64_t i = idx % inner;
    const int64_t rest = idx / inner;
    const int64_t k = rest % output_length;
    const int64_t o = rest / output_length;
    const int64_t src = ((o * inner + i) * dft_length + k) * 2;
    const T re = a[src];
    if (real_output) {
      y[idx] = re * scale;
    } else {
      const T im = a[src + 1];
      y[2 * idx] = re * scale;
      y[2 * idx + 1] = im * scale;
    }
  }
}

// One block per frame; threads stride over the frame so no per-element index division is needed.
template <typename T>
__global__ void StftFramesKernel(const T* __restrict__ signal, bool complex_input, bool complex_output,
                                 int64_t signal_length, int64_t num_frames, int64_t frame_step, int64_t frame_length,
                                 const T* __restrict__ window, int64_t num_rows, T* __restrict__ frames) {
  const int64_t components = complex_input ? 2 : 1;
  for (int64_t row = blockIdx.x; row < num_rows; row += gridDim.x) {
    const int64_t b = row / num_frames;
    const int64_t f = row - b * num_frames;
    const T* src = signal + (b * signal_length + f * frame_step) * components;
    T* dst = frames + row * frame_length * (complex_output ? 2 : 1);
    for (int64_t n = threadIdx.x; n < frame_length; n += blockDim.x) {
      const T w = window != nullptr ? window[n] : T(1);
      if (complex_output) {
        dst[2 * n] = src[n * components] * w;
        dst[2 * n + 1] = complex_input ? src[n * components + 1] * w : T(0);
      } else {
        dst[n] = src[n] * w;
      }
    }
  }
}

}  // namespace

template <typename T>
void DftPrepareRows(cudaStream_t stream, const T* x, bool complex_input, int64_t outer, int64_t num_samples,
                    int64_t inner, int64_t dft_length, int64_t irfft_conj_end, T* a) {
  const int64_t total = outer * inner * dft_length;
  if (total == 0) return;
  DftPrepareRowsKernel<T><<<GridSize(total), kThreadsPerBlock, 0, stream>>>(
      x, complex_input, num_samples, inner, dft_length, irfft_conj_end, total, a);
}

template <typename T>
void DftFinalizeRows(cudaStream_t stream, const T* a, int64_t outer, int64_t dft_length, int64_t inner,
                     int64_t output_length, bool real_output, T scale, T* y) {
  const int64_t total = outer * output_length * inner;
  if (total == 0) return;
  DftFinalizeRowsKernel<T><<<GridSize(total), kThreadsPerBlock, 0, stream>>>(
      a, dft_length, inner, output_length, real_output, scale, total, y);
}

template <typename T>
void StftFrames(cudaStream_t stream, const T* signal, bool complex_input, bool complex_output, int64_t batch,
                int64_t signal_length, int64_t num_frames, int64_t frame_step, int64_t frame_length,
                const T* window, T* frames) {
  const int64_t num_rows = batch * num_frames;
  if (num_rows == 0 || frame_length == 0) return;
  const int threads = frame_length >= kThreadsPerBlock ? kThreadsPerBlock : 128;
  const int blocks = static_cast<int>(std::min<int64_t>(num_rows, 65535LL * 16));
  StftFramesKernel<T><<<blocks, threads, 0, stream>>>(
      signal, complex_input, complex_output, signal_length, num_frames, frame_step, frame_length, window, num_rows,
      frames);
}

#define SPECIALIZE_DFT_IMPL(T)                                                                                    \
  template void DftPrepareRows<T>(cudaStream_t, const T*, bool, int64_t, int64_t, int64_t, int64_t, int64_t, T*); \
  template void DftFinalizeRows<T>(cudaStream_t, const T*, int64_t, int64_t, int64_t, int64_t, bool, T, T*);      \
  template void StftFrames<T>(cudaStream_t, const T*, bool, bool, int64_t, int64_t, int64_t, int64_t, int64_t,    \
                              const T*, T*);

SPECIALIZE_DFT_IMPL(float)
SPECIALIZE_DFT_IMPL(double)

#undef SPECIALIZE_DFT_IMPL

}  // namespace cuda
}  // namespace onnxruntime
