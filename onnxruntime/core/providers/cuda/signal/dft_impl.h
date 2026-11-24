// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace onnxruntime {
namespace cuda {

// Builds complex rows a[(o * inner + i) * dft_length + j] from X viewed as [outer, num_samples, inner, components].
// Samples beyond min(num_samples, dft_length) are zero. Positions dft_length - k for k in [1, irfft_conj_end)
// take conj(X[k]), which rebuilds the negative frequencies of a one-sided spectrum for the inverse transform.
template <typename T>
void DftPrepareRows(cudaStream_t stream, const T* x, bool complex_input, int64_t outer, int64_t num_samples,
                    int64_t inner, int64_t dft_length, int64_t irfft_conj_end, T* a);

// Scatters complex rows a[(o * inner + i) * dft_length + k] into Y viewed as [outer, output_length, inner, 1 or 2],
// multiplying by scale. With real_output, only the real part is written. a and y may alias when inner == 1 and
// output_length == dft_length because every element is then read and written by the same thread.
template <typename T>
void DftFinalizeRows(cudaStream_t stream, const T* a, int64_t outer, int64_t dft_length, int64_t inner,
                     int64_t output_length, bool real_output, T scale, T* y);

// Writes frames[(b * num_frames + f) * frame_length + n] = signal[b, f * frame_step + n] * window[n].
// complex_output selects interleaved complex frames; otherwise the input must be real.
template <typename T>
void StftFrames(cudaStream_t stream, const T* signal, bool complex_input, bool complex_output, int64_t batch,
                int64_t signal_length, int64_t num_frames, int64_t frame_step, int64_t frame_length,
                const T* window, T* frames);

}  // namespace cuda
}  // namespace onnxruntime
