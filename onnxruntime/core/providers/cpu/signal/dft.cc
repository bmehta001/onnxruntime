// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/signal/dft.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <type_traits>

#include "core/common/inlined_containers.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"
#include "core/providers/cpu/signal/utils.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    DFT,
    17, 19,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraints<float, double>())
        .TypeConstraint("T2", BuildKernelDefConstraints<int32_t, int64_t>()),
    DFT);

ONNX_CPU_OPERATOR_KERNEL(DFT, 20,
                         KernelDefBuilder()
                             .TypeConstraint("T1", BuildKernelDefConstraints<float, double>())
                             .TypeConstraint("T2", BuildKernelDefConstraints<int32_t, int64_t>()),
                         DFT);

ONNX_CPU_OPERATOR_KERNEL(STFT, 17,
                         KernelDefBuilder()
                             .MayInplace(0, 0)
                             .TypeConstraint("T1", BuildKernelDefConstraints<float, double>())
                             .TypeConstraint("T2", BuildKernelDefConstraints<int32_t, int64_t>()),
                         STFT);

namespace {

using signal::fft::Complex;
using signal::fft::ComplexFftPlan;
using signal::fft::PlanCache;
using signal::fft::RealFftPlan;

template <typename T>
inline Complex<T> LoadComplex(T x) { return {x, 0}; }

template <typename T>
inline Complex<T> LoadComplex(const std::complex<T>& x) { return {x.real(), x.imag()}; }

enum class DftMode {
  kForward,
  kInverse,
  // Inverse of a one-sided spectrum producing a real signal (IRFFT).
  kInverseOnesided,
};

// Everything needed to run one 1-D transform; shared by all signals of a Compute call.
template <typename T>
struct DftSetup {
  DftMode mode;
  size_t n;            // transform length
  size_t samples;      // input samples used per signal: min(input length, n)
  size_t output_size;  // output elements written per signal
  // Forward transforms of real input and IRFFTs of exactly n/2+1 bins use the real-input plan.
  std::shared_ptr<const RealFftPlan<T>> real_plan;
  std::shared_ptr<const ComplexFftPlan<T>> complex_plan;
  size_t buffer_size;
  size_t scratch_size;
};

template <typename T, typename U>
DftSetup<T> MakeDftSetup(PlanCache& cache, DftMode mode, size_t n, size_t input_length, size_t output_size) {
  // IRFFT follows numpy.fft.irfft: only the first n/2+1 bins are used and missing bins are treated as zero.
  const size_t max_samples = mode == DftMode::kInverseOnesided ? n / 2 + 1 : n;
  DftSetup<T> setup{mode, n, std::min(input_length, max_samples), output_size, nullptr, nullptr, 0, 0};
  constexpr bool kRealInput = std::is_same_v<U, T>;
  bool use_real = false;
  if (mode == DftMode::kForward) {
    use_real = kRealInput;
  } else if (mode == DftMode::kInverseOnesided) {
    use_real = n % 2 == 0;
  }
  if (use_real) {
    setup.real_plan = cache.GetReal<T>(n);
    setup.buffer_size = setup.real_plan->BufferSize();
    setup.scratch_size = setup.real_plan->ScratchSize();
  } else {
    setup.complex_plan = cache.GetComplex<T>(n);
    setup.buffer_size = n;
    setup.scratch_size = setup.complex_plan->ScratchSize();
  }
  return setup;
}

// Transforms one signal. `x` is read with a stride of `x_stride` elements of U. Output element k is written to
// y[k * y_step] (and y[k * y_step + 1] for the imaginary part of complex outputs).
template <typename T, typename U>
void RunDft(const DftSetup<T>& s, const U* x, size_t x_stride, const T* window, T* y, size_t y_step,
            Complex<T>* buffer, Complex<T>* scratch) {
  const size_t n = s.n;
  const size_t samples = s.samples;

  if (s.mode == DftMode::kInverseOnesided) {
    const T inv_n = static_cast<T>(1.0 / static_cast<double>(n));
    if (s.real_plan) {
      for (size_t k = 0; k < samples; ++k) {
        buffer[k] = LoadComplex(x[k * x_stride]);
      }
      std::fill(buffer + samples, buffer + n / 2 + 1, Complex<T>{0, 0});
      s.real_plan->Backward(buffer, scratch);
      const T* r = reinterpret_cast<const T*>(buffer);
      for (size_t j = 0; j < n; ++j) {
        y[j * y_step] = r[j] * inv_n;
      }
      return;
    }
    // Odd n: rebuild the full Hermitian spectrum. Bin k < (n+1)/2 mirrors to n-k; the imaginary part of DC
    // only contributes to the discarded imaginary output.
    for (size_t k = 0; k < samples; ++k) {
      buffer[k] = LoadComplex(x[k * x_stride]);
    }
    std::fill(buffer + samples, buffer + n, Complex<T>{0, 0});
    const size_t conjugate_end = std::min(samples, (n + 1) / 2);
    for (size_t k = 1; k < conjugate_end; ++k) {
      buffer[n - k] = signal::fft::Conj(buffer[k]);
    }
    s.complex_plan->Backward(buffer, scratch);
    for (size_t j = 0; j < n; ++j) {
      y[j * y_step] = buffer[j].re * inv_n;
    }
    return;
  }

  if constexpr (std::is_same_v<U, T>) {
    if (s.real_plan) {
      if (s.real_plan->IsPacked()) {
        T* r = reinterpret_cast<T*>(buffer);
        for (size_t j = 0; j < samples; ++j) {
          r[j] = window ? x[j * x_stride] * window[j] : x[j * x_stride];
        }
        std::fill(r + samples, r + n, T{0});
      } else {
        for (size_t j = 0; j < samples; ++j) {
          buffer[j] = {window ? x[j * x_stride] * window[j] : x[j * x_stride], 0};
        }
        std::fill(buffer + samples, buffer + n, Complex<T>{0, 0});
      }
      s.real_plan->Forward(buffer, scratch);
      const size_t half = n / 2;
      for (size_t k = 0; k < s.output_size; ++k) {
        // Bins above n/2 of a real signal's spectrum are conjugates of the lower bins.
        const Complex<T> v = k <= half ? buffer[k] : signal::fft::Conj(buffer[n - k]);
        y[k * y_step] = v.re;
        y[k * y_step + 1] = v.im;
      }
      return;
    }
  }

  for (size_t j = 0; j < samples; ++j) {
    const Complex<T> v = LoadComplex(x[j * x_stride]);
    buffer[j] = window ? signal::fft::Scale(v, window[j]) : v;
  }
  std::fill(buffer + samples, buffer + n, Complex<T>{0, 0});
  if (s.mode == DftMode::kForward) {
    s.complex_plan->Forward(buffer, scratch);
    for (size_t k = 0; k < s.output_size; ++k) {
      y[k * y_step] = buffer[k].re;
      y[k * y_step + 1] = buffer[k].im;
    }
  } else {
    s.complex_plan->Backward(buffer, scratch);
    const T inv_n = static_cast<T>(1.0 / static_cast<double>(n));
    for (size_t k = 0; k < s.output_size; ++k) {
      y[k * y_step] = buffer[k].re * inv_n;
      y[k * y_step + 1] = buffer[k].im * inv_n;
    }
  }
}

// Runs `count` independent transforms on the operator thread pool. get_signal(i) returns the input and output base
// pointers of signal i. Each worker allocates its buffers once per range of signals.
template <typename T, typename U, typename GetSignal>
void RunDfts(concurrency::ThreadPool* thread_pool, const DftSetup<T>& setup, size_t count, size_t x_stride,
             const T* window, size_t y_step, const GetSignal& get_signal) {
  const double work = static_cast<double>(std::max(setup.n, setup.scratch_size));
  const TensorOpCost cost{static_cast<double>(setup.samples * sizeof(U)),
                          static_cast<double>(setup.output_size * 2 * sizeof(T)),
                          5.0 * work * std::max(1.0, std::log2(work))};
  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(count), cost, [&](std::ptrdiff_t first, std::ptrdiff_t last) {
        auto memory = std::make_unique<Complex<T>[]>(setup.buffer_size + setup.scratch_size);
        Complex<T>* buffer = memory.get();
        Complex<T>* scratch = buffer + setup.buffer_size;
        for (std::ptrdiff_t i = first; i < last; ++i) {
          const U* x;
          T* y;
          get_signal(static_cast<size_t>(i), x, y);
          RunDft<T, U>(setup, x, x_stride, window, y, y_step, buffer, scratch);
        }
      });
}

template <typename T, typename U>
Status DiscreteFourierTransform(OpKernelContext* ctx, PlanCache& cache, const Tensor* X, Tensor* Y, size_t axis,
                                size_t dft_length, bool is_onesided, bool inverse) {
  const auto& X_shape = X->Shape();
  const auto& Y_shape = Y->Shape();
  const size_t rank = X_shape.NumDimensions() - 1;
  const size_t input_factor = narrow<size_t>(X_shape[rank]);
  const size_t input_length = narrow<size_t>(X_shape[axis]);
  const size_t output_length = narrow<size_t>(Y_shape[axis]);
  const size_t total_dfts = narrow<size_t>(X_shape.SizeToDimension(rank)) / input_length;

  const DftMode mode = !inverse ? DftMode::kForward : (is_onesided ? DftMode::kInverseOnesided : DftMode::kInverse);
  const auto setup = MakeDftSetup<T, U>(cache, mode, dft_length, input_length, output_length);

  // Strides are in units of U for the input and of T for the output.
  const size_t x_stride = narrow<size_t>(X_shape.SizeFromDimension(axis + 1)) / input_factor;
  const size_t y_step = narrow<size_t>(Y_shape.SizeFromDimension(axis + 1));
  InlinedVector<size_t> dims, x_strides, y_strides;
  for (size_t r = 0; r < rank; ++r) {
    if (r == axis) {
      continue;
    }
    dims.push_back(narrow<size_t>(X_shape[r]));
    x_strides.push_back(narrow<size_t>(X_shape.SizeFromDimension(r + 1)) / input_factor);
    y_strides.push_back(narrow<size_t>(Y_shape.SizeFromDimension(r + 1)));
  }

  const U* X_data = reinterpret_cast<const U*>(X->Data<T>());
  T* Y_data = Y->MutableData<T>();
  RunDfts<T, U>(ctx->GetOperatorThreadPool(), setup, total_dfts, x_stride, nullptr, y_step,
                [&](size_t i, const U*& x, T*& y) {
                  size_t x_offset = 0;
                  size_t y_offset = 0;
                  for (size_t d = dims.size(); d-- > 0;) {
                    const size_t index = i % dims[d];
                    i /= dims[d];
                    x_offset += index * x_strides[d];
                    y_offset += index * y_strides[d];
                  }
                  x = X_data + x_offset;
                  y = Y_data + y_offset;
                });
  return Status::OK();
}

Status DiscreteFourierTransform(OpKernelContext* ctx, PlanCache& cache, int64_t axis, bool is_onesided,
                                bool inverse) {
  const auto* X = ctx->Input<Tensor>(0);
  const auto* dft_length = ctx->Input<Tensor>(1);
  const auto& X_shape = X->Shape();
  const auto is_real_valued = X_shape[X_shape.NumDimensions() - 1] == 1;
  const auto is_complex_valued = X_shape[X_shape.NumDimensions() - 1] == 2;

  axis = HandleNegativeAxis(axis, X_shape.NumDimensions());
  ORT_RETURN_IF(axis == static_cast<int64_t>(X_shape.NumDimensions()) - 1,
                "DFT axis must refer to a signal dimension, not the last (real/imag) dimension.");

  if (inverse && is_onesided) {
    ORT_RETURN_IF(!is_complex_valued,
                  "Inverse one-sided DFT (IRFFT) requires complex-valued input (last dimension must be 2)");
  } else {
    ORT_RETURN_IF(!(is_real_valued || is_complex_valued),
                  "Input layout not supported. The input tensor must have a last dimension of 1 (real) or 2 (complex).");
  }

  int64_t number_of_samples = static_cast<int64_t>(X_shape[onnxruntime::narrow<size_t>(axis)]);
  if (inverse && is_onesided) {
    number_of_samples = (number_of_samples - 1) << 1;
  }
  if (dft_length) {
    const auto& dft_length_shape = dft_length->Shape();
    ORT_RETURN_IF(!dft_length_shape.IsScalar(), "dft_length must be a scalar value.");
    number_of_samples = signal::get_scalar_value_from_tensor<int64_t>(dft_length);
    ORT_RETURN_IF(number_of_samples <= 0, "dft_length must be greater than zero.");
  }

  // Forward one-sided transforms only return the unique bins; IRFFT returns a full-length real signal.
  auto dft_output_size = (is_onesided && !inverse) ? ((number_of_samples >> 1) + 1) : number_of_samples;

  auto Y_shape = onnxruntime::TensorShape(X_shape);
  Y_shape[Y_shape.NumDimensions() - 1] = (inverse && is_onesided) ? 1 : 2;
  Y_shape[onnxruntime::narrow<size_t>(axis)] = dft_output_size;
  auto Y = ctx->Output(0, Y_shape);

  if (Y_shape.Size() == 0) {
    return Status::OK();
  }

  ORT_RETURN_IF(X_shape[onnxruntime::narrow<size_t>(axis)] == 0,
                "DFT input signal dimension must be greater than zero when the output is non-empty.");

  const size_t axis_index = narrow<size_t>(axis);
  const size_t n = narrow<size_t>(number_of_samples);
  const auto element_size = X->DataType()->Size();
  if (element_size == sizeof(float)) {
    if (is_real_valued) {
      return DiscreteFourierTransform<float, float>(ctx, cache, X, Y, axis_index, n, is_onesided, inverse);
    }
    return DiscreteFourierTransform<float, std::complex<float>>(ctx, cache, X, Y, axis_index, n, is_onesided,
                                                                inverse);
  }
  if (element_size == sizeof(double)) {
    if (is_real_valued) {
      return DiscreteFourierTransform<double, double>(ctx, cache, X, Y, axis_index, n, is_onesided, inverse);
    }
    return DiscreteFourierTransform<double, std::complex<double>>(ctx, cache, X, Y, axis_index, n, is_onesided,
                                                                  inverse);
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported input data type of ", X->DataType());
}

template <typename T, typename U>
Status ShortTimeFourierTransform(OpKernelContext* ctx, PlanCache& cache, bool is_onesided) {
  // Input(0, "signal") type = T1
  // Input(1, "frame_step") type = T2
  // Input(2, "window") type = T1, optional
  // Input(3, "frame_length") type = T2, optional
  const auto* signal = ctx->Input<Tensor>(0);
  const auto frame_step = signal::get_scalar_value_from_tensor<int64_t>(ctx->Input<Tensor>(1));
  ORT_RETURN_IF_NOT(frame_step > 0, "frame_step must be greater than zero.");
  const auto* window = ctx->Input<Tensor>(2);
  const auto* frame_length_tensor = ctx->Input<Tensor>(3);

  const auto& signal_shape = signal->Shape();
  const auto batch_size = signal_shape[0];
  const auto signal_size = signal_shape[1];
  const auto signal_components = signal_shape.NumDimensions() == 2   ? 1
                                 : signal_shape.NumDimensions() == 3 ? signal_shape[2]
                                                                     : 0;  // error
  ORT_ENFORCE(signal_components == 1 || signal_components == 2,
              "signal shape must end in 1 (real) or 2 (real, imaginary).");

  int64_t frame_length = std::numeric_limits<int64_t>::min();
  if (frame_length_tensor) {
    frame_length = signal::get_scalar_value_from_tensor<int64_t>(frame_length_tensor);
  }

  int64_t window_length = std::numeric_limits<int64_t>::min();
  if (window) {
    window_length = window->Shape()[0];
  }

  // The frame_length and window inputs are generally used interchangeably, and should match!
  if (frame_length != std::numeric_limits<int64_t>::min() && window_length != std::numeric_limits<int64_t>::min()) {
    ORT_ENFORCE(
        frame_length == window_length,
        "If both frame_length and window are set, then the size of the window must be equal to the frame_length.");
  }

  // Calculate the window size with preference to the window input.
  const auto window_size = window ? window->Shape()[0] : frame_length;
  ORT_RETURN_IF(window_size <= 0, "STFT requires a window or frame_length with a positive length.");
  ORT_ENFORCE(window_size <= signal_size, "Ensure that the dft size is smaller than the signal.");

  const int64_t n_dfts = (signal_size - window_size) / frame_step + 1;

  // Onesided output only contains the unique bins.
  const auto dft_output_size = is_onesided ? (window_size >> 1) + 1 : window_size;

  constexpr int64_t output_components = 2;
  auto Y = ctx->Output(0, onnxruntime::TensorShape({batch_size, n_dfts, dft_output_size, output_components}));
  if (batch_size == 0) {
    return Status::OK();
  }

  const auto n = narrow<size_t>(window_size);
  const auto setup = MakeDftSetup<T, U>(cache, DftMode::kForward, n, n, narrow<size_t>(dft_output_size));
  const U* signal_data = reinterpret_cast<const U*>(signal->Data<T>());
  T* Y_data = Y->MutableData<T>();
  const T* window_data = window ? window->Data<T>() : nullptr;
  const size_t frames = narrow<size_t>(n_dfts);
  const size_t step = narrow<size_t>(frame_step);
  const size_t batch_stride = narrow<size_t>(signal_size);
  const size_t frame_output_size = narrow<size_t>(dft_output_size * output_components);

  // Every (batch, frame) pair is an independent transform. signal_data is U*, so one increment advances one input
  // sample, including both lanes for complex input.
  RunDfts<T, U>(ctx->GetOperatorThreadPool(), setup, narrow<size_t>(batch_size) * frames, 1, window_data,
                output_components, [&](size_t i, const U*& x, T*& y) {
                  const size_t batch_idx = i / frames;
                  const size_t frame_idx = i % frames;
                  x = signal_data + batch_idx * batch_stride + frame_idx * step;
                  y = Y_data + i * frame_output_size;
                });
  return Status::OK();
}

}  // namespace

Status DFT::Compute(OpKernelContext* ctx) const {
  int64_t axis = axis_;
  if (opset_ >= 20 && ctx->InputCount() >= 3) {
    const Tensor* axes_tensor = ctx->Input<Tensor>(2);
    axis = axes_tensor->Data<int64_t>()[0];
  }

  return DiscreteFourierTransform(ctx, plan_cache_, axis, is_onesided_, is_inverse_);
}

Status STFT::Compute(OpKernelContext* ctx) const {
  const auto* signal = ctx->Input<Tensor>(0);
  const auto& signal_shape = signal->Shape();
  const auto is_real_valued = signal_shape.NumDimensions() == 2 || signal_shape[signal_shape.NumDimensions() - 1] == 1;
  const auto is_complex_valued = signal_shape.NumDimensions() > 2 && signal_shape[signal_shape.NumDimensions() - 1] == 2;
  ORT_RETURN_IF_NOT(is_real_valued || is_complex_valued,
                    "Unsupported input signal shape. The signal's first dimension must be the batch dimension and its "
                    "second dimension must be the signal length dimension. It may optionally include a 3rd dimension "
                    "of size 2 for complex inputs.");

  const auto element_size = signal->DataType()->Size();
  if (element_size == sizeof(float)) {
    return is_real_valued ? ShortTimeFourierTransform<float, float>(ctx, plan_cache_, is_onesided_)
                          : ShortTimeFourierTransform<float, std::complex<float>>(ctx, plan_cache_, is_onesided_);
  }
  if (element_size == sizeof(double)) {
    return is_real_valued ? ShortTimeFourierTransform<double, double>(ctx, plan_cache_, is_onesided_)
                          : ShortTimeFourierTransform<double, std::complex<double>>(ctx, plan_cache_, is_onesided_);
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported input data type of ", signal->DataType());
}

}  // namespace onnxruntime
