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
using signal::fft::LanePtr;
using signal::fft::LaneTraits;
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

// Lane type used to transform several signals of type T at once; T itself when no SIMD type is available.
template <typename T>
struct VectorLane {
  using type = T;
};

#if defined(ORT_FFT_FLOAT32X4)
template <>
struct VectorLane<float> {
  using type = signal::fft::Float32x4;
};
#endif

// Real and imaginary parts of lane l (signal l) of element k of a lane buffer.
template <typename V>
inline auto& LaneRe(Complex<V>* b, size_t k, size_t l) { return LanePtr(b + k)[l]; }

template <typename V>
inline auto& LaneIm(Complex<V>* b, size_t k, size_t l) { return LanePtr(b + k)[LaneTraits<V>::kLanes + l]; }

// Sample j of lane l of a real signal packed two samples per complex element.
template <typename V>
inline auto& PackedSample(Complex<V>* b, size_t j, size_t l) {
  return LanePtr(b + j / 2)[(j & 1) * LaneTraits<V>::kLanes + l];
}

// Transforms LaneTraits<V>::kLanes signals, one per lane of V. Signal l is read from xs[l] with a stride of
// `x_stride` elements of U. Its output element k is written to ys[l][k * y_step] (and ys[l][k * y_step + 1] for the
// imaginary part of complex outputs). Only the first `active` lanes are stored; the inputs of the other lanes must
// still be readable. Lanes are independent, so a signal's result does not depend on the other lanes.
template <typename T, typename U, typename V>
void RunDft(const DftSetup<T>& s, const U* const* xs, size_t x_stride, const T* window, T* const* ys, size_t y_step,
            Complex<V>* buffer, Complex<V>* scratch, size_t active) {
  constexpr size_t kLanes = LaneTraits<V>::kLanes;
  const size_t n = s.n;
  const size_t samples = s.samples;

  auto load = [&](size_t end) {
    for (size_t j = 0; j < samples; ++j) {
      for (size_t l = 0; l < kLanes; ++l) {
        const U* x = xs[l];
        if constexpr (std::is_same_v<U, T>) {
          LaneRe(buffer, j, l) = window ? x[j * x_stride] * window[j] : x[j * x_stride];
          LaneIm(buffer, j, l) = T{0};
        } else {
          Complex<T> v = LoadComplex(x[j * x_stride]);
          if (window) {
            v = signal::fft::Scale(v, window[j]);
          }
          LaneRe(buffer, j, l) = v.re;
          LaneIm(buffer, j, l) = v.im;
        }
      }
    }
    std::fill(buffer + samples, buffer + end, Complex<V>{});
  };

  if (s.mode == DftMode::kInverseOnesided) {
    const T inv_n = static_cast<T>(1.0 / static_cast<double>(n));
    if (s.real_plan) {
      load(n / 2 + 1);
      s.real_plan->Backward(buffer, scratch);
      for (size_t j = 0; j < n; ++j) {
        for (size_t l = 0; l < active; ++l) {
          ys[l][j * y_step] = PackedSample(buffer, j, l) * inv_n;
        }
      }
      return;
    }
    // Odd n: rebuild the full Hermitian spectrum. Bin k < (n+1)/2 mirrors to n-k; the imaginary part of DC
    // only contributes to the discarded imaginary output.
    load(n);
    const size_t conjugate_end = std::min(samples, (n + 1) / 2);
    for (size_t k = 1; k < conjugate_end; ++k) {
      buffer[n - k] = signal::fft::Conj(buffer[k]);
    }
    s.complex_plan->Backward(buffer, scratch);
    for (size_t j = 0; j < n; ++j) {
      for (size_t l = 0; l < active; ++l) {
        ys[l][j * y_step] = LaneRe(buffer, j, l) * inv_n;
      }
    }
    return;
  }

  if constexpr (std::is_same_v<U, T>) {
    if (s.real_plan) {
      if (s.real_plan->IsPacked()) {
        for (size_t j = 0; j < samples; ++j) {
          for (size_t l = 0; l < kLanes; ++l) {
            PackedSample(buffer, j, l) = window ? xs[l][j * x_stride] * window[j] : xs[l][j * x_stride];
          }
        }
        for (size_t j = samples; j < n; ++j) {
          for (size_t l = 0; l < kLanes; ++l) {
            PackedSample(buffer, j, l) = T{0};
          }
        }
      } else {
        load(n);
      }
      s.real_plan->Forward(buffer, scratch);
      const size_t half = n / 2;
      // Bins above n/2 of a real signal's spectrum are conjugates of the lower bins.
      const size_t direct_end = std::min(s.output_size, half + 1);
      for (size_t k = 0; k < direct_end; ++k) {
        for (size_t l = 0; l < active; ++l) {
          ys[l][k * y_step] = LaneRe(buffer, k, l);
          ys[l][k * y_step + 1] = LaneIm(buffer, k, l);
        }
      }
      for (size_t k = direct_end; k < s.output_size; ++k) {
        for (size_t l = 0; l < active; ++l) {
          ys[l][k * y_step] = LaneRe(buffer, n - k, l);
          ys[l][k * y_step + 1] = -LaneIm(buffer, n - k, l);
        }
      }
      return;
    }
  }

  load(n);
  const bool forward = s.mode == DftMode::kForward;
  if (forward) {
    s.complex_plan->Forward(buffer, scratch);
  } else {
    s.complex_plan->Backward(buffer, scratch);
  }
  if (forward) {
    for (size_t k = 0; k < s.output_size; ++k) {
      for (size_t l = 0; l < active; ++l) {
        ys[l][k * y_step] = LaneRe(buffer, k, l);
        ys[l][k * y_step + 1] = LaneIm(buffer, k, l);
      }
    }
    return;
  }
  const T inv_n = static_cast<T>(1.0 / static_cast<double>(n));
  for (size_t k = 0; k < s.output_size; ++k) {
    for (size_t l = 0; l < active; ++l) {
      ys[l][k * y_step] = LaneRe(buffer, k, l) * inv_n;
      ys[l][k * y_step + 1] = LaneIm(buffer, k, l) * inv_n;
    }
  }
}

// Work buffers for the ranges of one parallel loop, allocated up front in one block from the kernel's temp-space
// allocator. A running range claims a free slot with an atomic flag and releases it when done; at most one range per
// thread runs at a time, so `slots` = degree of parallelism suffices. Allocating a fresh large buffer in every range
// made the workers fault in new pages concurrently, and a mutex-based pool parked threads under contention; both kept
// the lane path from scaling.
template <typename V>
class LaneBufferSlots {
 public:
  LaneBufferSlots(const AllocatorPtr& allocator, size_t size, size_t slots)
      : size_(size),
        slots_(slots),
        memory_(IAllocator::MakeUniquePtr<Complex<V>>(allocator, SafeInt<size_t>(size) * slots)),
        busy_(std::make_unique<std::atomic<bool>[]>(slots)) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LaneBufferSlots);

  // Returns a free slot index, or slots_ if none is free; the caller then uses its own buffer.
  size_t Acquire() {
    for (size_t i = 0; i < slots_; ++i) {
      bool expected = false;
      if (!busy_[i].load(std::memory_order_relaxed) &&
          busy_[i].compare_exchange_strong(expected, true, std::memory_order_acquire)) {
        return i;
      }
    }
    return slots_;
  }

  Complex<V>* Buffer(size_t slot) const { return memory_.get() + slot * size_; }
  size_t Slots() const { return slots_; }
  void Release(size_t slot) { busy_[slot].store(false, std::memory_order_release); }

 private:
  size_t size_;
  size_t slots_;
  IAllocatorUniquePtr<Complex<V>> memory_;
  std::unique_ptr<std::atomic<bool>[]> busy_;
};

// Runs `count` independent transforms on the operator thread pool. get_signal(i) returns the input and output base
// pointers of signal i. Signals are transformed in groups of kLanes using the SIMD lane type. A partial last group
// fills its unused lanes with a copy of its first signal and discards their results, so every signal takes the same
// code path regardless of its position, the batch size or the thread count.
template <typename T, typename U, typename GetSignal>
void RunDfts(concurrency::ThreadPool* thread_pool, AllocatorPtr allocator, const DftSetup<T>& setup, size_t count,
             size_t x_stride, const T* window, size_t y_step, const GetSignal& get_signal) {
  using Vec = typename VectorLane<T>::type;
  constexpr size_t kLanes = LaneTraits<Vec>::kLanes;
  const size_t groups = (count + kLanes - 1) / kLanes;
  const size_t memory_size = setup.buffer_size + setup.scratch_size;
  const auto parallelism = static_cast<size_t>(concurrency::ThreadPool::DegreeOfParallelism(thread_pool));
  LaneBufferSlots<Vec> buffers(allocator, memory_size, std::max<size_t>(1, std::min(groups, parallelism)));

  // Charge a group the work of kLanes transforms. SIMD lanes speed up the arithmetic, but a group still takes several
  // times longer than one scalar transform; a lower estimate makes the pool use too few threads for small lengths.
  const double work = static_cast<double>(std::max(setup.n, setup.scratch_size));
  const TensorOpCost cost{static_cast<double>(kLanes * setup.samples * sizeof(U)),
                          static_cast<double>(kLanes * setup.output_size * 2 * sizeof(T)),
                          static_cast<double>(kLanes) * 5.0 * work * std::max(1.0, std::log2(work))};
  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(groups), cost, [&](std::ptrdiff_t first, std::ptrdiff_t last) {
        const size_t slot = buffers.Acquire();
        std::unique_ptr<Complex<Vec>[]> fallback;
        Complex<Vec>* memory = nullptr;
        if (slot < buffers.Slots()) {
          memory = buffers.Buffer(slot);
        } else {
          fallback = std::make_unique<Complex<Vec>[]>(memory_size);
          memory = fallback.get();
        }
        const U* xs[kLanes];
        T* ys[kLanes];
        for (std::ptrdiff_t g = first; g < last; ++g) {
          const size_t begin = static_cast<size_t>(g) * kLanes;
          const size_t active = std::min(kLanes, count - begin);
          for (size_t l = 0; l < kLanes; ++l) {
            if (l < active) {
              get_signal(begin + l, xs[l], ys[l]);
            } else {
              xs[l] = xs[0];
              ys[l] = nullptr;
            }
          }
          RunDft<T, U, Vec>(setup, xs, x_stride, window, ys, y_step, memory, memory + setup.buffer_size, active);
        }
        if (slot < buffers.Slots()) {
          buffers.Release(slot);
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
  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&allocator));
  RunDfts<T, U>(ctx->GetOperatorThreadPool(), std::move(allocator), setup, total_dfts, x_stride, nullptr, y_step,
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
  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&allocator));
  RunDfts<T, U>(ctx->GetOperatorThreadPool(), std::move(allocator), setup, narrow<size_t>(batch_size) * frames, 1,
                window_data, output_components, [&](size_t i, const U*& x, T*& y) {
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
