// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/signal/dft.h"

#include <algorithm>
#include <cstring>

#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/providers/cuda/signal/dft_impl.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DFT,
    kOnnxDomain,
    17, 19,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T1", BuildKernelDefConstraints<float, double>())
        .TypeConstraint("T2", BuildKernelDefConstraints<int32_t, int64_t>()),
    DFT);

ONNX_OPERATOR_KERNEL_EX(
    DFT,
    kOnnxDomain,
    20,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .TypeConstraint("T1", BuildKernelDefConstraints<float, double>())
        .TypeConstraint("T2", BuildKernelDefConstraints<int32_t, int64_t>()),
    DFT);

ONNX_OPERATOR_KERNEL_EX(
    STFT,
    kOnnxDomain,
    17,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .TypeConstraint("T1", BuildKernelDefConstraints<float, double>())
        .TypeConstraint("T2", BuildKernelDefConstraints<int32_t, int64_t>()),
    STFT);

namespace {

int64_t GetScalarInt64(const Tensor& tensor) {
  ORT_ENFORCE(tensor.Shape().Size() == 1, "Expected a scalar value.");
  if (tensor.IsDataType<int64_t>()) {
    return *tensor.Data<int64_t>();
  }
  return static_cast<int64_t>(*tensor.Data<int32_t>());
}

template <typename T>
struct CufftDataTypes;

template <>
struct CufftDataTypes<float> {
  static constexpr cudaDataType kReal = CUDA_R_32F;
  static constexpr cudaDataType kComplex = CUDA_C_32F;
};

template <>
struct CufftDataTypes<double> {
  static constexpr cudaDataType kReal = CUDA_R_64F;
  static constexpr cudaDataType kComplex = CUDA_C_64F;
};

// Runs a batched 1-D transform of length n on contiguous rows. Real input is only used for forward R2C, which
// writes the n / 2 + 1 non-redundant bins of each row.
template <typename T>
Status ExecuteFft(contrib::cuda::CuFFTPlanCache& cache, cudaStream_t stream, bool real_input, int64_t n,
                  int64_t batch, const T* input, T* output, int direction) {
  contrib::cuda::FFTState state;
  std::memset(&state, 0, sizeof(state));
  state.signal_ndim = 1;
  state.signal_dims[0] = n;
  state.itype = real_input ? CufftDataTypes<T>::kReal : CufftDataTypes<T>::kComplex;
  state.otype = CufftDataTypes<T>::kComplex;
  state.batch_size = batch;
  state.exec_type = CufftDataTypes<T>::kComplex;

  const contrib::cuda::CufftPlanInfo plan_info = cache.TryEmplaceValue(state);
  CUFFT_RETURN_IF_ERROR(cufftSetStream(plan_info.plan, stream));
  CUFFT_RETURN_IF_ERROR(cufftXtExec(plan_info.plan, const_cast<T*>(input), output, direction));
  return Status::OK();
}

}  // namespace

template <typename T>
Status DFT::ComputeImpl(OpKernelContext* context, const Tensor& input, Tensor& output, int64_t axis,
                        int64_t dft_length) const {
  const auto& input_shape = input.Shape();
  const size_t rank = input_shape.NumDimensions();
  const bool complex_input = input_shape[rank - 1] == 2;
  const int64_t num_samples = input_shape[onnxruntime::narrow<size_t>(axis)];
  const int64_t outer = input_shape.SizeToDimension(onnxruntime::narrow<size_t>(axis));
  const int64_t inner = input_shape.SizeFromDimension(onnxruntime::narrow<size_t>(axis) + 1) / input_shape[rank - 1];
  const bool irfft = is_inverse_ && is_onesided_;
  const int64_t output_length = output.Shape()[onnxruntime::narrow<size_t>(axis)];
  const int direction = is_inverse_ ? CUFFT_INVERSE : CUFFT_FORWARD;

  cudaStream_t stream = Stream(context);
  const T* x = input.Data<T>();
  T* y = output.MutableData<T>();

  // Contiguous transforms without padding or truncation write straight into the output tensor.
  if (inner == 1 && num_samples == dft_length) {
    if (!complex_input && !is_inverse_ && is_onesided_) {
      return ExecuteFft(cufft_cache_, stream, true, dft_length, outer, x, y, CUFFT_FORWARD);
    }
    if (complex_input && !is_onesided_) {
      ORT_RETURN_IF_ERROR(ExecuteFft(cufft_cache_, stream, false, dft_length, outer, x, y, direction));
      if (is_inverse_) {
        DftFinalizeRows(stream, y, outer, dft_length, 1, dft_length, false, T(1) / static_cast<T>(dft_length), y);
      }
      return CUDA_CALL(cudaGetLastError());
    }
  }

  const int64_t rows = outer * inner;
  auto buffer = GetScratchBuffer<T>(SafeInt<size_t>(rows) * dft_length * 2, GetComputeStream(context));

  int64_t irfft_conj_end = 0;
  if (irfft) {
    const int64_t used_samples = std::min(num_samples, dft_length);
    irfft_conj_end = dft_length % 2 == 0 ? used_samples - 1 : used_samples;
  }
  DftPrepareRows(stream, x, complex_input, outer, num_samples, inner, dft_length, irfft_conj_end, buffer.get());
  ORT_RETURN_IF_ERROR(ExecuteFft(cufft_cache_, stream, false, dft_length, rows, buffer.get(), buffer.get(),
                                 direction));
  const T scale = is_inverse_ ? T(1) / static_cast<T>(dft_length) : T(1);
  DftFinalizeRows(stream, buffer.get(), outer, dft_length, inner, output_length, irfft, scale, y);
  return CUDA_CALL(cudaGetLastError());
}

Status DFT::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* dft_length_tensor = context->Input<Tensor>(1);
  const auto& input_shape = input->Shape();
  const int64_t rank = static_cast<int64_t>(input_shape.NumDimensions());
  ORT_RETURN_IF(rank < 2, "DFT input must have at least 2 dimensions.");

  int64_t axis = axis_;
  if (opset_ >= 20) {
    const Tensor* axis_tensor = context->Input<Tensor>(2);
    if (axis_tensor != nullptr) {
      axis = GetScalarInt64(*axis_tensor);
    }
  }
  axis = HandleNegativeAxis(axis, rank);
  ORT_RETURN_IF(axis == rank - 1, "DFT axis must refer to a signal dimension, not the last (real/imag) dimension.");

  const int64_t components = input_shape[onnxruntime::narrow<size_t>(rank - 1)];
  const bool irfft = is_inverse_ && is_onesided_;
  if (irfft) {
    ORT_RETURN_IF(components != 2,
                  "Inverse one-sided DFT (IRFFT) requires complex-valued input (last dimension must be 2)");
  } else {
    ORT_RETURN_IF(components != 1 && components != 2,
                  "Input layout not supported. The input tensor must have a last dimension of 1 (real) or 2 "
                  "(complex).");
  }

  const int64_t num_samples = input_shape[onnxruntime::narrow<size_t>(axis)];
  int64_t dft_length = irfft ? (num_samples - 1) * 2 : num_samples;
  if (dft_length_tensor != nullptr) {
    ORT_RETURN_IF(!dft_length_tensor->Shape().IsScalar(), "dft_length must be a scalar value.");
    dft_length = GetScalarInt64(*dft_length_tensor);
    ORT_RETURN_IF(dft_length <= 0, "dft_length must be greater than zero.");
  }
  ORT_RETURN_IF(dft_length < 0, "DFT length must not be negative.");

  const int64_t output_length = (is_onesided_ && !is_inverse_) ? (dft_length >> 1) + 1 : dft_length;
  TensorShape output_shape(input_shape);
  output_shape[onnxruntime::narrow<size_t>(rank - 1)] = irfft ? 1 : 2;
  output_shape[onnxruntime::narrow<size_t>(axis)] = output_length;
  Tensor* output = context->Output(0, output_shape);
  if (output_shape.Size() == 0) {
    return Status::OK();
  }
  ORT_RETURN_IF(num_samples == 0,
                "DFT input signal dimension must be greater than zero when the output is non-empty.");

  if (input->IsDataType<float>()) {
    return ComputeImpl<float>(context, *input, *output, axis, dft_length);
  }
  return ComputeImpl<double>(context, *input, *output, axis, dft_length);
}

template <typename T>
Status STFT::ComputeImpl(OpKernelContext* context, const Tensor& signal, const Tensor* window, Tensor& output,
                         int64_t frame_step, int64_t frame_length) const {
  const auto& signal_shape = signal.Shape();
  const bool complex_input = signal_shape.NumDimensions() == 3 && signal_shape[2] == 2;
  const int64_t batch = signal_shape[0];
  const int64_t signal_length = signal_shape[1];
  const int64_t num_frames = output.Shape()[1];
  const int64_t output_bins = output.Shape()[2];
  const int64_t rows = batch * num_frames;

  cudaStream_t stream = Stream(context);
  const T* x = signal.Data<T>();
  const T* w = window != nullptr ? window->Data<T>() : nullptr;
  T* y = output.MutableData<T>();

  if (!complex_input && is_onesided_) {
    // R2C emits exactly the one-sided [frames, bins, 2] layout, so cuFFT writes the output tensor directly.
    auto frames = GetScratchBuffer<T>(SafeInt<size_t>(rows) * frame_length, GetComputeStream(context));
    StftFrames(stream, x, false, false, batch, signal_length, num_frames, frame_step, frame_length, w, frames.get());
    ORT_RETURN_IF_ERROR(ExecuteFft(cufft_cache_, stream, true, frame_length, rows, frames.get(), y, CUFFT_FORWARD));
    return CUDA_CALL(cudaGetLastError());
  }

  auto frames = GetScratchBuffer<T>(SafeInt<size_t>(rows) * frame_length * 2, GetComputeStream(context));
  StftFrames(stream, x, complex_input, true, batch, signal_length, num_frames, frame_step, frame_length, w,
             frames.get());
  if (!is_onesided_) {
    ORT_RETURN_IF_ERROR(ExecuteFft(cufft_cache_, stream, false, frame_length, rows, frames.get(), y, CUFFT_FORWARD));
  } else {
    ORT_RETURN_IF_ERROR(
        ExecuteFft(cufft_cache_, stream, false, frame_length, rows, frames.get(), frames.get(), CUFFT_FORWARD));
    DftFinalizeRows(stream, frames.get(), rows, frame_length, 1, output_bins, false, T(1), y);
  }
  return CUDA_CALL(cudaGetLastError());
}

Status STFT::ComputeInternal(OpKernelContext* context) const {
  const Tensor* signal = context->Input<Tensor>(0);
  const Tensor* frame_step_tensor = context->Input<Tensor>(1);
  const Tensor* window = context->Input<Tensor>(2);
  const Tensor* frame_length_tensor = context->Input<Tensor>(3);

  const auto& signal_shape = signal->Shape();
  const size_t signal_rank = signal_shape.NumDimensions();
  ORT_RETURN_IF(signal_rank != 2 && signal_rank != 3, "STFT signal must have shape [batch, length] or "
                                                      "[batch, length, 1 or 2].");
  const int64_t components = signal_rank == 3 ? signal_shape[2] : 1;
  ORT_RETURN_IF(components != 1 && components != 2, "signal shape must end in 1 (real) or 2 (real, imaginary).");

  const int64_t frame_step = GetScalarInt64(*frame_step_tensor);
  ORT_RETURN_IF_NOT(frame_step > 0, "frame_step must be greater than zero.");

  int64_t window_length = -1;
  if (window != nullptr) {
    ORT_RETURN_IF(window->Shape().NumDimensions() != 1, "STFT window must be 1-D.");
    window_length = window->Shape()[0];
  }
  int64_t frame_length = -1;
  if (frame_length_tensor != nullptr) {
    frame_length = GetScalarInt64(*frame_length_tensor);
  }
  ORT_RETURN_IF(window != nullptr && frame_length_tensor != nullptr && frame_length != window_length,
                "If both frame_length and window are set, then the size of the window must be equal to the "
                "frame_length.");

  const int64_t window_size = window != nullptr ? window_length : frame_length;
  const int64_t signal_length = signal_shape[1];
  ORT_RETURN_IF(window_size <= 0, "STFT requires a non-empty window or a positive frame_length.");
  ORT_RETURN_IF(window_size > signal_length, "Ensure that the dft size is smaller than the signal.");

  const int64_t num_frames = (signal_length - window_size) / frame_step + 1;
  const int64_t output_bins = is_onesided_ ? (window_size >> 1) + 1 : window_size;
  Tensor* output = context->Output(0, TensorShape({signal_shape[0], num_frames, output_bins, 2}));
  if (output->Shape().Size() == 0) {
    return Status::OK();
  }

  if (signal->IsDataType<float>()) {
    return ComputeImpl<float>(context, *signal, window, *output, frame_step, window_size);
  }
  return ComputeImpl<double>(context, *signal, window, *output, frame_step, window_size);
}

}  // namespace cuda
}  // namespace onnxruntime
