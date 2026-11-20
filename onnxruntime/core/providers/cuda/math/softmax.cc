// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/softmax.h"

#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/shared_inc/accumulation_type.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename TOut, bool is_log_softmax>
Status SoftMaxComputeHelper(
    SoftmaxComputeStreamT stream,
    const T* X,
    const TensorShape& input_shape,
    TOut* Y,
    int64_t axis) {
  typedef typename ToCudaType<T>::MappedType CudaT_IN;
  typedef typename ToCudaType<TOut>::MappedType CudaT_OUT;
  typedef typename ToCudaType<T>::MappedType CudaT_ACCUM;

  int64_t N = input_shape.SizeToDimension(axis);
  int64_t D = input_shape.SizeFromDimension(axis);
  auto Y_data = reinterpret_cast<CudaT_OUT*>(Y);
  auto X_data = reinterpret_cast<const CudaT_IN*>(X);
  // According to nsight compute profiling, softmax_warp_forward_resource_efficient is better than dispatch_blockwise_softmax_forward when 1024 < D <=2048 and N >= 8192.
  const bool use_softmax_warp_forward_resource_efficient = 1024 < D && D <= 2048 && D * sizeof(T) <= 4096 && N >= 8192;
  if ((D <= 1024 && D * sizeof(T) <= 4096) or use_softmax_warp_forward_resource_efficient) {
    return dispatch_warpwise_softmax_forward<
        CudaT_IN, CudaT_OUT, AccumulationType_t<CudaT_ACCUM>, is_log_softmax>(
        stream, Y_data, X_data, gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(N));
  }

  return dispatch_blockwise_softmax_forward<CudaT_IN, CudaT_OUT, AccumulationType_t<CudaT_ACCUM>, is_log_softmax>(
      stream, Y_data, X_data, gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(D), gsl::narrow_cast<int>(D),
      gsl::narrow_cast<int>(N));
}

#define SPECIALIZED_SOFTMAX_HELPER_IMPL(T, TOut)                                                         \
  template Status SoftMaxComputeHelper<T, TOut, false>(SoftmaxComputeStreamT stream, const T* input,     \
                                                       const TensorShape& shape, TOut* Y, int64_t axis); \
  template Status SoftMaxComputeHelper<T, TOut, true>(SoftmaxComputeStreamT stream, const T* input,      \
                                                      const TensorShape& shape, TOut* Y, int64_t axis);

SPECIALIZED_SOFTMAX_HELPER_IMPL(MLFloat16, float)
SPECIALIZED_SOFTMAX_HELPER_IMPL(float, float)
SPECIALIZED_SOFTMAX_HELPER_IMPL(double, double)
SPECIALIZED_SOFTMAX_HELPER_IMPL(MLFloat16, MLFloat16)
SPECIALIZED_SOFTMAX_HELPER_IMPL(BFloat16, BFloat16)

#define REGISTER_KERNEL_TYPED(T)                                                           \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      Softmax,                                                                             \
      kOnnxDomain,                                                                         \
      1, 10,                                                                               \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);                                                                         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      Softmax,                                                                             \
      kOnnxDomain,                                                                         \
      11, 12,                                                                              \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);                                                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      Softmax,                                                                             \
      kOnnxDomain,                                                                         \
      13,                                                                                  \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);                                                                         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      LogSoftmax,                                                                          \
      kOnnxDomain,                                                                         \
      1, 10,                                                                               \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);                                                                         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      LogSoftmax,                                                                          \
      kOnnxDomain,                                                                         \
      11, 12,                                                                              \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);                                                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      LogSoftmax,                                                                          \
      kOnnxDomain,                                                                         \
      13,                                                                                  \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Softmax<T>);

template <typename T>
Status Softmax<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* X = ctx->Input<Tensor>(0);
  const TensorShape& input_shape{X->Shape()};
  size_t rank = input_shape.NumDimensions();
  Tensor* Y = ctx->Output(0, input_shape);

  // special case when there is a dim value of 0 in the shape.
  if (input_shape.Size() == 0)
    return Status::OK();

  // handle negative and enforce axis is valid
  const size_t axis = static_cast<size_t>(HandleNegativeAxis(axis_, rank));

  // Before opset 13 the input is coerced to 2D at `axis`, which is always a contiguous [N, D] softmax.
  // From opset 13 softmax runs over the single dim `axis`, i.e. over the middle dim of [outer, axis_dim, inner].
  // When inner == 1 that is the same contiguous layout; otherwise use the strided kernel, which avoids
  // transposing the input and output.
  const int64_t inner = (opset_ >= 13) ? input_shape.SizeFromDimension(axis + 1) : 1;
  auto compute_stream = Stream(ctx);
  if (inner == 1) {
    if (log_softmax_) {
      return SoftMaxComputeHelper<T, T, true>(compute_stream, X->Data<T>(), input_shape, Y->MutableData<T>(),
                                              static_cast<int64_t>(axis));
    }
    return SoftMaxComputeHelper<T, T, false>(compute_stream, X->Data<T>(), input_shape, Y->MutableData<T>(),
                                             static_cast<int64_t>(axis));
  }

  const int64_t outer = input_shape.SizeToDimension(axis);
  const int64_t axis_dim = input_shape[axis];
  auto X_data = reinterpret_cast<const CudaT*>(X->Data<T>());
  auto Y_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());
  if (log_softmax_) {
    return dispatch_strided_softmax_forward<CudaT, CudaT, AccumulationType_t<CudaT>, true>(
        compute_stream, Y_data, X_data, outer, axis_dim, inner);
  }
  return dispatch_strided_softmax_forward<CudaT, CudaT, AccumulationType_t<CudaT>, false>(
      compute_stream, Y_data, X_data, outer, axis_dim, inner);
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status Softmax<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)
SPECIALIZED_COMPUTE(BFloat16)

}  // namespace cuda
}  // namespace onnxruntime
