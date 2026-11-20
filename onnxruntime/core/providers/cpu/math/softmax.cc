// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/softmax.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>

#include "core/common/safeint.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/tensor/transpose.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Softmax,
    1,
    10,
    float,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Softmax,
    11,
    12,
    float,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

// Opset 13 changed the semantic meaning of the axis attribute.
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Softmax,
    13,
    float,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Softmax,
    1,
    10,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Softmax,
    11,
    12,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);

// Opset 13 changed the semantic meaning of the axis attribute.
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Softmax,
    13,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    LogSoftmax,
    1,
    10,
    float,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    LogSoftmax,
    11,
    12,
    float,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

// Opset 13 changed the semantic meaning of the axis attribute.
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    LogSoftmax,
    13,
    float,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    LogSoftmax,
    1,
    10,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    LogSoftmax,
    11,
    12,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);

// Opset 13 changed the semantic meaning of the axis attribute.
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    LogSoftmax,
    13,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);

// opset-12 and below
template <typename T>
Status Softmax<T>::ComputeImpl(const Tensor& input, Tensor& output, size_t axis,
                               concurrency::ThreadPool* thread_pool,
                               const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* mlas_backend_kernel_selector_config) const {
  const auto& X_shape = input.Shape();
  const size_t N = onnxruntime::narrow<size_t>(X_shape.SizeToDimension(axis));
  const size_t D = onnxruntime::narrow<size_t>(X_shape.SizeFromDimension(axis));

  return SoftmaxCPU<T>(N, D, input.Data<T>(), output.MutableData<T>(), log_softmax_, thread_pool, mlas_backend_kernel_selector_config);
}

namespace {

constexpr size_t kMaxBlock = 256;
constexpr size_t kMinBlock = 64;

template <typename T>
void ExpInPlace(T* data, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    data[i] = std::exp(data[i]);
  }
}

template <>
void ExpInPlace<float>(float* data, size_t count) {
  MlasComputeExp(data, data, count);
}

// LogSoftmax of `len` columns spaced `inner` apart in rows of `axis_dim`, reading X twice instead of three times: a
// single pass keeps a running max and a sum of exp(x - max) rescaled whenever the max grows, then Y = X - (max +
// log(sum)). The rescale is skipped for rows where no column's max changed. scratch holds 2 * len values. Y may
// alias X as each element of Y is written after its last read from X.
template <typename T>
void LogSoftmaxColumnsOnline(const T* x, T* y, size_t axis_dim, size_t inner, size_t len, T* max_vals, T* sums,
                             T* scratch) {
  std::copy_n(x, len, max_vals);
  std::fill_n(sums, len, T{1});
  for (size_t a = 1; a < axis_dim; ++a) {
    const T* x_row = x + a * inner;
    bool max_changed = false;
    for (size_t j = 0; j < len; ++j) {
      max_changed |= x_row[j] > max_vals[j];
    }
    if (max_changed) {
      for (size_t j = 0; j < len; ++j) {
        const T m = std::max(max_vals[j], x_row[j]);
        scratch[j] = max_vals[j] - m;
        scratch[len + j] = x_row[j] - m;
        max_vals[j] = m;
      }
      ExpInPlace(scratch, 2 * len);
      for (size_t j = 0; j < len; ++j) {
        sums[j] = sums[j] * scratch[j] + scratch[len + j];
      }
    } else {
      for (size_t j = 0; j < len; ++j) {
        scratch[j] = x_row[j] - max_vals[j];
      }
      ExpInPlace(scratch, len);
      for (size_t j = 0; j < len; ++j) {
        sums[j] += scratch[j];
      }
    }
  }

  for (size_t j = 0; j < len; ++j) {
    if constexpr (std::is_same_v<T, float>) {
      sums[j] = max_vals[j] + std::log(sums[j]);
    } else {
      sums[j] = max_vals[j] + std::log(std::fmax(sums[j], 1e-20));
    }
  }
  for (size_t a = 0; a < axis_dim; ++a) {
    const T* x_row = x + a * inner;
    T* y_row = y + a * inner;
    for (size_t j = 0; j < len; ++j) {
      y_row[j] = x_row[j] - sums[j];
    }
  }
}

// Softmax/LogSoftmax along the middle axis of a tensor viewed as [outer, axis_dim, inner] with inner > 1.
// Each task reduces a block of contiguous inner columns across axis_dim, so every access is unit-stride and no
// transposed copies are needed. Softmax arithmetic matches SoftmaxCPU for each element type. Y may alias X: an
// element of Y is only written after the last read of the same element of X.
template <typename T>
void SoftmaxInnerStrided(const T* X, T* Y, size_t outer, size_t axis_dim, size_t inner, bool log_softmax,
                         concurrency::ThreadPool* thread_pool) {
  // Narrow the column blocks when there are too few of them to keep every thread busy.
  const size_t target_tasks = 4 * static_cast<size_t>(concurrency::ThreadPool::DegreeOfParallelism(thread_pool));
  size_t block = kMaxBlock;
  while (block > kMinBlock && outer * ((inner + block - 1) / block) < target_tasks) {
    block /= 2;
  }
  const size_t blocks_per_outer = (inner + block - 1) / block;
  const size_t num_tasks = SafeInt<size_t>(outer) * blocks_per_outer;
  const double block_elements = static_cast<double>(axis_dim) * static_cast<double>(std::min(inner, block));
  const TensorOpCost cost{3.0 * block_elements * sizeof(T), block_elements * sizeof(T), 24.0 * block_elements};

  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(num_tasks), cost,
      [=](std::ptrdiff_t first, std::ptrdiff_t last) {
        T max_vals[kMaxBlock];
        T sums[kMaxBlock];
        T scratch[2 * kMaxBlock];

        for (std::ptrdiff_t task = first; task < last; ++task) {
          const size_t o = static_cast<size_t>(task) / blocks_per_outer;
          const size_t col = (static_cast<size_t>(task) % blocks_per_outer) * block;
          const size_t len = std::min(block, inner - col);
          const size_t base = o * axis_dim * inner + col;
          const T* x = X + base;
          T* y = Y + base;

          if (log_softmax) {
            LogSoftmaxColumnsOnline(x, y, axis_dim, inner, len, max_vals, sums, scratch);
            continue;
          }

          std::copy_n(x, len, max_vals);
          for (size_t a = 1; a < axis_dim; ++a) {
            const T* x_row = x + a * inner;
            for (size_t j = 0; j < len; ++j) {
              max_vals[j] = std::max(max_vals[j], x_row[j]);
            }
          }

          std::fill_n(sums, len, T{0});
          for (size_t a = 0; a < axis_dim; ++a) {
            const T* x_row = x + a * inner;
            T* exp_row = y + a * inner;
            for (size_t j = 0; j < len; ++j) {
              exp_row[j] = x_row[j] - max_vals[j];
            }
            ExpInPlace(exp_row, len);
            for (size_t j = 0; j < len; ++j) {
              sums[j] += exp_row[j];
            }
          }

          if constexpr (std::is_same_v<T, float>) {
            for (size_t j = 0; j < len; ++j) {
              sums[j] = 1.0f / sums[j];
            }
          }
          for (size_t a = 0; a < axis_dim; ++a) {
            T* y_row = y + a * inner;
            for (size_t j = 0; j < len; ++j) {
              if constexpr (std::is_same_v<T, float>) {
                y_row[j] *= sums[j];
              } else {
                y_row[j] /= sums[j];
              }
            }
          }
        }
      });
}

// Softmax/LogSoftmax along a non-innermost axis by swapping it with the innermost axis, using SoftmaxCPU on the
// transposed copy and transposing back.
template <typename T>
Status SoftmaxViaTranspose(const Tensor& input, Tensor& output, size_t axis, bool log_softmax,
                           concurrency::ThreadPool* thread_pool, OpKernelContext* ctx,
                           const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* mlas_backend_kernel_selector_config) {
  const auto& X_shape = input.Shape();
  const size_t rank = X_shape.NumDimensions();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));

  InlinedVector<size_t> permutation(rank);
  std::iota(permutation.begin(), permutation.end(), size_t{0});
  permutation[axis] = rank - 1;
  permutation[rank - 1] = axis;

  TensorShapeVector transposed_dims;
  transposed_dims.reserve(rank);
  for (auto e : permutation) {
    transposed_dims.push_back(X_shape[e]);
  }
  const TensorShape transposed_shape(transposed_dims);

  Tensor transposed_input(input.DataType(), transposed_shape, alloc);
  ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(permutation, input, transposed_input));
  Tensor transposed_output(output.DataType(), transposed_shape, alloc);

  const size_t N = onnxruntime::narrow<size_t>(transposed_shape.SizeToDimension(rank - 1));
  const size_t D = onnxruntime::narrow<size_t>(transposed_shape[rank - 1]);
  ORT_RETURN_IF_ERROR(SoftmaxCPU<T>(N, D, transposed_input.Data<T>(), transposed_output.MutableData<T>(), log_softmax,
                                    thread_pool, mlas_backend_kernel_selector_config));

  return TransposeBase::DoTranspose(permutation, transposed_output, output);
}

}  // namespace

// opset-13 and above
template <typename T>
Status Softmax<T>::ComputeImplOpset13(const Tensor& input, Tensor& output, size_t axis,
                                      concurrency::ThreadPool* thread_pool, OpKernelContext* ctx,
                                      const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* mlas_backend_kernel_selector_config) const {
  // From opset 13 the softmax is computed along `axis` only, rather than over all dims from `axis` onwards.
  const auto& X_shape = input.Shape();
  const size_t outer = onnxruntime::narrow<size_t>(X_shape.SizeToDimension(axis));
  const size_t axis_dim = onnxruntime::narrow<size_t>(X_shape[axis]);
  const size_t inner = onnxruntime::narrow<size_t>(X_shape.SizeFromDimension(axis + 1));

  if (inner == 1) {
    return SoftmaxCPU<T>(outer, axis_dim, input.Data<T>(), output.MutableData<T>(), log_softmax_, thread_pool,
                         mlas_backend_kernel_selector_config);
  }

  // With few inner columns the strided kernel is dominated by per-row overhead, and transposing to the vectorized
  // innermost-axis kernel is faster.
  if (inner < kMinBlock) {
    return SoftmaxViaTranspose<T>(input, output, axis, log_softmax_, thread_pool, ctx,
                                  mlas_backend_kernel_selector_config);
  }

  SoftmaxInnerStrided<T>(input.Data<T>(), output.MutableData<T>(), outer, axis_dim, inner, log_softmax_,
                         thread_pool);
  return Status::OK();
}

// compute method of Softmax
template <typename T>
Status Softmax<T>::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  size_t rank = X_shape.NumDimensions();
  auto* Y = ctx->Output(0, X_shape);

  // edge case. one or more dims with value of 0. nothing to do
  if (X_shape.Size() == 0) {
    return Status::OK();
  }

  const size_t axis = static_cast<size_t>(HandleNegativeAxis(axis_, rank));
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  if (opset_ < 13) {
    return ComputeImpl(*X, *Y, axis, thread_pool, &mlas_backend_kernel_selector_config_);
  } else {
    return ComputeImplOpset13(*X, *Y, axis, thread_pool, ctx, &mlas_backend_kernel_selector_config_);
  }
}

}  // namespace onnxruntime
