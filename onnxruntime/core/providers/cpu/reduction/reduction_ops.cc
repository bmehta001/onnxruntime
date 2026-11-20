// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/reduction/reduction_ops.h"

#include "core/common/inlined_containers.h"
#include "core/common/narrow.h"
#include "core/common/span_utils.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/common.h"

#if (defined(_M_AMD64) && !defined(_M_ARM64EC)) || defined(__x86_64__)
#include <emmintrin.h>
#define REDUCTION_OPS_USE_SSE2 1
#else
#define REDUCTION_OPS_USE_SSE2 0
#endif
// TODO: fix the warnings
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26451)
#endif
using namespace std;
namespace onnxruntime {
#define REGISTER_UNARY_ELEMENTWISE_KERNEL(x, sinceVersion)                            \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                     \
      x,                                                                              \
      sinceVersion,                                                                   \
      float,                                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),   \
      x<float>);                                                                      \
                                                                                      \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                     \
      x,                                                                              \
      sinceVersion,                                                                   \
      int32_t,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()), \
      x<int32_t>);

#define REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(x, startVer, endVer)              \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                           \
      x,                                                                              \
      startVer,                                                                       \
      endVer,                                                                         \
      float,                                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),   \
      x<float>);                                                                      \
                                                                                      \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                           \
      x,                                                                              \
      startVer,                                                                       \
      endVer,                                                                         \
      int32_t,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()), \
      x<int32_t>);

#define REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(x, sinceVersion)               \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                    \
      x,                                                                             \
      sinceVersion,                                                                  \
      double,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()), \
      x<double>);

#define REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(x, startVer, endVer) \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                          \
      x,                                                                             \
      startVer,                                                                      \
      endVer,                                                                        \
      double,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()), \
      x<double>);

#define REGISTER_UNARY_ELEMENTWISE_KERNEL_INT64_ONLY(x, sinceVersion)                 \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                     \
      x,                                                                              \
      sinceVersion,                                                                   \
      int64_t,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()), \
      x<int64_t>);

#define REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(x, startVer, endVer)   \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                           \
      x,                                                                              \
      startVer,                                                                       \
      endVer,                                                                         \
      int64_t,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()), \
      x<int64_t>);

#define REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT8_ONLY(x, startVer, endVer)   \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                          \
      x,                                                                             \
      startVer,                                                                      \
      endVer,                                                                        \
      int8_t,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>()), \
      x<int8_t>);

#define REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_UINT8_ONLY(x, startVer, endVer)   \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                           \
      x,                                                                              \
      startVer,                                                                       \
      endVer,                                                                         \
      uint8_t,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()), \
      x<uint8_t>);

#define REGISTER_UNARY_ELEMENTWISE_KERNEL_INT8_ONLY(x, sinceVersion)                 \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                    \
      x,                                                                             \
      sinceVersion,                                                                  \
      int8_t,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>()), \
      x<int8_t>);

#define REGISTER_UNARY_ELEMENTWISE_KERNEL_UINT8_ONLY(x, sinceVersion)                 \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                     \
      x,                                                                              \
      sinceVersion,                                                                   \
      uint8_t,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()), \
      x<uint8_t>);

#define REGISTER_UNARY_ELEMENTWISE_KERNEL_BOOL_ONLY(x, sinceVersion)               \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                  \
      x,                                                                           \
      sinceVersion,                                                                \
      bool,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()), \
      x<bool>);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceL1, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceL1, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceL1, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceL1, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceL1, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceL1, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceL1, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceL1, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceL1, 13, 17);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceL1, 18);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ReduceL1, 18);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT64_ONLY(ReduceL1, 18);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceL2, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceL2, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceL2, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceL2, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceL2, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceL2, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceL2, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceL2, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceL2, 13, 17);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceL2, 18);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ReduceL2, 18);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT64_ONLY(ReduceL2, 18);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceLogSum, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceLogSum, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceLogSum, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceLogSum, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceLogSum, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceLogSum, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceLogSum, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceLogSum, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceLogSum, 13, 17);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceLogSum, 18);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ReduceLogSum, 18);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT64_ONLY(ReduceLogSum, 18);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceLogSumExp, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceLogSumExp, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceLogSumExp, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceLogSumExp, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceLogSumExp, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceLogSumExp, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceLogSumExp, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceLogSumExp, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceLogSumExp, 13, 17);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceLogSumExp, 18);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ReduceLogSumExp, 18);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT64_ONLY(ReduceLogSumExp, 18);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMax, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMax, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMax, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMax, 11, 11);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMax, 11, 11);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMax, 11, 11);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMax, 12, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMax, 12, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMax, 12, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT8_ONLY(ReduceMax, 12, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_UINT8_ONLY(ReduceMax, 12, 12);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMax, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMax, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMax, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT8_ONLY(ReduceMax, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_UINT8_ONLY(ReduceMax, 13, 17);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMax, 18, 19);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMax, 18, 19);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMax, 18, 19);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT8_ONLY(ReduceMax, 18, 19);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_UINT8_ONLY(ReduceMax, 18, 19);

REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceMax, 20);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT64_ONLY(ReduceMax, 20);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ReduceMax, 20);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT8_ONLY(ReduceMax, 20);
REGISTER_UNARY_ELEMENTWISE_KERNEL_UINT8_ONLY(ReduceMax, 20);
REGISTER_UNARY_ELEMENTWISE_KERNEL_BOOL_ONLY(ReduceMax, 20);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMean, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMean, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMean, 13, 17);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceMean, 18);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMean, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMean, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMean, 13, 17);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ReduceMean, 18);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMean, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMean, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMean, 13, 17);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT64_ONLY(ReduceMean, 18);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMin, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMin, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMin, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMin, 11, 11);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMin, 11, 11);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMin, 11, 11);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMin, 12, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMin, 12, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMin, 12, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT8_ONLY(ReduceMin, 12, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_UINT8_ONLY(ReduceMin, 12, 12);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMin, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMin, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMin, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT8_ONLY(ReduceMin, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_UINT8_ONLY(ReduceMin, 13, 17);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceMin, 18, 19);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceMin, 18, 19);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceMin, 18, 19);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT8_ONLY(ReduceMin, 18, 19);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_UINT8_ONLY(ReduceMin, 18, 19);

REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceMin, 20);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT64_ONLY(ReduceMin, 20);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ReduceMin, 20);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT8_ONLY(ReduceMin, 20);
REGISTER_UNARY_ELEMENTWISE_KERNEL_UINT8_ONLY(ReduceMin, 20);
REGISTER_UNARY_ELEMENTWISE_KERNEL_BOOL_ONLY(ReduceMin, 20);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceProd, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceProd, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceProd, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceProd, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceProd, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceProd, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceProd, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceProd, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceProd, 13, 17);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceProd, 18);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ReduceProd, 18);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT64_ONLY(ReduceProd, 18);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceSum, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceSum, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceSum, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceSum, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceSum, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceSum, 11, 12);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceSum, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT64_ONLY(ReduceSum, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ReduceSum, 13);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceSumSquare, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceSumSquare, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceSumSquare, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceSumSquare, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceSumSquare, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceSumSquare, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ReduceSumSquare, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ReduceSumSquare, 13, 17);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ReduceSumSquare, 13, 17);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ReduceSumSquare, 18);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ReduceSumSquare, 18);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT64_ONLY(ReduceSumSquare, 18);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ArgMax, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ArgMax, 1, 10)
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT8_ONLY(ArgMax, 1, 10)
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ArgMax, 1, 10)
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_UINT8_ONLY(ArgMax, 1, 10)
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ArgMax, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ArgMax, 11, 12)
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT8_ONLY(ArgMax, 11, 12)
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ArgMax, 11, 12)
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_UINT8_ONLY(ArgMax, 11, 12)
REGISTER_UNARY_ELEMENTWISE_KERNEL(ArgMax, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ArgMax, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT64_ONLY(ArgMax, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT8_ONLY(ArgMax, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_UINT8_ONLY(ArgMax, 13);

REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ArgMin, 1, 10);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ArgMin, 1, 10)
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT8_ONLY(ArgMin, 1, 10)
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ArgMin, 1, 10)
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_UINT8_ONLY(ArgMin, 1, 10)
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL(ArgMin, 11, 12);
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_DOUBLE_ONLY(ArgMin, 11, 12)
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT8_ONLY(ArgMin, 11, 12)
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_INT64_ONLY(ArgMin, 11, 12)
REGISTER_UNARY_ELEMENTWISE_VERSIONED_KERNEL_UINT8_ONLY(ArgMin, 11, 12)
REGISTER_UNARY_ELEMENTWISE_KERNEL(ArgMin, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_DOUBLE_ONLY(ArgMin, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT64_ONLY(ArgMin, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_INT8_ONLY(ArgMin, 13);
REGISTER_UNARY_ELEMENTWISE_KERNEL_UINT8_ONLY(ArgMin, 13);

FastReduceKind operator|(FastReduceKind a, FastReduceKind b) {
  return static_cast<FastReduceKind>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

bool operator==(FastReduceKind a, FastReduceKind b) {
  return static_cast<uint8_t>(a) == static_cast<uint8_t>(b);
}

bool operator!=(FastReduceKind a, FastReduceKind b) {
  return static_cast<uint8_t>(a) != static_cast<uint8_t>(b);
}

bool ResultsNoTransposePrepareForReduce::equal(gsl::span<const int64_t> local_input_shape,
                                               gsl::span<const int64_t> local_reduced_axes) {
  if (!SpanEq(gsl::make_span(input_shape), local_input_shape))
    return false;
  if (!SpanEq(gsl::make_span(reduced_axes), local_reduced_axes))
    return false;
  return true;
}

void ResultsNoTransposePrepareForReduce::ValidateNotEmpty() {
  ORT_ENFORCE(last_loop_red_size > 0);
  ORT_ENFORCE(last_loop_size > 0);
  ORT_ENFORCE(projected_index.size() > 0);
}

static void ValidateMustBeOverloaded() {
  ORT_ENFORCE(false, "must be overloaded.");
}

static void ValidateFastReduceKR(const gsl::span<const int64_t>& fast_shape, const Tensor& output) {
  ORT_ENFORCE(fast_shape.size() == 2, "Only works on matrices with two dimensions.");
  ORT_ENFORCE(fast_shape[0] == output.Shape().Size(), "Output size mismatch.");
}

static void ValidateFastReduceRK(const gsl::span<const int64_t>& fast_shape, const Tensor& output) {
  ORT_ENFORCE(fast_shape.size() == 2, "Only works on matrices with two dimensions.");
  ORT_ENFORCE(fast_shape[1] == output.Shape().Size(), "Output size mismatch.");
}

static void ValidateFastReduceKRK(const gsl::span<const int64_t>& fast_shape, const Tensor& output) {
  ORT_ENFORCE(fast_shape.size() == 3, "Only works on matrices with three dimensions.");
  ORT_ENFORCE(fast_shape[0] * fast_shape[2] == output.Shape().Size(), "Output size mismatch.");
}

static void ValidateFastReduceRKR(const gsl::span<const int64_t>& fast_shape, const Tensor& output) {
  ORT_ENFORCE(fast_shape.size() == 3, "Only works on matrices with three dimensions.");
  ORT_ENFORCE(fast_shape[1] == output.Shape().Size(), "Output size mismatch.");
}

void ReduceAggregatorBase::FastReduceKR(const Tensor&, const gsl::span<const int64_t>&, Tensor&, concurrency::ThreadPool*) {
  ValidateMustBeOverloaded();
}
void ReduceAggregatorBase::FastReduceRK(const Tensor&, const gsl::span<const int64_t>&, Tensor&, concurrency::ThreadPool*) {
  ValidateMustBeOverloaded();
}
void ReduceAggregatorBase::FastReduceKRK(const Tensor&, const gsl::span<const int64_t>&, Tensor&, concurrency::ThreadPool*) {
  ValidateMustBeOverloaded();
}
void ReduceAggregatorBase::FastReduceRKR(const Tensor&, const gsl::span<const int64_t>&, Tensor&, concurrency::ThreadPool*) {
  ValidateMustBeOverloaded();
}

namespace {

// Number of elements passed to one exp call. The scratch buffers live on the stack.
constexpr int64_t kLogSumExpChunk = 1024;

template <typename T>
void ExpInPlace(T* buf, int64_t n) {
  if constexpr (std::is_same_v<T, float>) {
    MlasComputeExp(buf, buf, onnxruntime::narrow<size_t>(n));
  } else {
    EigenVectorArrayMap<T> a(buf, onnxruntime::narrow<size_t>(n));
    a = a.exp();
  }
}

// Combines the maximum m and sum(exp(x - m)) into log(sum(exp(x))). A NaN or infinite maximum is the result:
// any NaN gives NaN, otherwise any +inf gives +inf, and all -inf gives -inf.
template <typename T>
inline T FinishLogSumExp(T m, T sum) {
  return std::isfinite(m) ? static_cast<T>(std::log(sum)) + m : m;
}

template <typename T>
inline T MaxPropagateNaN(const T* p, int64_t n) {
  return ConstEigenVectorArrayMap<T>(p, onnxruntime::narrow<size_t>(n)).template maxCoeff<Eigen::PropagateNaN>();
}

}  // namespace

// Rows are processed in groups whose layout in the scratch buffer depends only on the shape, so the result does
// not depend on the number of threads.
template <typename T>
void LogSumExpFastReduceKR(const T* data, T* out, int64_t n_rows, int64_t N, concurrency::ThreadPool* tp) {
  if (n_rows == 0 || N == 0) {
    return;
  }
  const int64_t rows_per_group = std::max<int64_t>(1, kLogSumExpChunk / N);
  const int64_t n_groups = (n_rows + rows_per_group - 1) / rows_per_group;
  const TensorOpCost cost{static_cast<double>(rows_per_group * N * sizeof(T)),
                          static_cast<double>(rows_per_group * sizeof(T)),
                          static_cast<double>(rows_per_group * N * 16)};
  concurrency::ThreadPool::TryParallelFor(
      tp, onnxruntime::narrow<std::ptrdiff_t>(n_groups), cost, [=](std::ptrdiff_t first, std::ptrdiff_t last) {
        alignas(64) T buf[kLogSumExpChunk];
        alignas(64) T maxes[kLogSumExpChunk];
        for (std::ptrdiff_t g = first; g < last; ++g) {
          const int64_t r0 = g * rows_per_group;
          const int64_t nr = std::min(rows_per_group, n_rows - r0);
          if (N > kLogSumExpChunk) {
            const T* x = data + r0 * N;
            const T m = MaxPropagateNaN(x, N);
            T sum = 0;
            if (std::isfinite(m)) {
              for (int64_t c0 = 0; c0 < N; c0 += kLogSumExpChunk) {
                const int64_t w = std::min(kLogSumExpChunk, N - c0);
                EigenVectorArrayMap<T>(buf, w) = ConstEigenVectorArrayMap<T>(x + c0, w) - m;
                ExpInPlace(buf, w);
                sum += ConstEigenVectorArrayMap<T>(buf, w).sum();
              }
            }
            out[r0] = FinishLogSumExp(m, sum);
            continue;
          }
          for (int64_t i = 0; i < nr; ++i) {
            const T* x = data + (r0 + i) * N;
            maxes[i] = MaxPropagateNaN(x, N);
            const T shift = std::isfinite(maxes[i]) ? maxes[i] : static_cast<T>(0);
            EigenVectorArrayMap<T>(buf + i * N, N) = ConstEigenVectorArrayMap<T>(x, N) - shift;
          }
          ExpInPlace(buf, nr * N);
          for (int64_t i = 0; i < nr; ++i) {
            out[r0 + i] = FinishLogSumExp(maxes[i], ConstEigenVectorArrayMap<T>(buf + i * N, N).sum());
          }
        }
      });
}

// Reduces the middle axis of a [n_batch, n_rows, N] tensor into [n_batch, N]. Like ReduceSum's RK path, the work is
// split into (batch, row block, column tile) tasks whose partition depends only on the shape. The first pass computes
// partial column maxima, the second pass partial sums of exp(x - max), and the last pass merges them in a fixed order.
template <typename T>
void LogSumExpFastReduceRK(const T* data, T* out, int64_t n_batch, int64_t n_rows, int64_t N,
                           concurrency::ThreadPool* tp) {
  constexpr int64_t kMinElementsPerTask = 32768;
  constexpr int64_t kElementsPerTask = 65536;
  if (n_batch == 0 || n_rows == 0 || N == 0) {
    return;
  }
  if (n_batch * n_rows * N <= kMinElementsPerTask) {
    tp = nullptr;
  }
  const int64_t n_col_tiles = (N + kLogSumExpChunk - 1) / kLogSumExpChunk;
  const int64_t col_tile = (N + n_col_tiles - 1) / n_col_tiles;
  const int64_t rows_per_block = std::min(n_rows, std::max<int64_t>(1, kElementsPerTask / col_tile));
  const int64_t n_row_blocks = (n_rows + rows_per_block - 1) / rows_per_block;
  const int64_t tasks_per_batch = n_row_blocks * n_col_tiles;
  const auto n_tasks = onnxruntime::narrow<std::ptrdiff_t>(n_batch * tasks_per_batch);

  // partials[batch][row_block][N] holds the partial maxima and then the partial sums.
  std::vector<T> partials(SafeInt<size_t>(n_batch) * n_row_blocks * onnxruntime::narrow<size_t>(N));
  std::vector<T> col_max(SafeInt<size_t>(n_batch) * onnxruntime::narrow<size_t>(N));
  // Per-column shift for exp; equal to the maximum where it is finite.
  std::vector<T> shift(col_max.size());
  T* partials_data = partials.data();
  T* col_max_data = col_max.data();
  T* shift_data = shift.data();

  struct Task {
    int64_t batch, row_block, c0, width, r_begin, r_end;
  };
  auto get_task = [=](std::ptrdiff_t task) {
    Task t;
    t.batch = task / tasks_per_batch;
    t.row_block = (task % tasks_per_batch) / n_col_tiles;
    t.c0 = (task % n_col_tiles) * col_tile;
    t.width = std::min(col_tile, N - t.c0);
    t.r_begin = t.row_block * rows_per_block;
    t.r_end = std::min(n_rows, t.r_begin + rows_per_block);
    return t;
  };

  concurrency::ThreadPool::TrySimpleParallelFor(tp, n_tasks, [=](std::ptrdiff_t task) {
    const Task t = get_task(task);
    EigenVectorArrayMap<T> acc(partials_data + (t.batch * n_row_blocks + t.row_block) * N + t.c0, t.width);
    const T* p = data + (t.batch * n_rows + t.r_begin) * N + t.c0;
    acc = ConstEigenVectorArrayMap<T>(p, t.width);
    for (int64_t r = t.r_begin + 1; r < t.r_end; ++r) {
      p += N;
      acc = acc.template max<Eigen::PropagateNaN>(ConstEigenVectorArrayMap<T>(p, t.width));
    }
  });

  concurrency::ThreadPool::TrySimpleParallelFor(
      tp, onnxruntime::narrow<std::ptrdiff_t>(n_batch * n_col_tiles), [=](std::ptrdiff_t task) {
        const int64_t batch = task / n_col_tiles;
        const int64_t c0 = (task % n_col_tiles) * col_tile;
        const int64_t width = std::min(col_tile, N - c0);
        EigenVectorArrayMap<T> m(col_max_data + batch * N + c0, width);
        const T* partial = partials_data + batch * n_row_blocks * N + c0;
        m = ConstEigenVectorArrayMap<T>(partial, width);
        for (int64_t b = 1; b < n_row_blocks; ++b) {
          partial += N;
          m = m.template max<Eigen::PropagateNaN>(ConstEigenVectorArrayMap<T>(partial, width));
        }
        // Non-finite maxima are the final result, so any finite shift works for those columns.
        EigenVectorArrayMap<T>(shift_data + batch * N + c0, width) = m.isFinite().select(m, static_cast<T>(0));
      });

  concurrency::ThreadPool::TrySimpleParallelFor(tp, n_tasks, [=](std::ptrdiff_t task) {
    const Task t = get_task(task);
    alignas(64) T buf[kLogSumExpChunk];
    EigenVectorArrayMap<T> acc(partials_data + (t.batch * n_row_blocks + t.row_block) * N + t.c0, t.width);
    ConstEigenVectorArrayMap<T> m(shift_data + t.batch * N + t.c0, t.width);
    acc.setZero();
    const int64_t rows_per_chunk = kLogSumExpChunk / t.width;
    const T* p = data + (t.batch * n_rows + t.r_begin) * N + t.c0;
    for (int64_t r = t.r_begin; r < t.r_end; r += rows_per_chunk) {
      const int64_t nr = std::min(rows_per_chunk, t.r_end - r);
      for (int64_t i = 0; i < nr; ++i, p += N) {
        EigenVectorArrayMap<T>(buf + i * t.width, t.width) = ConstEigenVectorArrayMap<T>(p, t.width) - m;
      }
      ExpInPlace(buf, nr * t.width);
      for (int64_t i = 0; i < nr; ++i) {
        acc += ConstEigenVectorArrayMap<T>(buf + i * t.width, t.width);
      }
    }
  });

  concurrency::ThreadPool::TrySimpleParallelFor(
      tp, onnxruntime::narrow<std::ptrdiff_t>(n_batch * n_col_tiles), [=](std::ptrdiff_t task) {
        const int64_t batch = task / n_col_tiles;
        const int64_t c0 = (task % n_col_tiles) * col_tile;
        const int64_t width = std::min(col_tile, N - c0);
        const T* m = col_max_data + batch * N + c0;
        const T* partial = partials_data + batch * n_row_blocks * N + c0;
        T* o = out + batch * N + c0;
        for (int64_t c = 0; c < width; ++c) {
          T sum = partial[c];
          for (int64_t b = 1; b < n_row_blocks; ++b) {
            sum += partial[b * N + c];
          }
          o[c] = FinishLogSumExp(m[c], sum);
        }
      });
}

template void LogSumExpFastReduceKR<float>(const float*, float*, int64_t, int64_t, concurrency::ThreadPool*);
template void LogSumExpFastReduceKR<double>(const double*, double*, int64_t, int64_t, concurrency::ThreadPool*);
template void LogSumExpFastReduceRK<float>(const float*, float*, int64_t, int64_t, int64_t,
                                           concurrency::ThreadPool*);
template void LogSumExpFastReduceRK<double>(const double*, double*, int64_t, int64_t, int64_t,
                                            concurrency::ThreadPool*);

namespace {

// acc = IsMax ? ((acc < p) ? p : acc) : ((acc > p) ? p : acc), element-wise. This is the update used by the generic
// ReduceMax/ReduceMin loops: a NaN accumulator is kept, a NaN input is ignored, and ties keep the accumulator, so
// the first of +0/-0 wins.
template <typename T, bool IsMax>
inline void MaxMinFold(T* acc, const T* p, int64_t w) {
  int64_t j = 0;
#if REDUCTION_OPS_USE_SSE2
  // _mm_max_ps(b, a) returns (b > a) ? b : a, which is exactly the update above.
  if constexpr (std::is_same_v<T, float>) {
    for (; j + 4 <= w; j += 4) {
      const __m128 a = _mm_loadu_ps(acc + j);
      const __m128 b = _mm_loadu_ps(p + j);
      _mm_storeu_ps(acc + j, IsMax ? _mm_max_ps(b, a) : _mm_min_ps(b, a));
    }
  } else if constexpr (std::is_same_v<T, double>) {
    for (; j + 2 <= w; j += 2) {
      const __m128d a = _mm_loadu_pd(acc + j);
      const __m128d b = _mm_loadu_pd(p + j);
      _mm_storeu_pd(acc + j, IsMax ? _mm_max_pd(b, a) : _mm_min_pd(b, a));
    }
  }
#endif
  for (; j < w; ++j) {
    if constexpr (IsMax) {
      acc[j] = acc[j] < p[j] ? p[j] : acc[j];
    } else {
      acc[j] = acc[j] > p[j] ? p[j] : acc[j];
    }
  }
}

// Folding a row block that does not start at row 0 from -inf (+inf for min) and then folding the block results in
// order gives the same result as one left fold over all rows: NaN only sticks when it is in row 0, and the first of
// several equal values (e.g. +0/-0) is kept.
template <typename T, bool IsMax>
struct MaxMinRowOp {
  static void Init(T* acc, const T* p, int64_t w, bool first_block) {
    if (first_block) {
      std::copy_n(p, w, acc);
    } else {
      std::fill_n(acc, w, IsMax ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity());
      MaxMinFold<T, IsMax>(acc, p, w);
    }
  }
  static void Accumulate(T* acc, const T* p, int64_t w) { MaxMinFold<T, IsMax>(acc, p, w); }
  static void Merge(T* acc, const T* partial, int64_t w) { MaxMinFold<T, IsMax>(acc, partial, w); }
  static void Finalize(T*, int64_t) {}
};

enum class RowSumKind { kAbs,
                        kSquare };

template <typename T, RowSumKind Kind>
inline T RowSumTransform(T v) {
  if constexpr (Kind == RowSumKind::kAbs) {
    return std::abs(v);
  } else {
    return v * v;
  }
}

template <typename T, RowSumKind Kind, bool Sqrt>
struct RowSumOp {
  static void Init(T* acc, const T* p, int64_t w, bool) {
    for (int64_t j = 0; j < w; ++j) {
      acc[j] = RowSumTransform<T, Kind>(p[j]);
    }
  }
  static void Accumulate(T* acc, const T* p, int64_t w) {
    for (int64_t j = 0; j < w; ++j) {
      acc[j] += RowSumTransform<T, Kind>(p[j]);
    }
  }
  static void Merge(T* acc, const T* partial, int64_t w) {
    for (int64_t j = 0; j < w; ++j) {
      acc[j] += partial[j];
    }
  }
  static void Finalize(T* acc, int64_t w) {
    if constexpr (Sqrt) {
      for (int64_t j = 0; j < w; ++j) {
        acc[j] = std::sqrt(acc[j]);
      }
    }
  }
};

// Reduces the middle axis of a [n_batch, n_rows, N] tensor into [n_batch, N] with an element-wise row operation.
// The work is split into (batch, row block, column tile) tasks whose partition depends only on the shape, and the
// row block results are merged in a fixed order, so the result does not depend on the number of threads.
template <typename T, typename Op>
void TiledRowReduceRK(const T* data, T* out, int64_t n_batch, int64_t n_rows, int64_t N,
                      concurrency::ThreadPool* tp) {
  constexpr int64_t kMaxColTile = 1024;
  constexpr int64_t kMinElementsPerTask = 32768;
  // About kTargetTasks tasks for mid-sized inputs, with the task size clamped to [kMinTaskElements, kMaxTaskElements].
  constexpr int64_t kTargetTasks = 16;
  constexpr int64_t kMinTaskElements = 65536;
  constexpr int64_t kMaxTaskElements = 262144;
  if (n_batch == 0 || n_rows == 0 || N == 0) {
    return;
  }
  if (n_batch * n_rows * N <= kMinElementsPerTask) {
    tp = nullptr;
  }
  const int64_t n_col_tiles = (N + kMaxColTile - 1) / kMaxColTile;
  const int64_t col_tile = (N + n_col_tiles - 1) / n_col_tiles;
  const int64_t elements_per_task =
      std::clamp(n_batch * n_rows * N / kTargetTasks, kMinTaskElements, kMaxTaskElements);
  const int64_t rows_per_block = std::min(n_rows, std::max<int64_t>(1, elements_per_task / col_tile));
  const int64_t n_row_blocks = (n_rows + rows_per_block - 1) / rows_per_block;
  const int64_t tasks_per_batch = n_row_blocks * n_col_tiles;
  const auto n_tasks = onnxruntime::narrow<std::ptrdiff_t>(n_batch * tasks_per_batch);

  // With a single row block, tasks write the output directly.
  std::vector<T> partials;
  if (n_row_blocks > 1) {
    partials.resize(SafeInt<size_t>(n_batch) * n_row_blocks * onnxruntime::narrow<size_t>(N));
  }
  T* partials_data = partials.data();

  concurrency::ThreadPool::TrySimpleParallelFor(tp, n_tasks, [=](std::ptrdiff_t task) {
    const int64_t batch = task / tasks_per_batch;
    const int64_t row_block = (task % tasks_per_batch) / n_col_tiles;
    const int64_t c0 = (task % n_col_tiles) * col_tile;
    const int64_t width = std::min(col_tile, N - c0);
    const int64_t r_begin = row_block * rows_per_block;
    const int64_t r_end = std::min(n_rows, r_begin + rows_per_block);
    T* acc = n_row_blocks > 1 ? partials_data + (batch * n_row_blocks + row_block) * N + c0 : out + batch * N + c0;
    const T* p = data + (batch * n_rows + r_begin) * N + c0;
    Op::Init(acc, p, width, row_block == 0);
    for (int64_t r = r_begin + 1; r < r_end; ++r) {
      p += N;
      Op::Accumulate(acc, p, width);
    }
    if (n_row_blocks == 1) {
      Op::Finalize(acc, width);
    }
  });

  if (n_row_blocks == 1) {
    return;
  }

  concurrency::ThreadPool::TrySimpleParallelFor(
      tp, onnxruntime::narrow<std::ptrdiff_t>(n_batch * n_col_tiles), [=](std::ptrdiff_t task) {
        const int64_t batch = task / n_col_tiles;
        const int64_t c0 = (task % n_col_tiles) * col_tile;
        const int64_t width = std::min(col_tile, N - c0);
        T* o = out + batch * N + c0;
        const T* partial = partials_data + batch * n_row_blocks * N + c0;
        std::copy_n(partial, width, o);
        for (int64_t b = 1; b < n_row_blocks; ++b) {
          partial += N;
          Op::Merge(o, partial, width);
        }
        Op::Finalize(o, width);
      });
}

}  // namespace

template <typename T>
void MaxFastReduceRK(const T* data, T* out, int64_t n_batch, int64_t n_rows, int64_t N,
                     concurrency::ThreadPool* tp) {
  TiledRowReduceRK<T, MaxMinRowOp<T, true>>(data, out, n_batch, n_rows, N, tp);
}

template <typename T>
void MinFastReduceRK(const T* data, T* out, int64_t n_batch, int64_t n_rows, int64_t N,
                     concurrency::ThreadPool* tp) {
  TiledRowReduceRK<T, MaxMinRowOp<T, false>>(data, out, n_batch, n_rows, N, tp);
}

template <typename T>
void SumAbsFastReduceRK(const T* data, T* out, int64_t n_batch, int64_t n_rows, int64_t N,
                        concurrency::ThreadPool* tp) {
  TiledRowReduceRK<T, RowSumOp<T, RowSumKind::kAbs, false>>(data, out, n_batch, n_rows, N, tp);
}

template <typename T>
void SumSquareFastReduceRK(const T* data, T* out, int64_t n_batch, int64_t n_rows, int64_t N,
                           concurrency::ThreadPool* tp) {
  TiledRowReduceRK<T, RowSumOp<T, RowSumKind::kSquare, false>>(data, out, n_batch, n_rows, N, tp);
}

template <typename T>
void L2NormFastReduceRK(const T* data, T* out, int64_t n_batch, int64_t n_rows, int64_t N,
                        concurrency::ThreadPool* tp) {
  TiledRowReduceRK<T, RowSumOp<T, RowSumKind::kSquare, true>>(data, out, n_batch, n_rows, N, tp);
}

#define INSTANTIATE_ROW_REDUCE_RK(fn)                                                                 \
  template void fn<float>(const float*, float*, int64_t, int64_t, int64_t, concurrency::ThreadPool*); \
  template void fn<double>(const double*, double*, int64_t, int64_t, int64_t, concurrency::ThreadPool*);

INSTANTIATE_ROW_REDUCE_RK(MaxFastReduceRK)
INSTANTIATE_ROW_REDUCE_RK(MinFastReduceRK)
INSTANTIATE_ROW_REDUCE_RK(SumAbsFastReduceRK)
INSTANTIATE_ROW_REDUCE_RK(SumSquareFastReduceRK)
INSTANTIATE_ROW_REDUCE_RK(L2NormFastReduceRK)

#undef INSTANTIATE_ROW_REDUCE_RK

void NoTransposePrepareForReduce(const TensorShape& new_input_shape,
                                 gsl::span<const int64_t> reduced_axes,
                                 ResultsNoTransposePrepareForReduce& results) {
  // Common initialisation for the indices.
  auto cumulative_shape = new_input_shape.AsShapeVector();
  cumulative_shape[cumulative_shape.size() - 1] = 1;
  for (int i = static_cast<int>(cumulative_shape.size()) - 2; i >= 0; --i) {
    cumulative_shape[i] = cumulative_shape[i + 1] * new_input_shape[i + 1];
  }
  int64_t projection_size = 1;
  for (auto a : reduced_axes) {
    projection_size *= new_input_shape[onnxruntime::narrow<size_t>(a)];
  }

  int last_reduced_axis = static_cast<int>(reduced_axes.size()) - 1;
  int loop_reduced_axis = 1;
  results.last_loop_red_size = new_input_shape[onnxruntime::narrow<size_t>(reduced_axes[onnxruntime::narrow<size_t>(last_reduced_axis)])];
  results.last_loop_red_inc = cumulative_shape[onnxruntime::narrow<size_t>(reduced_axes[onnxruntime::narrow<size_t>(last_reduced_axis)])];
  projection_size /= new_input_shape[onnxruntime::narrow<size_t>(reduced_axes[onnxruntime::narrow<size_t>(last_reduced_axis)])];
  --last_reduced_axis;
  while (last_reduced_axis >= 0) {
    if (reduced_axes[last_reduced_axis] != reduced_axes[last_reduced_axis + 1] - 1)
      break;
    results.last_loop_red_size *= new_input_shape[onnxruntime::narrow<size_t>(reduced_axes[onnxruntime::narrow<size_t>(last_reduced_axis)])];
    projection_size /= new_input_shape[onnxruntime::narrow<size_t>(reduced_axes[onnxruntime::narrow<size_t>(last_reduced_axis)])];
    --last_reduced_axis;
    ++loop_reduced_axis;
  }

  // Builds the list of indices projected into the same sum.
  int reduced_axes_size = static_cast<int>(reduced_axes.size()) - loop_reduced_axis;
  if (reduced_axes_size == 0) {
    results.projected_index.resize(1, 0);
  } else {
    results.projected_index.resize(onnxruntime::narrow<size_t>(projection_size));
    TensorShapeVector projected_indices(reduced_axes_size, 0);
    int64_t current_index = 0;
    size_t current_pos = 0;
    int j;
    for (current_pos = 0; current_pos < results.projected_index.size(); ++current_pos) {
      results.projected_index[current_pos] = current_index;
      ++projected_indices[projected_indices.size() - 1];
      current_index += cumulative_shape[onnxruntime::narrow<size_t>(reduced_axes[onnxruntime::narrow<size_t>(reduced_axes_size - 1)])];
      for (j = reduced_axes_size - 1; j > 0; --j) {
        if (projected_indices[onnxruntime::narrow<size_t>(j)] < new_input_shape[onnxruntime::narrow<size_t>(reduced_axes[onnxruntime::narrow<size_t>(j)])])
          break;
        projected_indices[j] -= new_input_shape[onnxruntime::narrow<size_t>(reduced_axes[onnxruntime::narrow<size_t>(j)])];
        current_index -= new_input_shape[onnxruntime::narrow<size_t>(reduced_axes[onnxruntime::narrow<size_t>(j)])] * cumulative_shape[onnxruntime::narrow<size_t>(reduced_axes[onnxruntime::narrow<size_t>(j)])];
        ++projected_indices[j - 1];
        current_index += cumulative_shape[onnxruntime::narrow<size_t>(reduced_axes[onnxruntime::narrow<size_t>(j - 1)])];
      }
    }
  }

  // Builds the list of indices for the unprojected sum.
  TensorShapeVector unreduced_axes;
  for (int64_t i = 0; i < static_cast<int64_t>(cumulative_shape.size()); ++i) {
    if (std::find(reduced_axes.begin(), reduced_axes.end(), i) != reduced_axes.end())
      continue;
    unreduced_axes.push_back(i);
  }
  int64_t unprojection_size = 1;
  for (auto a : unreduced_axes) {
    unprojection_size *= new_input_shape[onnxruntime::narrow<size_t>(a)];
  }
  if (unprojection_size == 0) {
    return;
  }
  TensorShapeVector unprojected_indices(unreduced_axes.size(), 0);

  // The last index is usually an image size.
  // We differently process the last unprojected dimension.
  results.last_loop_size = new_input_shape[onnxruntime::narrow<size_t>(unreduced_axes[onnxruntime::narrow<size_t>(unreduced_axes.size() - 1)])];
  int64_t unprojection_size_before_last = unprojection_size / results.last_loop_size;
  results.unprojected_index.reserve(onnxruntime::narrow<size_t>(unprojection_size_before_last));
  results.last_loop_inc = cumulative_shape[onnxruntime::narrow<size_t>(unreduced_axes[onnxruntime::narrow<size_t>(unreduced_axes.size() - 1)])];
  if (unprojected_indices.size() <= 1) {
    results.unprojected_index.push_back(0);
  } else {
    int64_t current_index = 0;
    int j;
    for (int64_t pos = 0; pos < unprojection_size_before_last; ++pos) {
      results.unprojected_index.push_back(current_index);
      ++unprojected_indices[unprojected_indices.size() - 2];
      current_index += cumulative_shape[onnxruntime::narrow<size_t>(unreduced_axes[onnxruntime::narrow<size_t>(unreduced_axes.size() - 2)])];
      for (j = static_cast<int>(unreduced_axes.size()) - 2; j > 0; --j) {
        if (unprojected_indices[j] < new_input_shape[onnxruntime::narrow<size_t>(unreduced_axes[onnxruntime::narrow<size_t>(j)])])
          break;
        unprojected_indices[j] -= new_input_shape[onnxruntime::narrow<size_t>(unreduced_axes[onnxruntime::narrow<size_t>(j)])];
        current_index -= new_input_shape[onnxruntime::narrow<size_t>(unreduced_axes[onnxruntime::narrow<size_t>(j)])] * cumulative_shape[onnxruntime::narrow<size_t>(unreduced_axes[onnxruntime::narrow<size_t>(j)])];
        ++unprojected_indices[j - 1];
        current_index += cumulative_shape[onnxruntime::narrow<size_t>(unreduced_axes[onnxruntime::narrow<size_t>(j - 1)])];
      }
    }
  }
}

void ValidateNoTransposeReduce(int64_t count) {
  ORT_ENFORCE(count == 1, "Reduction on all axes, output size should be 1.");
}

template <typename AGG>
struct ParallelizedData {
  int64_t denominator;
  int64_t loop_size;
  ResultsNoTransposePrepareForReduce* last_results;
  const typename AGG::input_type* from_data;
  typename AGG::value_type* to_data;
};

template <typename AGG>
void NoTransposeReduce1Loop(Tensor* output, const TensorShape& new_input_shape, const Tensor& input,
                            gsl::span<const int64_t> reduced_axes, concurrency::ThreadPool* tp,
                            ResultsNoTransposePrepareForReduce& last_results) {
  auto output_shape = output->Shape();
  const typename AGG::input_type* from_data = input.Data<typename AGG::input_type>();
  typename AGG::value_type* to_data = output->MutableData<typename AGG::value_type>();
  int64_t count = output_shape.Size();

  if (reduced_axes.size() == 0 || reduced_axes.size() == new_input_shape.NumDimensions()) {
    ValidateNoTransposeReduce(count);
    int64_t input_size = new_input_shape.Size();
    to_data[0] = AGG(input_size, from_data[0]).aggall(from_data);
    return;
  }

  if (!last_results.equal(new_input_shape.GetDims(), reduced_axes)) {
    NoTransposePrepareForReduce(new_input_shape, reduced_axes, last_results);
    if (last_results.last_loop_red_size == 0 || last_results.last_loop_size == 0)
      return;
  }
  last_results.ValidateNotEmpty();

  ParallelizedData<AGG> data;
  data.denominator = last_results.last_loop_red_size * last_results.projected_index.size();
  data.loop_size = last_results.last_loop_red_size * last_results.last_loop_red_inc;
  data.last_results = &last_results;
  data.from_data = from_data;
  data.to_data = to_data;

  auto fn = [&data](std::ptrdiff_t first, std::ptrdiff_t end) {
    const typename AGG::input_type* loop_red_ptr;
    const ResultsNoTransposePrepareForReduce& last_results = *data.last_results;
    int64_t main_index = first / last_results.last_loop_size;
    int64_t loop = first % last_results.last_loop_size;
    int64_t origin = last_results.unprojected_index[onnxruntime::narrow<size_t>(main_index)] + loop * last_results.last_loop_inc;
    for (int64_t main_index_last_loop = first; main_index_last_loop < end; ++main_index_last_loop) {
      AGG accumulator(data.denominator, data.from_data[origin + last_results.projected_index[0]]);
      for (auto it = last_results.projected_index.begin(); it != last_results.projected_index.end(); ++it) {
        loop_red_ptr = data.from_data + (origin + *it);
        for (int64_t red = 0; red < data.loop_size; red += last_results.last_loop_red_inc) {
          accumulator.update(loop_red_ptr[red]);
        }
      }
      data.to_data[main_index_last_loop] = accumulator.get_value();

      ++loop;
      if (loop >= last_results.last_loop_size) {
        loop = 0;
        ++main_index;
        if (main_index < static_cast<int64_t>(last_results.unprojected_index.size())) {
          origin = last_results.unprojected_index[onnxruntime::narrow<size_t>(main_index)];
        }
      } else {
        origin += last_results.last_loop_inc;
      }
    }
  };

  auto cost = ParallelReduceFastCost(1,
                                     last_results.projected_index.size() * last_results.last_loop_red_size,
                                     sizeof(typename AGG::input_type), 6);
  concurrency::ThreadPool::TryParallelFor(tp, onnxruntime::narrow<std::ptrdiff_t>(count), cost, fn);
}

template <typename AGG>
void NoTransposeReduce2Loops(Tensor* output, const TensorShape& new_input_shape, const Tensor& input,
                             gsl::span<const int64_t> reduced_axes, concurrency::ThreadPool* tp,
                             ResultsNoTransposePrepareForReduce& last_results) {
  auto output_shape = output->Shape();
  const typename AGG::input_type* from_data = input.Data<typename AGG::input_type>();
  typename AGG::value_type* to_data = output->MutableData<typename AGG::value_type>();
  int64_t count = output_shape.Size();

  if (reduced_axes.size() == 0 || reduced_axes.size() == new_input_shape.NumDimensions()) {
    ValidateNoTransposeReduce(count);
    int64_t input_size = new_input_shape.Size();
    to_data[0] = AGG(input_size, from_data[0]).aggall(from_data);
    return;
  }

  if (!last_results.equal(new_input_shape.GetDims(), reduced_axes)) {
    NoTransposePrepareForReduce(new_input_shape, reduced_axes, last_results);
    if (last_results.last_loop_red_size == 0 || last_results.last_loop_size == 0)
      return;
  }
  last_results.ValidateNotEmpty();

  ParallelizedData<AGG> data;
  data.denominator = last_results.last_loop_red_size * last_results.projected_index.size();
  data.loop_size = last_results.last_loop_red_size * last_results.last_loop_red_inc;
  data.last_results = &last_results;
  data.from_data = from_data;
  data.to_data = to_data;

  auto fn = [&](std::ptrdiff_t first, std::ptrdiff_t end) {
    const typename AGG::input_type* loop_red_ptr;
    const ResultsNoTransposePrepareForReduce& last_results = *data.last_results;
    int64_t main_index = first / last_results.last_loop_size;
    int64_t loop = first % last_results.last_loop_size;
    int64_t origin = last_results.unprojected_index[onnxruntime::narrow<size_t>(main_index)] + loop * last_results.last_loop_inc;
    for (int64_t main_index_last_loop = first; main_index_last_loop < end; ++main_index_last_loop) {
      AGG accumulator(data.denominator, data.from_data[origin + last_results.projected_index[0]]);
      for (auto it = last_results.projected_index.begin(); it != last_results.projected_index.end(); ++it) {
        loop_red_ptr = data.from_data + (origin + *it);
        for (int64_t red = 0; red < data.loop_size; red += last_results.last_loop_red_inc) {
          accumulator.update0(loop_red_ptr[red]);
        }
      }

      for (auto it = last_results.projected_index.begin(); it != last_results.projected_index.end(); ++it) {
        loop_red_ptr = data.from_data + (origin + *it);
        for (int64_t red = 0; red < data.loop_size; red += last_results.last_loop_red_inc) {
          accumulator.update(loop_red_ptr[red]);
        }
      }
      data.to_data[main_index_last_loop] = accumulator.get_value();

      ++loop;
      if (loop >= last_results.last_loop_size) {
        loop = 0;
        ++main_index;
        if (main_index < static_cast<int64_t>(last_results.unprojected_index.size())) {
          origin = last_results.unprojected_index[onnxruntime::narrow<size_t>(main_index)];
        }
      } else {
        origin += last_results.last_loop_inc;
      }
    }
  };

  auto cost = ParallelReduceFastCost(1,
                                     last_results.projected_index.size() * last_results.last_loop_red_size,
                                     sizeof(typename AGG::input_type), 8);
  concurrency::ThreadPool::TryParallelFor(tp, onnxruntime::narrow<std::ptrdiff_t>(count), cost, fn);
}

void DropDimensions(const gsl::span<const int64_t>& input_shape,
                    const gsl::span<const int64_t>& axes,
                    TensorShapeVector& dropped_axes) {
  TensorShapeVector dropped_dims = ToShapeVector(input_shape);
  for (auto i : axes) {
    dropped_dims[onnxruntime::narrow<size_t>(i)] = -1;
  }
  for (auto it = dropped_dims.begin(); it != dropped_dims.end(); ++it) {
    if (*it != -1) {
      dropped_axes.push_back(*it);
    }
  }
}

FastReduceKind OptimizeShapeForFastReduce(gsl::span<const int64_t> input_shape,
                                          gsl::span<const int64_t> reduced_axes,
                                          TensorShapeVector& fast_shape,
                                          TensorShapeVector& fast_output_shape,
                                          TensorShapeVector& fast_axes,
                                          bool keep_dims, bool noop_with_empty_axes) {
  if (input_shape.empty()) {
    fast_shape.clear();
    fast_output_shape.clear();
    // XXX: Should we enforce the absence of the axes in the scalar input case?
    // The operator spec refers to Numpy which returns error because axes can not possibly contain any valid
    // value in scalar case, but pytorch simply ignores it.
    // ORT_ENFORCE(reduced_axes.empty(), "With scalar input shape, axis can not contain valid values");
    fast_axes.clear();
    return FastReduceKind::kEmpty;
  }

  InlinedHashSet<int64_t> axes;
  const auto input_shape_size = narrow<int64_t>(input_shape.size());
  if (reduced_axes.size() == 0 && !noop_with_empty_axes) {
    for (int64_t i = 0; i < input_shape_size; ++i) {
      axes.insert(i);
    }
  } else {
    for (auto ax : reduced_axes) {
      axes.insert(HandleNegativeAxis(ax, input_shape_size));
    }
  }

  fast_output_shape.clear();
  fast_output_shape.reserve(onnxruntime::narrow<size_t>(input_shape_size));
  bool empty_reduce = false;
  InlinedVector<bool> reduce(onnxruntime::narrow<size_t>(input_shape_size));
  for (int64_t i = 0; i < input_shape_size; ++i) {
    reduce[onnxruntime::narrow<size_t>(i)] = axes.find(i) != axes.end();
    if (reduce[onnxruntime::narrow<size_t>(i)]) {
      empty_reduce |= input_shape[onnxruntime::narrow<size_t>(i)] == 0;
      if (keep_dims)
        fast_output_shape.push_back(input_shape[onnxruntime::narrow<size_t>(i)] > 0 ? 1 : 0);
    } else {
      fast_output_shape.push_back(input_shape[onnxruntime::narrow<size_t>(i)]);
    }
  }

  if (empty_reduce) {
    return FastReduceKind::kEmpty;
  }

  if (reduced_axes.empty()) {
    fast_shape.resize(1);
    fast_shape[0] = 1;
    for (auto a : input_shape) {
      fast_shape[0] *= a;
    }
    if (noop_with_empty_axes) {
      fast_axes.clear();
      fast_output_shape.assign(input_shape.begin(), input_shape.end());
      return FastReduceKind::kK;
    } else {
      if (keep_dims) {
        fast_output_shape.resize(onnxruntime::narrow<size_t>(input_shape_size), 1);
      } else {
        fast_output_shape.clear();
      }
      fast_axes.resize(1);
      fast_axes[0] = 0;
      return FastReduceKind::kR;
    }
  }

  fast_shape.clear();
  fast_axes.clear();
  fast_shape.reserve(onnxruntime::narrow<size_t>(input_shape_size));
  fast_axes.reserve(reduced_axes.size());

  fast_shape.push_back(input_shape[0]);
  if (reduce[0])
    fast_axes.push_back(0);
  for (int64_t i = 1; i < input_shape_size; ++i) {
    if (reduce[onnxruntime::narrow<size_t>(i)] == reduce[onnxruntime::narrow<size_t>(i - 1)]) {
      fast_shape[onnxruntime::narrow<size_t>(fast_shape.size() - 1)] *= input_shape[onnxruntime::narrow<size_t>(i)];
    } else {
      if (reduce[onnxruntime::narrow<size_t>(i)]) {
        fast_axes.push_back(onnxruntime::narrow<int64_t>(fast_shape.size()));
      }
      fast_shape.push_back(input_shape[onnxruntime::narrow<size_t>(i)]);
    }
  }
  if (fast_shape.size() == 1) {
    return reduce[0] ? FastReduceKind::kR : FastReduceKind::kK;
  }
  if (fast_shape.size() == 2) {
    return reduce[0] ? FastReduceKind::kRK : FastReduceKind::kKR;
  }
  if (fast_shape.size() == 3) {
    return reduce[0] ? FastReduceKind::kRKR : FastReduceKind::kKRK;
  }
  return FastReduceKind::kNone;
}

// template <typename T, typename TVAL>
bool CommonFastReduceCopy(OpKernelContext* ctx, TensorShapeVector& input_axes, bool noop_with_empty_axes) {
  if (ctx->InputCount() == 2) {
    // second input holds the axes.
    // the argument is optional
    const Tensor* axes_tensor = ctx->Input<Tensor>(1);

    if (axes_tensor != nullptr) {
      ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1,
                  "An axes tensor must be a vector tensor.");

      const auto data_span = axes_tensor->DataAsSpan<int64_t>();
      input_axes.assign(data_span.begin(), data_span.end());
    } else {
      input_axes.clear();
    }
    // noop_with_empty_axes is handled upstream by ApplyNoopEmptyAxesElementwise().
    // Return false for clarity and to prevent unsafe memcpy fallback.
    if (input_axes.empty() && noop_with_empty_axes) {
      return false;
    }
  }
  return false;
}

typedef void fast_reduce_fct(const Tensor& input, const gsl::span<const int64_t>& fast_shape,
                             Tensor& output, concurrency::ThreadPool* tp);

bool CommonFastReduceSwitch(OpKernelContext* ctx,
                            const gsl::span<const int64_t>& axes_,
                            int64_t keepdims_,
                            bool noop_with_empty_axes,
                            FastReduceKind& fast_kind,
                            TensorShapeVector& fast_shape,
                            TensorShapeVector& output_shape,
                            TensorShapeVector& fast_axes,
                            FastReduceKind which_fast_reduce,
                            bool rk_krk_scale_to_any_shape,
                            fast_reduce_fct* case_kr,
                            fast_reduce_fct* case_rk,
                            fast_reduce_fct* case_krk,
                            fast_reduce_fct* case_rkr) {
  const Tensor* input = ctx->Input<Tensor>(0);
  auto reduced_dims = input->Shape().GetDims();
  TensorShapeVector input_axes;

  if (CommonFastReduceCopy(ctx, input_axes, noop_with_empty_axes)) {
    return true;
  }

  fast_kind = OptimizeShapeForFastReduce(
      reduced_dims, input_axes.empty() ? axes_ : input_axes,
      fast_shape, output_shape, fast_axes, keepdims_ != 0, noop_with_empty_axes);
  if (which_fast_reduce != FastReduceKind::kNone) {
    if (IsFastReduceKindAvailable(fast_kind, which_fast_reduce)) {
      Tensor* output = ctx->Output(0, output_shape);
      switch (fast_kind) {
        case FastReduceKind::kKR: {
          ValidateFastReduceKR(fast_shape, *output);
          case_kr(*input, fast_shape, *output, ctx->GetOperatorThreadPool());
          return true;
        }
        case FastReduceKind::kRK: {
          ValidateFastReduceRK(fast_shape, *output);
          if (rk_krk_scale_to_any_shape ||
              ((fast_shape[0] > concurrency::ThreadPool::DegreeOfParallelism(ctx->GetOperatorThreadPool()) * 16) &&
               (std::max(fast_shape[0], fast_shape[1]) >
                concurrency::ThreadPool::DegreeOfParallelism(ctx->GetOperatorThreadPool()) * 256))) {
            // See benchmarks in PR #7719.
            case_rk(*input, fast_shape, *output, ctx->GetOperatorThreadPool());
            return true;
          } else {
            break;
          }
        }
        case FastReduceKind::kKRK:
          ValidateFastReduceKRK(fast_shape, *output);
          if (rk_krk_scale_to_any_shape ||
              fast_shape[0] >= std::max(2, concurrency::ThreadPool::DegreeOfParallelism(ctx->GetOperatorThreadPool()))) {
            // See benchmarks in PR #7719.
            case_krk(*input, fast_shape, *output, ctx->GetOperatorThreadPool());
            return true;
          } else {
            break;
          }
        case FastReduceKind::kRKR:
          ValidateFastReduceRKR(fast_shape, *output);
          if (fast_shape[1] >= std::max(2, concurrency::ThreadPool::DegreeOfParallelism(ctx->GetOperatorThreadPool()))) {
            case_rkr(*input, fast_shape, *output, ctx->GetOperatorThreadPool());
            return true;
          } else {
            break;
          }
        case FastReduceKind::kR:
        case FastReduceKind::kK:
        case FastReduceKind::kNone:
        default:
          // Former implementation prevails in this case.
          break;
      }
    }
  }
  return false;
}

template <typename AGG>
bool CommonFastReduce(OpKernelContext* ctx,
                      const gsl::span<const int64_t>& axes_,
                      int64_t keepdims_,
                      bool noop_with_empty_axes,
                      FastReduceKind& fast_kind,
                      TensorShapeVector& fast_shape,
                      TensorShapeVector& output_shape,
                      TensorShapeVector& fast_axes) {
  return CommonFastReduceSwitch(ctx, axes_, keepdims_, noop_with_empty_axes,
                                fast_kind, fast_shape, output_shape, fast_axes,
                                AGG::WhichFastReduce(), AGG::FastReduceRKAndKRKScaleToAnyShape(),
                                &AGG::FastReduceKR, &AGG::FastReduceRK,
                                &AGG::FastReduceKRK, &AGG::FastReduceRKR);
}

static void ValidateKeepDims(const TensorShape& shape, int64_t keepdims) {
  ORT_ENFORCE(keepdims,
              "Can't reduce on dim with value of 0 if 'keepdims' is false. "
              "Invalid output shape would be produced. input_shape:",
              shape);
}

static void ValidateKeepDims(const Tensor* input, int64_t keepdims) {
  ValidateKeepDims(input->Shape(), keepdims);
}

template <typename AGG>
bool check_and_reduce_empty_set_input(OpKernelContext* ctx, const gsl::span<const int64_t> axes, bool keepdims) {
  const Tensor* input = ctx->Input<Tensor>(0);
  const TensorShape& input_shape = input->Shape();
  if (input_shape.Size() != 0) {
    return false;
  }

  // input is an empty set
  std::vector<int64_t> input_axes;
  if (ctx->InputCount() == 2) {
    ORT_ENFORCE(axes.empty(), "Axes input and attribute should not both be present for reduction.");
    // second input holds the axes.
    const Tensor* axes_tensor = ctx->Input<Tensor>(1);
    if (axes_tensor != nullptr) {
      ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1,
                  "An axes tensor must be a vector tensor.");
      const auto axes_data = axes_tensor->DataAsSpan<int64_t>();
      input_axes.assign(axes_data.begin(), axes_data.end());
    }
  } else {
    input_axes.resize(axes.size());
    std::copy(axes.begin(), axes.end(), input_axes.begin());
  }

  gsl::span<const int64_t> shape_dims = input_shape.GetDims();
  const int64_t input_shape_size = narrow<int64_t>(shape_dims.size());
  for (auto& axis : input_axes) {
    axis = HandleNegativeAxis(axis, input_shape_size);
  }

  TensorShapeVector output_shape_vector;
  for (int64_t i = 0; i < input_shape_size; ++i) {
    if (input_axes.empty() || std::find(input_axes.begin(), input_axes.end(), i) != input_axes.end()) {
      if (keepdims) {
        output_shape_vector.push_back(1);
      }
    } else {
      output_shape_vector.push_back(input_shape[onnxruntime::narrow<size_t>(i)]);
    }
  }

  TensorShape output_shape(output_shape_vector);
  Tensor* output = ctx->Output(0, output_shape);
  if (output_shape.Size() != 0) {
    AGG::fill_for_empty_set(*output);
  }
  return true;
}

// Handles the ONNX noop_with_empty_axes=1 behavior when axes=[] for reduction operators.
// In this case no reduction is performed. Instead, for aggregators with element-wise
// transforms (PreOp/PostOp), the transform is applied to each input element.
// For identity aggregators, the input is copied directly to the output.
template <typename AGG>
inline void ApplyNoopEmptyAxesElementwise(OpKernelContext* ctx) {
  const Tensor* X = ctx->Input<Tensor>(0);
  const auto& shape = X->Shape();
  Tensor* Y = ctx->Output(0, shape);

  if constexpr (!ReduceAggTraits<AGG>::kHasPreOp && !ReduceAggTraits<AGG>::kHasPostOp) {
    if (X->SizeInBytes() > 0) {
      std::memcpy(Y->MutableDataRaw(), X->DataRaw(), X->SizeInBytes());
    }

  } else {
    using Tin = typename AGG::input_type;
    using Tacc = typename AGG::value_type;
    const Tin* x = X->Data<Tin>();
    Tacc* y = Y->MutableData<Tacc>();
    const int64_t n = shape.Size();

    for (int64_t i = 0; i < n; ++i) {
      AGG agg(1, x[i]);
      agg.update0(x[i]);
      agg.update(x[i]);
      y[i] = agg.get_value();
    }
  }
}

// Returns the effective reduction axes.
// If an axes input tensor is provided, it takes precedence over the axes attribute.
inline gsl::span<const int64_t> GetEffectiveAxes(
    OpKernelContext* ctx,
    const gsl::span<const int64_t> axes_attr,
    TensorShapeVector& axes_from_input) {
  axes_from_input.clear();

  if (ctx->InputCount() > 1) {
    const Tensor* axes_tensor = ctx->Input<Tensor>(1);
    if (axes_tensor) {
      ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1, "An axes tensor must be a vector tensor.");
      const auto span = axes_tensor->DataAsSpan<int64_t>();
      axes_from_input.assign(span.begin(), span.end());
      return gsl::make_span(axes_from_input);
    }
  }

  return axes_attr;
}

template <typename AGG>
void CommonReduce1Loop(OpKernelContext* ctx,
                       const gsl::span<const int64_t>& axes_, int64_t keepdims_,
                       bool noop_with_empty_axes) {
  TensorShapeVector tmp_axes;
  auto effective_axes = GetEffectiveAxes(ctx, axes_, tmp_axes);
  if (effective_axes.empty() && noop_with_empty_axes) {
    ApplyNoopEmptyAxesElementwise<AGG>(ctx);
    return;
  }

  if (check_and_reduce_empty_set_input<AGG>(ctx, axes_, keepdims_ != 0)) {
    return;
  }

  FastReduceKind fast_kind;
  TensorShapeVector fast_shape;
  TensorShapeVector output_shape;
  TensorShapeVector fast_axes;
  if (CommonFastReduce<AGG>(ctx, axes_, keepdims_, noop_with_empty_axes,
                            fast_kind, fast_shape, output_shape, fast_axes)) {
    return;
  }

  const Tensor* input = ctx->Input<Tensor>(0);
  Tensor* output = ctx->Output(0, output_shape);

  if (fast_kind == FastReduceKind::kEmpty) {
    const TensorShape& input_shape = input->Shape();
    if (input_shape.Size() == 1) {
      const typename AGG::input_type* from_data = input->Data<typename AGG::input_type>();
      typename AGG::value_type* to_data = output->MutableData<typename AGG::value_type>();
      AGG agg(1, *from_data);
      agg.update(*from_data);
      *to_data = agg.get_value();
    } else {
      ValidateKeepDims(input, keepdims_);
    }
    return;
  }

  ResultsNoTransposePrepareForReduce last_results;
  NoTransposeReduce1Loop<AGG>(output, fast_shape, *input, fast_axes, ctx->GetOperatorThreadPool(), last_results);
}

template <typename AGG>
void CommonReduce2Loops(OpKernelContext* ctx,
                        const gsl::span<const int64_t>& axes_, int64_t keepdims_,
                        bool noop_with_empty_axes) {
  TensorShapeVector tmp_axes;
  auto effective_axes = GetEffectiveAxes(ctx, axes_, tmp_axes);
  if (effective_axes.empty() && noop_with_empty_axes) {
    ApplyNoopEmptyAxesElementwise<AGG>(ctx);
    return;
  }

  if (check_and_reduce_empty_set_input<AGG>(ctx, axes_, keepdims_ != 0)) {
    return;
  }

  FastReduceKind fast_kind;
  TensorShapeVector fast_shape, output_shape, fast_axes;
  if (CommonFastReduce<AGG>(ctx, axes_, keepdims_, noop_with_empty_axes,
                            fast_kind, fast_shape, output_shape, fast_axes)) {
    return;
  }

  const Tensor* input = ctx->Input<Tensor>(0);
  Tensor* output = ctx->Output(0, output_shape);
  if (fast_kind == FastReduceKind::kEmpty) {
    const TensorShape& input_shape = input->Shape();
    if (input_shape.Size() == 1) {
      const typename AGG::input_type* from_data = input->Data<typename AGG::input_type>();
      typename AGG::value_type* to_data = output->MutableData<typename AGG::value_type>();
      AGG agg(1, *from_data);
      agg.update0(*from_data);
      agg.update(*from_data);
      *to_data = agg.get_value();
    } else {
      ValidateKeepDims(input, keepdims_);
    }
    return;
  }
  ResultsNoTransposePrepareForReduce last_results;
  NoTransposeReduce2Loops<AGG>(output, fast_shape, *input, fast_axes, ctx->GetOperatorThreadPool(), last_results);
}

template <typename T>
Status ReduceL1<T>::Compute(OpKernelContext* ctx) const {
  // The following variable does not change if the input tensor and the
  // axes do not either. It could be either cached in ctx or precomputed
  // in the constructor if shape and axes are known at this stage.
  CommonReduce1Loop<ReduceAggregatorL1<T>>(ctx, axes_, keepdims_, noop_with_empty_axes_);
  return Status::OK();
}

template <typename T>
Status ReduceL2<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce1Loop<ReduceAggregatorL2<T>>(ctx, axes_, keepdims_, noop_with_empty_axes_);
  return Status::OK();
}

template <typename T>
Status ReduceLogSum<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce1Loop<ReduceAggregatorLogSum<T>>(ctx, axes_, keepdims_, noop_with_empty_axes_);
  return Status::OK();
}

template <typename T>
Status ReduceLogSumExp<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce2Loops<ReduceAggregatorLogSumExp<T>>(ctx, axes_, keepdims_, noop_with_empty_axes_);
  return Status::OK();
}

template <typename T>
Status ReduceMax<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce1Loop<ReduceAggregatorMax<T>>(ctx, axes_, keepdims_, noop_with_empty_axes_);
  return Status::OK();
}

template <typename T>
Status ReduceMean<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce1Loop<ReduceAggregatorMean<T>>(ctx, axes_, keepdims_, noop_with_empty_axes_);
  return Status::OK();
}

template <typename T>
Status ReduceMin<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce1Loop<ReduceAggregatorMin<T>>(ctx, axes_, keepdims_, noop_with_empty_axes_);
  return Status::OK();
}

template <typename T>
Status ReduceProd<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce1Loop<ReduceAggregatorProd<T>>(ctx, axes_, keepdims_, noop_with_empty_axes_);
  return Status::OK();
}

template <typename T>
Status ReduceSum<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce1Loop<ReduceAggregatorSum<T>>(ctx, axes_, keepdims_, noop_with_empty_axes_);
  return Status::OK();
}

template <typename T>
std::unique_ptr<Tensor> ReduceSum<T>::Impl(const Tensor& input, gsl::span<const int64_t> reduce_axes,
                                           AllocatorPtr allocator, concurrency::ThreadPool* tp, bool keep_dims,
                                           const TensorShape* input_shape_override) {
  TensorShapeVector output_shape, fast_shape, fast_axes;
  TensorShape new_input_shape = input_shape_override == nullptr ? input.Shape() : *input_shape_override;
  auto reduced_dims = new_input_shape.GetDims();

  FastReduceKind fast_kind = OptimizeShapeForFastReduce(
      reduced_dims, reduce_axes, fast_shape, output_shape, fast_axes, keep_dims, false);

  auto output = std::make_unique<Tensor>(input.DataType(), keep_dims ? output_shape : TensorShapeVector(), allocator);

  if (fast_kind == FastReduceKind::kEmpty) {
    if (new_input_shape.Size() == 1) {
      const T* from_data = input.Data<T>();
      T* to_data = output->MutableData<T>();
      *to_data = *from_data;
    } else {
      ValidateKeepDims(new_input_shape, keep_dims);
    }
    return output;
  }

  if (IsFastReduceKindAvailable(fast_kind, ReduceAggregatorSum<T>::WhichFastReduce())) {
    switch (fast_kind) {
      case FastReduceKind::kKR: {
        ValidateFastReduceKR(fast_shape, *output);
        ReduceAggregatorSum<T>::FastReduceKR(input, fast_shape, *output, tp);
        return output;
      }
      case FastReduceKind::kRK:
        ValidateFastReduceRK(fast_shape, *output);
        if (ReduceAggregatorSum<T>::FastReduceRKAndKRKScaleToAnyShape() ||
            std::max(fast_shape[0], fast_shape[1]) >
                concurrency::ThreadPool::DegreeOfParallelism(tp) * 256) {
          // See benchmarks in PR #7719.
          ReduceAggregatorSum<T>::FastReduceRK(input, fast_shape, *output, tp);
          return output;
        } else {
          break;
        }
      case FastReduceKind::kKRK:
        ValidateFastReduceKRK(fast_shape, *output);
        if (ReduceAggregatorSum<T>::FastReduceRKAndKRKScaleToAnyShape() ||
            fast_shape[0] >= std::max(2, concurrency::ThreadPool::DegreeOfParallelism(tp))) {
          // See benchmarks in PR #7719.
          ReduceAggregatorSum<T>::FastReduceKRK(input, fast_shape, *output, tp);
          return output;
        } else {
          break;
        }
      case FastReduceKind::kRKR:
        ValidateFastReduceRKR(fast_shape, *output);
        if (fast_shape[0] >= std::max(2, concurrency::ThreadPool::DegreeOfParallelism(tp))) {
          ReduceAggregatorSum<T>::FastReduceRKR(input, fast_shape, *output, tp);
          return output;
        } else {
          break;
        }
      case FastReduceKind::kR:
      case FastReduceKind::kK:
      case FastReduceKind::kNone:
      default:
        // Former implementation prevails in this case.
        break;
    }
  }

  ResultsNoTransposePrepareForReduce last_results;
  NoTransposeReduce1Loop<ReduceAggregatorSum<T>>(output.get(), fast_shape, input, fast_axes, tp, last_results);
  return output;
}

template <typename T>
Status ReduceSumSquare<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce1Loop<ReduceAggregatorSumSquare<T>>(ctx, axes_, keepdims_, noop_with_empty_axes_);
  return Status::OK();
}

template <typename T>
Status ArgMax<T>::Compute(OpKernelContext* ctx) const {
  if (select_last_index_) {
    CommonReduce1Loop<ReduceAggregatorArgMaxLastIndex<T>>(ctx, axes_, keepdims_);
  } else {
    CommonReduce1Loop<ReduceAggregatorArgMax<T>>(ctx, axes_, keepdims_);
  }
  return Status::OK();
}

template <typename T>
Status ArgMin<T>::Compute(OpKernelContext* ctx) const {
  if (select_last_index_) {
    CommonReduce1Loop<ReduceAggregatorArgMinLastIndex<T>>(ctx, axes_, keepdims_);
  } else {
    CommonReduce1Loop<ReduceAggregatorArgMin<T>>(ctx, axes_, keepdims_);
  }
  return Status::OK();
}

// Explicit template instantiation -
// Even though there are kernels registered for ReduceSum op for these types,
// these are needed because we seem to get linker errors without these when the linker
// tries to resolve symbols in the einsum_auxiliary_ops obj file
template class ReduceSum<float>;
template class ReduceSum<int32_t>;
template class ReduceSum<double>;
template class ReduceSum<int64_t>;

template void CommonReduce1Loop<ReduceAggregatorSum<float>>(OpKernelContext* ctx,
                                                            const gsl::span<const int64_t>& axes_, int64_t keepdims_,
                                                            bool noop_with_empty_axes);
template void CommonReduce1Loop<ReduceAggregatorSum<int32_t>>(OpKernelContext* ctx,
                                                              const gsl::span<const int64_t>& axes_, int64_t keepdims_,
                                                              bool noop_with_empty_axes);
template void CommonReduce1Loop<ReduceAggregatorSum<double>>(OpKernelContext* ctx,
                                                             const gsl::span<const int64_t>& axes_, int64_t keepdims_,
                                                             bool noop_with_empty_axes);
template void CommonReduce1Loop<ReduceAggregatorSum<int64_t>>(OpKernelContext* ctx,
                                                              const gsl::span<const int64_t>& axes_, int64_t keepdims_,
                                                              bool noop_with_empty_axes);

}  // namespace onnxruntime
