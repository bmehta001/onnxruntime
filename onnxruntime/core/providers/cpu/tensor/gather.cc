// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather
#include "core/providers/cpu/tensor/gather.h"
#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel_type_control_utils.h"
#include "core/platform/threadpool.h"
#include "core/providers/op_kernel_type_control.h"

namespace onnxruntime {

using DefaultIndexTypes = TypeList<int32_t, int64_t>;

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Gather, Input, 1, DefaultIndexTypes);

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
// enable all types for layout transformation
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Gather, Input, 1, DefaultIndexTypes);
#else
ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Gather, Input, 1, int32_t, int64_t);
#endif
}  // namespace op_kernel_type_control

using EnabledIndexTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                         Gather, Input, 1);
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Gather,
    1,
    10,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<EnabledIndexTypes>()),
    Gather);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Gather,
    11,
    12,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<EnabledIndexTypes>()),
    Gather);

ONNX_CPU_OPERATOR_KERNEL(
    Gather,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<EnabledIndexTypes>()),
    Gather);

template <typename Tin>
Status GatherCopyData(const Tensor* indices_tensor, const uint8_t* src_base, uint8_t* dst_base, bool is_string_type,
                      const int64_t block_size, const int64_t M,
                      const int64_t N, const int64_t data_batch_bytes, const int64_t gathered_batch_bytes,
                      const TensorShape& input_data_shape, const int64_t axis, concurrency::ThreadPool* tp) {
  const Tin* indices_data = indices_tensor->Data<Tin>();

  // Check the indices first in case there's a out of bound index.
  auto axis_dim_limit = input_data_shape[narrow<size_t>(axis)];

  for (int64_t i = 0; i < N; ++i) {
    Tin idx = indices_data[i];
    if (idx < -axis_dim_limit || idx >= axis_dim_limit) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "indices element out of data bounds, idx=", idx,
                             " must be within the inclusive range [", -axis_dim_limit, ",", axis_dim_limit - 1, "]");
    }
  }

  // Iterate a [first, last) range of the flattened M x N gather positions incrementally, as per-element division
  // and an out-of-line memcpy dominate when block_size is small.
  auto copy_range = [&](ptrdiff_t first, ptrdiff_t last, auto copy_block) {
    // TryParallelFor may invoke fn(0, 0) for an empty range; avoid dividing by N == 0.
    if (first >= last) {
      return;
    }
    int64_t batch = static_cast<int64_t>(first / N);
    int64_t i = static_cast<int64_t>(first % N);
    const uint8_t* src_batch = src_base + batch * data_batch_bytes;
    uint8_t* dst = dst_base + batch * gathered_batch_bytes + i * block_size;
    for (ptrdiff_t index = first; index < last; ++index) {
      Tin idx = indices_data[i];
      idx = idx < 0 ? idx + static_cast<Tin>(axis_dim_limit) : idx;
      copy_block(dst, src_batch + idx * block_size);
      dst += block_size;
      if (++i == N) {
        i = 0;
        ++batch;
        src_batch += data_batch_bytes;
        dst = dst_base + batch * gathered_batch_bytes;
      }
    }
  };

  auto parallel_copy = [&](auto copy_block) {
    concurrency::ThreadPool::TryParallelFor(
        tp, SafeInt<ptrdiff_t>(M) * N, static_cast<double>(block_size),
        [&copy_range, &copy_block](ptrdiff_t first, ptrdiff_t last) { copy_range(first, last, copy_block); });
  };

  if (is_string_type) {
    parallel_copy([](uint8_t* dst, const uint8_t* src) {
      *reinterpret_cast<std::string*>(dst) = *reinterpret_cast<const std::string*>(src);
    });
  } else if (block_size == 1) {
    parallel_copy([](uint8_t* dst, const uint8_t* src) { *dst = *src; });
  } else if (block_size == 2) {
    parallel_copy([](uint8_t* dst, const uint8_t* src) { memcpy(dst, src, 2); });
  } else if (block_size == 4) {
    parallel_copy([](uint8_t* dst, const uint8_t* src) { memcpy(dst, src, 4); });
  } else if (block_size == 8) {
    parallel_copy([](uint8_t* dst, const uint8_t* src) { memcpy(dst, src, 8); });
  } else {
    const size_t block_bytes = narrow<size_t>(block_size);
    parallel_copy([block_bytes](uint8_t* dst, const uint8_t* src) { memcpy(dst, src, block_bytes); });
  }

  return Status::OK();
}

Status Gather::Compute(OpKernelContext* context) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(context, p));

  const TensorShape& input_data_shape = p.input_tensor->Shape();

  bool is_string_type = p.input_tensor->IsDataTypeString();

  const size_t element_bytes = p.input_tensor->DataType()->Size();
  const int64_t block = input_data_shape.SizeFromDimension(SafeInt<size_t>(p.axis) + 1);
  const int64_t block_size = SafeInt<int64_t>(element_bytes) * block;
  const int64_t M = input_data_shape.SizeToDimension(narrow<size_t>(p.axis));
  const int64_t N = p.indices_tensor->Shape().Size();
  const int64_t data_batch_bytes = input_data_shape.SizeFromDimension(narrow<size_t>(p.axis)) * element_bytes;
  const int64_t gathered_batch_bytes = N * block * SafeInt<int64_t>(element_bytes);

  const auto* src_base = static_cast<const uint8_t*>(p.input_tensor->DataRaw());
  auto* dst_base = static_cast<uint8_t*>(p.output_tensor->MutableDataRaw());

  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();

  if (utils::HasType<EnabledIndexTypes, int32_t>() &&
      p.indices_tensor->IsDataType<int32_t>()) {
    return GatherCopyData<int32_t>(p.indices_tensor, src_base, dst_base, is_string_type,
                                   block_size, M, N, data_batch_bytes, gathered_batch_bytes, input_data_shape, p.axis, tp);
  }
  if (utils::HasType<EnabledIndexTypes, int64_t>() &&
      p.indices_tensor->IsDataType<int64_t>()) {
    return GatherCopyData<int64_t>(p.indices_tensor, src_base, dst_base, is_string_type,
                                   block_size, M, N, data_batch_bytes, gathered_batch_bytes, input_data_shape, p.axis, tp);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Gather Tind type not supported in this build.");
}

}  // namespace onnxruntime
