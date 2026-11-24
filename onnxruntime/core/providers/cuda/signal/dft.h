// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/math/cufft_plan_cache.h"

namespace onnxruntime {
namespace cuda {

class DFT final : public CudaKernel {
 public:
  explicit DFT(const OpKernelInfo& info) : CudaKernel(info) {
    is_onesided_ = info.GetAttrOrDefault<int64_t>("onesided", 0) != 0;
    opset_ = info.node().SinceVersion();
    axis_ = opset_ < 20 ? info.GetAttrOrDefault<int64_t>("axis", 1) : -2;
    is_inverse_ = info.GetAttrOrDefault<int64_t>("inverse", 0) != 0;
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext* context, const Tensor& input, Tensor& output, int64_t axis,
                     int64_t dft_length) const;

  int opset_;
  bool is_onesided_;
  int64_t axis_;
  bool is_inverse_;
  mutable contrib::cuda::CuFFTPlanCache cufft_cache_;
};

class STFT final : public CudaKernel {
 public:
  explicit STFT(const OpKernelInfo& info) : CudaKernel(info) {
    is_onesided_ = info.GetAttrOrDefault<int64_t>("onesided", 1) != 0;
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext* context, const Tensor& signal, const Tensor* window, Tensor& output,
                     int64_t frame_step, int64_t frame_length) const;

  bool is_onesided_;
  mutable contrib::cuda::CuFFTPlanCache cufft_cache_;
};

}  // namespace cuda
}  // namespace onnxruntime
