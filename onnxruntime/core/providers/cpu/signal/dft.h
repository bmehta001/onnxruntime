// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/signal/fft.h"

namespace onnxruntime {

class DFT final : public OpKernel {
  int opset_;
  bool is_onesided_ = true;
  int64_t axis_ = 0;
  bool is_inverse_ = false;
  mutable signal::fft::PlanCache plan_cache_;

 public:
  explicit DFT(const OpKernelInfo& info) : OpKernel(info) {
    is_onesided_ = static_cast<bool>(info.GetAttrOrDefault<int64_t>("onesided", 0));
    opset_ = info.node().SinceVersion();
    if (opset_ < 20)
      axis_ = info.GetAttrOrDefault<int64_t>("axis", 1);
    else
      axis_ = -2;  // default axis of DFT(20)
    is_inverse_ = info.GetAttrOrDefault<int64_t>("inverse", 0);
  }
  Status Compute(OpKernelContext* ctx) const override;
};

class STFT final : public OpKernel {
  bool is_onesided_ = true;
  mutable signal::fft::PlanCache plan_cache_;

 public:
  explicit STFT(const OpKernelInfo& info) : OpKernel(info) {
    is_onesided_ = static_cast<bool>(info.GetAttrOrDefault<int64_t>("onesided", 1));
  }
  Status Compute(OpKernelContext* ctx) const override;
};

}  // namespace onnxruntime
