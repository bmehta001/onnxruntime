// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"
#include "core/mlas/lib/mlasi.h"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {

// Same tolerances as the AVX512F GELU(erf) kernel test.
constexpr float kGeluAbsoluteTolerance = 2e-6f;
constexpr float kGeluRelativeTolerance = 2e-5f;

// Bound against a double-precision erf reference.
constexpr double kReferenceAbsoluteTolerance = 1e-6;
constexpr double kReferenceRelativeTolerance = 1e-5;

constexpr float kTailSentinel = 12345.0f;

constexpr std::array<size_t, 33> kSizes = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33,
    63, 64, 65, 127, 128, 129, 255, 511, 1023, 1024, 1025, 3071, 4095, 4097};

bool OutputsMatch(float actual, float expected) {
  if (std::isnan(expected)) {
    return std::isnan(actual);
  }
  if (std::isinf(expected)) {
    return std::isinf(actual) && std::signbit(actual) == std::signbit(expected);
  }
  if (actual == 0.0f && expected == 0.0f) {
    return std::signbit(actual) == std::signbit(expected);
  }
  const float diff = std::fabs(actual - expected);
  if (diff <= kGeluAbsoluteTolerance) {
    return true;
  }
  const float scale = std::max(std::fabs(actual), std::fabs(expected));
  return diff <= scale * kGeluRelativeTolerance;
}

const std::vector<float>& GetSpecialValues() {
  constexpr float kSqrt2 = 1.41421356237309504880f;
  static const std::vector<float> values = {
      std::numeric_limits<float>::quiet_NaN(),
      -std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::max(),
      std::numeric_limits<float>::min(),
      -std::numeric_limits<float>::min(),
      std::numeric_limits<float>::denorm_min(),
      -std::numeric_limits<float>::denorm_min(),
      1.0e30f,
      -1.0e30f,
      1.0e4f,
      -1.0e4f,
      0.0f,
      -0.0f,
      // erf small/large polynomial split and upper clamp, mapped back through x / sqrt(2).
      0.921875f * kSqrt2,
      -0.921875f * kSqrt2,
      std::nextafter(0.921875f * kSqrt2, 0.0f),
      3.925f * kSqrt2,
      -3.925f * kSqrt2,
      -5.5f,
      -10.0f,
      10.0f,
  };
  return values;
}

void FillInput(float* input, size_t size, uint32_t seed, float min_value, float max_value) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(min_value, max_value);
  for (size_t i = 0; i < size + 1; ++i) {
    input[i] = distribution(generator);
  }
  const auto& special_values = GetSpecialValues();
  // Rotate special values through every lane position across seeds.
  for (size_t i = 0; i < std::min(size, special_values.size()); ++i) {
    input[(i + seed) % size] = special_values[i];
  }
}

// Unfused GELU(erf) built from the portable erf kernel; this is what the
// generic GELU(erf) kernel computed before it was fused into a single pass.
void ComputeMultiPassGeluErf(const float* input, float* output, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    output[i] = input[i] * 0.70710678118654752440f;
  }
  MlasErfKernel(output, output, size);
  for (size_t i = 0; i < size; ++i) {
    output[i] = 0.5f * input[i] * (output[i] + 1.0f);
  }
}

// Checks a GELU(erf) kernel against an expected MLAS result and a double-precision erf reference.
void CheckGeluErf(const char* name, const float* input, const float* actual, const float* expected, size_t size) {
  ASSERT_EQ(actual[size], kTailSentinel) << name << " wrote past the end, size=" << size;

  for (size_t i = 0; i < size; ++i) {
    ASSERT_TRUE(OutputsMatch(actual[i], expected[i]))
        << name << " mismatch vs expected at index " << i << " of " << size
        << ", input=" << input[i] << ", actual=" << actual[i] << ", expected=" << expected[i];

    const double x = static_cast<double>(input[i]);
    if (std::isfinite(x) && std::fabs(x) <= 1.0e4) {
      const double reference = 0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440));
      const double diff = std::fabs(static_cast<double>(actual[i]) - reference);
      ASSERT_TRUE(diff <= kReferenceAbsoluteTolerance || diff <= std::fabs(reference) * kReferenceRelativeTolerance)
          << name << " mismatch vs double reference at index " << i << " of " << size
          << ", input=" << input[i] << ", actual=" << actual[i] << ", reference=" << reference;
    }
  }
}

template <typename Runner>
void RunGeluErfCases(Runner&& run) {
  for (size_t size : kSizes) {
    for (uint32_t seed = 0; seed < 3; ++seed) {
      run(size, seed * 7919u + static_cast<uint32_t>(size), -10.0f, 10.0f);
      run(size, seed * 104729u + static_cast<uint32_t>(size), -1.5f, 1.5f);
    }
  }
}

class MlasGeluErfGenericTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> input_buffer_;
  MatrixGuardBuffer<float> expected_buffer_;
  MatrixGuardBuffer<float> output_buffer_;

  void Execute(size_t size, uint32_t seed, float min_value, float max_value) {
    // Allocate one extra element to detect writes past the tail.
    float* input = input_buffer_.GetBuffer(size + 1);
    float* expected = expected_buffer_.GetBuffer(size + 1);
    float* output = output_buffer_.GetBuffer(size + 1);

    FillInput(input, size, seed, min_value, max_value);
    for (size_t i = 0; i < size + 1; ++i) {
      output[i] = kTailSentinel;
    }

    ComputeMultiPassGeluErf(input, expected, size);
    MlasGeluErfKernel(input, output, size);

    CheckGeluErf("Generic GELU(erf)", input, output, expected, size);
  }

 public:
  static const char* GetTestSuiteName() {
    return "GeluErfGeneric";
  }

  void ExecuteShort() override {
    RunGeluErfCases([this](size_t size, uint32_t seed, float min_value, float max_value) {
      Execute(size, seed, min_value, max_value);
    });
  }
};

#if defined(MLAS_TARGET_AMD64)

class MlasGeluErfFma3Test : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> input_buffer_;
  MatrixGuardBuffer<float> generic_output_buffer_;
  MatrixGuardBuffer<float> fma3_output_buffer_;

  void Execute(size_t size, uint32_t seed, float min_value, float max_value) {
    // Allocate one extra element to detect writes past the tail.
    float* input = input_buffer_.GetBuffer(size + 1);
    float* generic_output = generic_output_buffer_.GetBuffer(size + 1);
    float* fma3_output = fma3_output_buffer_.GetBuffer(size + 1);

    FillInput(input, size, seed, min_value, max_value);
    for (size_t i = 0; i < size + 1; ++i) {
      fma3_output[i] = kTailSentinel;
    }

    MlasGeluErfKernel(input, generic_output, size);
    MlasGeluErfKernelFma3(input, fma3_output, size);

    CheckGeluErf("FMA3 GELU(erf)", input, fma3_output, generic_output, size);
  }

 public:
  static const char* GetTestSuiteName() {
    return "GeluErfFma3";
  }

  void ExecuteShort() override {
    if (!GetMlasPlatform().Avx2Supported_) {
      GTEST_SKIP() << "AVX2/FMA3 is not available on this machine.";
    }

    RunGeluErfCases([this](size_t size, uint32_t seed, float min_value, float max_value) {
      Execute(size, seed, min_value, max_value);
    });
  }
};

class MlasGeluErfFma3DispatchTest : public MlasTestBase {
 public:
  static const char* GetTestSuiteName() {
    return "GeluErfFma3Dispatch";
  }

  void ExecuteShort() override {
    const auto& platform = GetMlasPlatform();
    if (!platform.Avx2Supported_) {
      GTEST_SKIP() << "AVX2/FMA3 is not available on this machine.";
    }
    if (platform.GeluErfKernelRoutine == MlasGeluErfKernelAvx512F) {
      GTEST_SKIP() << "AVX512F GELU(erf) kernel takes precedence on this machine.";
    }
    EXPECT_EQ(platform.GeluErfKernelRoutine, MlasGeluErfKernelFma3);
  }
};

#endif

}  // namespace

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasGeluErfGenericTest>::RegisterShortExecute();
#if defined(MLAS_TARGET_AMD64)
    count += MlasDirectShortExecuteTests<MlasGeluErfFma3Test>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasGeluErfFma3DispatchTest>::RegisterShortExecute();
#endif
  }
  return count;
});
