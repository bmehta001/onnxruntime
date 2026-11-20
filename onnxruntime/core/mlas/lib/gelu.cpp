/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    gelu.cpp

Abstract:

    This module implements routines to compute the exact Gelu function.

--*/

#include <cstring>
#include <numbers>

#include "mlasi.h"
#include "erf_float32x4.h"

#if defined(MLAS_USE_SVE)

static
void
MlasGeluErfKernelMultiPass(
    const float* Input,
    float* Output,
    size_t N
    )
{
    // Not buffer alias safe: Output holds the erf intermediate while Input is
    // still needed by the final pass.
    for (size_t i = 0; i < N; ++i) {
        Output[i] = Input[i] * (1.0f / std::numbers::sqrt2_v<float>);
    }

    MlasComputeErf(Output, Output, N);

    for (size_t i = 0; i < N; ++i) {
        Output[i] = 0.5f * Input[i] * (Output[i] + 1.0f);
    }
}

#endif

struct MLAS_GELU_ERF_FLOAT32X4_CONSTANTS : MLAS_ERF_FLOAT32X4_CONSTANTS {
    MLAS_FLOAT32X4 Half;
    MLAS_FLOAT32X4 InvSqrt2;

    MLAS_GELU_ERF_FLOAT32X4_CONSTANTS()
        : Half(MlasBroadcastFloat32x4(MlasErfConstants.Gelu_Half)),
          InvSqrt2(MlasBroadcastFloat32x4(MlasErfConstants.Gelu_InvSqrt2))
    {
    }
};

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasGeluErfFloat32x4(
    MLAS_FLOAT32X4 Value,
    const MLAS_GELU_ERF_FLOAT32X4_CONSTANTS& C
    )
{
    MLAS_FLOAT32X4 ErfValue = MlasErfFloat32x4(MlasMultiplyFloat32x4(Value, C.InvSqrt2), C);

    // Keep 0.5 * x * (erf + 1) unfused so that the result matches the
    // multi-pass formulation, including the sign of zero for large negative x.
    return MlasMultiplyFloat32x4(MlasMultiplyFloat32x4(C.Half, Value), MlasAddFloat32x4(ErfValue, C.One));
}

void
MLASCALL
MlasGeluErfKernel(
    const float* Input,
    float* Output,
    size_t N
    )
{
    // Callers must guarantee that Input and Output do not overlap (see mlas.h
    // for aliasing requirements).
#if defined(MLAS_USE_SVE)
    // Keep the dispatched SVE erf kernel when it is available.
    if (GetMlasPlatform().ErfKernelRoutine != MlasErfKernel) {
        MlasGeluErfKernelMultiPass(Input, Output, N);
        return;
    }
#endif

    const MLAS_GELU_ERF_FLOAT32X4_CONSTANTS Constants;

    while (N >= 4) {
        MlasStoreFloat32x4(Output, MlasGeluErfFloat32x4(MlasLoadFloat32x4(Input), Constants));
        Input += 4;
        Output += 4;
        N -= 4;
    }

    if (N > 0) {
        // Run the tail through the vector path as well so that every element
        // uses the same approximation.
        float Buffer[4] = {};
        std::memcpy(Buffer, Input, N * sizeof(float));
        MlasStoreFloat32x4(Buffer, MlasGeluErfFloat32x4(MlasLoadFloat32x4(Buffer), Constants));
        std::memcpy(Output, Buffer, N * sizeof(float));
    }
}

void
MLASCALL
MlasComputeGeluErf(
    const float* Input,
    float* Output,
    size_t N
    )
{
#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_RISCV64)
    GetMlasPlatform().GeluErfKernelRoutine(Input, Output, N);
#else
    MlasGeluErfKernel(Input, Output, N);
#endif
}

void
MLASCALL
MlasComputeFP16Gelu(const MLAS_FP16* input,
                    MLAS_FP16* output,
                    MLAS_FP16* temp,
                    size_t count,
                    MLAS_GELU_ALGORITHM algo)
{
    if(GetMlasPlatform().GeluFP16KernelRoutine){
        GetMlasPlatform().GeluFP16KernelRoutine(input, output, temp, count, algo);
        return;
    }
    MLAS_UNREFERENCED_PARAMETER(temp); // 'temp' is only used by vectorized kernel implementations and it is unused in the scalar fallback path.
    for (size_t i = 0; i < count; ++i) {
        float x = static_cast<float>(input[i]);
        float gelu_val;

        if (algo == MlasGeluTanh) {
            // GELU approximation (tanh)
            const float B = 0.7978845608f;
            const float C = 0.044715f * B;
            float tanh_arg = x * (B + C * x * x);
            float tanh_res = std::tanh(tanh_arg);
            gelu_val = 0.5f * x * (1.0f + tanh_res);
        } else {
            // GELU exact (erf)
            gelu_val = 0.5f * x *
                (1.0f + std::erf(x * (1.0f / std::numbers::sqrt2_v<float>)));
        }

        output[i] = MLAS_FP16(gelu_val);
    }
}