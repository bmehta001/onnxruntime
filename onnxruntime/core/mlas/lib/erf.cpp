/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    erf.cpp

Abstract:

    This module implements routines to compute the hyperbolic tangent function.

    This implementation uses the same polynomial coefficients and algorithm as
    found in: https://stackoverflow.com/questions/35148198/efficient-faithfully-rounded-implementation-of-error-function-erff
    Our usage requires building platform specific versions of
    the algorithm to target different instruction sets. The implementation below
    targets the base instruction set (typically SSE2) while assembly
    implementations target newer instruction sets (such as FMA3).

--*/

#include "erf_float32x4.h"

//
// Bundles the constants for use by kernels written in assembly.
//

MLAS_INTERNAL_DATA const MLAS_ERF_CONSTANTS MlasErfConstants = {
    3.925f,
    0.921875f,
    -5.99104969e-4f,
    4.99339588e-3f,
    -2.67667342e-2f,
    1.12818025e-1f,
    -3.76124859e-1f,
    1.28379151e-1f,
    0.0f,
    1.72948930e-5f,
    -3.83208680e-4f,
    3.88393435e-3f,
    -2.42545605e-2f,
    1.06777847e-1f,
    6.34846687e-1f,
    1.28717512e-1f,
    -0.0f,
    1.0f,

    // Independent parameters to calculate Exp for Erff()
    88.3762626647950f,
    -88.3762626647949f,
    1.44269504088896341f,
    -6.93145752e-1f,
    -1.42860677e-6f,
    1.38319808e-3f,
    8.37550033e-3f,
    4.16689515e-2f,
    1.66664466e-1f,
    4.99999851e-1f,
    1.00000000e+0f,
    1.00000000e+0f,
    1.25829120e+7f,
    127,

    // Parameters for the fused exact Gelu kernel.
    0.5f,
    0.70710678118654752440f,
};

void
MLASCALL
MlasErfKernel(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine implements the generic kernel for the error function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    const MLAS_ERF_FLOAT32X4_CONSTANTS Constants;

    while (N >= 4) {
        MlasStoreFloat32x4(Output, MlasErfFloat32x4(MlasLoadFloat32x4(Input), Constants));

        Input += 4;
        Output += 4;
        N -= 4;
    }

    while (N > 0) {
        float Value = *Input++;
        float AbsValue = fabsf(Value);

        float r;
        if (AbsValue > MlasErfConstants.ErfSplitBoundary) {
            AbsValue = std::min(MlasErfConstants.ErfUpperAbsRange, AbsValue);
            float r_big = MlasErfConstants.ErfBIG_P0;
            r_big = r_big * AbsValue + MlasErfConstants.ErfBIG_P1;
            r_big = r_big * AbsValue + MlasErfConstants.ErfBIG_P2;
            r_big = r_big * AbsValue + MlasErfConstants.ErfBIG_P3;
            r_big = r_big * AbsValue + MlasErfConstants.ErfBIG_P4;
            r_big = r_big * AbsValue + MlasErfConstants.ErfBIG_P5;
            r_big = r_big * AbsValue + MlasErfConstants.ErfBIG_P6_Minus_One;
            r_big = r_big * AbsValue + AbsValue;

            r_big = std::max(-r_big, MlasErfConstants.Exp_LowerRange);
            r = MlasErfConstants.Exp_Log2Reciprocal * r_big + MlasErfConstants.Exp_C;
            r -= MlasErfConstants.Exp_C;
            float fx = r * MlasErfConstants.Exp_log2_hi + r_big;
            fx = r * MlasErfConstants.Exp_log2_lo + fx;

            float y = MlasErfConstants.Exp_P0;
            y = y * fx + MlasErfConstants.Exp_P1;
            y = y * fx + MlasErfConstants.Exp_P2;
            y = y * fx + MlasErfConstants.Exp_P3;
            y = y * fx + MlasErfConstants.Exp_P4;
            y = y * fx + MlasErfConstants.Exp_P5;
            y = y * fx + MlasErfConstants.Exp_P6;

            r = 1.0f - ldexpf(y, (int)r);
            r = (Value <= -0.0f) ? -r : r;
        }
        else {
            float SquareValue = AbsValue * AbsValue;
            r = MlasErfConstants.ErfSMALL_P0;
            r = r * SquareValue + MlasErfConstants.ErfSMALL_P1;
            r = r * SquareValue + MlasErfConstants.ErfSMALL_P2;
            r = r * SquareValue + MlasErfConstants.ErfSMALL_P3;
            r = r * SquareValue + MlasErfConstants.ErfSMALL_P4;
            r = r * SquareValue + MlasErfConstants.ErfSMALL_P5_Minus_One;
            r = r * Value + Value;
        }

        *Output++ = r;
        N -= 1;
    }
}

void
MLASCALL
MlasComputeErf(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine computes the error function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
#if defined(MLAS_TARGET_AMD64) || defined(MLAS_USE_SVE) || defined(MLAS_TARGET_RISCV64)
    GetMlasPlatform().ErfKernelRoutine(Input, Output, N);
#else
    MlasErfKernel(Input, Output, N);
#endif
}

void
MLASCALL
MlasComputeFP16Erf(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    float* Input_tmp_fp32,
    float* Output_tmp_fp32,
    size_t N
    )
{
    if(GetMlasPlatform().ErfFP16KernelRoutine){
        GetMlasPlatform().ErfFP16KernelRoutine(Input, Output, N);
        return;
    }

    MlasConvertHalfToFloatBuffer(Input, Input_tmp_fp32, N);
    MlasComputeErf(Input_tmp_fp32, Output_tmp_fp32, N);
    MlasConvertFloatToHalfBuffer(Output_tmp_fp32, Output, N);
}