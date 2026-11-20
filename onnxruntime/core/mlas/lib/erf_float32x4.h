/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    erf_float32x4.h

Abstract:

    This module implements the portable MLAS_FLOAT32X4 error function shared
    by the generic erf and GELU(erf) kernels.

--*/

#pragma once

#include "mlasi.h"
#include "elementwise_constants.h"

MLAS_INTERNAL_DATA const MLAS_ERF_CONSTANTS MlasErfConstants;

//
// Broadcast copies of MlasErfConstants. Kernels build this once per call and
// pass it to MlasErfFloat32x4 so that some compilers do not re-broadcast every
// constant on each loop iteration.
//

struct MLAS_ERF_FLOAT32X4_CONSTANTS {
    MLAS_FLOAT32X4 NegZero;
    MLAS_FLOAT32X4 UpperAbsRange;
    MLAS_FLOAT32X4 SplitBoundary;
    MLAS_FLOAT32X4 SmallP0;
    MLAS_FLOAT32X4 SmallP1;
    MLAS_FLOAT32X4 SmallP2;
    MLAS_FLOAT32X4 SmallP3;
    MLAS_FLOAT32X4 SmallP4;
    MLAS_FLOAT32X4 SmallP5_Minus_One;
    MLAS_FLOAT32X4 BigP0;
    MLAS_FLOAT32X4 BigP1;
    MLAS_FLOAT32X4 BigP2;
    MLAS_FLOAT32X4 BigP3;
    MLAS_FLOAT32X4 BigP4;
    MLAS_FLOAT32X4 BigP5;
    MLAS_FLOAT32X4 BigP6_Minus_One;
    MLAS_FLOAT32X4 One;
    MLAS_FLOAT32X4 ExpLowerRange;
    MLAS_FLOAT32X4 ExpLog2Reciprocal;
    MLAS_FLOAT32X4 ExpLog2Hi;
    MLAS_FLOAT32X4 ExpLog2Lo;
    MLAS_FLOAT32X4 ExpP0;
    MLAS_FLOAT32X4 ExpP1;
    MLAS_FLOAT32X4 ExpP2;
    MLAS_FLOAT32X4 ExpP3;
    MLAS_FLOAT32X4 ExpP4;
    MLAS_FLOAT32X4 ExpP5;
    MLAS_FLOAT32X4 ExpP6;
    MLAS_FLOAT32X4 ExpC;

    MLAS_ERF_FLOAT32X4_CONSTANTS()
        : NegZero(MlasBroadcastFloat32x4(MlasErfConstants.ErfNegZero)),
          UpperAbsRange(MlasBroadcastFloat32x4(MlasErfConstants.ErfUpperAbsRange)),
          SplitBoundary(MlasBroadcastFloat32x4(MlasErfConstants.ErfSplitBoundary)),
          SmallP0(MlasBroadcastFloat32x4(MlasErfConstants.ErfSMALL_P0)),
          SmallP1(MlasBroadcastFloat32x4(MlasErfConstants.ErfSMALL_P1)),
          SmallP2(MlasBroadcastFloat32x4(MlasErfConstants.ErfSMALL_P2)),
          SmallP3(MlasBroadcastFloat32x4(MlasErfConstants.ErfSMALL_P3)),
          SmallP4(MlasBroadcastFloat32x4(MlasErfConstants.ErfSMALL_P4)),
          SmallP5_Minus_One(MlasBroadcastFloat32x4(MlasErfConstants.ErfSMALL_P5_Minus_One)),
          BigP0(MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P0)),
          BigP1(MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P1)),
          BigP2(MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P2)),
          BigP3(MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P3)),
          BigP4(MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P4)),
          BigP5(MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P5)),
          BigP6_Minus_One(MlasBroadcastFloat32x4(MlasErfConstants.ErfBIG_P6_Minus_One)),
          One(MlasBroadcastFloat32x4(MlasErfConstants.ErfOne)),
          ExpLowerRange(MlasBroadcastFloat32x4(MlasErfConstants.Exp_LowerRange)),
          ExpLog2Reciprocal(MlasBroadcastFloat32x4(MlasErfConstants.Exp_Log2Reciprocal)),
          ExpLog2Hi(MlasBroadcastFloat32x4(MlasErfConstants.Exp_log2_hi)),
          ExpLog2Lo(MlasBroadcastFloat32x4(MlasErfConstants.Exp_log2_lo)),
          ExpP0(MlasBroadcastFloat32x4(MlasErfConstants.Exp_P0)),
          ExpP1(MlasBroadcastFloat32x4(MlasErfConstants.Exp_P1)),
          ExpP2(MlasBroadcastFloat32x4(MlasErfConstants.Exp_P2)),
          ExpP3(MlasBroadcastFloat32x4(MlasErfConstants.Exp_P3)),
          ExpP4(MlasBroadcastFloat32x4(MlasErfConstants.Exp_P4)),
          ExpP5(MlasBroadcastFloat32x4(MlasErfConstants.Exp_P5)),
          ExpP6(MlasBroadcastFloat32x4(MlasErfConstants.Exp_P6)),
          ExpC(MlasBroadcastFloat32x4(MlasErfConstants.Exp_C))
    {
    }
};

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasErfFloat32x4(
    MLAS_FLOAT32X4 Value,
    const MLAS_ERF_FLOAT32X4_CONSTANTS& C
    )
{
    MLAS_FLOAT32X4 SignMask = MlasAndFloat32x4(Value, C.NegZero);
    MLAS_FLOAT32X4 AbsValue = MlasAndNotFloat32x4(C.NegZero, Value);
    AbsValue = MlasMinimumFloat32x4(C.UpperAbsRange, AbsValue);
    MLAS_FLOAT32X4 SquareValue = MlasMultiplyFloat32x4(AbsValue, AbsValue);

    MLAS_FLOAT32X4 r_small = C.SmallP0;
    r_small = MlasMultiplyAddFloat32x4(r_small, SquareValue, C.SmallP1);
    r_small = MlasMultiplyAddFloat32x4(r_small, SquareValue, C.SmallP2);
    r_small = MlasMultiplyAddFloat32x4(r_small, SquareValue, C.SmallP3);
    r_small = MlasMultiplyAddFloat32x4(r_small, SquareValue, C.SmallP4);
    r_small = MlasMultiplyAddFloat32x4(r_small, SquareValue, C.SmallP5_Minus_One);
    r_small = MlasMultiplyAddFloat32x4(r_small, AbsValue, AbsValue);
    MLAS_FLOAT32X4 split_mask = MlasGreaterThanFloat32x4(AbsValue, C.SplitBoundary);
    r_small = MlasAndNotFloat32x4(split_mask, r_small);

    AbsValue = MlasAndFloat32x4(split_mask, AbsValue); // clear smaller value into zero for bigger number calculation
    MLAS_FLOAT32X4 r_big = C.BigP0;
    r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, C.BigP1);
    r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, C.BigP2);
    r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, C.BigP3);
    r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, C.BigP4);
    r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, C.BigP5);
    r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, C.BigP6_Minus_One);
    r_big = MlasMultiplyAddFloat32x4(r_big, AbsValue, AbsValue);

    // 1.0 - exp(-r_big), no need to do min()
    r_big = MlasXorFloat32x4(r_big, C.NegZero); // -r_big
    r_big = MlasMaximumFloat32x4(C.ExpLowerRange, r_big);
    MLAS_FLOAT32X4 r = MlasMultiplyAddFloat32x4(C.ExpLog2Reciprocal, r_big, C.ExpC);
    r = MlasSubtractFloat32x4(r, C.ExpC);

    MLAS_FLOAT32X4 fx = MlasMultiplyAddFloat32x4(r, C.ExpLog2Hi, r_big);
    fx = MlasMultiplyAddFloat32x4(r, C.ExpLog2Lo, fx);
    // y = exp(fx)
    MLAS_FLOAT32X4 y = C.ExpP0;
    y = MlasMultiplyAddFloat32x4(y, fx, C.ExpP1);
    y = MlasMultiplyAddFloat32x4(y, fx, C.ExpP2);
    y = MlasMultiplyAddFloat32x4(y, fx, C.ExpP3);
    y = MlasMultiplyAddFloat32x4(y, fx, C.ExpP4);
    y = MlasMultiplyAddFloat32x4(y, fx, C.ExpP5);
    y = MlasMultiplyAddFloat32x4(y, fx, C.ExpP6);
    // 1.0 - exp(fx) * 2^INT(r)
    y = MlasMultiplyFloat32x4(y, MlasPowerOf2Float32x4(r));
    y = MlasSubtractFloat32x4(C.One, y);

    // merge two splits results
    y = MlasOrFloat32x4(r_small, y);
    y = MlasOrFloat32x4(y, SignMask);

    return y;
}