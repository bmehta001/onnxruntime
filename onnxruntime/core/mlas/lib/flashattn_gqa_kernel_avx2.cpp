/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    flashattn_gqa_kernel_avx2.cpp

Abstract:

    AVX2/FMA3 GEMV kernels for single-token GroupQueryAttention decode.

    All query heads of a GQA group share one K/V head, so each kernel applies
    a K or V row to several query heads while the row is in registers/L1.
    HeadSize must be a multiple of 8.

    Every dot product and every output element is accumulated in the same
    order regardless of how heads and rows are tiled, so a result does not
    depend on the group size or on the row range a caller passes in.

--*/

#include "mlasi.h"

#if defined(MLAS_TARGET_AMD64)

#include <immintrin.h>

namespace {

// Returns [sum(a0), ..., sum(a7)]. Each lane is ((x0+x1)+(x2+x3))+((x4+x5)+(x6+x7)) of its
// input, independent of the other inputs, so zero padding does not change any result.
MLAS_FORCEINLINE __m256
ReduceAdd8(const __m256* a)
{
    const __m256 h01 = _mm256_hadd_ps(a[0], a[1]);
    const __m256 h23 = _mm256_hadd_ps(a[2], a[3]);
    const __m256 h45 = _mm256_hadd_ps(a[4], a[5]);
    const __m256 h67 = _mm256_hadd_ps(a[6], a[7]);
    const __m256 h0123 = _mm256_hadd_ps(h01, h23);
    const __m256 h4567 = _mm256_hadd_ps(h45, h67);
    const __m256 lo = _mm256_permute2f128_ps(h0123, h4567, 0x20);
    const __m256 hi = _mm256_permute2f128_ps(h0123, h4567, 0x31);
    return _mm256_add_ps(lo, hi);
}

template <size_t R, size_t G>
MLAS_FORCEINLINE void
QKTile(
    const float* Query,
    const float* Key,
    size_t HeadSize,
    float Scale,
    float* Scores,
    size_t ScoresStride
)
{
    static_assert(R * G <= 8, "tile exceeds one reduction");

    __m256 acc[8];
    for (size_t i = 0; i < 8; ++i) {
        acc[i] = _mm256_setzero_ps();
    }

    for (size_t h = 0; h < HeadSize; h += 8) {
        __m256 q[G];
        for (size_t g = 0; g < G; ++g) {
            q[g] = _mm256_loadu_ps(Query + g * HeadSize + h);
        }
        for (size_t r = 0; r < R; ++r) {
            const __m256 k = _mm256_loadu_ps(Key + r * HeadSize + h);
            for (size_t g = 0; g < G; ++g) {
                acc[r * G + g] = _mm256_fmadd_ps(q[g], k, acc[r * G + g]);
            }
        }
    }

    alignas(32) float sums[8];
    _mm256_store_ps(sums, _mm256_mul_ps(ReduceAdd8(acc), _mm256_set1_ps(Scale)));
    for (size_t r = 0; r < R; ++r) {
        for (size_t g = 0; g < G; ++g) {
            Scores[g * ScoresStride + r] = sums[r * G + g];
        }
    }
}

template <size_t G>
void
QKHeadTile(
    const float* Query,
    const float* Key,
    size_t RowCount,
    size_t HeadSize,
    float Scale,
    float* Scores,
    size_t ScoresStride
)
{
    constexpr size_t R = (G >= 3) ? 2 : 4;
    size_t t = 0;
    for (; t + R <= RowCount; t += R) {
        QKTile<R, G>(Query, Key + t * HeadSize, HeadSize, Scale, Scores + t, ScoresStride);
    }
    for (; t < RowCount; ++t) {
        QKTile<1, G>(Query, Key + t * HeadSize, HeadSize, Scale, Scores + t, ScoresStride);
    }
}

template <size_t G, size_t HB>
MLAS_FORCEINLINE void
SVTile(
    const float* Probs,
    size_t ProbsStride,
    const float* Value,
    size_t RowCount,
    size_t HeadSize,
    float* Output,
    size_t OutputStride
)
{
    __m256 acc[G][HB];
    for (size_t g = 0; g < G; ++g) {
        for (size_t j = 0; j < HB; ++j) {
            acc[g][j] = _mm256_setzero_ps();
        }
    }

    for (size_t t = 0; t < RowCount; ++t) {
        __m256 v[HB];
        for (size_t j = 0; j < HB; ++j) {
            v[j] = _mm256_loadu_ps(Value + t * HeadSize + j * 8);
        }
        for (size_t g = 0; g < G; ++g) {
            const __m256 p = _mm256_broadcast_ss(Probs + g * ProbsStride + t);
            for (size_t j = 0; j < HB; ++j) {
                acc[g][j] = _mm256_fmadd_ps(p, v[j], acc[g][j]);
            }
        }
    }

    for (size_t g = 0; g < G; ++g) {
        for (size_t j = 0; j < HB; ++j) {
            _mm256_storeu_ps(Output + g * OutputStride + j * 8, acc[g][j]);
        }
    }
}

template <size_t G, size_t HB>
void
SVHeadTile(
    const float* Probs,
    size_t ProbsStride,
    const float* Value,
    size_t RowCount,
    size_t HeadSize,
    float* Output,
    size_t OutputStride
)
{
    size_t h = 0;
    for (; h + HB * 8 <= HeadSize; h += HB * 8) {
        SVTile<G, HB>(Probs, ProbsStride, Value + h, RowCount, HeadSize, Output + h, OutputStride);
    }
    for (; h < HeadSize; h += 8) {
        SVTile<G, 1>(Probs, ProbsStride, Value + h, RowCount, HeadSize, Output + h, OutputStride);
    }
}

// Splits a head group into register tiles: prefer tiles of 4, then 2, then 3 so that common
// group sizes (2, 3, 4, 6, 8) need no single-head tile.
MLAS_FORCEINLINE size_t
NextHeadTile(size_t Remaining)
{
    if (Remaining % 4 == 0) {
        return 4;
    }
    if (Remaining % 2 == 0) {
        return 2;
    }
    if (Remaining % 3 == 0) {
        return 3;
    }
    return Remaining > 4 ? 4 : 1;
}

}  // namespace

void
MLASCALL
MlasGQADecodeQKKernelAvx2(
    const float* Query,
    size_t GroupSize,
    const float* Key,
    size_t RowCount,
    size_t HeadSize,
    float Scale,
    float* Scores,
    size_t ScoresStride
)
{
    size_t g = 0;
    while (g < GroupSize) {
        const size_t tile = NextHeadTile(GroupSize - g);
        const float* q = Query + g * HeadSize;
        float* s = Scores + g * ScoresStride;
        switch (tile) {
            case 4:
                QKHeadTile<4>(q, Key, RowCount, HeadSize, Scale, s, ScoresStride);
                break;
            case 3:
                QKHeadTile<3>(q, Key, RowCount, HeadSize, Scale, s, ScoresStride);
                break;
            case 2:
                QKHeadTile<2>(q, Key, RowCount, HeadSize, Scale, s, ScoresStride);
                break;
            default:
                QKHeadTile<1>(q, Key, RowCount, HeadSize, Scale, s, ScoresStride);
                break;
        }
        g += tile;
    }
}

void
MLASCALL
MlasGQADecodeSVKernelAvx2(
    const float* Probs,
    size_t ProbsStride,
    size_t GroupSize,
    const float* Value,
    size_t RowCount,
    size_t HeadSize,
    float* Output,
    size_t OutputStride
)
{
    size_t g = 0;
    while (g < GroupSize) {
        const size_t tile = NextHeadTile(GroupSize - g);
        const float* p = Probs + g * ProbsStride;
        float* o = Output + g * OutputStride;
        switch (tile) {
            case 4:
                SVHeadTile<4, 2>(p, ProbsStride, Value, RowCount, HeadSize, o, OutputStride);
                break;
            case 3:
                SVHeadTile<3, 2>(p, ProbsStride, Value, RowCount, HeadSize, o, OutputStride);
                break;
            case 2:
                SVHeadTile<2, 4>(p, ProbsStride, Value, RowCount, HeadSize, o, OutputStride);
                break;
            default:
                SVHeadTile<1, 4>(p, ProbsStride, Value, RowCount, HeadSize, o, OutputStride);
                break;
        }
        g += tile;
    }
}

#endif  // MLAS_TARGET_AMD64
