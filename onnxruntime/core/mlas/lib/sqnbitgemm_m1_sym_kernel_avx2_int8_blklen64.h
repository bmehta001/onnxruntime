#pragma once
#include <algorithm>
#include <cassert>
#include <utility>

#include "qnbitgemm.h"
#include "sqnbitgemm_kernel_avx_common.h"


static MLAS_FORCEINLINE void
accumulate_blklen64_r1c1blk1_zp_avx2(
    const __m256i& av00_32_epi8,
    const __m256i& av01_32_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a,
    const float* scale_b,
    const std::byte* QuantBZeroPointPtr,
    const bool is_lower_half_byte_zp,
    __m256& acc0,
    const __m256i& low_mask
)
{
    // | v0  v32 | v1  v33 | ... | v30 v62 | v31 v63 |
    const __m256i bv_packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
    __m256i bv0_32_epi8 = _mm256_and_si256(bv_packed, low_mask);                          // 0, 1,...30, 31
    __m256i bv1_32_epi8 = _mm256_srli_epi16(_mm256_sub_epi8(bv_packed, bv0_32_epi8), 4);  // 32, 33,...62, 63

    const __m256i bzp8 = _mm256_set1_epi8(get_zp<true>(is_lower_half_byte_zp, QuantBZeroPointPtr));
    bv0_32_epi8 = _mm256_sub_epi8(bv0_32_epi8, bzp8);
    bv1_32_epi8 = _mm256_sub_epi8(bv1_32_epi8, bzp8);

    const __m256i dot0_16_epi16 = _mm256_maddubs_epi16(_mm256_sign_epi8(bv0_32_epi8, bv0_32_epi8), _mm256_sign_epi8(av00_32_epi8, bv0_32_epi8));
    const __m256i dot1_16_epi16 = _mm256_maddubs_epi16(_mm256_sign_epi8(bv1_32_epi8, bv1_32_epi8), _mm256_sign_epi8(av01_32_epi8, bv1_32_epi8));
    const __m256i sum_16_epi16 = _mm256_hadd_epi16(dot0_16_epi16, dot1_16_epi16);

    __m256i one_16_epi16 = _mm256_srli_epi16(_mm256_cmpeq_epi16(bv0_32_epi8, bv0_32_epi8), 15);
    const __m256i sum_8_epi32 = _mm256_madd_epi16(one_16_epi16, sum_16_epi16);
    const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);

    __m256 scale_a_8_ps = _mm256_broadcast_ss(scale_a);
    __m256 scale_b_8_ps = _mm256_broadcast_ss(scale_b);

    acc0 = _mm256_fmadd_ps(sum_ps, _mm256_mul_ps(scale_a_8_ps, scale_b_8_ps), acc0);
}

static MLAS_FORCEINLINE void
accumulate_blklen64_r1c1blk1_zp_is_8_avx2(
    const __m256i& av00_32_epi8,
    const __m256i& av01_32_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a,
    const float* scale_b,
    __m256& acc0,
    const __m256i& low_mask,
    const __m256i& bzp8
)
{
    // | v0  v32 | v1  v33 | ... | v30 v62 | v31 v63 |
    const __m256i bv_packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
    __m256i bv0_32_epi8 = _mm256_and_si256(bv_packed, low_mask);                          // 0, 1,...30, 31
    __m256i bv1_32_epi8 = _mm256_srli_epi16(_mm256_sub_epi8(bv_packed, bv0_32_epi8), 4);  // 32, 33,...62, 63

    bv0_32_epi8 = _mm256_sub_epi8(bv0_32_epi8, bzp8);
    bv1_32_epi8 = _mm256_sub_epi8(bv1_32_epi8, bzp8);

    const __m256i dot0_16_epi16 = _mm256_maddubs_epi16(_mm256_sign_epi8(bv0_32_epi8, bv0_32_epi8), _mm256_sign_epi8(av00_32_epi8, bv0_32_epi8));
    const __m256i dot1_16_epi16 = _mm256_maddubs_epi16(_mm256_sign_epi8(bv1_32_epi8, bv1_32_epi8), _mm256_sign_epi8(av01_32_epi8, bv1_32_epi8));
    const __m256i sum_16_epi16 = _mm256_hadd_epi16(dot0_16_epi16, dot1_16_epi16);

    __m256i one_16_epi16 = _mm256_srli_epi16(_mm256_cmpeq_epi16(bv0_32_epi8, bv0_32_epi8), 15);
    const __m256i sum_8_epi32 = _mm256_madd_epi16(one_16_epi16, sum_16_epi16);
    const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);

    __m256 scale_a_8_ps = _mm256_broadcast_ss(scale_a);
    __m256 scale_b_8_ps = _mm256_broadcast_ss(scale_b);

    acc0 = _mm256_fmadd_ps(sum_ps, _mm256_mul_ps(scale_a_8_ps, scale_b_8_ps), acc0);
}

// Unsigned-weight variant of the two functions above: sum(a * (b - zp)) = sum(a * b) - zp * sum(a),
// with zp_ps * sum_a_ps (8 * sum_a_ps when symmetric) precomputed or applied in float. All values are
// integers below 2^24, so the result equals the signed formulation exactly.
template <bool HasZeroPoint>
static MLAS_FORCEINLINE void
accumulate_blklen64_r1c1blk1_u_avx2(
    const __m256i& av00_32_epi8,
    const __m256i& av01_32_epi8,
    const __m256& sum_a_ps,
    const std::byte* QuantBDataPtr,
    const float* scale_a,
    const float* scale_b,
    [[maybe_unused]] const __m256& zp_ps,
    __m256& acc0,
    const __m256i& low_mask,
    const __m256i& one_16_epi16
)
{
    const __m256i bv_packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
    const __m256i bv0_32_epi8 = _mm256_and_si256(bv_packed, low_mask);                          // 0, 1,...30, 31
    const __m256i bv1_32_epi8 = _mm256_srli_epi16(_mm256_sub_epi8(bv_packed, bv0_32_epi8), 4);  // 32, 33,...62, 63

    const __m256i dot0_16_epi16 = _mm256_maddubs_epi16(bv0_32_epi8, av00_32_epi8);
    const __m256i dot1_16_epi16 = _mm256_maddubs_epi16(bv1_32_epi8, av01_32_epi8);
    const __m256i sum_8_epi32 = _mm256_madd_epi16(one_16_epi16, _mm256_hadd_epi16(dot0_16_epi16, dot1_16_epi16));
    __m256 sum_ps;
    if constexpr (HasZeroPoint) {
        sum_ps = _mm256_fnmadd_ps(zp_ps, sum_a_ps, _mm256_cvtepi32_ps(sum_8_epi32));
    } else {
        sum_ps = _mm256_sub_ps(_mm256_cvtepi32_ps(sum_8_epi32), sum_a_ps);
    }

    const __m256 scale_a_8_ps = _mm256_broadcast_ss(scale_a);
    const __m256 scale_b_8_ps = _mm256_broadcast_ss(scale_b);

    acc0 = _mm256_fmadd_ps(sum_ps, _mm256_mul_ps(scale_a_8_ps, scale_b_8_ps), acc0);
}

template <bool HasZeroPoint>
MLAS_FORCEINLINE void
Q4Int8GemmM1C4BlkLen64Avx2(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias)
{
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;
    constexpr size_t SubblkLen64 = 64;

    const size_t BlkDataSizeInBytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    const size_t PerBlkSubblkCount = BlkLen / SubblkLen64;
    const size_t SubblkDataSizeInBytes = BlkDataSizeInBytes / PerBlkSubblkCount;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;

    assert(CountN % NCols4 == 0);

    const std::byte* QuantBDataColPtr = QuantBData;
    const float* QuantBScaleColPtr = QuantBScale;
    const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;
    const float* BiasPtr = Bias;
    auto* SumPtr = C;

    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i one_32_epi8 = _mm256_set1_epi8(1);
    const __m256i one_16_epi16 = _mm256_set1_epi16(1);
    const size_t StrideQuantBData1 = 1 * SubblkDataSizeInBytes;
    const size_t StrideQuantBScale1 = 1;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

    for (size_t n = 0; n < CountN; n += NCols4) {
        const std::byte* QuantAPtr = QuantA;
        const float* QuantAScalePtr = QuantAScale;

        const std::byte* QuantBDataPtr = QuantBDataColPtr;
        const float* QuantBScalePtr = QuantBScaleColPtr;
        const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

        __m256 acc[NCols4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (size_t k = 0; k < BlockCountK; ++k) {
            [[maybe_unused]] const bool is_lower_half_byte_zp = (k % 2) == 0;
            for (size_t kk = 0; kk < PerBlkSubblkCount; kk++) {
                const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantAPtr);
                const __m256i av_01_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + 32));
                __m256i sum_a_16_epi16 = _mm256_hadd_epi16(_mm256_maddubs_epi16(one_32_epi8, av_00_epi8), _mm256_maddubs_epi16(one_32_epi8, av_01_epi8));
                if constexpr (!HasZeroPoint) {
                    sum_a_16_epi16 = _mm256_slli_epi16(sum_a_16_epi16, 3);
                }
                const __m256 sum_a_ps = _mm256_cvtepi32_ps(_mm256_madd_epi16(one_16_epi16, sum_a_16_epi16));
                for (size_t c = 0; c < NCols4; ++c) {
                    __m256 zp_ps = sum_a_ps;
                    if constexpr (HasZeroPoint) {
                        const uint8_t zp = std::to_integer<uint8_t>(QuantBZeroPointPtr[c * StrideQuantBZeroPoint]);
                        zp_ps = _mm256_set1_ps(static_cast<float>(is_lower_half_byte_zp ? (zp & 0x0F) : (zp >> 4)));
                    }
                    accumulate_blklen64_r1c1blk1_u_avx2<HasZeroPoint>(av_00_epi8, av_01_epi8, sum_a_ps, QuantBDataPtr + c * StrideQuantBData1, QuantAScalePtr, QuantBScalePtr + c * StrideQuantBScale1, zp_ps, acc[c], low_mask, one_16_epi16);
                }

                // increment block pointers
                QuantAPtr += SubblkLen64;
                QuantBDataPtr += NCols4 * SubblkDataSizeInBytes;
            }
            QuantAScalePtr++;
            QuantBScalePtr += NCols4;
            if constexpr (HasZeroPoint) {
                QuantBZeroPointPtr += k % 2;
            }
        }

        __m128 acc_r0 = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);
        if (BiasPtr != nullptr) {
            acc_r0 = _mm_add_ps(acc_r0, _mm_loadu_ps(BiasPtr));
        }

        _mm_storeu_ps(SumPtr, acc_r0);

        // move to next NCols columns
        QuantBDataColPtr += NCols4 * StrideQuantBData;
        QuantBScaleColPtr += NCols4 * StrideQuantBScale;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointColPtr += NCols4 * StrideQuantBZeroPoint;
        }
        BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
        SumPtr += NCols4;
    }
}

template <bool HasZeroPoint>
MLAS_FORCEINLINE void
Q4Int8GemmM1C1BlkLen64Avx2(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias)
{
    constexpr size_t BlkBitWidth4 = 4;
    [[maybe_unused]] constexpr size_t NCols4 = 4;
    [[maybe_unused]] constexpr size_t NRows2 = 2;
    constexpr size_t SubblkLen = 64;

    const size_t BlkDataSizeInBytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    const size_t PerBlkSubblkCount = BlkLen / SubblkLen;
    const size_t SubblkDataSizeInBytes = BlkDataSizeInBytes / PerBlkSubblkCount;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

    assert(CountN < NCols4);

    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    [[maybe_unused]] const __m256i bzp8 = _mm256_set1_epi8(8);

    const std::byte* QuantBDataColPtr = QuantBData;
    const float* QuantBScaleColPtr = QuantBScale;
    const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;
    const float* BiasPtr = Bias;
    auto* SumPtr = C;

    for (size_t n = 0; n < CountN; n++) {
        const std::byte* QuantAPtr = QuantA;
        const float* QuantAScalePtr = QuantAScale;
        const std::byte* QuantBDataPtr = QuantBDataColPtr;
        const float* QuantBScalePtr = QuantBScaleColPtr;
        const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

        __m256 acc0 = _mm256_setzero_ps();
        for (size_t k = 0; k < BlockCountK; ++k) {
            [[maybe_unused]] const bool is_lower_half_byte_zp = (k % 2) == 0;
            for (size_t kk = 0; kk < PerBlkSubblkCount; kk++) {
                const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantAPtr);
                const __m256i av_01_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + 32));

                if constexpr (HasZeroPoint) {
                    accumulate_blklen64_r1c1blk1_zp_avx2(av_00_epi8, av_01_epi8, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, QuantBZeroPointPtr, is_lower_half_byte_zp, acc0, low_mask);
                } else {
                    accumulate_blklen64_r1c1blk1_zp_is_8_avx2(av_00_epi8, av_01_epi8, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc0, low_mask, bzp8);
                }

                // increment block pointers
                QuantAPtr += SubblkLen;
                QuantBDataPtr += SubblkDataSizeInBytes;
            }
            QuantAScalePtr++;
            QuantBScalePtr++;
            if constexpr (HasZeroPoint) {
                QuantBZeroPointPtr += k % 2;
            }
        }

        *SumPtr = hsum_float_8(acc0);
        if (BiasPtr) {
            *SumPtr += *BiasPtr;
        }

        // move to next column
        QuantBDataColPtr += StrideQuantBData;
        QuantBScaleColPtr += StrideQuantBScale;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointColPtr += StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? 1 : 0;
        SumPtr += 1;
    }
}

template <bool HasZeroPoint>
MLAS_FORCEINLINE void
MlasQ4Int8GemmKernelBlkLen64Avx2(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias
)
{
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

    size_t remainingCols = CountN % NCols4;
    size_t multipleCols = CountN - remainingCols;

    if (multipleCols > 0) {
        Q4Int8GemmM1C4BlkLen64Avx2<HasZeroPoint>(
            BlkLen,
            QuantA,
            QuantAScale,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            multipleCols,
            BlockCountK,
            Bias);
    }

    if (remainingCols > 0) {
        Q4Int8GemmM1C1BlkLen64Avx2<HasZeroPoint>(
            BlkLen,
            QuantA,
            QuantAScale,
            QuantBData + multipleCols * StrideQuantBData,
            QuantBScale + multipleCols * StrideQuantBScale,
            QuantBZeroPoint ? QuantBZeroPoint + multipleCols * StrideQuantBZeroPoint : nullptr,
            C + multipleCols,
            remainingCols,
            BlockCountK,
            Bias ? Bias + multipleCols : nullptr);
    }
}
