#pragma once
#include <cassert>
#include <cstdint>

#include "qnbitgemm.h"
#include "sqnbitgemm_kernel_avx_common.h"

//
// M = 1 (GEMV) kernel for 8-bit weights with int8 activations (SQNBIT_CompInt8), BlkLen >= 32, AVX2 without VNNI.
//
// The result is bit-identical to MlasQ8Int8GemmKernelBlkLen{32,64}Avx2 followed by the float block-sum GEMM
// (C += ABlockSum * QuantBBlkSum) in SQ8BitGemmKernel_BlkSum_CompInt8_avx2:
//  - every int32 lane holds the same exact integer sum(a * b) over the same 4 bytes, computed as
//    sum(a * (b - 128)) + 128 * sum(a). b - 128 is a signed byte, so the products go through the abs/sign + maddubs
//    trick without saturating (|a| <= 127), and 128 * sum(a) is shared by all columns. This replaces the two masked
//    maddubs/madd per 32 bytes of the reference kernel.
//  - the per-lane float FMAs, the horizontal folds and the bias add happen in the same order as the reference kernel.
//  - the block-sum term is accumulated per column with a sequential FMA chain over k and added last, which is what
//    the FMA SGEMM kernel computes for one row, so the separate pass over C is fused away.
//
// Packed B layout (Q8PackQuantB with SubBlkLen 64): full groups of 4 columns store, for every 64 rows,
// 64 bytes of column 0, then columns 1, 2 and 3. With BlkLen 32 and an odd block count the last block
// stores 32 bytes per column instead. Columns of a trailing partial group keep the original column-major
// order. Scales follow the data order (see Q8ComputePackBlkSum). QuantBBlkSum is stored in panels of 16 columns:
// [N / 16][BlockCountK][16].
//

// 8 int32 partial sums of a . (b - 128) over 32 bytes, 4 bytes per lane.
static MLAS_FORCEINLINE __m256i
Q8M1DotChunkAvx2(const __m256i& av_32_epi8, const std::byte* QuantBDataPtr, const __m256i& flip, const __m256i& one_16_epi16)
{
    const __m256i bv = _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr)), flip);
    const __m256i dot_16_epi16 = _mm256_maddubs_epi16(_mm256_abs_epi8(bv), _mm256_sign_epi8(av_32_epi8, bv));
    return _mm256_madd_epi16(dot_16_epi16, one_16_epi16);
}

// 8 int32 lanes of 128 * sum(a) over the same 4 bytes as Q8M1DotChunkAvx2.
static MLAS_FORCEINLINE __m256i
Q8M1Sum128Avx2(const __m256i& av_32_epi8, const __m256i& one_32_epi8, const __m256i& c128_16_epi16)
{
    return _mm256_madd_epi16(_mm256_maddubs_epi16(one_32_epi8, av_32_epi8), c128_16_epi16);
}

MLAS_FORCEINLINE void
Q8Int8GemmM1C4Avx2(
    size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    const float* ABlockSum,
    const float* QuantBBlkSum
)
{
    constexpr size_t NCols4 = 4;
    constexpr size_t SubBlkLen64 = 64;
    constexpr size_t SubBlkStride = SubBlkLen64 * NCols4;
    constexpr size_t BlkSumPanel = 16;

    assert(CountN % NCols4 == 0);
    assert(BlkLen == 32 || BlkLen % SubBlkLen64 == 0);

    const size_t StrideQuantBDataCol = BlockCountK * BlkLen;
    const __m256i flip = _mm256_set1_epi8(static_cast<char>(0x80));
    const __m256i one_16_epi16 = _mm256_set1_epi16(1);
    const __m256i one_32_epi8 = _mm256_set1_epi8(1);
    const __m256i c128_16_epi16 = _mm256_set1_epi16(128);

    for (size_t n = 0; n < CountN; n += NCols4) {
        const std::byte* QuantAPtr = QuantA;
        const float* QuantAScalePtr = QuantAScale;
        const std::byte* QuantBDataPtr = QuantBData + n * StrideQuantBDataCol;
        const float* QuantBScalePtr = QuantBScale + n * BlockCountK;
        const float* BlkSumPtr = QuantBBlkSum + (n / BlkSumPanel) * BlockCountK * BlkSumPanel + n % BlkSumPanel;

        __m256 acc[NCols4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        __m128 acc_blksum = _mm_setzero_ps();

        if (BlkLen == 32) {
            size_t k = 0;
            for (; k + 2 <= BlockCountK; k += 2) {
                const __m256i av0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantAPtr));
                const __m256i av1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantAPtr + 32));

                // scales of (col 0, blk k), (col 0, blk k + 1), (col 1, blk k), ...
                const __m256 scale_a = _mm256_castpd_ps(_mm256_broadcast_sd(reinterpret_cast<const double*>(QuantAScalePtr)));
                const __m256 scale = _mm256_mul_ps(_mm256_loadu_ps(QuantBScalePtr), scale_a);

                const __m256i corr = _mm256_hadd_epi32(
                    Q8M1Sum128Avx2(av0, one_32_epi8, c128_16_epi16), Q8M1Sum128Avx2(av1, one_32_epi8, c128_16_epi16)
                );

                for (size_t c = 0; c < NCols4; ++c) {
                    const __m256i dot0 = Q8M1DotChunkAvx2(av0, QuantBDataPtr + c * SubBlkLen64, flip, one_16_epi16);
                    const __m256i dot1 = Q8M1DotChunkAvx2(av1, QuantBDataPtr + c * SubBlkLen64 + 32, flip, one_16_epi16);
                    const __m256i sum = _mm256_add_epi32(_mm256_hadd_epi32(dot0, dot1), corr);  // 00 11 00 11
                    const int l0 = static_cast<int>(2 * c);
                    const __m256 scale_c = _mm256_permutevar8x32_ps(scale, _mm256_setr_epi32(l0, l0, l0 + 1, l0 + 1, l0, l0, l0 + 1, l0 + 1));
                    acc[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum), scale_c, acc[c]);
                }

                acc_blksum = _mm_fmadd_ps(_mm_set1_ps(ABlockSum[k]), _mm_loadu_ps(BlkSumPtr + k * BlkSumPanel), acc_blksum);
                acc_blksum = _mm_fmadd_ps(_mm_set1_ps(ABlockSum[k + 1]), _mm_loadu_ps(BlkSumPtr + (k + 1) * BlkSumPanel), acc_blksum);

                QuantAPtr += 64;
                QuantAScalePtr += 2;
                QuantBDataPtr += SubBlkStride;
                QuantBScalePtr += 2 * NCols4;
            }

            if (k < BlockCountK) {
                const __m256i av0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantAPtr));
                const __m128 scale = _mm_mul_ps(_mm_loadu_ps(QuantBScalePtr), _mm_set1_ps(*QuantAScalePtr));
                const __m256i corr = Q8M1Sum128Avx2(av0, one_32_epi8, c128_16_epi16);

                for (size_t c = 0; c < NCols4; ++c) {
                    const __m256i dot0 = _mm256_add_epi32(Q8M1DotChunkAvx2(av0, QuantBDataPtr + c * 32, flip, one_16_epi16), corr);
                    const __m256 scale_c = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(scale), _mm256_set1_epi32(static_cast<int>(c)));
                    acc[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot0), scale_c, acc[c]);
                }

                acc_blksum = _mm_fmadd_ps(_mm_set1_ps(ABlockSum[k]), _mm_loadu_ps(BlkSumPtr + k * BlkSumPanel), acc_blksum);
            }
        } else {
            const size_t SubBlksPerBlk = BlkLen / SubBlkLen64;
            for (size_t k = 0; k < BlockCountK; ++k) {
                const __m256 scale = _mm256_castps128_ps256(_mm_mul_ps(_mm_loadu_ps(QuantBScalePtr), _mm_set1_ps(*QuantAScalePtr)));
                __m256 scale_c[NCols4];
                for (size_t c = 0; c < NCols4; ++c) {
                    scale_c[c] = _mm256_permutevar8x32_ps(scale, _mm256_set1_epi32(static_cast<int>(c)));
                }

                for (size_t j = 0; j < SubBlksPerBlk; ++j) {
                    const __m256i av0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantAPtr));
                    const __m256i av1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantAPtr + 32));
                    const __m256i corr = _mm256_add_epi32(
                        Q8M1Sum128Avx2(av0, one_32_epi8, c128_16_epi16), Q8M1Sum128Avx2(av1, one_32_epi8, c128_16_epi16)
                    );
                    for (size_t c = 0; c < NCols4; ++c) {
                        const __m256i dot0 = Q8M1DotChunkAvx2(av0, QuantBDataPtr + c * SubBlkLen64, flip, one_16_epi16);
                        const __m256i dot1 = Q8M1DotChunkAvx2(av1, QuantBDataPtr + c * SubBlkLen64 + 32, flip, one_16_epi16);
                        const __m256i sum = _mm256_add_epi32(_mm256_add_epi32(dot0, dot1), corr);
                        acc[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum), scale_c[c], acc[c]);
                    }
                    QuantAPtr += SubBlkLen64;
                    QuantBDataPtr += SubBlkStride;
                }

                acc_blksum = _mm_fmadd_ps(_mm_set1_ps(ABlockSum[k]), _mm_loadu_ps(BlkSumPtr + k * BlkSumPanel), acc_blksum);

                QuantAScalePtr += 1;
                QuantBScalePtr += NCols4;
            }
        }

        __m128 acc_r0 = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);
        if (Bias != nullptr) {
            acc_r0 = _mm_add_ps(acc_r0, _mm_loadu_ps(Bias + n));
        }
        _mm_storeu_ps(C + n, _mm_add_ps(acc_blksum, acc_r0));
    }
}

// Columns of a trailing partial group of 4, stored in the original column-major order. ColumnOffset is the index
// of the first column relative to the start of QuantBBlkSum.
MLAS_FORCEINLINE void
Q8Int8GemmM1C1Avx2(
    size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    const float* ABlockSum,
    const float* QuantBBlkSum,
    size_t ColumnOffset
)
{
    constexpr size_t SubBlkLen64 = 64;
    constexpr size_t BlkSumPanel = 16;

    assert(BlkLen == 32 || BlkLen % SubBlkLen64 == 0);

    const size_t StrideQuantBDataCol = BlockCountK * BlkLen;
    const __m256i flip = _mm256_set1_epi8(static_cast<char>(0x80));
    const __m256i one_16_epi16 = _mm256_set1_epi16(1);
    const __m256i one_32_epi8 = _mm256_set1_epi8(1);
    const __m256i c128_16_epi16 = _mm256_set1_epi16(128);

    for (size_t n = 0; n < CountN; ++n) {
        const std::byte* QuantAPtr = QuantA;
        const std::byte* QuantBDataPtr = QuantBData + n * StrideQuantBDataCol;
        const float* QuantBScalePtr = QuantBScale + n * BlockCountK;
        const size_t col = ColumnOffset + n;
        const float* BlkSumPtr = QuantBBlkSum + (col / BlkSumPanel) * BlockCountK * BlkSumPanel + col % BlkSumPanel;

        __m256 acc = _mm256_setzero_ps();
        __m128 acc_blksum = _mm_setzero_ps();

        if (BlkLen == 32) {
            size_t k = 0;
            for (; k + 2 <= BlockCountK; k += 2) {
                const __m256i av0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantAPtr));
                const __m256i av1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantAPtr + 32));
                const __m256i corr = _mm256_hadd_epi32(
                    Q8M1Sum128Avx2(av0, one_32_epi8, c128_16_epi16), Q8M1Sum128Avx2(av1, one_32_epi8, c128_16_epi16)
                );
                const __m256i dot0 = Q8M1DotChunkAvx2(av0, QuantBDataPtr, flip, one_16_epi16);
                const __m256i dot1 = Q8M1DotChunkAvx2(av1, QuantBDataPtr + 32, flip, one_16_epi16);
                const __m256i sum = _mm256_add_epi32(_mm256_hadd_epi32(dot0, dot1), corr);  // 00 11 00 11

                const __m256 scale_b = _mm256_castpd_ps(_mm256_broadcast_sd(reinterpret_cast<const double*>(QuantBScalePtr + k)));
                const __m256 scale_a = _mm256_castpd_ps(_mm256_broadcast_sd(reinterpret_cast<const double*>(QuantAScale + k)));
                const __m256 scale_ab = _mm256_mul_ps(scale_b, scale_a);
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum), _mm256_shuffle_ps(scale_ab, scale_ab, _MM_SHUFFLE(1, 1, 0, 0)), acc);

                acc_blksum = _mm_fmadd_ss(_mm_set_ss(ABlockSum[k]), _mm_load_ss(BlkSumPtr + k * BlkSumPanel), acc_blksum);
                acc_blksum = _mm_fmadd_ss(_mm_set_ss(ABlockSum[k + 1]), _mm_load_ss(BlkSumPtr + (k + 1) * BlkSumPanel), acc_blksum);

                QuantAPtr += 64;
                QuantBDataPtr += 64;
            }

            if (k < BlockCountK) {
                const __m256i av0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantAPtr));
                const __m256i sum = _mm256_add_epi32(
                    Q8M1DotChunkAvx2(av0, QuantBDataPtr, flip, one_16_epi16), Q8M1Sum128Avx2(av0, one_32_epi8, c128_16_epi16)
                );
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum), _mm256_set1_ps(QuantAScale[k] * QuantBScalePtr[k]), acc);
                acc_blksum = _mm_fmadd_ss(_mm_set_ss(ABlockSum[k]), _mm_load_ss(BlkSumPtr + k * BlkSumPanel), acc_blksum);
            }
        } else {
            const size_t SubBlksPerBlk = BlkLen / SubBlkLen64;
            for (size_t k = 0; k < BlockCountK; ++k) {
                const __m256 scale = _mm256_set1_ps(QuantBScalePtr[k] * QuantAScale[k]);
                for (size_t j = 0; j < SubBlksPerBlk; ++j) {
                    const __m256i av0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantAPtr));
                    const __m256i av1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantAPtr + 32));
                    const __m256i corr = _mm256_add_epi32(
                        Q8M1Sum128Avx2(av0, one_32_epi8, c128_16_epi16), Q8M1Sum128Avx2(av1, one_32_epi8, c128_16_epi16)
                    );
                    const __m256i dot0 = Q8M1DotChunkAvx2(av0, QuantBDataPtr, flip, one_16_epi16);
                    const __m256i dot1 = Q8M1DotChunkAvx2(av1, QuantBDataPtr + 32, flip, one_16_epi16);
                    const __m256i sum = _mm256_add_epi32(_mm256_add_epi32(dot0, dot1), corr);
                    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum), scale, acc);
                    QuantAPtr += SubBlkLen64;
                    QuantBDataPtr += SubBlkLen64;
                }
                acc_blksum = _mm_fmadd_ss(_mm_set_ss(ABlockSum[k]), _mm_load_ss(BlkSumPtr + k * BlkSumPanel), acc_blksum);
            }
        }

        float result = hsum_float_8(acc);
        if (Bias != nullptr) {
            result += Bias[n];
        }
        C[n] = _mm_cvtss_f32(_mm_add_ss(acc_blksum, _mm_set_ss(result)));
    }
}

MLAS_FORCEINLINE void
MlasQ8Int8GemmM1KernelAvx2(
    size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    const float* ABlockSum,
    const float* QuantBBlkSum
)
{
    constexpr size_t NCols4 = 4;
    const size_t RemainingCols = CountN % NCols4;
    const size_t MultipleCols = CountN - RemainingCols;

    if (MultipleCols > 0) {
        Q8Int8GemmM1C4Avx2(
            BlkLen, QuantA, QuantAScale, QuantBData, QuantBScale, C, MultipleCols, BlockCountK, Bias, ABlockSum, QuantBBlkSum
        );
    }

    if (RemainingCols > 0) {
        Q8Int8GemmM1C1Avx2(
            BlkLen,
            QuantA,
            QuantAScale,
            QuantBData + MultipleCols * BlockCountK * BlkLen,
            QuantBScale + MultipleCols * BlockCountK,
            C + MultipleCols,
            RemainingCols,
            BlockCountK,
            Bias != nullptr ? Bias + MultipleCols : nullptr,
            ABlockSum,
            QuantBBlkSum,
            MultipleCols
        );
    }
}
