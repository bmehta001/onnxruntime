/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    flashattn_gqa.cpp

Abstract:

    Flash Attention kernel for the non-quantized (FP32) GroupQueryAttention
    KV cache.

    Adapts the online-softmax tiled algorithm from flashattn.cpp to operate on
    an FP32 present K/V cache laid out as BNSH
    ([batch, kv_num_heads, seqlen_present, head_size]) and to support GQA head
    grouping (num_heads % kv_num_heads == 0), causal masking, local window
    attention, additive attention bias, an attention sink (head_sink or smooth
    softmax), and an optional flash-decoding split over the KV sequence
    dimension for single-token decode.

    For multi-token prefill (sequence_length > 1) QK^T and S*V use the
    single-threaded SGEMM primitive MlasSgemmOperation. For single-token decode
    (sequence_length == 1) the M == 1 GEMVs use grouped GEMV kernels that apply
    each K/V row to every query head of its GQA group, avoiding both SGEMM
    packing overhead and re-reading the shared K/V head once per query head.
    The outer parallelism is provided by MlasExecuteThreaded.

--*/

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>

#include "mlasi.h"

//
// Decode (sequence_length == 1) grouped GEMV helpers.
//
// With a single query token the QK^T and S*V products degenerate into
// matrix-vector products. All query heads of a GQA group read the same K/V
// head, so the helpers apply each K/V row to the whole group while it is
// cached instead of streaming the K/V head once per query head. These portable
// versions live in the baseline-ISA translation unit; the inner loops use
// independent accumulator lanes and map-style updates so the compiler can
// vectorize them without -ffast-math. Platforms may provide faster kernels via
// GQADecodeQKKernel / GQADecodeSVKernel.
//

static void
MLASCALL
MlasGQADecodeQKKernel(
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
    constexpr size_t kLanes = 8;
    for (size_t t = 0; t < RowCount; ++t) {
        const float* krow = Key + t * HeadSize;
        for (size_t g = 0; g < GroupSize; ++g) {
            const float* q = Query + g * HeadSize;
            float acc[kLanes] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            size_t h = 0;
            for (; h + kLanes <= HeadSize; h += kLanes) {
                for (size_t j = 0; j < kLanes; ++j) {
                    acc[j] += q[h + j] * krow[h + j];
                }
            }
            float sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) +
                        ((acc[4] + acc[5]) + (acc[6] + acc[7]));
            for (; h < HeadSize; ++h) {
                sum += q[h] * krow[h];
            }
            Scores[g * ScoresStride + t] = sum * Scale;
        }
    }
}

static void
MLASCALL
MlasGQADecodeSVKernel(
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
    for (size_t g = 0; g < GroupSize; ++g) {
        std::fill_n(Output + g * OutputStride, HeadSize, 0.0f);
    }
    for (size_t t = 0; t < RowCount; ++t) {
        const float* vrow = Value + t * HeadSize;
        for (size_t g = 0; g < GroupSize; ++g) {
            const float p = Probs[g * ProbsStride + t];
            float* out = Output + g * OutputStride;
            for (size_t h = 0; h < HeadSize; ++h) {
                out[h] += p * vrow[h];
            }
        }
    }
}

constexpr size_t MlasGQADecodeMaxHeadTile = 16;

static float
MlasGQAHeadSink(const MlasFlashAttentionGQAArgs* args, ptrdiff_t head_idx)
{
    return args->head_sink != nullptr ? args->head_sink[head_idx] : 0.0f;
}

// Folds the sink logit into the softmax statistics: the sink adds exp(sink - m) to the
// denominator and contributes no value. Returns the factor that maps the unnormalized
// output (accumulated relative to `m`) to the final output.
static float
MlasGQAFinalScale(float m, float l, bool has_sink, float sink)
{
    if (has_sink) {
        if (!(m > std::numeric_limits<float>::lowest())) {
            return 0.0f;
        }
        const float m_final = std::max(m, sink);
        const float rescale = std::exp(m - m_final);
        const float denominator = l * rescale + std::exp(sink - m_final);
        return rescale / denominator;
    }
    return l > 0.0f ? 1.0f / l : 0.0f;
}

void
MlasFlashAttentionGQAThreaded(
    void* argptr,
    std::ptrdiff_t thread_id
)
{
    const MlasFlashAttentionGQAArgs* args =
        reinterpret_cast<MlasFlashAttentionGQAArgs*>(argptr);

    const ptrdiff_t q_block_size = static_cast<ptrdiff_t>(args->q_block_size);
    const ptrdiff_t kv_block_size = static_cast<ptrdiff_t>(args->kv_block_size);
    const ptrdiff_t batch_size = static_cast<ptrdiff_t>(args->batch_size);
    const ptrdiff_t num_heads = static_cast<ptrdiff_t>(args->num_heads);
    const ptrdiff_t kv_num_heads = static_cast<ptrdiff_t>(args->kv_num_heads);
    const ptrdiff_t sequence_length = static_cast<ptrdiff_t>(args->sequence_length);
    const ptrdiff_t valid_query_length =
        args->valid_query_length < 0 ? sequence_length : args->valid_query_length;
    const ptrdiff_t total_seqlen = static_cast<ptrdiff_t>(args->total_seqlen);
    const ptrdiff_t head_size = static_cast<ptrdiff_t>(args->head_size);
    const ptrdiff_t past_seqlen = static_cast<ptrdiff_t>(args->past_seqlen);
    const ptrdiff_t local_window_size = static_cast<ptrdiff_t>(args->local_window_size);
    const float scale = args->scale;

    float* buffer = args->buffer;
    const ptrdiff_t buffer_size_per_thread = static_cast<ptrdiff_t>(args->buffer_size_per_thread);
    const ptrdiff_t thread_count = static_cast<ptrdiff_t>(args->thread_count);

    const size_t kv_num_heads_factor = static_cast<size_t>(num_heads / kv_num_heads);
    const size_t kv_head_stride =
        static_cast<size_t>(args->seqlen_present_kv) * static_cast<size_t>(head_size);

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)
    auto&& mlas_platform = GetMlasPlatform();
#endif

    // Total tasks: one per (batch, head, q_block)
    const ptrdiff_t q_chunk_count = (valid_query_length + q_block_size - 1) / q_block_size;
    const ptrdiff_t total_task_count = batch_size * num_heads * q_chunk_count;

    ptrdiff_t task_start = 0;
    ptrdiff_t task_end = 0;
    ptrdiff_t quotient = total_task_count / thread_count;
    ptrdiff_t remainder = total_task_count % thread_count;
    if (thread_id < remainder) {
        task_start = (quotient + 1) * thread_id;
        task_end = task_start + quotient + 1;
    } else {
        task_start = quotient * thread_id + remainder;
        task_end = task_start + quotient;
    }

    for (ptrdiff_t task_index = task_start; task_index < task_end; ++task_index) {
        ptrdiff_t batch_idx = task_index;
        ptrdiff_t q_idx = (batch_idx % q_chunk_count) * q_block_size;
        batch_idx /= q_chunk_count;
        ptrdiff_t head_idx = batch_idx % num_heads;
        batch_idx /= num_heads;

        // Per-thread buffer layout:
        //   l[q_block_size]                       - running sum for online softmax
        //   m[q_block_size]                       - running max for online softmax
        //   scores[q_block_size * kv_block_size]  - QK scores (S)
        //   temp_output[q_block_size * head_size] - accumulated output
        char* buffer_ptr = reinterpret_cast<char*>(buffer) + thread_id * buffer_size_per_thread;
        float* l = reinterpret_cast<float*>(buffer_ptr);
        float* m = l + q_block_size;
        float* scores = m + q_block_size;
        float* temp_output = scores + q_block_size * kv_block_size;

        const size_t row_size_q = static_cast<size_t>(std::min(q_block_size, valid_query_length - q_idx));

        // Initialize only the active rows; scratch tile strides remain fixed.
        for (size_t t = 0; t < row_size_q; ++t) {
            m[t] = std::numeric_limits<float>::lowest();
            l[t] = 0.0f;
        }
        memset(temp_output, 0, row_size_q * static_cast<size_t>(head_size) * sizeof(float));

        // Determine KV head index for GQA head sharing
        const size_t kv_head_idx = static_cast<size_t>(head_idx) / kv_num_heads_factor;

        // K/V cache pointers. Layout: [batch, kv_num_heads, seqlen_present, head_size]
        const size_t kv_batch_head_offset =
            (static_cast<size_t>(batch_idx) * static_cast<size_t>(kv_num_heads) + kv_head_idx) *
            kv_head_stride;
        const float* k_cache_head = args->k_cache + kv_batch_head_offset;
        const float* v_cache_head = args->v_cache + kv_batch_head_offset;

        // Q pointer: layout [batch, num_heads, seq, head_size]. The batch stride is
        // supplied separately (args->q_batch_stride) so the kernel works with both the
        // standard BNSH layout and packed-QKV input where Q/K/V are interleaved per batch.
        const float* q_ptr = args->query +
            static_cast<size_t>(batch_idx) * args->q_batch_stride +
            static_cast<size_t>(head_idx) * static_cast<size_t>(sequence_length) * static_cast<size_t>(head_size) +
            static_cast<size_t>(q_idx) * static_cast<size_t>(head_size);

        // Causal early-termination bound: the largest global query position in this
        // q_block is (past_seqlen + q_idx + row_size_q - 1), so it can attend to KV
        // positions up to that index inclusive. Any KV block starting at or beyond
        // (past_seqlen + q_idx + row_size_q) is fully causally masked for every row in
        // the block, so it contributes nothing and can be skipped. This avoids the
        // wasted QK/SV GEMMs over the causal upper triangle during prefill.
        const ptrdiff_t kv_causal_limit =
            past_seqlen + q_idx + static_cast<ptrdiff_t>(row_size_q);

        // Iterate over KV blocks
        for (ptrdiff_t ir = 0; ir < total_seqlen; ir += kv_block_size) {
            if (ir >= kv_causal_limit) {
                break;
            }
            const size_t row_size_kv = static_cast<size_t>(std::min(kv_block_size, total_seqlen - ir));

            // Step 1: QK^T GEMM with FP32 K block
            const float* k_block = k_cache_head + static_cast<size_t>(ir) * static_cast<size_t>(head_size);
            MlasSgemmOperation(
                CblasNoTrans,
                CblasTrans,
                row_size_q,                         // M
                row_size_kv,                        // N
                static_cast<size_t>(head_size),     // K
                scale,                              // alpha
                q_ptr,                              // A (FP32 query)
                static_cast<size_t>(head_size),     // lda
                k_block,                            // B (FP32 K block)
                static_cast<size_t>(head_size),     // ldb
                0.0f,                               // beta
                scores,                             // C (output scores)
                row_size_kv                         // ldc
            );

            // Step 1b: Apply attention bias (additive) if present
            if (args->attention_bias != nullptr) {
                const ptrdiff_t bias_seqlen_stride =
                    static_cast<ptrdiff_t>(args->attention_bias_seqlen_stride);
                const ptrdiff_t bias_matrix_size =
                    static_cast<ptrdiff_t>(sequence_length) * bias_seqlen_stride;
                // The bias tensor has shape [batch|1, num_heads|1, S, T]; the batch
                // stride uses the actual head extent (1 when the head dim is broadcast).
                const ptrdiff_t bias_head_extent =
                    args->attention_bias_broadcast_head ? 1 : static_cast<ptrdiff_t>(num_heads);
                ptrdiff_t bias_offset = 0;
                if (!args->attention_bias_broadcast_batch) {
                    bias_offset += static_cast<ptrdiff_t>(batch_idx) *
                                   bias_head_extent * bias_matrix_size;
                }
                if (!args->attention_bias_broadcast_head) {
                    bias_offset += static_cast<ptrdiff_t>(head_idx) * bias_matrix_size;
                }
                // Add bias tile: bias[q_idx + irow, ir + jcol]
                for (ptrdiff_t irow = 0; irow < static_cast<ptrdiff_t>(row_size_q); ++irow) {
                    const float* bias_row = args->attention_bias + bias_offset +
                        (q_idx + irow) * bias_seqlen_stride + ir;
                    float* s_row = scores + irow * static_cast<ptrdiff_t>(row_size_kv);
                    for (ptrdiff_t jcol = 0; jcol < static_cast<ptrdiff_t>(row_size_kv); ++jcol) {
                        s_row[jcol] += bias_row[jcol];
                    }
                }
            }

            // Step 2: Apply causal mask and Step 3: Online softmax update
            for (ptrdiff_t irow = 0; irow < static_cast<ptrdiff_t>(row_size_q); ++irow) {
                float* p = scores + irow * static_cast<ptrdiff_t>(row_size_kv);
                const ptrdiff_t global_q_pos = past_seqlen + q_idx + irow;
                const ptrdiff_t causal_limit = global_q_pos + 1;  // can attend to positions [0, causal_limit)

                // Apply causal masking
                for (ptrdiff_t jcol = 0; jcol < static_cast<ptrdiff_t>(row_size_kv); ++jcol) {
                    ptrdiff_t kv_pos = ir + jcol;
                    if (kv_pos >= causal_limit) {
                        p[jcol] = std::numeric_limits<float>::lowest();
                    }
                }

                // Apply local window masking if enabled
                if (local_window_size >= 0) {
                    const ptrdiff_t window_start =
                        (causal_limit > local_window_size) ? (causal_limit - local_window_size) : 0;
                    for (ptrdiff_t jcol = 0; jcol < static_cast<ptrdiff_t>(row_size_kv); ++jcol) {
                        ptrdiff_t kv_pos = ir + jcol;
                        if (kv_pos < window_start) {
                            p[jcol] = std::numeric_limits<float>::lowest();
                        }
                    }
                }

                // Online softmax: find row max, update running max
#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)
                float rowmax = mlas_platform.ReduceMaximumF32Kernel(p, row_size_kv);
#else
                float rowmax = MlasReduceMaximumF32Kernel(p, row_size_kv);
#endif

                // If the entire row is masked (all scores are -inf), zero the scores
                // so the S*V GEMM contributes nothing and skip the softmax state update.
                if (rowmax == std::numeric_limits<float>::lowest()) {
                    memset(p, 0, row_size_kv * sizeof(float));
                    continue;
                }

                float m_old = m[irow];
                m[irow] = std::max(m[irow], rowmax);
                float m_diff = m_old - m[irow];  // <= 0

                // Compute exp(score - m_new) for each element
                float negmax = -m[irow];
#if defined(MLAS_TARGET_AMD64)
                float rowsum = mlas_platform.ComputeSumExpF32Kernel(p, p, row_size_kv, &negmax);
#else
                float rowsum = MlasComputeSumExpF32Kernel(p, p, row_size_kv, &negmax);
#endif

                // Rescale previous state
                if (ir != 0) {
                    float exp_diff = std::exp(m_diff);
                    l[irow] = exp_diff * l[irow] + rowsum;

                    // Rescale accumulated output
                    float* out_row = temp_output + irow * head_size;
                    for (ptrdiff_t icol = 0; icol < head_size; ++icol) {
                        out_row[icol] *= exp_diff;
                    }
                } else {
                    l[irow] = rowsum;
                }
            }

            // Step 4: Accumulate O += S_exp * V_block
            const float* v_block = v_cache_head + static_cast<size_t>(ir) * static_cast<size_t>(head_size);
            MlasSgemmOperation(
                CblasNoTrans,
                CblasNoTrans,
                row_size_q,                         // M
                static_cast<size_t>(head_size),     // N
                row_size_kv,                        // K
                1.0f,                               // alpha
                scores,                             // A (exp softmax scores)
                row_size_kv,                        // lda
                v_block,                            // B (FP32 V block)
                static_cast<size_t>(head_size),     // ldb
                ir == 0 ? 0.0f : 1.0f,              // beta (accumulate after first block)
                temp_output,                        // C (accumulated output)
                static_cast<size_t>(head_size)      // ldc
            );
        }

        // Final: normalize output by l (softmax denominator)
        // Output layout: [batch, sequence_length, num_heads, head_size]
        float* output_row = args->output +
            (static_cast<size_t>(batch_idx) * static_cast<size_t>(sequence_length) +
             static_cast<size_t>(q_idx)) * static_cast<size_t>(num_heads) * static_cast<size_t>(head_size) +
            static_cast<size_t>(head_idx) * static_cast<size_t>(head_size);
        const ptrdiff_t output_row_stride = num_heads * head_size;

        const bool has_sink = args->head_sink != nullptr || args->use_smooth_softmax;
        const float sink = has_sink ? MlasGQAHeadSink(args, head_idx) : 0.0f;
        for (ptrdiff_t irow = 0; irow < static_cast<ptrdiff_t>(row_size_q); ++irow) {
            const float inv_l = MlasGQAFinalScale(m[irow], l[irow], has_sink, sink);
            float* src = temp_output + irow * head_size;
            for (ptrdiff_t icol = 0; icol < head_size; ++icol) {
                output_row[icol] = src[icol] * inv_l;
            }
            output_row += output_row_stride;
        }
    }
}
//
// Single-token decode (sequence_length == 1).
//
// Tasks are (batch, kv_head, kv_chunk) with the chunk varying fastest. Each task
// computes all query heads of the GQA group against one chunk of the shared K/V
// head, so K/V rows are streamed once per group rather than once per query head.
// The chunking depends only on the arguments (never on the thread count), and
// partial results are combined in fixed chunk order, so results are identical
// for any thread count. Tasks are claimed dynamically to tolerate uneven worker
// availability; results do not depend on which worker runs a task.
//
// With kv_chunk_count > 1 each task writes (m, l, unnormalized output) for every
// head into flash_decoding_partials and MlasFlashDecodingGQAReduceThreaded combines
// them. With a single chunk the normalized output is written directly.
//
struct MlasGQADecodeContext {
    const MlasFlashAttentionGQAArgs* args;
    MLAS_GQA_DECODE_QK_KERNEL* qk_kernel;
    MLAS_GQA_DECODE_SV_KERNEL* sv_kernel;
    ptrdiff_t chunk_count;
    ptrdiff_t chunk_size;
    ptrdiff_t task_count;
    size_t head_tile;  // query heads processed per QK/SV call, bounded by the scratch size
    std::atomic<ptrdiff_t> next_task;
};

static void
MlasGQADecodeThreaded(
    void* argptr,
    std::ptrdiff_t thread_id
)
{
    auto* ctx = reinterpret_cast<MlasGQADecodeContext*>(argptr);
    const MlasFlashAttentionGQAArgs* args = ctx->args;

    const ptrdiff_t num_heads = static_cast<ptrdiff_t>(args->num_heads);
    const ptrdiff_t kv_num_heads = static_cast<ptrdiff_t>(args->kv_num_heads);
    const ptrdiff_t total_seqlen = static_cast<ptrdiff_t>(args->total_seqlen);
    const ptrdiff_t head_size = static_cast<ptrdiff_t>(args->head_size);
    const ptrdiff_t local_window_size = static_cast<ptrdiff_t>(args->local_window_size);
    const ptrdiff_t group_size = num_heads / kv_num_heads;
    const size_t kv_head_stride =
        static_cast<size_t>(args->seqlen_present_kv) * static_cast<size_t>(head_size);
    const ptrdiff_t chunk_count = ctx->chunk_count;
    const ptrdiff_t chunk_size = ctx->chunk_size;
    const bool use_partials = chunk_count > 1;
    const ptrdiff_t partial_stride = 2 + head_size;
    const bool has_sink = args->head_sink != nullptr || args->use_smooth_softmax;

    // The new token attends to positions [window_start, causal_limit).
    const ptrdiff_t causal_limit = std::min(total_seqlen, static_cast<ptrdiff_t>(args->past_seqlen) + 1);
    const ptrdiff_t window_start =
        (local_window_size >= 0 && causal_limit > local_window_size) ? causal_limit - local_window_size : 0;

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)
    auto&& mlas_platform = GetMlasPlatform();
#endif

    float* scores = reinterpret_cast<float*>(
        reinterpret_cast<char*>(args->buffer) + thread_id * args->buffer_size_per_thread);

    for (;;) {
        const ptrdiff_t task_index = ctx->next_task.fetch_add(1, std::memory_order_relaxed);
        if (task_index >= ctx->task_count) {
            break;
        }

        const ptrdiff_t chunk_idx = task_index % chunk_count;
        const ptrdiff_t kv_head_idx = (task_index / chunk_count) % kv_num_heads;
        const ptrdiff_t batch_idx = task_index / (chunk_count * kv_num_heads);

        const ptrdiff_t chunk_start = chunk_idx * chunk_size;
        const ptrdiff_t chunk_end = std::min(chunk_start + chunk_size, total_seqlen);
        const ptrdiff_t row_start = std::max(chunk_start, window_start);
        const ptrdiff_t row_end = std::min(chunk_end, causal_limit);
        const ptrdiff_t row_count = row_end - row_start;

        const size_t kv_offset =
            (static_cast<size_t>(batch_idx) * static_cast<size_t>(kv_num_heads) + static_cast<size_t>(kv_head_idx)) *
                kv_head_stride +
            static_cast<size_t>(std::max<ptrdiff_t>(row_start, 0)) * static_cast<size_t>(head_size);
        const float* k_rows = args->k_cache + kv_offset;
        const float* v_rows = args->v_cache + kv_offset;

        const ptrdiff_t first_head = kv_head_idx * group_size;

        // Output (or partial output) for head h of this task, and the stride between heads.
        auto head_output = [&](ptrdiff_t head_idx) -> float* {
            if (use_partials) {
                const ptrdiff_t partial_index = (batch_idx * num_heads + head_idx) * chunk_count + chunk_idx;
                return args->flash_decoding_partials + partial_index * partial_stride + 2;
            }
            return args->output +
                   (static_cast<size_t>(batch_idx) * static_cast<size_t>(num_heads) + static_cast<size_t>(head_idx)) *
                       static_cast<size_t>(head_size);
        };
        const size_t output_stride = use_partials ? static_cast<size_t>(chunk_count * partial_stride)
                                                  : static_cast<size_t>(head_size);

        if (row_count <= 0) {
            // Entire chunk is outside the visible range.
            for (ptrdiff_t g = 0; g < group_size; ++g) {
                float* out = head_output(first_head + g);
                std::fill_n(out, head_size, 0.0f);
                if (use_partials) {
                    out[-2] = std::numeric_limits<float>::lowest();
                    out[-1] = 0.0f;
                }
            }
            continue;
        }

        for (ptrdiff_t g0 = 0; g0 < group_size; g0 += static_cast<ptrdiff_t>(ctx->head_tile)) {
            const size_t tile = static_cast<size_t>(std::min<ptrdiff_t>(ctx->head_tile, group_size - g0));
            const ptrdiff_t tile_head = first_head + g0;
            const float* q_ptr = args->query + static_cast<size_t>(batch_idx) * args->q_batch_stride +
                                 static_cast<size_t>(tile_head) * static_cast<size_t>(head_size);

            ctx->qk_kernel(q_ptr, tile, k_rows, static_cast<size_t>(row_count), static_cast<size_t>(head_size),
                           args->scale, scores, static_cast<size_t>(row_count));

            float row_max[MlasGQADecodeMaxHeadTile];
            float row_sum[MlasGQADecodeMaxHeadTile];
            for (size_t g = 0; g < tile; ++g) {
                const ptrdiff_t head_idx = tile_head + static_cast<ptrdiff_t>(g);
                float* head_scores = scores + g * static_cast<size_t>(row_count);

                if (args->attention_bias != nullptr) {
                    // Bias shape [batch|1, num_heads|1, S=1, T].
                    const ptrdiff_t bias_row_size = static_cast<ptrdiff_t>(args->attention_bias_seqlen_stride);
                    const ptrdiff_t bias_head_extent = args->attention_bias_broadcast_head ? 1 : num_heads;
                    ptrdiff_t bias_offset = 0;
                    if (!args->attention_bias_broadcast_batch) {
                        bias_offset += batch_idx * bias_head_extent * bias_row_size;
                    }
                    if (!args->attention_bias_broadcast_head) {
                        bias_offset += head_idx * bias_row_size;
                    }
                    const float* bias_row = args->attention_bias + bias_offset + row_start;
                    for (ptrdiff_t t = 0; t < row_count; ++t) {
                        head_scores[t] += bias_row[t];
                    }
                }

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)
                float m = mlas_platform.ReduceMaximumF32Kernel(head_scores, static_cast<size_t>(row_count));
#else
                float m = MlasReduceMaximumF32Kernel(head_scores, static_cast<size_t>(row_count));
#endif
                float l = 0.0f;
                if (m > std::numeric_limits<float>::lowest()) {
                    float negmax = -m;
#if defined(MLAS_TARGET_AMD64)
                    l = mlas_platform.ComputeSumExpF32Kernel(head_scores, head_scores, static_cast<size_t>(row_count),
                                                             &negmax);
#else
                    l = MlasComputeSumExpF32Kernel(head_scores, head_scores, static_cast<size_t>(row_count), &negmax);
#endif
                } else {
                    // Every visible score is masked: give the rows zero weight.
                    m = std::numeric_limits<float>::lowest();
                    std::fill_n(head_scores, row_count, 0.0f);
                }
                row_max[g] = m;
                row_sum[g] = l;
            }

            float* out = head_output(tile_head);
            ctx->sv_kernel(scores, static_cast<size_t>(row_count), tile, v_rows, static_cast<size_t>(row_count),
                           static_cast<size_t>(head_size), out, output_stride);

            for (size_t g = 0; g < tile; ++g) {
                float* head_out = out + g * output_stride;
                if (use_partials) {
                    head_out[-2] = row_max[g];
                    head_out[-1] = row_sum[g];
                } else {
                    const ptrdiff_t head_idx = tile_head + static_cast<ptrdiff_t>(g);
                    const float factor =
                        MlasGQAFinalScale(row_max[g], row_sum[g], has_sink, has_sink ? MlasGQAHeadSink(args, head_idx) : 0.0f);
                    for (ptrdiff_t h = 0; h < head_size; ++h) {
                        head_out[h] *= factor;
                    }
                }
            }
        }
    }
}

//
// Flash Decoding: Phase 2 - reduce partials for each (batch, head) into final output.
//
void
MlasFlashDecodingGQAReduceThreaded(
    void* argptr,
    std::ptrdiff_t thread_id
)
{
    const MlasFlashAttentionGQAArgs* args =
        reinterpret_cast<MlasFlashAttentionGQAArgs*>(argptr);

    const ptrdiff_t batch_size = static_cast<ptrdiff_t>(args->batch_size);
    const ptrdiff_t num_heads = static_cast<ptrdiff_t>(args->num_heads);
    const ptrdiff_t head_size = static_cast<ptrdiff_t>(args->head_size);
    const ptrdiff_t kv_chunk_count = static_cast<ptrdiff_t>(args->kv_chunk_count);
    const ptrdiff_t thread_count = static_cast<ptrdiff_t>(args->thread_count);
    const ptrdiff_t partial_stride = 2 + head_size;
    const bool has_sink = args->head_sink != nullptr || args->use_smooth_softmax;

    // Total reduction tasks: one per (batch, head)
    const ptrdiff_t total_task_count = batch_size * num_heads;

    ptrdiff_t task_start = 0;
    ptrdiff_t task_end = 0;
    ptrdiff_t quotient = total_task_count / thread_count;
    ptrdiff_t remainder = total_task_count % thread_count;
    if (thread_id < remainder) {
        task_start = (quotient + 1) * thread_id;
        task_end = task_start + quotient + 1;
    } else {
        task_start = quotient * thread_id + remainder;
        task_end = task_start + quotient;
    }

    for (ptrdiff_t task_index = task_start; task_index < task_end; ++task_index) {
        ptrdiff_t head_idx = task_index % num_heads;
        ptrdiff_t batch_idx = task_index / num_heads;

        // Pointer to this (batch, head)'s partials: kv_chunk_count entries
        const float* partials_base = args->flash_decoding_partials +
            task_index * kv_chunk_count * partial_stride;

        // Find global max across all chunks
        float global_m = std::numeric_limits<float>::lowest();
        for (ptrdiff_t c = 0; c < kv_chunk_count; ++c) {
            float chunk_m = partials_base[c * partial_stride];
            global_m = std::max(global_m, chunk_m);
        }

        // Output layout: [batch, sequence_length=1, num_heads, head_size]
        float* output_ptr = args->output +
            static_cast<size_t>(batch_idx) * static_cast<size_t>(num_heads) * static_cast<size_t>(head_size) +
            static_cast<size_t>(head_idx) * static_cast<size_t>(head_size);

        // If all chunks are masked, output zeros
        if (global_m == std::numeric_limits<float>::lowest()) {
            memset(output_ptr, 0, static_cast<size_t>(head_size) * sizeof(float));
            continue;
        }

        // Accumulate rescaled outputs and l values
        float global_l = 0.0f;
        memset(output_ptr, 0, static_cast<size_t>(head_size) * sizeof(float));

        for (ptrdiff_t c = 0; c < kv_chunk_count; ++c) {
            const float* partial = partials_base + c * partial_stride;
            float chunk_m = partial[0];
            float chunk_l = partial[1];
            const float* chunk_output = partial + 2;

            if (chunk_l <= 0.0f) {
                continue;  // masked chunk contributes nothing
            }

            float rescale = std::exp(chunk_m - global_m);
            global_l += rescale * chunk_l;

            // partial_output = S_exp * V where sum(S_exp) = l_c (unnormalized).
            // Rescale by exp(m_c - global_m) to align all chunks to the same max.
            for (ptrdiff_t i = 0; i < head_size; ++i) {
                output_ptr[i] += rescale * chunk_output[i];
            }
        }

        // output = sum_c(rescale_c * partial_output_c) / global_l
        const float factor = MlasGQAFinalScale(global_m, global_l, has_sink,
                                               has_sink ? MlasGQAHeadSink(args, head_idx) : 0.0f);
        for (ptrdiff_t i = 0; i < head_size; ++i) {
            output_ptr[i] *= factor;
        }
    }
}

void
MLASCALL
MlasFlashAttentionGQA(
    MlasFlashAttentionGQAArgs* args,
    MLAS_THREADPOOL* ThreadPool
)
{
    if (args->valid_query_length == 0) {
        return;
    }
    if (args->sequence_length == 1) {
        // Decode: M = 1, use the grouped GEMV kernels (no SGEMM packing overhead).
        const size_t group_size = static_cast<size_t>(args->num_heads / args->kv_num_heads);
        const bool split = args->flash_decoding_partials != nullptr && args->kv_chunk_count > 1;

        MlasGQADecodeContext ctx;
        ctx.args = args;
        ctx.qk_kernel = MlasGQADecodeQKKernel;
        ctx.sv_kernel = MlasGQADecodeSVKernel;
        const auto& platform = GetMlasPlatform();
        if (args->head_size % 8 == 0 && platform.GQADecodeQKKernel != nullptr &&
            platform.GQADecodeSVKernel != nullptr) {
            ctx.qk_kernel = platform.GQADecodeQKKernel;
            ctx.sv_kernel = platform.GQADecodeSVKernel;
        }
        ctx.chunk_count = split ? args->kv_chunk_count : 1;
        ctx.chunk_size = split ? args->kv_block_size : std::max(args->total_seqlen, 1);
        ctx.task_count = static_cast<ptrdiff_t>(args->batch_size) * args->kv_num_heads * ctx.chunk_count;
        const size_t scratch_rows = args->buffer_size_per_thread / sizeof(float);
        ctx.head_tile = std::max<size_t>(
            1, std::min({group_size, MlasGQADecodeMaxHeadTile, scratch_rows / static_cast<size_t>(ctx.chunk_size)}));
        ctx.next_task.store(0, std::memory_order_relaxed);

        MlasExecuteThreaded(
            MlasGQADecodeThreaded,
            static_cast<void*>(&ctx),
            std::min<std::ptrdiff_t>(args->thread_count, ctx.task_count),
            ThreadPool
        );
        if (split) {
            // Combine the per-chunk partials in fixed chunk order (parallel over batch * heads).
            MlasExecuteThreaded(
                MlasFlashDecodingGQAReduceThreaded,
                static_cast<void*>(args),
                static_cast<std::ptrdiff_t>(args->thread_count),
                ThreadPool
            );
        }
    } else {
        // Prefill (sequence_length > 1): tiled online-softmax SGEMM kernel.
        MlasExecuteThreaded(
            MlasFlashAttentionGQAThreaded,
            static_cast<void*>(args),
            static_cast<std::ptrdiff_t>(args->thread_count),
            ThreadPool
        );
    }
}
