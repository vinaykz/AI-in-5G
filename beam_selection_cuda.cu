/**
 * beam_selection_cuda.cu
 * ============================================================================
 * CUDA kernels for RSRP-based beam selection
 *
 * GPU target : NVIDIA L4 (sm_89, Ada Lovelace)
 *   • 7680 CUDA cores, 4th-gen Tensor Cores, 24 GB GDDR6 @ 300 GB/s
 *   • No NVLink — single-GPU design
 *   • L2 cache : 48 MB  →  keep working sets resident when possible
 *
 * Compile:
 *   nvcc -O3 -arch=sm_89 --use_fast_math -lineinfo \
 *        -Xcompiler "-march=native -fPIC" \
 *        beam_selection_cuda.cu -o beam_selection_cuda.o
 * ============================================================================
 */

#include "beam_selection.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <cooperative_groups.h>
#include <math_constants.h>
#include <stdio.h>
#include <assert.h>

namespace cg = cooperative_groups;

/* ─────────────────── compile-time tile dimensions ──────────────────────── */
#define RSRP_THREADS_PER_BLOCK  256   /* tuned for L4 occupancy              */
#define BEAM_BLOCK_X            16    /* beams processed per CTA (x-dim)     */
#define SC_TILE                 256   /* subcarriers per tile                 */

/* ─────────────────────────────────────────────────────────────────────────
 * KERNEL 1: computeBeamRSRP
 *
 * Computes RSRP per beam in linear scale.
 *
 * For CSI-RS port p on subcarrier k, symbol l:
 *
 *   ĥ(k,l,p) = y(k,l,p) / s(k,p)          [LS channel estimate]
 *
 *   RSRP_linear = (1 / N_RE) * Σ_{k,l,p} |ĥ(k,l,p)|²
 *
 * where N_RE = num_sc * num_symbols * num_ports.
 *
 * Grid  : (num_beams, ceil(num_sc / SC_TILE))
 * Block : (RSRP_THREADS_PER_BLOCK, 1)
 * Smem  : RSRP_THREADS_PER_BLOCK * sizeof(float)
 * ───────────────────────────────────────────────────────────────────────── */
__global__ void computeBeamRSRP(
    const cuComplex * __restrict__ rx_iq,       /* [nBeams × nSC × nSym]    */
    const cuComplex * __restrict__ csirs_ref,   /* [nBeams × nSC]  — ref seq */
    float           * __restrict__ rsrp_accum,  /* [nBeams] partial sums     */
    const int nBeams,
    const int nSC,
    const int nSym
) {
    /* Each block handles one beam (gridDim.x) across a tile of SC (gridDim.y) */
    const int beam  = blockIdx.x;
    const int sc_base = blockIdx.y * SC_TILE;

    if (beam >= nBeams) return;

    extern __shared__ float smem[];  /* [RSRP_THREADS_PER_BLOCK] */

    float thread_sum = 0.0f;

    /* Stride loop over subcarriers assigned to this block */
    for (int sc = sc_base + threadIdx.x; sc < sc_base + SC_TILE && sc < nSC;
         sc += blockDim.x)
    {
        /* Reference sample for this beam / subcarrier */
        const cuComplex ref = csirs_ref[beam * nSC + sc];
        const float ref_pwr = ref.x * ref.x + ref.y * ref.y;

        if (ref_pwr < 1e-12f) continue;   /* skip if reference is zero      */

        for (int sym = 0; sym < nSym; sym++) {
            const cuComplex y = rx_iq[(beam * nSC + sc) * nSym + sym];

            /*
             * LS estimate: ĥ = y·ref* / |ref|²
             * |ĥ|² = |y·ref*|² / |ref|⁴  = |y|²/|ref|²  (if |ref|=1, simplified)
             *
             * General form used here (handles non-unit amplitude references):
             *   h_re = (y.re·ref.re + y.im·ref.im) / ref_pwr
             *   h_im = (y.im·ref.re − y.re·ref.im) / ref_pwr
             */
            const float inv_ref = __frcp_rn(ref_pwr);  /* fast HW reciprocal */
            const float h_re = (y.x * ref.x + y.y * ref.y) * inv_ref;
            const float h_im = (y.y * ref.x - y.x * ref.y) * inv_ref;

            thread_sum += h_re * h_re + h_im * h_im;
        }
    }

    /* ── parallel reduction within the block ── */
    smem[threadIdx.x] = thread_sum;
    __syncthreads();

#pragma unroll
    for (int stride = RSRP_THREADS_PER_BLOCK / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }

    /* Warp-level reduction for last 32 threads (no __syncthreads needed) */
    if (threadIdx.x < 32) {
        volatile float *vsmem = smem;
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 32];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 16];
        vsmem[threadIdx.x] += vsmem[threadIdx.x +  8];
        vsmem[threadIdx.x] += vsmem[threadIdx.x +  4];
        vsmem[threadIdx.x] += vsmem[threadIdx.x +  2];
        vsmem[threadIdx.x] += vsmem[threadIdx.x +  1];
    }

    /* One thread per block atomically accumulates into global per-beam sum */
    if (threadIdx.x == 0)
        atomicAdd(&rsrp_accum[beam], smem[0]);
}


/* ─────────────────────────────────────────────────────────────────────────
 * KERNEL 2: finalizeRSRP
 *
 * Converts the accumulated |ĥ|² sums to:
 *   • RSRP_linear  (normalised by N_RE)
 *   • RSRP_dBm     (10·log10(rsrp_linear) + 30)
 *   • RSRQ_dB      = RSRP_dBm − RSSI_dBm  (simplified: uses noise_var)
 *   • SINR_dB      = RSRP_linear / noise_var_avg
 *
 * Marks beams below rsrp_floor_dBm as invalid.
 *
 * Grid : (1), Block : (num_beams) — small kernel, one thread per beam.
 * ───────────────────────────────────────────────────────────────────────── */
__global__ void finalizeRSRP(
    const float           * __restrict__ rsrp_accum,    /* [nBeams] raw sums  */
    const float           * __restrict__ noise_var,     /* [nSC] per-SC noise */
    BeamMeasurement_t     * __restrict__ measurements,  /* [nBeams] output     */
    const int nBeams,
    const int nSC,
    const int nSym,
    const float rsrp_floor_dBm,
    const uint64_t slot_number
) {
    const int beam = threadIdx.x + blockIdx.x * blockDim.x;
    if (beam >= nBeams) return;

    const float n_re      = (float)(nSC * nSym);
    const float rsrp_lin  = rsrp_accum[beam] / n_re;

    /* Noise average across all subcarriers (all beams share same noise) */
    /* In production, per-beam noise estimated from null subcarriers.     */
    float noise_avg = 0.0f;
    for (int sc = 0; sc < nSC; sc++)
        noise_avg += noise_var[sc];
    noise_avg /= (float)nSC;

    /* dBm: 10·log10(RSRP_watts) + 30 dB */
    const float rsrp_dBm  = 10.0f * log10f(rsrp_lin + 1e-20f) + 30.0f;

    /* RSRQ = N_RB × RSRP / RSSI  (simplified: RSSI ≈ RSRP + noise) */
    const float rssi_lin  = rsrp_lin + noise_avg;
    const float rsrq_dB   = 10.0f * log10f(rsrp_lin / rssi_lin + 1e-20f);

    /* SINR = signal power / noise power */
    const float sinr_dB   = 10.0f * log10f(rsrp_lin / (noise_avg + 1e-20f));

    measurements[beam].beam_id      = (uint8_t)beam;
    measurements[beam].rsrp_linear  = rsrp_lin;
    measurements[beam].rsrp_dBm     = rsrp_dBm;
    measurements[beam].rsrq_dB      = rsrq_dB;
    measurements[beam].sinr_dB      = sinr_dB;
    measurements[beam].is_valid     = (rsrp_dBm >= rsrp_floor_dBm);
    measurements[beam].slot_number  = slot_number;
}


/* ─────────────────────────────────────────────────────────────────────────
 * KERNEL 3: selectBestBeam
 *
 * Parallel reduction to find beam_id with maximum RSRP among valid beams.
 * Uses cooperative groups for warp-level primitives.
 *
 * Grid : (1), Block : (num_beams, padded to warp multiple)
 * ───────────────────────────────────────────────────────────────────────── */
__global__ void selectBestBeam(
    const BeamMeasurement_t * __restrict__ measurements,
    const float                            hysteresis_dB,
    const float                            current_beam_rsrp_dBm, /* prev best */
    uint8_t                              * best_beam_out,
    const int                              nBeams
) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());

    const int tid  = threadIdx.x;
    const int beam = tid;

    /* Load valid RSRP; set invalid beams to -∞ */
    float rsrp = (beam < nBeams && measurements[beam].is_valid)
                 ? measurements[beam].rsrp_dBm
                 : -CUDART_INF_F;

    /*
     * Apply hysteresis: current serving beam gets a "bonus" so we
     * don't thrash on small RSRP differences (3GPP A3 event equivalent).
     * current_beam_rsrp_dBm == -∞ means no current beam (initial selection).
     */
    if (beam < nBeams && measurements[beam].is_valid &&
        fabsf(measurements[beam].rsrp_dBm - current_beam_rsrp_dBm) < hysteresis_dB)
    {
        /* Bias toward current beam if within hysteresis window */
        rsrp += hysteresis_dB;
    }

    /* Warp-level argmax */
    float  best_rsrp  = rsrp;
    int    best_idx   = beam;

    for (int offset = warp.size() / 2; offset > 0; offset >>= 1) {
        float  peer_rsrp = warp.shfl_down(best_rsrp, offset);
        int    peer_idx  = warp.shfl_down(best_idx,  offset);
        if (peer_rsrp > best_rsrp) {
            best_rsrp = peer_rsrp;
            best_idx  = peer_idx;
        }
    }

    /* First lane in warp 0 writes result */
    if (warp.thread_rank() == 0 && threadIdx.x == 0)
        *best_beam_out = (uint8_t)best_idx;
}


/* ─────────────────────────────────────────────────────────────────────────
 * KERNEL 4: updateHistoryBuffer
 *
 * Appends the current slot's RSRP vector to a circular history buffer.
 * This buffer is consumed by the AI feature extraction step.
 *
 * Layout: history_buf[BEAM_SEL_HISTORY_DEPTH × nBeams]  (row = slot, oldest first)
 * ───────────────────────────────────────────────────────────────────────── */
__global__ void updateHistoryBuffer(
    float                          * __restrict__ history_buf,
    const BeamMeasurement_t        * __restrict__ measurements,
    const int                                     nBeams,
    const int                                     write_idx   /* 0 … DEPTH-1 */
) {
    const int beam = threadIdx.x + blockIdx.x * blockDim.x;
    if (beam >= nBeams) return;

    history_buf[write_idx * nBeams + beam] = measurements[beam].rsrp_dBm;
}


/* ─────────────────────────────────────────────────────────────────────────
 * KERNEL 5: buildAIFeatureVector
 *
 * Constructs the input tensor for the TensorRT beam prediction model.
 *
 * Feature layout per beam (total: BEAM_SEL_AI_FEATURE_DIM floats):
 *   [0]       current RSRP_dBm
 *   [1]       current RSRQ_dB
 *   [2]       RSRP slope  = (rsrp_now − rsrp[H-1]) / H  (dBm/slot)
 *   [3]       RSRP std_dev over history window
 *
 * grid : (1, 1), block : (nBeams, 1)
 * ───────────────────────────────────────────────────────────────────────── */
__global__ void buildAIFeatureVector(
    const float               * __restrict__ history_buf,   /* [DEPTH × nBeams]  */
    const BeamMeasurement_t   * __restrict__ measurements,  /* current slot       */
    float                     * __restrict__ feature_vec,   /* output [4 × nBeams]*/
    const int                               nBeams,
    const int                               history_depth,
    const int                               write_idx       /* latest slot index  */
) {
    const int beam = threadIdx.x + blockIdx.x * blockDim.x;
    if (beam >= nBeams) return;

    /* Current RSRP / RSRQ */
    const float rsrp_now  = measurements[beam].rsrp_dBm;
    const float rsrq_now  = measurements[beam].rsrq_dB;

    /* Compute mean and slope over history */
    float sum  = 0.0f, sum_sq = 0.0f;
    for (int d = 0; d < history_depth; d++) {
        /* oldest → newest:  (write_idx+1+d) % history_depth  */
        const int slot_idx = (write_idx + 1 + d) % history_depth;
        const float v = history_buf[slot_idx * nBeams + beam];
        sum    += v;
        sum_sq += v * v;
    }
    const float mean    = sum    / (float)history_depth;
    const float var     = sum_sq / (float)history_depth - mean * mean;
    const float std_dev = sqrtf(fmaxf(var, 0.0f));

    /* Slope: difference between newest and oldest normalised by depth */
    const int oldest_idx = (write_idx + 1) % history_depth;
    const float rsrp_old = history_buf[oldest_idx * nBeams + beam];
    const float slope    = (rsrp_now - rsrp_old) / (float)history_depth;

    /* Pack into feature vector  [4 features × nBeams] */
    feature_vec[0 * nBeams + beam] = rsrp_now;
    feature_vec[1 * nBeams + beam] = rsrq_now;
    feature_vec[2 * nBeams + beam] = slope;
    feature_vec[3 * nBeams + beam] = std_dev;
}


/* ─────────────────────────────────────────────────────────────────────────
 * Host-side launcher functions
 * (called from beam_manager.c through the context dispatch table)
 * ───────────────────────────────────────────────────────────────────────── */

/**
 * launchComputeRSRP
 *
 * Schedules kernels 1–3 on the provided CUDA stream:
 *   1. Zero rsrp_accum_d
 *   2. computeBeamRSRP  — accumulate |ĥ|² per beam
 *   3. finalizeRSRP     — convert to dBm, mark validity
 *   4. selectBestBeam   — find argmax with hysteresis
 *
 * All buffers are device pointers.  No cudaDeviceSynchronize here;
 * caller streams an event or syncs at its own cadence.
 */
extern "C"
BeamSelStatus_t launchComputeRSRP(
    const cuComplex     *rx_iq_d,
    const cuComplex     *csirs_ref_d,
    const float         *noise_var_d,
    float               *rsrp_accum_d,        /* zeroed in this function     */
    BeamMeasurement_t   *measurements_d,
    uint8_t             *best_beam_d,
    float               *history_buf_d,
    float               *feature_vec_d,
    const int            nBeams,
    const int            nSC,
    const int            nSym,
    const float          rsrp_floor_dBm,
    const float          hysteresis_dB,
    const float          current_rsrp_dBm,
    const uint64_t       slot_number,
    const int            history_write_idx,
    cudaStream_t         stream
) {
    cudaError_t err;

    /* ── 1. Zero per-beam accumulator ── */
    err = cudaMemsetAsync(rsrp_accum_d, 0, nBeams * sizeof(float), stream);
    if (err != cudaSuccess) return BEAM_SEL_ERR_CUDA;

    /* ── 2. RSRP accumulation kernel ── */
    const int num_sc_tiles = (nSC + SC_TILE - 1) / SC_TILE;
    dim3 grid2(nBeams, num_sc_tiles);
    dim3 block2(RSRP_THREADS_PER_BLOCK);
    const size_t smem2 = RSRP_THREADS_PER_BLOCK * sizeof(float);

    computeBeamRSRP<<<grid2, block2, smem2, stream>>>(
        rx_iq_d, csirs_ref_d, rsrp_accum_d, nBeams, nSC, nSym);

    err = cudaGetLastError();
    if (err != cudaSuccess) return BEAM_SEL_ERR_CUDA;

    /* ── 3. Finalize: convert to dBm, fill BeamMeasurement_t ── */
    const int beam_block = 32;  /* one warp */
    const int beam_grid  = (nBeams + beam_block - 1) / beam_block;

    finalizeRSRP<<<beam_grid, beam_block, 0, stream>>>(
        rsrp_accum_d, noise_var_d, measurements_d,
        nBeams, nSC, nSym, rsrp_floor_dBm, slot_number);

    err = cudaGetLastError();
    if (err != cudaSuccess) return BEAM_SEL_ERR_CUDA;

    /* ── 4. Argmax with hysteresis ── */
    /*    Block size padded to next warp multiple (max 64 beams → 64 threads) */
    const int sel_threads = ((nBeams + 31) / 32) * 32;
    selectBestBeam<<<1, sel_threads, 0, stream>>>(
        measurements_d, hysteresis_dB, current_rsrp_dBm, best_beam_d, nBeams);

    err = cudaGetLastError();
    if (err != cudaSuccess) return BEAM_SEL_ERR_CUDA;

    /* ── 5. Update circular history buffer ── */
    updateHistoryBuffer<<<beam_grid, beam_block, 0, stream>>>(
        history_buf_d, measurements_d, nBeams, history_write_idx);

    /* ── 6. Build AI feature vector (runs every slot; AI inference gated separately) ── */
    buildAIFeatureVector<<<beam_grid, beam_block, 0, stream>>>(
        history_buf_d, measurements_d, feature_vec_d,
        nBeams, BEAM_SEL_HISTORY_DEPTH, history_write_idx);

    err = cudaGetLastError();
    if (err != cudaSuccess) return BEAM_SEL_ERR_CUDA;

    return BEAM_SEL_OK;
}
