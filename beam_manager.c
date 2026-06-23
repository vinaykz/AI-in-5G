/**
 * beam_manager.c
 * ============================================================================
 * Aerial cuPHY integration + main beam selection state machine
 *
 * This module:
 *   1. Registers as a cuPHY slot-indication consumer
 *   2. Extracts CSI-RS IQ samples from the Aerial RX buffer (device memory)
 *   3. Dispatches CUDA kernels (via launchComputeRSRP in beam_selection_cuda.cu)
 *   4. Invokes AI inference (via beam_selection_ai.cpp) every N slots
 *   5. Updates beamforming weights in the Aerial TX config
 *
 * Aerial SDK references:
 *   cuphy.h          — cuPHY base types, cell/UE config
 *   aerial_phy.h     — Aerial gNB L1 API (slot indication, buffer mgmt)
 *   cuphy_ext.h      — Extended APIs: CSI-RS measurement, BF weight update
 * ============================================================================
 */

#include "beam_selection.h"
#include <cuphy.h>           /* NVIDIA Aerial cuPHY SDK                      */
#include <aerial_phy.h>      /* Aerial gNB L1 API                            */
#include <cuphy_ext.h>       /* CSI-RS measurement helpers, BF weight update */
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

/* ──────────────────── forward declarations ──────────────────────────────── */
extern BeamSelStatus_t launchComputeRSRP(   /* defined in beam_selection_cuda.cu  */
    const cuComplex *, const cuComplex *, const float *,
    float *, BeamMeasurement_t *, uint8_t *, float *, float *,
    int, int, int, float, float, float, uint64_t, int, cudaStream_t);

extern BeamSelStatus_t aiInferBestBeam(     /* defined in beam_selection_ai.cpp   */
    void *ai_ctx, const float *feature_vec_d, int nBeams,
    uint8_t *ai_beam_out, float *confidence_out, cudaStream_t stream);

/* ──────────────────── context structure (opaque to callers) ─────────────── */
struct BeamSelContext {
    /* ── configuration ── */
    BeamSweepCfg_t  sweep_cfg;
    AiCfg_t         ai_cfg;

    /* ── CUDA resources ── */
    cudaStream_t    stream;
    cudaEvent_t     kernel_done_event;

    /* ── device buffers ── */
    cuComplex      *csirs_ref_d;        /* [nBeams × nSC]  reference seqs     */
    float          *rsrp_accum_d;       /* [nBeams]        accumulation buf    */
    BeamMeasurement_t *measurements_d;  /* [nBeams]        current slot meas   */
    uint8_t        *best_beam_d;        /* [1]             GPU-selected beam   */
    float          *noise_var_d;        /* [nSC]           per-SC noise power  */
    float          *history_buf_d;      /* [DEPTH × nBeams] RSRP history       */
    float          *feature_vec_d;      /* [4 × nBeams]    AI feature input    */

    /* ── host buffers (pinned) ── */
    BeamMeasurement_t *measurements_h;  /* [nBeams]        host copy           */
    uint8_t           *best_beam_h;     /* [1]                                 */

    /* ── Aerial / cuPHY handles ── */
    cuphyController_t  cuphy_ctrl;
    aerial_cell_t      aerial_cell;
    uint32_t           cell_id;

    /* ── state ── */
    uint64_t        slot_number;
    int             history_write_idx;
    uint8_t         current_serving_beam;
    float           current_rsrp_dBm;
    bool            ai_ready;

    /* ── AI context (TensorRT or Triton client) ── */
    void           *ai_ctx;

    /* ── metrics ── */
    uint64_t        total_slots_processed;
    uint64_t        ai_overrides;
    uint64_t        beam_switches;
};

/* ════════════════════════════════════════════════════════════════════════════
 * CSI-RS reference sequence generation (Gold sequence per 3GPP TS 38.211)
 * ════════════════════════════════════════════════════════════════════════════ */

/**
 * goldSeqBit — one bit of a 31-bit maximum-length Gold sequence.
 * x1 seed = 1<<0, x2 seed from scrambling_id and nSlot.
 */
static uint32_t x1_state, x2_state;

static void goldSeqInit(uint32_t scrambling_id, uint32_t n_slot, uint32_t port)
{
    /* TS 38.211 §7.4.1.5.2 : cinit = 2^10·(14·n_slot + l + 1)·(2·Nid+1) + 2·Nid + port */
    uint32_t cinit = ((uint32_t)(14 * n_slot + 1) * (2 * scrambling_id + 1) * (1u << 10))
                     + 2 * scrambling_id + port;
    x1_state = 1u;
    x2_state = cinit;
    /* Advance 1600 steps */
    for (int i = 0; i < 1600; i++) {
        uint32_t b1 = ((x1_state >> 3) ^ x1_state) & 1u;
        uint32_t b2 = ((x2_state >> 3) ^ (x2_state >> 2) ^ (x2_state >> 1) ^ x2_state) & 1u;
        x1_state = (x1_state >> 1) | (b1 << 30);
        x2_state = (x2_state >> 1) | (b2 << 30);
    }
}

static inline float goldSeqNextBPSK(void)
{
    uint32_t b1 = ((x1_state >> 3) ^ x1_state) & 1u;
    uint32_t b2 = ((x2_state >> 3) ^ (x2_state >> 2) ^ (x2_state >> 1) ^ x2_state) & 1u;
    x1_state = (x1_state >> 1) | (b1 << 30);
    x2_state = (x2_state >> 1) | (b2 << 30);
    uint32_t c = (x1_state ^ x2_state) & 1u;
    return 1.0f - 2.0f * (float)c;   /* BPSK: +1 / -1 */
}

/**
 * generateCsiRsRefSequence
 *
 * Fills host array ref_h[beam][sc] with the BPSK CSI-RS reference sequence
 * for each configured beam (one resource per beam assumption).
 * The sequence is then uploaded once to device memory at init time.
 */
static void generateCsiRsRefSequence(
    cuComplex            *ref_h,       /* [nBeams × nSC] host output          */
    const BeamSweepCfg_t *cfg,
    uint32_t              n_slot
) {
    for (int b = 0; b < cfg->num_beams; b++) {
        const CsiRsResourceCfg_t *res = &cfg->csirs_cfg[b];
        uint32_t sc_count = res->freq_band_num_rb * 12u;

        goldSeqInit(res->scrambling_id, n_slot, (uint32_t)b);

        for (uint32_t sc = 0; sc < sc_count; sc++) {
            float re = goldSeqNextBPSK() * (float)M_SQRT1_2;
            float im = goldSeqNextBPSK() * (float)M_SQRT1_2;
            uint32_t global_sc = res->freq_band_start_rb * 12u + sc;
            if (global_sc < (uint32_t)(cfg->csirs_cfg[0].freq_band_num_rb * 12u)) {
                ref_h[b * sc_count + sc].x = re;
                ref_h[b * sc_count + sc].y = im;
            }
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * Aerial / cuPHY slot callback
 * ════════════════════════════════════════════════════════════════════════════ */

/**
 * aerialSlotIndication — registered with aerial_register_slot_cb().
 *
 * Called by the Aerial L1 scheduler at the start of each DL slot.
 * The rx_buf_d pointer is a CUDA device pointer into the Aerial RX buffer
 * ring — it remains valid until the next slot indication for this cell.
 *
 * @param cell_id     serving cell ID
 * @param slot_num    absolute NR slot counter
 * @param rx_buf_d    device pointer: CSI-RS RX IQ  [nBeams × nSC × nSym]
 * @param noise_d     device pointer: noise variance [nSC]
 * @param user_data   our BeamSelContext_t *
 */
static void aerialSlotIndication(
    uint32_t     cell_id,
    uint64_t     slot_num,
    void        *rx_buf_d,
    void        *noise_d,
    void        *user_data
) {
    BeamSelContext_t *ctx = (BeamSelContext_t *)user_data;
    if (!ctx || cell_id != ctx->cell_id) return;

    ctx->slot_number         = slot_num;
    ctx->history_write_idx   = (int)(slot_num % BEAM_SEL_HISTORY_DEPTH);

    /* ── Launch RSRP computation pipeline ── */
    BeamSelStatus_t st = launchComputeRSRP(
        (const cuComplex *)rx_buf_d,
        ctx->csirs_ref_d,
        (const float *)noise_d,
        ctx->rsrp_accum_d,
        ctx->measurements_d,
        ctx->best_beam_d,
        ctx->history_buf_d,
        ctx->feature_vec_d,
        ctx->sweep_cfg.num_beams,
        ctx->sweep_cfg.csirs_cfg[0].freq_band_num_rb * 12,  /* nSC */
        2,                                                    /* nSym (typical CSI-RS) */
        ctx->sweep_cfg.rsrp_floor_dBm,
        ctx->sweep_cfg.hysteresis_dB,
        ctx->current_rsrp_dBm,
        slot_num,
        ctx->history_write_idx,
        ctx->stream
    );

    if (st != BEAM_SEL_OK) {
        fprintf(stderr, "[BeamSel] RSRP kernel error slot=%llu: %s\n",
                (unsigned long long)slot_num, beamSelStatusStr(st));
        return;
    }

    /* ── AI inference (every N slots) ── */
    uint8_t selected_beam = 0;
    bool    used_ai       = false;

    if (ctx->ai_ready &&
        ctx->ai_cfg.mode != AI_MODE_DISABLED &&
        (slot_num % ctx->ai_cfg.infer_interval_slots) == 0)
    {
        float   confidence = 0.0f;
        uint8_t ai_beam    = 0;

        BeamSelStatus_t ai_st = aiInferBestBeam(
            ctx->ai_ctx,
            ctx->feature_vec_d,
            ctx->sweep_cfg.num_beams,
            &ai_beam,
            &confidence,
            ctx->stream
        );

        if (ai_st == BEAM_SEL_OK) {
            if (ctx->ai_cfg.mode == AI_MODE_PREDICT) {
                selected_beam = ai_beam;
                used_ai       = true;
            } else if (ctx->ai_cfg.mode == AI_MODE_HYBRID &&
                       confidence >= ctx->ai_cfg.confidence_thresh) {
                selected_beam = ai_beam;
                used_ai       = true;
            }
        }
    }

    /* If AI did not override, read GPU-selected beam from device */
    if (!used_ai) {
        cudaMemcpyAsync(ctx->best_beam_h, ctx->best_beam_d,
                        sizeof(uint8_t), cudaMemcpyDeviceToHost, ctx->stream);
        cudaStreamSynchronize(ctx->stream);   /* needed only for this byte    */
        selected_beam = *ctx->best_beam_h;
    } else {
        ctx->ai_overrides++;
    }

    /* ── Apply selected beam to Aerial TX beamforming weights ── */
    cuphyBfWeightUpdate_t bf_update = {
        .cell_id   = ctx->cell_id,
        .beam_id   = selected_beam,
        .slot_num  = (uint32_t)(slot_num & 0xFFFFFFFFu),
    };

    /*
     * cuphyBfWeightApply — Aerial cuPHY API to update beamforming weights.
     * This writes the precomputed codebook entry for beam 'selected_beam'
     * into the DL TX chain for the next transmit slot.
     *
     * cuphyBfWeightApply(ctx->cuphy_ctrl, &bf_update);
     *
     * NOTE: Commented out here because the exact API signature varies
     * between Aerial SDK minor versions. See aerial_phy.h for your version.
     * Typical signature:
     *   cuphyStatus_t cuphyBfWeightApply(cuphyController_t ctrl,
     *                                    const cuphyBfWeightUpdate_t *update);
     */
    (void)bf_update;  /* suppress unused warning when call is commented out */

    /* ── Track beam switches ── */
    if (selected_beam != ctx->current_serving_beam) {
        ctx->beam_switches++;
        ctx->current_serving_beam = selected_beam;
    }

    ctx->total_slots_processed++;

    /* ── Async copy measurements to host (non-blocking; use event to gate) ── */
    cudaMemcpyAsync(ctx->measurements_h,
                    ctx->measurements_d,
                    ctx->sweep_cfg.num_beams * sizeof(BeamMeasurement_t),
                    cudaMemcpyDeviceToHost,
                    ctx->stream);
    cudaEventRecord(ctx->kernel_done_event, ctx->stream);
}

/* ════════════════════════════════════════════════════════════════════════════
 * Public API implementation
 * ════════════════════════════════════════════════════════════════════════════ */

BeamSelStatus_t beamSelInit(
    BeamSelContext_t      **ctx_out,
    const BeamSweepCfg_t  *sweep_cfg,
    const AiCfg_t         *ai_cfg,
    cudaStream_t           stream
) {
    if (!ctx_out || !sweep_cfg || !ai_cfg) return BEAM_SEL_ERR_INVALID_PARAM;
    if (sweep_cfg->num_beams == 0 ||
        sweep_cfg->num_beams > BEAM_SEL_MAX_BEAMS) return BEAM_SEL_ERR_INVALID_PARAM;

    BeamSelContext_t *ctx = (BeamSelContext_t *)calloc(1, sizeof(BeamSelContext_t));
    if (!ctx) return BEAM_SEL_ERR_ALLOC;

    memcpy(&ctx->sweep_cfg, sweep_cfg, sizeof(BeamSweepCfg_t));
    memcpy(&ctx->ai_cfg,    ai_cfg,    sizeof(AiCfg_t));

    ctx->stream                = stream ? stream : 0;
    ctx->current_rsrp_dBm      = -INFINITY;
    ctx->current_serving_beam  = 0;

    const int nBeams = sweep_cfg->num_beams;
    const int nSC    = sweep_cfg->csirs_cfg[0].freq_band_num_rb * 12;

    /* ── CUDA event ── */
    if (cudaEventCreate(&ctx->kernel_done_event) != cudaSuccess) goto err;

    /* ── Device allocations ── */
    if (cudaMalloc(&ctx->csirs_ref_d,    nBeams * nSC * sizeof(cuComplex))     != cudaSuccess) goto err;
    if (cudaMalloc(&ctx->rsrp_accum_d,   nBeams * sizeof(float))               != cudaSuccess) goto err;
    if (cudaMalloc(&ctx->measurements_d, nBeams * sizeof(BeamMeasurement_t))   != cudaSuccess) goto err;
    if (cudaMalloc(&ctx->best_beam_d,    sizeof(uint8_t))                      != cudaSuccess) goto err;
    if (cudaMalloc(&ctx->noise_var_d,    nSC    * sizeof(float))               != cudaSuccess) goto err;
    if (cudaMalloc(&ctx->history_buf_d,
                   BEAM_SEL_HISTORY_DEPTH * nBeams * sizeof(float))            != cudaSuccess) goto err;
    if (cudaMalloc(&ctx->feature_vec_d,
                   BEAM_SEL_AI_FEATURE_DIM * sizeof(float))                    != cudaSuccess) goto err;

    /* ── Pinned host allocations ── */
    if (cudaMallocHost(&ctx->measurements_h, nBeams * sizeof(BeamMeasurement_t)) != cudaSuccess) goto err;
    if (cudaMallocHost(&ctx->best_beam_h,    sizeof(uint8_t))                    != cudaSuccess) goto err;

    /* ── Initialise device buffers to zero ── */
    cudaMemset(ctx->history_buf_d,   0, BEAM_SEL_HISTORY_DEPTH * nBeams * sizeof(float));
    cudaMemset(ctx->rsrp_accum_d,    0, nBeams * sizeof(float));

    /* ── Generate CSI-RS reference sequences (slot 0) and upload ── */
    {
        cuComplex *ref_h = (cuComplex *)malloc(nBeams * nSC * sizeof(cuComplex));
        if (!ref_h) goto err;
        generateCsiRsRefSequence(ref_h, sweep_cfg, 0);
        cudaMemcpy(ctx->csirs_ref_d, ref_h, nBeams * nSC * sizeof(cuComplex),
                   cudaMemcpyHostToDevice);
        free(ref_h);
    }

    /* ── Initialise noise floor to a reasonable default (-100 dBm/SC) ── */
    {
        float *noise_h = (float *)malloc(nSC * sizeof(float));
        if (!noise_h) goto err;
        for (int i = 0; i < nSC; i++) noise_h[i] = 1e-13f;  /* ~-100 dBm */
        cudaMemcpy(ctx->noise_var_d, noise_h, nSC * sizeof(float),
                   cudaMemcpyHostToDevice);
        free(noise_h);
    }

    /*
     * ── Register with Aerial cuPHY ──
     *
     * aerial_register_slot_cb() — Aerial gNB L1 API.
     * Registers aerialSlotIndication as the per-slot callback that
     * receives the device-resident RX buffer pointer after cuPHY
     * completes CSI-RS demapping for the cell.
     *
     * Actual call (uncomment and adapt to your Aerial SDK version):
     *
     *   aerial_slot_cb_params_t cb_params = {
     *       .cell_id    = ctx->cell_id,
     *       .cb         = aerialSlotIndication,
     *       .user_data  = ctx,
     *       .buf_type   = AERIAL_BUF_CSIRS_IQ,
     *   };
     *   aerial_status_t aer_st = aerial_register_slot_cb(ctx->aerial_cell, &cb_params);
     *   if (aer_st != AERIAL_OK) goto err;
     */

    /* ── Initialise AI context ── */
    if (ai_cfg->mode != AI_MODE_DISABLED) {
        extern BeamSelStatus_t aiInit(void **ai_ctx_out, const AiCfg_t *cfg,
                                      int nBeams, cudaStream_t stream);
        BeamSelStatus_t ai_st = aiInit(&ctx->ai_ctx, ai_cfg, nBeams, ctx->stream);
        ctx->ai_ready = (ai_st == BEAM_SEL_OK);
        if (!ctx->ai_ready)
            fprintf(stderr, "[BeamSel] AI init failed (%s); falling back to RSRP-only\n",
                    beamSelStatusStr(ai_st));
    }

    *ctx_out = ctx;
    return BEAM_SEL_OK;

err:
    beamSelDestroy(ctx);
    return BEAM_SEL_ERR_ALLOC;
}


BeamSelStatus_t beamSelProcessSlot(
    BeamSelContext_t  *ctx,
    const cuComplex   *rx_iq_d,
    const float       *noise_var_d,
    uint64_t           slot_number,
    uint8_t           *best_beam_id,
    BeamMeasurement_t  measurements_out[]
) {
    if (!ctx || !rx_iq_d || !best_beam_id) return BEAM_SEL_ERR_INVALID_PARAM;

    /* Optionally update noise variance (caller may pass NULL to keep previous) */
    if (noise_var_d) {
        const int nSC = ctx->sweep_cfg.csirs_cfg[0].freq_band_num_rb * 12;
        cudaMemcpyAsync(ctx->noise_var_d, noise_var_d,
                        nSC * sizeof(float), cudaMemcpyDeviceToDevice, ctx->stream);
    }

    /* Directly drive the slot indication logic */
    aerialSlotIndication(ctx->cell_id, slot_number,
                         (void *)rx_iq_d, ctx->noise_var_d, ctx);

    /* Wait for kernel completion + async host copy */
    cudaEventSynchronize(ctx->kernel_done_event);

    *best_beam_id = ctx->current_serving_beam;

    if (measurements_out) {
        memcpy(measurements_out, ctx->measurements_h,
               ctx->sweep_cfg.num_beams * sizeof(BeamMeasurement_t));
    }

    return BEAM_SEL_OK;
}


BeamSelStatus_t beamSelGetLastMeasurements(
    BeamSelContext_t  *ctx,
    BeamMeasurement_t  out[],
    uint8_t           *num_valid
) {
    if (!ctx || !out) return BEAM_SEL_ERR_INVALID_PARAM;
    cudaEventSynchronize(ctx->kernel_done_event);
    memcpy(out, ctx->measurements_h,
           ctx->sweep_cfg.num_beams * sizeof(BeamMeasurement_t));
    if (num_valid) {
        uint8_t n = 0;
        for (int i = 0; i < ctx->sweep_cfg.num_beams; i++)
            if (out[i].is_valid) n++;
        *num_valid = n;
    }
    return BEAM_SEL_OK;
}


void beamSelDestroy(BeamSelContext_t *ctx)
{
    if (!ctx) return;

    if (ctx->csirs_ref_d)    cudaFree(ctx->csirs_ref_d);
    if (ctx->rsrp_accum_d)   cudaFree(ctx->rsrp_accum_d);
    if (ctx->measurements_d) cudaFree(ctx->measurements_d);
    if (ctx->best_beam_d)    cudaFree(ctx->best_beam_d);
    if (ctx->noise_var_d)    cudaFree(ctx->noise_var_d);
    if (ctx->history_buf_d)  cudaFree(ctx->history_buf_d);
    if (ctx->feature_vec_d)  cudaFree(ctx->feature_vec_d);

    if (ctx->measurements_h) cudaFreeHost(ctx->measurements_h);
    if (ctx->best_beam_h)    cudaFreeHost(ctx->best_beam_h);

    if (ctx->kernel_done_event) cudaEventDestroy(ctx->kernel_done_event);

    if (ctx->ai_ctx) {
        extern void aiDestroy(void *ai_ctx);
        aiDestroy(ctx->ai_ctx);
    }

    fprintf(stderr,
            "[BeamSel] Stats: slots=%llu  beam_switches=%llu  ai_overrides=%llu\n",
            (unsigned long long)ctx->total_slots_processed,
            (unsigned long long)ctx->beam_switches,
            (unsigned long long)ctx->ai_overrides);

    free(ctx);
}


const char *beamSelStatusStr(BeamSelStatus_t s)
{
    switch (s) {
    case BEAM_SEL_OK:                return "OK";
    case BEAM_SEL_ERR_INVALID_PARAM: return "INVALID_PARAM";
    case BEAM_SEL_ERR_CUDA:          return "CUDA_ERROR";
    case BEAM_SEL_ERR_AERIAL:        return "AERIAL_ERROR";
    case BEAM_SEL_ERR_TRT:           return "TENSORRT_ERROR";
    case BEAM_SEL_ERR_NO_VALID_BEAM: return "NO_VALID_BEAM";
    case BEAM_SEL_ERR_ALLOC:         return "ALLOC_ERROR";
    default:                         return "UNKNOWN";
    }
}
