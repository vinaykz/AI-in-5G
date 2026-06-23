/**
 * beam_selection.h
 * ============================================================================
 * Beam Selection System — NVIDIA Aerial + L4 GPU
 * RSRP-based beam selection with AI/TensorRT predictive extension
 *
 * Platform  : NVIDIA L4 (Ada Lovelace, 24 GB GDDR6, 4th-gen Tensor Cores)
 * SDK       : NVIDIA Aerial cuPHY, CUDA 12.x, TensorRT 10.x
 * Standard  : 3GPP NR TS 38.213 / 38.215 — Beam Management P1/P2/P3
 * ============================================================================
 */

#ifndef BEAM_SELECTION_H
#define BEAM_SELECTION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

/* ─────────────────────────── compile-time limits ────────────────────────── */
#define BEAM_SEL_MAX_BEAMS          64      /* NR FR2 max SSB beams            */
#define BEAM_SEL_MAX_SUBCARRIERS    3276    /* 273 RBs × 12 SC                 */
#define BEAM_SEL_MAX_CSIRS_PORTS    32
#define BEAM_SEL_MAX_SYMBOLS        14      /* per slot                        */
#define BEAM_SEL_HISTORY_DEPTH      32      /* slots kept for AI feature vector */
#define BEAM_SEL_AI_FEATURE_DIM     (BEAM_SEL_MAX_BEAMS * 4)   /* rsrp+rsrq+trend+pos */

/* ─────────────────────────── return codes ───────────────────────────────── */
typedef enum {
    BEAM_SEL_OK                  =  0,
    BEAM_SEL_ERR_INVALID_PARAM   = -1,
    BEAM_SEL_ERR_CUDA            = -2,
    BEAM_SEL_ERR_AERIAL          = -3,
    BEAM_SEL_ERR_TRT             = -4,
    BEAM_SEL_ERR_NO_VALID_BEAM   = -5,
    BEAM_SEL_ERR_ALLOC           = -6,
} BeamSelStatus_t;

/* ─────────────────────────── beam measurement ───────────────────────────── */
/**
 * Per-beam measurement result.
 * Populated after computeRSRP() completes on device; copied to host
 * on demand (not every slot — see AI path).
 */
typedef struct {
    uint8_t  beam_id;           /* 0 … BEAM_SEL_MAX_BEAMS-1                  */
    uint8_t  port_index;        /* TX antenna port                            */
    bool     is_valid;          /* false if below RSRP floor                  */
    float    rsrp_dBm;          /* Reference Signal Received Power (dBm)      */
    float    rsrq_dB;           /* Reference Signal Received Quality (dB)     */
    float    sinr_dB;           /* SINR estimate (dB)                         */
    float    rsrp_linear;       /* RSRP in linear scale (W) — GPU-internal    */
    uint64_t slot_number;       /* absolute NR slot counter                   */
} BeamMeasurement_t;

/* ─────────────────────────── CSI-RS resource config ────────────────────── */
/**
 * Mirrors 3GPP TS 38.211 Table 7.4.1.5.3-1 for one CSI-RS resource.
 */
typedef struct {
    uint8_t  num_ports;         /* 1,2,4,8,12,16,24,32                        */
    uint8_t  row;               /* CSI-RS table row                           */
    uint8_t  freq_domain_alloc; /* bitmap (up to 12 bits)                     */
    uint8_t  symbol_l0;         /* first OFDM symbol in slot                  */
    uint8_t  symbol_l1;         /* second symbol (if row requires it)         */
    uint8_t  cdm_type;          /* 0=noCDM, 1=fd-CDM2, 2=cdm4-FD2-TD2        */
    uint8_t  density;           /* 0.5, 1, or 3 (× 2 for fixed-point)        */
    uint16_t freq_band_start_rb;
    uint16_t freq_band_num_rb;
    uint32_t scrambling_id;
} CsiRsResourceCfg_t;

/* ─────────────────────────── beam sweeping config ───────────────────────── */
typedef struct {
    uint8_t           num_beams;
    uint8_t           num_csirs_resources;      /* one per beam typically     */
    CsiRsResourceCfg_t csirs_cfg[BEAM_SEL_MAX_BEAMS];
    float             rsrp_floor_dBm;           /* ignore beams below this    */
    float             hysteresis_dB;            /* beam switch hysteresis     */
    uint8_t           p2_refinement_beams;      /* top-N beams from P1 → P2  */
} BeamSweepCfg_t;

/* ─────────────────────────── AI model config ────────────────────────────── */
typedef enum {
    AI_MODE_DISABLED   = 0,   /* pure RSRP greedy selection                  */
    AI_MODE_PREDICT    = 1,   /* TensorRT model predicts next best beam       */
    AI_MODE_HYBRID     = 2,   /* AI overrides only when confidence > thresh   */
    AI_MODE_EXTERNAL   = 3,   /* forward features to external inference server*/
} AiMode_t;

typedef struct {
    AiMode_t   mode;
    char       engine_path[256];   /* path to serialised TRT engine (.trt)    */
    char       onnx_path[256];     /* fallback: rebuild engine from ONNX      */
    float      confidence_thresh;  /* AI_MODE_HYBRID: min softmax confidence  */
    uint32_t   infer_interval_slots; /* run AI every N slots                  */
    bool       enable_int8;        /* INT8 calibration for L4 Tensor Cores    */
    bool       enable_dla;         /* L4 has no DLA; set false                */
    /* external server (AI_MODE_EXTERNAL) */
    char       server_addr[64];    /* e.g. "192.168.1.10:8001"               */
    char       model_name[64];     /* Triton model repo name                  */
} AiCfg_t;

/* ─────────────────────────── top-level context ─────────────────────────── */
/**
 * Opaque handle created by beamSelInit().
 * All GPU memory and TRT objects live inside this structure.
 */
typedef struct BeamSelContext BeamSelContext_t;

/* ─────────────────────────── public API ────────────────────────────────── */

/**
 * beamSelInit — allocate GPU buffers, build TRT engine, register with cuPHY.
 *
 * @param ctx        [out] handle on success
 * @param sweep_cfg  beam sweep / CSI-RS configuration
 * @param ai_cfg     AI configuration (may have mode = AI_MODE_DISABLED)
 * @param cuda_stream CUDA stream to use for all async ops; pass 0 for default
 * @return BEAM_SEL_OK or error code
 */
BeamSelStatus_t beamSelInit(BeamSelContext_t      **ctx,
                             const BeamSweepCfg_t  *sweep_cfg,
                             const AiCfg_t         *ai_cfg,
                             cudaStream_t           cuda_stream);

/**
 * beamSelProcessSlot — main per-slot entry point.
 *
 * Called from the Aerial cuPHY slot-indication callback.
 * Performs:
 *   1. Extract CSI-RS IQ samples from cuPHY RX buffer (device pointer)
 *   2. Run RSRP CUDA kernels
 *   3. Optionally run AI inference
 *   4. Write selected beam index into *best_beam_id
 *
 * @param ctx            beam selection context
 * @param rx_iq_d        device pointer to RX IQ buffer (cuComplex, row-major)
 *                       dimensions: [num_beams][num_sc][num_symbols]
 * @param noise_var_d    device pointer to per-subcarrier noise variance [float]
 * @param slot_number    current NR slot number (for history buffer)
 * @param best_beam_id   [out] selected beam index (0 … num_beams-1)
 * @param measurements_h [out, optional] host-side measurements array
 *                       (NULL = skip host copy)
 * @return BEAM_SEL_OK or error code
 */
BeamSelStatus_t beamSelProcessSlot(BeamSelContext_t  *ctx,
                                    const cuComplex   *rx_iq_d,
                                    const float       *noise_var_d,
                                    uint64_t           slot_number,
                                    uint8_t           *best_beam_id,
                                    BeamMeasurement_t  measurements_h[]);

/**
 * beamSelGetLastMeasurements — copy latest measurement array to host.
 * Non-blocking: uses the context's CUDA stream + cudaMemcpyAsync.
 */
BeamSelStatus_t beamSelGetLastMeasurements(BeamSelContext_t  *ctx,
                                            BeamMeasurement_t  out[],
                                            uint8_t           *num_valid);

/**
 * beamSelDestroy — free all GPU memory, destroy TRT engine, deregister.
 */
void beamSelDestroy(BeamSelContext_t *ctx);

/* ─────────────────────────── utility ───────────────────────────────────── */
const char *beamSelStatusStr(BeamSelStatus_t status);

#ifdef __cplusplus
}
#endif
#endif /* BEAM_SELECTION_H */
