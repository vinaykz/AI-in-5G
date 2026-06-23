# Beam Selection System Design
## NVIDIA Aerial + L4 GPU · RSRP-Based with AI Extension
### MAHA 6G Program — Challenge 1: AI/ML-Native RAN

---

## 1. Overview

This document describes a complete beam selection implementation for a 5G NR / 6G gNB running on **NVIDIA Aerial SDK** with an **NVIDIA L4 GPU** (Ada Lovelace, sm_89). The design covers:

- RSRP computation using custom CUDA kernels
- Integration with Aerial's cuPHY CSI-RS pipeline
- Beam selection with hysteresis (3GPP A3-equivalent)
- TensorRT-based AI beam prediction (on-device)
- Optional external AI via NVIDIA Triton Inference Server

---

## 2. Platform Characteristics

| Component | Details |
|---|---|
| GPU | NVIDIA L4 · Ada Lovelace · sm_89 |
| CUDA Cores | 7,680 |
| Tensor Cores | 4th Gen (INT8/FP16/TF32/FP8) |
| VRAM | 24 GB GDDR6 @ 300 GB/s |
| L2 Cache | 48 MB |
| TDP | 72W (low-power, datacenter rack) |
| SDK | NVIDIA Aerial cuPHY + CUDA 12.x + TensorRT 10.x |

The L4's 4th-gen Tensor Cores deliver **120 TOPS INT8** and **60 TFLOPS FP16**, sufficient to run the RSRP pipeline and a beam prediction model simultaneously alongside cuPHY's L1 processing — all on a single GPU.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          NVIDIA L4 GPU (sm_89)                              │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    NVIDIA Aerial cuPHY                               │   │
│  │  RF Frontend → ADC → [OFDM Demod] → [CSI-RS Demapping] → IQ buf     │   │
│  │                                           │ device ptr (rx_iq_d)     │   │
│  └───────────────────────────────────────────┼──────────────────────────┘   │
│                                              │ slot indication callback      │
│  ┌───────────────────────────────────────────▼──────────────────────────┐   │
│  │               Beam Selection Pipeline (CUDA stream)                  │   │
│  │                                                                      │   │
│  │  Kernel 1: computeBeamRSRP                                           │   │
│  │    Grid(nBeams, nSC_tiles) × Block(256)                              │   │
│  │    LS channel estimate: ĥ = y·ref* / |ref|²                         │   │
│  │    Shared-mem reduction → rsrp_accum_d[nBeams]                      │   │
│  │                                                                      │   │
│  │  Kernel 2: finalizeRSRP                                              │   │
│  │    → rsrp_dBm, rsrq_dB, sinr_dB per beam                            │   │
│  │    → validity flag (rsrp_dBm ≥ floor)                               │   │
│  │                                                                      │   │
│  │  Kernel 3: selectBestBeam                                            │   │
│  │    Warp-level argmax + hysteresis                                    │   │
│  │    → best_beam_d (uint8)                                             │   │
│  │                                                                      │   │
│  │  Kernel 4: updateHistoryBuffer                                       │   │
│  │    Circular buffer: history_buf_d[DEPTH=32 × nBeams]                │   │
│  │                                                                      │   │
│  │  Kernel 5: buildAIFeatureVector                                      │   │
│  │    → feature_vec_d[4 × nBeams]                                       │   │
│  │      [rsrp_now, rsrq_now, slope, std_dev] per beam                  │   │
│  └────────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                              │
│       ┌───────────────────────▼───────────────────────────┐                 │
│       │           AI Inference (every N slots)             │                 │
│       │                                                     │                │
│       │  ┌──────────────────────┐  ┌─────────────────────┐ │                │
│       │  │  LOCAL TensorRT      │  │  EXTERNAL Triton     │ │                │
│       │  │  engine (.trt/.onnx) │  │  gRPC to AI server   │ │                │
│       │  │  INT8 on Tensor Cores│  │  (foundation model)  │ │                │
│       │  │  ~0.15 ms latency    │  │  ~1–5 ms RTT         │ │                │
│       │  └──────────┬───────────┘  └──────────┬──────────┘ │                │
│       │             └──────────────────────────┘            │                │
│       │                    beam_probs[nBeams]                │                │
│       │                    confidence score                  │                │
│       └───────────────────────┬───────────────────────────┘                 │
│                               │                                              │
│  ┌────────────────────────────▼──────────────────────────────────────────┐  │
│  │          Beam Decision + cuphyBfWeightApply()                         │  │
│  │  AI_MODE_PREDICT : use AI beam always                                 │  │
│  │  AI_MODE_HYBRID  : use AI beam if confidence ≥ threshold              │  │
│  │  AI_MODE_DISABLED: use RSRP argmax (Kernel 3 output)                  │  │
│  │  → updates Aerial TX beamforming weights for next slot                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Key SDK / API References

### 4.1 NVIDIA Aerial cuPHY APIs

| API | Purpose |
|---|---|
| `cuphyController_t` | Main cuPHY controller handle |
| `aerial_register_slot_cb()` | Register per-slot IQ buffer callback |
| `cuphyBfWeightApply()` | Update TX beamforming codebook entry |
| `cuphyCsiRsMeasCreate()` | Optional: use cuPHY's built-in CSI-RS measurement (lower flexibility than custom kernels) |
| `cuphyCsiRsMeasRun()` | Run built-in CSI-RS measurement |
| `cuphyCsiRsMeasGetResults()` | Retrieve measurement results |
| `aerial_cell_t` | Cell configuration handle |
| `cuphyBfWeightUpdate_t` | BF weight update descriptor |

> **Note:** When using the custom RSRP kernels (recommended for research), bypass `cuphyCsiRsMeasRun()` and work directly with the raw IQ device pointer provided in the slot callback. This gives full flexibility for experimental channel estimators.

### 4.2 CUDA APIs Used

| API | Where Used |
|---|---|
| `cudaMalloc / cudaFree` | Device buffer lifecycle |
| `cudaMallocHost / cudaFreeHost` | Pinned host buffers (zero-copy D2H) |
| `cudaMemsetAsync` | Zero accumulator per slot |
| `cudaMemcpyAsync` | Non-blocking D2H copy of measurements |
| `cudaEventRecord / cudaEventSynchronize` | Gate host reads on kernel completion |
| `atomicAdd` | Lock-free RSRP per-beam accumulation |
| `__frcp_rn` | Fast hardware reciprocal (L4 SFU) |
| `cooperative_groups::tiled_partition<32>` | Warp-level argmax in selectBestBeam |
| `warp.shfl_down` | Warp shuffle for reduction without shared memory |

### 4.3 TensorRT 10 APIs

| API | Purpose |
|---|---|
| `createInferBuilder()` | Build TRT engine from ONNX |
| `INetworkDefinition::createNetworkV2()` | Explicit-batch network creation |
| `nvonnxparser::createParser()` | Parse ONNX model graph |
| `IBuilderConfig::setFlag(kINT8)` | Enable INT8 on L4 Tensor Cores |
| `IBuilderConfig::setFlag(kFP16)` | Enable FP16 fallback |
| `builder->buildSerializedNetwork()` | Compile to .trt engine |
| `IRuntime::deserializeCudaEngine()` | Load cached engine |
| `IExecutionContext::setTensorAddress()` | Named I/O binding (TRT 10 API) |
| `IExecutionContext::setInputShape()` | Set dynamic batch/sequence shape |
| `IExecutionContext::enqueueV3()` | Async inference on CUDA stream |

---

## 5. CUDA Kernel Design

### Kernel 1 — `computeBeamRSRP`

**Grid:** `(nBeams, ceil(nSC / 256))`  
**Block:** `(256)` threads  
**Shared memory:** `256 × sizeof(float)` = 1 KB per block

Each block handles one beam across a 256-subcarrier tile. The LS channel estimate per RE is:

```
ĥ(k,l) = y(k,l) · ref*(k) / |ref(k)|²
RSRP   += |ĥ|² / N_RE
```

`atomicAdd` accumulates partial sums across tiles into `rsrp_accum_d[beam]`. The fast hardware reciprocal `__frcp_rn` is used instead of division (L4 SFU, 1 cycle).

**L4 occupancy estimate:**  
- Registers: ~32 per thread  
- Shared memory: 1 KB per block  
- Theoretical occupancy: ~75% (limited by register file)  
- Effective memory bandwidth utilisation: ~70% of 300 GB/s peak

### Kernel 3 — `selectBestBeam`

Uses `cooperative_groups::tiled_partition<32>` for a warp-level parallel argmax. Avoids shared memory entirely. Hysteresis is applied as an RSRP bias to the current serving beam before comparison — equivalent to 3GPP A3 event offset.

### Kernel 5 — `buildAIFeatureVector`

Computes four features per beam from the `BEAM_SEL_HISTORY_DEPTH`-slot circular buffer:

| Feature | Description |
|---|---|
| `rsrp_now` | Current RSRP (dBm) |
| `rsrq_now` | Current RSRQ (dB) |
| `slope` | (rsrp_now − rsrp_oldest) / depth (dBm/slot) — trend direction |
| `std_dev` | RSRP standard deviation over history — stability indicator |

This feature vector is the input to the AI model.

---

## 6. 3GPP Beam Management Alignment

| 3GPP Procedure | Implementation |
|---|---|
| **P1** (initial beam sweep — SSB) | `num_beams` up to 64, RSRP computed for all simultaneously in one kernel dispatch |
| **P2** (beam refinement — CSI-RS) | `p2_refinement_beams`: top-N from P1 used for finer CSI-RS resources |
| **P3** (beam tracking — CSI-RS/SRS) | History buffer + AI slope feature enable predictive tracking |
| **A3 event** | Hysteresis parameter in `selectBestBeam` kernel |
| **Beam failure recovery** | `rsrp_floor_dBm` marks beams invalid; fallback to last valid beam |

---

## 7. AI Integration

### 7.1 Recommended Model Architecture

```
Input: [1, 4, nBeams]  — (batch, features, beams)
     rsrp_dBm, rsrq_dB, slope, std_dev per beam

Conv1d(4→32, k=1, ReLU)         — per-beam feature mixing (lightweight)
Conv1d(32→64, k=3, pad=1, ReLU) — cross-beam context window (k=3 captures adjacent beams)
GlobalAveragePool                — [1, 64]
Linear(64 → nBeams)             — beam logits
Softmax                          — beam probability distribution [nBeams]

Parameters: ~18,000  (negligible footprint, ~72 KB FP32)
Latency (L4 INT8): ~0.15 ms
```

**Training data:** Collect `(feature_vec, next_best_beam)` tuples from RSRP-only mode during a burn-in period (e.g., 1000 slots). Label = beam with highest RSRP two slots ahead (predictive label).

**Export to ONNX:**
```python
torch.onnx.export(model, dummy_input, "beam_predict.onnx",
    input_names=["features"], output_names=["beam_probs"],
    dynamic_axes={"features": {2: "nBeams"}},
    opset_version=17)
```

### 7.2 Operating Modes

| Mode | When to Use |
|---|---|
| `AI_MODE_DISABLED` | Baseline RSRP greedy; use during initial deployment / debugging |
| `AI_MODE_PREDICT` | Production: AI always selects beam; best for high-mobility UEs (vehicles, drones — MAHA 6G Challenge 2 overlap) |
| `AI_MODE_HYBRID` | Recommended default: AI overrides only when `confidence ≥ threshold` (e.g., 0.7). Falls back to RSRP otherwise |
| `AI_MODE_EXTERNAL` | Large foundation model (Ericsson AIM, Nokia MX-One, custom LLM-RAN) hosted on a separate AI cluster via Triton |

### 7.3 External AI Infrastructure (AI_MODE_EXTERNAL)

```
gNB L1 (L4 GPU)                  AI Inference Cluster
┌──────────────────┐              ┌──────────────────────────────┐
│ feature_vec_d    │──gRPC/HTTP──▶│  Triton Inference Server     │
│ [4 × nBeams]     │              │  Model: beam_predict_v2      │
│                  │◀─────────────│  Backend: TensorRT / PyTorch  │
│ beam_probs[nBeams]│             │  GPU: A100 / H100 (training) │
└──────────────────┘              └──────────────────────────────┘
     ~0.1 ms D2H                        ~1–5 ms round-trip
```

The Triton server can host a larger model (e.g., transformer-based sequence model over 32-slot history) that would be too costly to run on the L4 alongside cuPHY. Use `AI_MODE_HYBRID` with a conservative confidence threshold so that the gRPC RTT does not impact beam switch latency on the critical path.

---

## 8. Data Flow Summary (per slot, ~0.5 ms budget)

```
t=0.0 ms  Aerial cuPHY slot indication fires
           → aerialSlotIndication() callback called
           → rx_iq_d (device ptr) available

t=0.05 ms cudaMemsetAsync: zero rsrp_accum_d
t=0.10 ms computeBeamRSRP kernel: ~0.20 ms for 64 beams × 3276 SC
t=0.30 ms finalizeRSRP kernel: < 0.02 ms
t=0.32 ms selectBestBeam kernel: < 0.01 ms
t=0.33 ms updateHistoryBuffer: < 0.01 ms
t=0.34 ms buildAIFeatureVector: < 0.02 ms

t=0.36 ms [if AI slot] aiInferBestBeam (TRT): ~0.15 ms
t=0.51 ms beam decision + cuphyBfWeightApply()
t=0.52 ms cudaMemcpyAsync measurements → host (non-blocking)
           cudaEventRecord kernel_done_event

t=1.0 ms  Next slot — full pipeline headroom maintained
```

Total GPU time: **~0.5 ms** out of a 1 ms NR slot budget, leaving ample margin for cuPHY DL/UL processing on the same GPU.

---

## 9. File Structure

| File | Description |
|---|---|
| `beam_selection.h` | Public API, all structs, enums, constants |
| `beam_selection_cuda.cu` | 5 CUDA kernels + `launchComputeRSRP()` launcher |
| `beam_manager.c` | Aerial cuPHY slot callback, state machine, `beamSelInit/ProcessSlot/Destroy` |
| `beam_selection_ai.cpp` | TensorRT engine load/build, `aiInferBestBeam()`, Triton client stub |

### Build

```makefile
# Makefile excerpt
CUDA_ARCH := -arch=sm_89
CFLAGS    := -O3 -march=native -fPIC
NVCCFLAGS := $(CUDA_ARCH) --use_fast_math -lineinfo -O3

AERIAL_INC := -I$(AERIAL_SDK)/include
TRT_INC    := -I$(TRT_ROOT)/include
ONNX_INC   := -I$(ONNX_PARSER_ROOT)/include

LIBS := -lcuda -lcudart -lnvinfer -lnvonnxparser \
        -L$(AERIAL_SDK)/lib -lcuphy -laerial_phy

beam_selection.a: beam_selection_cuda.o beam_manager.o beam_selection_ai.o
    ar rcs $@ $^

beam_selection_cuda.o: beam_selection_cuda.cu beam_selection.h
    nvcc $(NVCCFLAGS) $(AERIAL_INC) -c $< -o $@

beam_manager.o: beam_manager.c beam_selection.h
    gcc $(CFLAGS) $(AERIAL_INC) -c $< -o $@

beam_selection_ai.o: beam_selection_ai.cpp beam_selection.h
    g++ $(CFLAGS) $(TRT_INC) $(ONNX_INC) -c $< -o $@
```

---

## 10. AI Recommendations Summary

| Recommendation | Rationale |
|---|---|
| Start with `AI_MODE_DISABLED` baseline | Validate RSRP pipeline correctness first; collect training data |
| Use `AI_MODE_HYBRID` in production | Avoids AI-induced beam thrash on model uncertainty |
| INT8 quantisation on L4 | 2× throughput vs FP16; model is small enough that INT8 accuracy loss is negligible |
| Feature: RSRP slope | Enables predictive selection for high-mobility UEs (drones, vehicles — MAHA 6G Challenge 2) |
| Online learning | After deployment, fine-tune model with operator data (federated if multi-cell) — aligns with MAHA 6G Challenge 4 (Operator Intelligence) |
| Triton for large models | Offload transformer/LLM-based beam models to A100/H100 cluster; L4 handles latency-critical RSRP path locally |
| History depth = 32 slots | Covers ~32 ms at SCS=30 kHz — sufficient for human-speed mobility; increase to 64 for vehicular |

---

## 11. Integration with MAHA 6G Challenges

| MAHA 6G Challenge | This Design's Contribution |
|---|---|
| **Challenge 1** (AI-native RAN) | AI beam prediction model (on-device TRT + external Triton), feature extraction from RSRP history, hybrid AI/RSRP decision |
| **Challenge 2** (ISAC) | RSRP pipeline extendable to ISAC sensing KPIs (detection, localisation) — same CSI-RS IQ buffer used for both |
| **Challenge 3** (Green 6G) | Hysteresis prevents unnecessary beam switches (saves TX power); INT8 inference minimises AI energy cost |
| **Challenge 4** (Operator Intelligence) | `measurements_h` log fed to operator data pipeline; online model adaptation via Triton update API |
| **Challenge 5** (NTN) | Large Doppler / delay shifts in NTN change RSRP dynamics; slope feature in AI model captures fast fading characteristic of LEO |
