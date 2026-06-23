# Beam Selection Architecture
## NVIDIA Aerial + L4 GPU · RSRP-Based with AI Extension

---

## Colour Legend

| Colour | Meaning |
|--------|---------|
| 🟢 Green | NVIDIA / SDK (pre-built) |
| 🔵 Blue | Your code (to develop) |
| 🟣 Purple | External AI infrastructure |
| 🟡 Amber | Shared / you train separately |

---

## Platform

**NVIDIA L4 GPU · sm_89 · 24 GB GDDR6**

---

## Layer 1 — RF Ingestion & Aerial cuPHY

| Component | Owner | Details |
|-----------|-------|---------|
| RF Frontend / ADC / OFDM demod | 🟢 NVIDIA | Via Aerial RU/DU |
| cuPHY Controller | 🟢 NVIDIA | `cuphy.h` · `aerial_phy.h` — CSI-RS demapping, slot scheduling |
| `cuphyBfWeightApply()` | 🟢 NVIDIA | `cuphy_ext.h` — TX beamforming weight update |
| `aerialSlotIndication()` callback | 🔵 Your code | `beam_manager.c` — registered via `aerial_register_slot_cb()`, receives `rx_iq_d` device ptr |

---

## Layer 2 — CUDA Kernel Pipeline

> File: **`beam_selection_cuda.cu`** 🔵 (your code)

| # | Kernel | Grid | Key Operation |
|---|--------|------|---------------|
| ① | `computeBeamRSRP` | `Grid(nBeams, SC_tiles)` · `Block(256)` | LS estimate `ĥ = y·ref* / |ref|²` · shared-mem reduction · `atomicAdd` |
| ② | `finalizeRSRP` | `Grid(nBeams/32)` · `Block(32)` | → `rsrp_dBm`, `rsrq_dB`, `sinr_dB`, validity flag |
| ③ | `selectBestBeam` | `Grid(1)` · `Block(nBeams)` | `cg::tiled_partition<32>` · `shfl_down` · argmax + A3 hysteresis |
| ④ | `updateHistoryBuffer` | `Grid(nBeams/32)` · `Block(32)` | Circular buffer `history_buf_d[DEPTH=32 × nBeams]` |
| ⑤ | `buildAIFeatureVector` | `Grid(nBeams/32)` · `Block(32)` | Outputs `[rsrp_now, rsrq_now, slope, std_dev]` per beam |

**Data flowing between layers:** `feature_vec_d [4 × nBeams]` · `best_beam_d`

---

## Layer 3 — AI Inference

### On-Device (TensorRT)

| Component | Owner | Details |
|-----------|-------|---------|
| `beam_selection_ai.cpp` — `aiInit()` / `aiInferBestBeam()` | 🔵 Your code | Engine load / ONNX build · `setTensorAddress()` · `enqueueV3()` |
| TensorRT 10 Runtime | 🟢 NVIDIA | `NvInfer.h` · `NvOnnxParser.h` · INT8 on Tensor Cores · ~0.15 ms · FP16 fallback |
| `beam_predict.onnx` | 🟡 You train | Conv1d(4→32) → Conv1d(32→64) → GlobalAvgPool → Linear(64→nBeams) → Softmax · ~18K params |

### AI Operating Modes

| Mode | Behaviour |
|------|-----------|
| `AI_MODE_DISABLED` | Pure RSRP greedy — Kernel ③ output only |
| `AI_MODE_HYBRID` | AI overrides if `confidence ≥ threshold`; falls back to RSRP |
| `AI_MODE_PREDICT` | AI always selects beam |
| `AI_MODE_EXTERNAL` | Forwards features to Triton server via gRPC |

### Beam Decision Logic

> `beam_manager.c` 🔵 — `confidence_thresh` gate → AI beam or RSRP fallback → `cuphyBfWeightApply()`

---

## Layer 4 — External AI Infrastructure (`AI_MODE_EXTERNAL`)

| Component | Owner | Details |
|-----------|-------|---------|
| Triton Inference Server | 🟣 External | NVIDIA Triton · gRPC/HTTP · hosted on A100 / H100 cluster |
| Triton Client Stub | 🔵 Your code | `beam_selection_ai.cpp` · `InferenceServerGrpcClient` · D2H copy → gRPC → `beam_probs` |
| Foundation / LLM-RAN Model | 🟣 External | Large transformer · full 32-slot sequence context · ~1–5 ms round-trip |

---

## Source Files

| File | Owner | Role |
|------|-------|------|
| `beam_selection.h` | 🔵 Your code | Public API, structs, constants |
| `beam_selection_cuda.cu` | 🔵 Your code | 5 CUDA kernels |
| `beam_manager.c` | 🔵 Your code | Aerial callback + state machine |
| `beam_selection_ai.cpp` | 🔵 Your code | TRT engine manager + Triton stub |
| `cuphy.h` | 🟢 NVIDIA | cuPHY base types |
| `aerial_phy.h` | 🟢 NVIDIA | Aerial gNB L1 API |
| `cuphy_ext.h` | 🟢 NVIDIA | CSI-RS measurement, BF weight APIs |
| `NvInfer.h` | 🟢 NVIDIA | TensorRT 10 runtime |
| `NvOnnxParser.h` | 🟢 NVIDIA | ONNX → TRT parser |
| `libtriton_client` | 🟣 External | Triton gRPC client library |

---

## CUDA APIs Used

```
atomicAdd              — lock-free per-beam RSRP accumulation
__frcp_rn              — fast HW reciprocal (L4 SFU, 1 cycle)
cg::tiled_partition<32>— warp-level argmax (cooperative groups)
shfl_down              — warp shuffle reduction (no shared mem)
cudaMemsetAsync        — zero accumulator per slot (non-blocking)
cudaMemcpyAsync        — non-blocking D2H measurement copy
cudaEventRecord        — gate host reads on kernel completion
cudaMallocHost         — pinned host buffers for zero-copy D2H
```

---

## Per-Slot Timing Budget (1 ms slot)

| Stage | Time |
|-------|------|
| cuPHY CSI-RS demapping | 0.05 ms |
| Kernels ①–⑤ | 0.35 ms |
| TensorRT inference (INT8) | 0.15 ms |
| BF weight apply | < 0.01 ms |
| **Total** | **~0.56 ms** |
| Headroom for cuPHY DL/UL | 0.44 ms |

---

## 3GPP Beam Management Alignment

| Procedure | Implementation |
|-----------|---------------|
| **P1** — Initial sweep (SSB) | Up to 64 beams measured simultaneously in one kernel dispatch |
| **P2** — Beam refinement (CSI-RS) | `p2_refinement_beams`: top-N from P1 fed into finer CSI-RS resources |
| **P3** — Beam tracking (CSI-RS/SRS) | History buffer + AI slope feature enable predictive tracking |
| **A3 event** | Hysteresis parameter in `selectBestBeam` kernel |
| **Beam failure recovery** | `rsrp_floor_dBm` marks beams invalid; fallback to last valid beam |
