/**
 * beam_selection_ai.cpp
 * ============================================================================
 * AI / TensorRT beam prediction engine
 *
 * Three integration modes (selected via AiCfg_t.mode):
 *
 *  AI_MODE_PREDICT  — TensorRT engine loaded from .trt or built from .onnx.
 *                     Input:  feature_vec [4 × nBeams] (device ptr)
 *                     Output: beam probability distribution [nBeams]
 *                     → selects argmax as next beam
 *
 *  AI_MODE_HYBRID   — Same engine; only overrides RSRP selection when
 *                     softmax confidence >= threshold.
 *
 *  AI_MODE_EXTERNAL — Forwards features to an NVIDIA Triton Inference Server
 *                     over HTTP/gRPC. Requires libtriton_client.so.
 *                     Useful when the AI model is too large for on-GPU
 *                     co-hosting with cuPHY (e.g., foundation models).
 *
 * Model architecture (recommended — see beam_predict_model.py):
 *
 *   Input  : [batch=1, 4, nBeams]   (rsrp, rsrq, slope, std per beam)
 *   Layer 1: Conv1d(4→32, k=1) + ReLU       — per-beam feature mixing
 *   Layer 2: Conv1d(32→64, k=3, pad=1) + ReLU  — cross-beam context
 *   Layer 3: GlobalAvgPool                   — aggregate
 *   Layer 4: Linear(64 → nBeams)             — beam logits
 *   Output : Softmax → beam probability distribution [nBeams]
 *
 * On L4 (sm_89, INT8, Tensor Cores):
 *   Estimated inference latency: ~0.15 ms per slot @ batch=1
 *   Memory: ~2 MB engine (nBeams=64 variant)
 *
 * ============================================================================
 */

#include "beam_selection.h"
#include <NvInfer.h>           /* TensorRT 10.x                               */
#include <NvOnnxParser.h>      /* ONNX → TRT graph parser                    */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <cassert>

using namespace nvinfer1;

/* ─────────────────── TRT logger ─────────────────────────────────────────── */
class BeamSelLogger : public ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING)
            fprintf(stderr, "[TRT][%s] %s\n",
                    severity == Severity::kERROR   ? "ERROR" :
                    severity == Severity::kWARNING ? "WARN"  : "INFO", msg);
    }
};

static BeamSelLogger gTrtLogger;

/* ─────────────────── AI context structure ──────────────────────────────── */
struct AiContext {
    /* TensorRT objects */
    IRuntime          *runtime;
    ICudaEngine       *engine;
    IExecutionContext *exec_ctx;

    /* I/O bindings (TRT 10 API: named bindings) */
    void  *input_buf_d;    /* device: [4 × nBeams] float — same as feature_vec */
    void  *output_buf_d;   /* device: [nBeams] float — softmax probabilities    */
    float *output_buf_h;   /* host  (pinned): [nBeams] float                    */

    int    nBeams;
    AiMode_t mode;
    float    confidence_thresh;

    /* External Triton client handle (AI_MODE_EXTERNAL) */
    void  *triton_client;
    char   triton_server_addr[64];
    char   triton_model_name[64];

    /* Inference call counter */
    uint64_t infer_count;
};

/* ─────────────────── helper: load serialised engine from file ───────────── */
static std::vector<char> loadEngineFile(const char *path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    file.read(buf.data(), size);
    return buf;
}

/* ─────────────────── helper: build engine from ONNX ─────────────────────── */
static ICudaEngine *buildEngineFromOnnx(
    const char *onnx_path,
    int         nBeams,
    bool        enable_int8
) {
    IBuilder *builder = createInferBuilder(gTrtLogger);
    if (!builder) return nullptr;

    INetworkDefinition *network = builder->createNetworkV2(
        1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    auto parser = nvonnxparser::createParser(*network, gTrtLogger);
    if (!parser->parseFromFile(onnx_path,
            static_cast<int>(ILogger::Severity::kWARNING))) {
        fprintf(stderr, "[BeamSelAI] ONNX parse failed: %s\n", onnx_path);
        return nullptr;
    }

    IBuilderConfig *config = builder->createBuilderConfig();

    /* L4-specific optimisation flags */
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 256u << 20); /* 256 MB */

    if (enable_int8) {
        /* INT8 calibration for L4 4th-gen Tensor Cores (2× throughput vs FP16) */
        config->setFlag(BuilderFlag::kINT8);
        config->setFlag(BuilderFlag::kFP16);   /* fallback layers              */
        /* TODO: Attach an IInt8Calibrator here for post-training calibration.
         * Recommended calibration data: ~500 slots of RSRP history.
         *   class BeamInt8Calibrator : public IInt8EntropyCalibrator2 { ... };
         */
    } else {
        config->setFlag(BuilderFlag::kFP16);   /* FP16 on Tensor Cores        */
    }

    /* Build serialised network */
    IHostMemory *serialised = builder->buildSerializedNetwork(*network, *config);
    if (!serialised) {
        fprintf(stderr, "[BeamSelAI] Engine build failed\n");
        return nullptr;
    }

    IRuntime *runtime = createInferRuntime(gTrtLogger);
    ICudaEngine *engine = runtime->deserializeCudaEngine(
        serialised->data(), serialised->size());

    /* Cache engine to disk for future runs */
    char cache_path[280];
    snprintf(cache_path, sizeof(cache_path), "%s_b%d_int%d.trt",
             onnx_path, nBeams, enable_int8 ? 8 : 16);
    std::ofstream cache(cache_path, std::ios::binary);
    cache.write(static_cast<const char*>(serialised->data()), serialised->size());
    fprintf(stderr, "[BeamSelAI] Engine cached to %s\n", cache_path);

    serialised->destroy();
    delete config;
    delete parser;
    delete network;
    delete builder;

    return engine;
}

/* ════════════════════════════════════════════════════════════════════════════
 * aiInit — build/load TRT engine, allocate I/O buffers
 * ════════════════════════════════════════════════════════════════════════════ */
extern "C"
BeamSelStatus_t aiInit(
    void         **ai_ctx_out,
    const AiCfg_t *cfg,
    int            nBeams,
    cudaStream_t   stream
) {
    AiContext *ai = new AiContext{};
    ai->nBeams             = nBeams;
    ai->mode               = cfg->mode;
    ai->confidence_thresh  = cfg->confidence_thresh;
    strncpy(ai->triton_server_addr, cfg->server_addr, sizeof(ai->triton_server_addr)-1);
    strncpy(ai->triton_model_name,  cfg->model_name,  sizeof(ai->triton_model_name)-1);

    if (cfg->mode == AI_MODE_EXTERNAL) {
        /*
         * ── Triton gRPC client setup ──
         *
         * Uses NVIDIA Triton Inference Server client library.
         * The external server hosts the full beam prediction model
         * (potentially a large foundation model like Ericsson/Nokia AIR).
         *
         * namespace tc = triton::client;
         * std::unique_ptr<tc::InferenceServerGrpcClient> client;
         * tc::InferenceServerGrpcClient::Create(&client, cfg->server_addr, false);
         * ai->triton_client = client.release();
         *
         * Allocate host-pinned input/output buffers for zero-copy transfer:
         */
        cudaMallocHost(&ai->input_buf_d,  4 * nBeams * sizeof(float));
        cudaMallocHost(&ai->output_buf_h, nBeams * sizeof(float));
        *ai_ctx_out = ai;
        return BEAM_SEL_OK;
    }

    /* ── TensorRT local inference path ── */
    ai->runtime = createInferRuntime(gTrtLogger);
    if (!ai->runtime) { delete ai; return BEAM_SEL_ERR_TRT; }

    /* Try loading prebuilt engine first, then fall back to ONNX build */
    std::vector<char> engine_data = loadEngineFile(cfg->engine_path);
    if (!engine_data.empty()) {
        ai->engine = ai->runtime->deserializeCudaEngine(
            engine_data.data(), engine_data.size());
        fprintf(stderr, "[BeamSelAI] Loaded engine from %s\n", cfg->engine_path);
    } else if (cfg->onnx_path[0]) {
        fprintf(stderr, "[BeamSelAI] Building engine from ONNX: %s\n", cfg->onnx_path);
        ai->engine = buildEngineFromOnnx(cfg->onnx_path, nBeams, cfg->enable_int8);
    }

    if (!ai->engine) {
        fprintf(stderr, "[BeamSelAI] No engine available\n");
        delete ai; return BEAM_SEL_ERR_TRT;
    }

    ai->exec_ctx = ai->engine->createExecutionContext();
    if (!ai->exec_ctx) { delete ai; return BEAM_SEL_ERR_TRT; }

    /* ── Allocate I/O device buffers ── */
    cudaMalloc(&ai->input_buf_d,  4 * nBeams * sizeof(float));
    cudaMalloc(&ai->output_buf_d, nBeams * sizeof(float));
    cudaMallocHost(&ai->output_buf_h, nBeams * sizeof(float));

    /*
     * Set dynamic tensor shapes (TRT 10 explicit-batch API).
     * Input tensor name must match what was exported in the ONNX model.
     * Adjust "features" and "beam_probs" to match your model's I/O names.
     */
    ai->exec_ctx->setInputShape("features",
        Dims3{1, 4, nBeams});  /* batch=1, features=4, beams=nBeams */

    *ai_ctx_out = ai;
    return BEAM_SEL_OK;
}

/* ════════════════════════════════════════════════════════════════════════════
 * aiInferBestBeam — run one inference forward pass
 *
 * feature_vec_d : device pointer [4 × nBeams] — output of buildAIFeatureVector
 * ai_beam_out   : host pointer   — best beam predicted by AI
 * confidence_out: host pointer   — softmax probability of predicted beam
 * ════════════════════════════════════════════════════════════════════════════ */
extern "C"
BeamSelStatus_t aiInferBestBeam(
    void         *ai_ctx,
    const float  *feature_vec_d,
    int           nBeams,
    uint8_t      *ai_beam_out,
    float        *confidence_out,
    cudaStream_t  stream
) {
    AiContext *ai = static_cast<AiContext*>(ai_ctx);
    if (!ai) return BEAM_SEL_ERR_TRT;

    /* ── External Triton path ── */
    if (ai->mode == AI_MODE_EXTERNAL) {
        /*
         * 1. D2H copy of feature vector to pinned host buffer
         * 2. Build Triton InferInput from host buffer
         * 3. Call Async() and wait (or use callback for pipelining)
         * 4. Read output probabilities back
         *
         * Example (Triton C++ client):
         *
         *   cudaMemcpyAsync(ai->input_buf_d, feature_vec_d,
         *       4 * nBeams * sizeof(float), cudaMemcpyDeviceToHost, stream);
         *   cudaStreamSynchronize(stream);
         *
         *   tc::InferInput *input;
         *   tc::InferInput::Create(&input, "features", {1, 4, nBeams}, "FP32");
         *   input->AppendRaw((const uint8_t*)ai->input_buf_d,
         *                    4 * nBeams * sizeof(float));
         *
         *   tc::InferRequestedOutput *output;
         *   tc::InferRequestedOutput::Create(&output, "beam_probs");
         *
         *   tc::InferOptions options(ai->triton_model_name);
         *   tc::InferResult *result;
         *   static_cast<tc::InferenceServerGrpcClient*>(ai->triton_client)
         *       ->Infer(&result, options, {input}, {output});
         *
         *   const uint8_t *out_ptr; size_t out_bytes;
         *   result->RawData("beam_probs", &out_ptr, &out_bytes);
         *   memcpy(ai->output_buf_h, out_ptr, out_bytes);
         *   delete result;
         */
        fprintf(stderr, "[BeamSelAI] External Triton inference: implement gRPC call\n");
        *ai_beam_out   = 0;
        *confidence_out = 0.0f;
        return BEAM_SEL_ERR_TRT;  /* replace with BEAM_SEL_OK after implementing */
    }

    /* ── TensorRT local inference path ── */

    /*
     * The feature vector is already on device (output of buildAIFeatureVector).
     * We bind it directly — zero-copy from cuPHY → TRT I/O pipeline.
     *
     * TensorRT 10 named-binding API:
     */
    ai->exec_ctx->setTensorAddress("features",   (void*)feature_vec_d);
    ai->exec_ctx->setTensorAddress("beam_probs",  ai->output_buf_d);

    /* Asynchronous inference on the shared CUDA stream */
    bool ok = ai->exec_ctx->enqueueV3(stream);
    if (!ok) {
        fprintf(stderr, "[BeamSelAI] enqueueV3 failed\n");
        return BEAM_SEL_ERR_TRT;
    }

    /* Copy output probabilities D→H (async, then sync for this function) */
    cudaMemcpyAsync(ai->output_buf_h, ai->output_buf_d,
                    nBeams * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    /* Find argmax and confidence */
    int   best_beam  = 0;
    float best_prob  = ai->output_buf_h[0];
    for (int b = 1; b < nBeams; b++) {
        if (ai->output_buf_h[b] > best_prob) {
            best_prob = ai->output_buf_h[b];
            best_beam = b;
        }
    }

    *ai_beam_out    = (uint8_t)best_beam;
    *confidence_out = best_prob;
    ai->infer_count++;

    return BEAM_SEL_OK;
}

/* ════════════════════════════════════════════════════════════════════════════
 * aiDestroy — release TRT resources
 * ════════════════════════════════════════════════════════════════════════════ */
extern "C"
void aiDestroy(void *ai_ctx)
{
    AiContext *ai = static_cast<AiContext*>(ai_ctx);
    if (!ai) return;

    fprintf(stderr, "[BeamSelAI] Total inferences: %llu\n",
            (unsigned long long)ai->infer_count);

    if (ai->exec_ctx)    delete ai->exec_ctx;
    if (ai->engine)      delete ai->engine;
    if (ai->runtime)     delete ai->runtime;
    if (ai->input_buf_d) {
        if (ai->mode == AI_MODE_EXTERNAL) cudaFreeHost(ai->input_buf_d);
        else                              cudaFree(ai->input_buf_d);
    }
    if (ai->output_buf_d) cudaFree(ai->output_buf_d);
    if (ai->output_buf_h) cudaFreeHost(ai->output_buf_h);

    delete ai;
}
