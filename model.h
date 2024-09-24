#include "ggml.h"

#define NANOGPT_MAX_NODES 4096

struct nanogpt_model_hparams {
    int64_t n_vocab;    // number of tokens in the vocabulary of the model
    int64_t n_batches;  // number of batches of tokenx (length = n_ctx) to eval
    int64_t n_ctx;      // number of tokens in the context
    int64_t n_embd;     // the embedding size
};

struct nanogpt_model {
    struct ggml_context * ctx;
    ggml_backend_t backend;
    ggml_backend_buffer_t buffer_w;
    
    struct nanogpt_model_hparams params;
    struct ggml_tensor * embeddings;
};

// initialize the model
bool 
nanogpt_model_init(
    struct nanogpt_model* model, 
    struct nanogpt_model_hparams params);

// initialize the model with random data
void
nanogpt_model_randomize(
    struct nanogpt_model* model);

// constructs the computation graph for the forward evaluation of the model
// with the output tensor 'logits' set as the output.
struct ggml_cgraph *
nanogpt_model_build_eval_graph(
    const struct nanogpt_model* model);


bool
nanogpt_model_calculate_loss(
    const struct nanogpt_model* model,
    const ggml_backend_t cpu_backend,
    struct ggml_cgraph* eval_graph,
    const float* targets,
    const int64_t targets_count,
    float* out_loss);