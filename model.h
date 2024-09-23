#include "ggml.h"

#define NANOGPT_MAX_NODES 4096

struct nanogpt_model_hparams {
    int64_t n_vocab;
    int64_t n_ctx;
    int64_t n_embd;
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