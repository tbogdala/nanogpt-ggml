#pragma once

#include "ggml.h"
#include "dataset.h"

#define NANOGPT_MAX_NODES 4096

struct nanogpt_model_hparams {
    int64_t n_vocab;    // number of tokens in the vocabulary of the model
    int64_t n_ctx;      // number of tokens in the context
    int64_t n_embd;     // the embedding size
};

struct nanogpt_model {
    struct ggml_context * ctx;
    ggml_backend_t backend;
    ggml_backend_buffer_t buffer_w;
    ggml_backend_t cpu_backend;
    
    int threads;
    struct nanogpt_model_hparams params;
    struct ggml_tensor* embeddings;
    struct ggml_tensor* token_ids_input;
    struct ggml_tensor* tokens_input_1d; 
};

// initialize the model
bool 
nanogpt_model_init(
    struct nanogpt_model* model, 
    struct nanogpt_model_hparams params,
    int threads,
    int batch_count);

// release the backend GGML structures used in the model
void
nanogpt_model_free(struct nanogpt_model* model);

// initialize the model with random data
void
nanogpt_model_randomize(
    struct nanogpt_model* model);

// constructs the computation graph for the forward evaluation of the model
// with the output tensor 'logits' set as the output.
struct ggml_cgraph *
nanogpt_model_build_eval_graph(
    const struct nanogpt_model* model,
    const int batches);

// takes a loaded model and the computed evaluation graph to pull 'logits'
// from, taking the last set of logits for each batch and putting them
// into `out_logits`. `out_logits` MUST BE be at least 
// `params.n_embd * params.n_batches` in size.
// overall success of the function is returned as a bool.
bool
nanogpt_model_get_last_logits(
    const struct nanogpt_model* model,
    const int batches,
    struct ggml_cgraph* eval_graph,
    float* out_logits,
    const int64_t out_logits_capacity);

// takes the loaded model, a cpu backend to do the work on, and the 
// computed evaluation graph to pull 'logits', a named tensor, from.
// the logits are compared against the `targets` and the loss is 
// placed in `out_loss` as an output value. overall success of the
// function is returned as a bool.
bool
nanogpt_model_calculate_loss(
    const struct nanogpt_model* model,
    const int batches,
    struct ggml_cgraph* eval_graph,
    const float* targets,
    const int64_t targets_count,
    float* out_loss);

bool
nanogpt_model_predict_batch(
    const struct nanogpt_model* model,
    const struct dataset_vocab* vocab_data, 
    const int batch_count,
    int64_t num_to_predict,
    ggml_gallocr_t allocr,
    const TokenId* input_token_ids, // input_tokens [T, B]
    int64_t input_token_ids_count,
    TokenId* output_tokens,
    int64_t output_tokens_size);