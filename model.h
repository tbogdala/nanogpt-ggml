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
    ggml_gallocr_t allocr;
    ggml_backend_buffer_t buffer_w;
    
    int threads;
    struct nanogpt_model_hparams params;
    struct ggml_tensor* embeddings;
};

// initialize the model and also initializes the backend used for calculations.
// this function allocates the tensors so that they can be loaded as well.
bool 
nanogpt_model_init(
    struct nanogpt_model* model, 
    struct nanogpt_model_hparams params,
    int threads);

// release the backend GGML structures used in the model.
void
nanogpt_model_free(struct nanogpt_model* model);

// initialize the model with random data.
void
nanogpt_model_randomize(
    struct nanogpt_model* model);

// constructs the computation graph for the forward evaluation of the model
// with the output tensor 'logits' set as the output. `input_token_count`
// indicates how many tokens are going to be passed as input and should
// not exceed the model's `n_ctx` parameter.
//
// a `ggml_cgraph` must be created and passed in. for training workflows,
// this graph can be extended with the `nano_model_calculate_loss()` function
// to add a targets and cross entropy loss tensor.
//
// this function returns the 'logits' tensor.
struct ggml_tensor*
nanogpt_model_build_eval_graph(
    struct ggml_context* ctx,
    struct ggml_cgraph* gf,
    const struct nanogpt_model* model,
    const int batches,
    const int input_token_count);

// takes the graph to pull 'logits' from and extends it with a 'targets' tensor.
// the logits are compared against the `targets` and the cross entropy loss is 
// placed in a tensor named 'loss' and returned.
//
// additionally, the 'logits' tensor from `nanogpt_model_build_eval_graph()`
// should be supplied. `input_token_count`
// indicates how many tokens are going to be passed as input and should
// not exceed the model's `n_ctx` parameter.
struct ggml_tensor*
nanogpt_model_calculate_loss(
    struct ggml_context* ctx,
    struct ggml_cgraph* gf,
    const struct nanogpt_model* model,
    struct ggml_tensor * logits_tensor,
    const int batches,
    const int input_token_count);

// predicts text based on the incoming token ids. client code
// should put the 'prompt' tokens into `tokens` and pass the number
// of prompt tokens as `input_token_ids_count` (which must be >0).
// `num_to_predict` should include input token size, so if you pass
// 8 tokens and want 100 new ones, `num_to_predict` should be 108.
// `tokens` should be sized to `num_to_predict`
bool
nanogpt_model_predict_batch(
    const struct nanogpt_model* model,
    const struct dataset_vocab* vocab_data, 
    const int batch_count,
    int64_t num_to_predict,
    int64_t input_token_ids_count,
    TokenId* tokens);