#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "model.h"
#include "utility.h"

static void ggml_log_callback_default(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

void assert_shape_2d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1) {
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == ne1);
    GGML_ASSERT(tensor->ne[2] == 1);
    GGML_ASSERT(tensor->ne[3] == 1);
}

bool 
nanogpt_model_init(
    struct nanogpt_model* model, 
    struct nanogpt_model_hparams hparams,
    int threads,
    int batch_count)
{
    size_t n_tensors = 8;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    model->threads = threads;
    model->ctx = ggml_init(params);
    if (!model->ctx) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return false;
    }

    // initialize the backend
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    model->backend = ggml_backend_cuda_init(0);
    if (!model->backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif

#ifdef GGML_USE_METAL
    fprintf(stderr, "%s: using Metal backend\n", __func__);
    ggml_backend_metal_log_set_callback(ggml_log_callback_default, NULL);
    model->backend = ggml_backend_metal_init();
    if (!model->backend) {
        fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
    }
#endif

    // fallback to CPU backend
    if (!model->backend) {
        fprintf(stderr, "%s: using CPU backend\n", __func__);
        model->backend = ggml_backend_cpu_init();
        model->cpu_backend = model->backend;
    } else {
        // ensure we have CPU backend too for some calculations
        model->cpu_backend = ggml_backend_cpu_init();
    }

    if (!model->backend) {
        fprintf(stderr, "%s: ggml_backend_cpu_init() failed\n", __func__);
        return false;
    }

    ggml_backend_cpu_set_n_threads(model->cpu_backend, model->threads);
#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model->backend)) {
        ggml_backend_metal_set_n_cb(model->backend, threads);
    }
#endif
    
    model->params = hparams;
    model->embeddings = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->params.n_embd, model->params.n_vocab);
   
    // allocate the model tensors in a backend buffer
    model->buffer_w = ggml_backend_alloc_ctx_tensors(model->ctx, model->backend);

    // setup our tensors for input tokens and their targets
    model->token_ids_input = ggml_new_tensor_2d(model->ctx, GGML_TYPE_I32, model->params.n_ctx, batch_count);
    ggml_set_input(model->token_ids_input);
    model->tokens_input_1d = ggml_view_1d(
        model->ctx,
        model->token_ids_input,
        model->token_ids_input->ne[0] * model->token_ids_input->ne[1],
        0);


    // from set_param_model() ...
    ggml_set_param(model->ctx, model->embeddings);  

    return true;  
}

void
nanogpt_model_free(struct nanogpt_model* model)
{
    ggml_free(model->ctx);
    ggml_backend_buffer_free(model->buffer_w);
    if (model->backend != model->cpu_backend) {
        ggml_backend_free(model->backend);
    }
    ggml_backend_free(model->cpu_backend);
}

void
nanogpt_model_randomize(
    struct nanogpt_model* model)
{
    for (int i1 = 0; i1 < model->embeddings->ne[1]; i1++) {
        for (int i0 = 0; i0 < model->embeddings->ne[0]; i0++) {
            float * dst = (float *) ((char *) model->embeddings->data + i0*model->embeddings->nb[0] + i1*model->embeddings->nb[1]);
            *dst = (float)rand() / (float)RAND_MAX;
        }
    }
}

struct ggml_cgraph *
nanogpt_model_build_eval_graph(
    const struct nanogpt_model* model,
    const int batches)
{
    // create the worst case graph for memory usage estimation
    size_t buf_size = ggml_tensor_overhead()*NANOGPT_MAX_NODES + ggml_graph_overhead_custom(NANOGPT_MAX_NODES, true);
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };

    struct ggml_context* ctx = ggml_init(params);
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, NANOGPT_MAX_NODES, true);

    // pull the logits straight from the embeddings tensor of the model.
    // should be shape: [vocab_size, batch_size*block_size]
    struct ggml_tensor* logits = ggml_get_rows(ctx, model->embeddings, model->tokens_input_1d);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    assert_shape_2d(logits, model->params.n_vocab, batches * model->params.n_ctx);

    // expand the graph so we can allocate and compute it
    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx);
    return gf;
}

bool
nanogpt_model_get_last_logits(
    const struct nanogpt_model* model,
    const int batches,
    struct ggml_cgraph* eval_graph,
    float* out_logits,
    const int64_t out_logits_capacity)
{
    // ensure we have the capacity to store our logits
    assert(out_logits_capacity >= model->params.n_embd * batches);

    // get a handle to the previous graph's 'logits' tensor
    struct ggml_tensor * logits_tensor = ggml_graph_get_tensor(eval_graph, "logits");
    if (logits_tensor == NULL) {
        fprintf(stderr, "%s: failed to get logits tensor by name.\n", __func__);    
        return false;
    }

    for (int k=0; k<batches; ++k) {
        const int64_t offset = k * model->params.n_embd;
        assert(offset < out_logits_capacity);
        ggml_backend_tensor_get(
            logits_tensor, 
            &out_logits[offset], 
            (offset*model->params.n_ctx + (model->params.n_ctx - 1) * model->params.n_embd) * sizeof(float), 
            model->params.n_embd * sizeof(float));
    }

    return true;
}

bool
nanogpt_model_calculate_loss(
    const struct nanogpt_model* model,
    const int batches,
    struct ggml_cgraph* eval_graph,
    const float* targets,
    const int64_t targets_count,
    float* out_loss)
{
    const int64_t total_batched_size = model->params.n_ctx * batches;

    // make a new set of cpu allocators, context, graphs and tensors to do the 
    // final cross_entropy calcuation based on the computed logits and the
    // expected targets.
    ggml_gallocr_t cpu_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model->cpu_backend));
    size_t cpu_buf_size = ggml_tensor_overhead()*NANOGPT_MAX_NODES + ggml_graph_overhead();
    struct ggml_init_params cpu_params = {
        /*.mem_size   =*/ cpu_buf_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };
    struct ggml_context* cpu_ctx = ggml_init(cpu_params);
    struct ggml_cgraph* cpu_gf = ggml_new_graph(cpu_ctx);

    struct ggml_tensor* cpu_targets = ggml_new_tensor_2d(
        cpu_ctx, 
        GGML_TYPE_F32, 
        model->params.n_vocab, 
        total_batched_size);
    ggml_set_param(cpu_ctx, cpu_targets);

    struct ggml_tensor* cpu_logits = ggml_new_tensor_2d(
        cpu_ctx, 
        GGML_TYPE_F32, 
        model->params.n_vocab, 
        total_batched_size);
    ggml_set_param(cpu_ctx, cpu_logits);

    // this is the final calculation we want so mark the resulting tensor
    // as an 'output' tensor.
    struct ggml_tensor * e = ggml_cross_entropy_loss(cpu_ctx, cpu_targets, cpu_logits);
    ggml_set_output(e);

    // expand the graph on the cpu for our cross entropy calculation
    ggml_build_forward_expand(cpu_gf, e);

    // do the allocations necessary on the cpu backend for this calculation
    bool success = ggml_gallocr_alloc_graph(cpu_allocr, cpu_gf);
    if (!success) {
        fprintf(stderr, "%s: call to ggml_gallocr_alloc_graph for cpu_gf failed!\n", __func__);
        return false;
    }

    // get a handle to the previous graph's 'logits' tensor
    struct ggml_tensor * logits_tensor = ggml_graph_get_tensor(eval_graph, "logits");
    if (logits_tensor == NULL) {
        fprintf(stderr, "%s: failed to get logits tensor by name.\n", __func__);    
        return false;
    }

    // pull the data from the backend tensor and place it in a cpu tensor
    int64_t total_targets_count = model->params.n_vocab * total_batched_size;
    float* gpu_logits_data = (float*) calloc(total_targets_count, sizeof(float));
    ggml_backend_tensor_set(cpu_targets, targets, 0, targets_count*sizeof(float));
    ggml_backend_tensor_get(logits_tensor, gpu_logits_data, 0, total_targets_count*sizeof(float));
    ggml_backend_tensor_set(cpu_logits, gpu_logits_data, 0, total_targets_count*sizeof(float));
    
    // run the computation for the cross entropy loss on the cpu
    ggml_backend_graph_compute(model->cpu_backend, cpu_gf);

    // get our results
    ggml_backend_tensor_get(e, out_loss, 0, sizeof(float));

    // free our allocations
    free(gpu_logits_data);
    ggml_free(cpu_ctx);
    ggml_gallocr_free(cpu_allocr);

    return true;
}

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
    int64_t output_tokens_size) 
{
    // for now we only support getting passed in a full context of tokens to
    // bootstrap the prediction process.
    assert(input_token_ids_count == model->params.n_ctx);
    assert(output_tokens_size >= batch_count * num_to_predict);
    memset(output_tokens, 0, sizeof(TokenId) * output_tokens_size);
    
    struct ggml_cgraph* eval_graph = nanogpt_model_build_eval_graph(model, batch_count);
    bool success = ggml_gallocr_alloc_graph(allocr, eval_graph);
    if (!success) {
        fprintf(stderr, "%s: call to ggml_gallocr_alloc_graph failed!.\n", __func__);
    }

    // for each token to predict ...
    for (int pred_i=0; pred_i<num_to_predict; ++pred_i) {
        // populate targets and tokens_input tensors in batches
        for (int k=0; k<batch_count; k++) {
            int n_tokens = model->token_ids_input->ne[0];
            assert(model->params.n_ctx == n_tokens);

            // make targets a series of one-hot vectors for the desired tokens
            for (int i=0; i<n_tokens; ++i) {
                TokenId token_id;
                // check to see if we're still within range of the initial input parameters
                if (pred_i + i < n_tokens) {
                    token_id = input_token_ids[k*n_tokens + pred_i + i];
                    //printf("  DEBUG: b1 => pred_i %d ; i %d ; n_tokens %d ==> %d", pred_i, i, n_tokens, k*n_tokens + pred_i + i);
                } else {
                    token_id = output_tokens[pred_i + i - n_tokens];
                    //printf(  "  DEBUG: b2 => pred_i %d ; i %d ; n_tokens %d ==> %d", pred_i, i, n_tokens, pred_i + i - n_tokens);
                }
                //printf(" --> %d : '%c'\n", token_id, vocab_data->itot[token_id]);
                
                ggml_backend_tensor_set(model->tokens_input_1d, &token_id, i*sizeof(TokenId), sizeof(TokenId));
            }
        }  

        // run the calculation for the expanded graph
        ggml_backend_graph_compute(model->backend, eval_graph);

        int64_t last_logits_cap = batch_count * model->params.n_embd;
        float last_logits[last_logits_cap];
        for (int i=0; i<last_logits_cap; ++i) {
            last_logits[i] = -42.42f;
        }
        bool success = nanogpt_model_get_last_logits(
            model,
            batch_count,
            eval_graph,
            last_logits,
            last_logits_cap);
        if (!success) {
            fprintf(stderr, "%s: failed to get logits with nanogpt_model_get_last_logits\n", __func__);
            return false;
        }

        // softmax the logits
        calc_softmax_inplace(batch_count, model->params.n_embd, last_logits);

        TokenId predictions[batch_count];
        int batch_success = multinominal_sample_batch(last_logits, model->params.n_embd, batch_count, predictions);
        if (batch_success < 0) {
            fprintf(stderr, "%s: failed sample next token with multinominal_sample_batch\n", __func__);
            return false;
        }

        // DEBUG
        // puts("  Batch predictions are as follows:");
        // for (int k=0; k<batch_count; ++k) {
        //     printf("  \t(Id:%d) ==> '%c'\n", predictions[k], dataset_vocab_decode(vocab_data, predictions[k]));
        // }

        // put the predictions into the output buffer, split according to batches
        for (int k=0; k<batch_count; ++k) {
            output_tokens[k*num_to_predict + pred_i] = predictions[k];
        }
    }

    return true;
}