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
    int threads)
{
    model->backend = NULL;

#ifdef GGML_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    model->backend = ggml_backend_cuda_init(0);
    if (!model->backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif

#ifdef GGML_METAL
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
        ggml_backend_cpu_set_n_threads(model->backend, model->threads);
    } 

    if (!model->backend) {
        fprintf(stderr, "%s: ggml_backend_cpu_init() failed\n", __func__);
        return false;
    }

    size_t n_tensors = 2;
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

#ifdef GGML_METAL
    if (ggml_backend_is_metal(model->backend)) {
        ggml_backend_metal_set_n_cb(model->backend, threads);
    }
#endif

    model->params = hparams;
    model->embeddings = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->params.n_embd, model->params.n_vocab);
    ggml_set_param(model->ctx, model->embeddings);  
   
    // allocate the model tensors in a backend buffer
    model->buffer_w = ggml_backend_alloc_ctx_tensors(model->ctx, model->backend);
    model->allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model->backend));

    return true;  
}

void
nanogpt_model_free(struct nanogpt_model* model)
{
    ggml_free(model->ctx);
    ggml_backend_buffer_free(model->buffer_w);
    ggml_backend_free(model->backend);
    ggml_gallocr_free(model->allocr);

    model->ctx = NULL;
    model->buffer_w = NULL;
    model->backend = NULL;
    model->allocr = NULL;
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

struct ggml_tensor*
nanogpt_model_build_eval_graph(
    struct ggml_context* ctx,
    struct ggml_cgraph* gf,
    const struct nanogpt_model* model,
    const int batches,
    const int input_token_count)
{
    assert(input_token_count <= model->params.n_ctx);

    // create new input token tensors that are dependent on the number of tokens being passed in
    struct ggml_tensor* token_ids_input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input_token_count * batches);
    ggml_set_name(token_ids_input, "token_ids_input");
    ggml_set_input(token_ids_input);

    // pull the logits straight from the embeddings tensor of the model.
    struct ggml_tensor* logits = ggml_get_rows(ctx, model->embeddings, token_ids_input);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    assert_shape_2d(logits, model->params.n_vocab, batches * input_token_count);

    return logits;
}

struct ggml_tensor*
nanogpt_model_calculate_loss(
    struct ggml_context* ctx,
    struct ggml_cgraph* gf,
    const struct nanogpt_model* model,
    struct ggml_tensor * logits_tensor,
    const int batches,
    const int input_token_count)
{
    const int64_t total_batched_size = input_token_count * batches;

    struct ggml_tensor* targets_tensor = ggml_new_tensor_2d(
        ctx, 
        GGML_TYPE_F32, 
        model->params.n_vocab, 
        total_batched_size);
    ggml_set_name(targets_tensor, "targets");
    ggml_set_param(ctx, targets_tensor);

    struct ggml_tensor * e = ggml_cross_entropy_loss(ctx, targets_tensor, logits_tensor);
    ggml_set_name(e, "loss");
    ggml_set_loss(e);

    return e;
}

bool
nanogpt_model_predict_batch(
    const struct nanogpt_model* model,
    const struct dataset_vocab* vocab_data, 
    const int batch_count,
    int64_t num_to_predict,
    int64_t input_token_ids_count,
    TokenId* tokens) // output_tokens [T, B]
{
    // need at least one token to base predictions on
    assert(input_token_ids_count > 0);

    // for each token to predict ...
    for (int pred_i=input_token_ids_count; pred_i<num_to_predict; ++pred_i) {
        int context_token_count = pred_i;
        if (context_token_count >= model->params.n_ctx) {
            context_token_count = model->params.n_ctx;
        }
        // create the worst case graph for memory usage estimation
        size_t buf_size = ggml_tensor_overhead()*NANOGPT_MAX_NODES + ggml_graph_overhead_custom(NANOGPT_MAX_NODES, false);
        struct ggml_init_params params = {
            /*.mem_size   =*/ buf_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
        };
        struct ggml_context* ctx = ggml_init(params);
        struct ggml_cgraph* eval_graph = ggml_new_graph_custom(ctx, NANOGPT_MAX_NODES, false);
        struct ggml_tensor* logits_tensor = nanogpt_model_build_eval_graph(ctx, eval_graph, model, batch_count, context_token_count);

        ggml_build_forward_expand(eval_graph, logits_tensor);
        bool success = ggml_gallocr_alloc_graph(model->allocr, eval_graph);
        if (!success) {
            fprintf(stderr, "%s: call to ggml_gallocr_alloc_graph failed!.\n", __func__);
        }
        struct ggml_tensor* token_ids_input = ggml_graph_get_tensor(eval_graph, "token_ids_input");
        if (token_ids_input == NULL) {
            fprintf(stderr, "%s: failed to get token_ids_input tensor by name.\n", __func__);    
            return false;
        }

        // populate targets and tokens_input tensors in batches
        int offset = 0;
        if (context_token_count == model->params.n_ctx) {
            offset = pred_i - model->params.n_ctx; 
        }

        for (int k=0; k<batch_count; ++k) {
            ggml_backend_tensor_set(
                token_ids_input,
                &tokens[k*(context_token_count + offset)],
                k*context_token_count*sizeof(TokenId),
                context_token_count*sizeof(TokenId));
        }

        // set backend options for threading in case it's changed
        if (ggml_backend_is_cpu(model->backend)) {
            ggml_backend_cpu_set_n_threads(model->backend, model->threads);
        }

#ifdef GGML_METAL
        if (ggml_backend_is_metal(model->backend)) {
            ggml_backend_metal_set_n_cb(model->backend, model->threads);
        }
#endif

        // run the calculation for the expanded graph
        ggml_backend_graph_compute(model->backend, eval_graph);

        const int64_t model_embd = model->params.n_embd;
        int64_t logits_cap = batch_count * context_token_count * model_embd;
        float logits[logits_cap];
        ggml_backend_tensor_get(logits_tensor, logits, 0, logits_cap*sizeof(float));

        // puts("\nOutputting logits:");
        // for (int j=0; j<batch_count*context_token_count; ++j) {
        //     printf("Logit #%d\n", j);
        //     for (int k=0; k<model_embd; ++k) {
        //         if (k!=0 && k%10==0) {
        //             puts("");
        //         }
        //         printf("%f ", logits[j*model->params.n_ctx + k]);
        //     }
        //     puts("");
        // }

        // softmax the logits
        float softmaxies[model_embd * batch_count];
        for (int k=0; k<batch_count; ++k) {
            //printf("Softmaxing offset for logits: %lld\n", (k+1)*(context_token_count-1)*model_embd);
            memcpy(&softmaxies[k*model_embd], &logits[(k+1)*(context_token_count-1)*model_embd], model_embd*sizeof(float));
        }

        // puts("\nBefore softmax logits:");
        // for (int k=0; k<model_embd; ++k) {
        //     if (k!=0 && k%10==0) {
        //         puts("");
        //     }
        //     printf("%f ", softmaxies[k]);
        // }
        // puts("");

        calc_softmax_inplace(batch_count, model_embd, softmaxies);

        // puts("\nAftare softmax logits:");
        // for (int k=0; k<model_embd; ++k) {
        //     if (k!=0 && k%10==0) {
        //         puts("");
        //     }
        //     printf("%f ", softmaxies[k]);
        // }
        // puts("");

        TokenId predictions[batch_count];
        int batch_success = multinominal_sample_batch(softmaxies, model_embd, batch_count, predictions);
        if (batch_success < 0) {
            fprintf(stderr, "%s: failed sample next token with multinominal_sample_batch\n", __func__);
            return false;
        }

        // puts("  Batch predictions are as follows:");
        // for (int k=0; k<batch_count; ++k) {
        //     printf("  \t(Id:%d) ==> '%c'\n", predictions[k], dataset_vocab_decode(vocab_data, predictions[k]));
        // }

        // put the predictions into the output buffer, split according to batches
        for (int k=0; k<batch_count; ++k) {
            tokens[k*num_to_predict + pred_i] = predictions[k];
        }

        ggml_free(ctx);
    }

    return true;
}