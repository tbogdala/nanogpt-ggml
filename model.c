#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "model.h"

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
    struct nanogpt_model_hparams hparams)
{
    //FIXME: pin down why 1 doesn't work here
    size_t n_tensors = 10;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

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

    if (!model->backend) {
        // fallback to CPU backend
        fprintf(stderr, "%s: using CPU backend\n", __func__);
        model->backend = ggml_backend_cpu_init();
    }

    if (!model->backend) {
        fprintf(stderr, "%s: ggml_backend_cpu_init() failed\n", __func__);
        return false;
    }

    model->params = hparams;
    model->embeddings = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, model->params.n_embd, model->params.n_vocab);
   
    // allocate the model tensors in a backend buffer
    model->buffer_w = ggml_backend_alloc_ctx_tensors(model->ctx, model->backend);

    // from set_param_model() ...
    ggml_set_param(model->ctx, model->embeddings);  

    return true;  
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
    const struct nanogpt_model* model)
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

    // setup our tensors for input tokens and their targets
    struct ggml_tensor* tokens_input = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, model->params.n_ctx, model->params.n_batches);
    ggml_set_name(tokens_input, "tokens_input");
    ggml_set_input(tokens_input);

    // pull the logits out of the embbedding table and mark it as our output
    struct ggml_tensor* tokens_input_1d = ggml_view_1d(ctx,
                                                tokens_input,
                                                tokens_input->ne[0] * tokens_input->ne[1],
                                                0);

    // pull the logits straight from the embeddings tensor of the model.
    // should be shape: [vocab_size, batch_size*block_size]
    struct ggml_tensor* logits = ggml_get_rows(ctx, model->embeddings, tokens_input_1d);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    assert_shape_2d(logits, model->params.n_vocab, model->params.n_batches * model->params.n_ctx);

    // expand the graph so we can allocate and compute it
    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx);
    return gf;
}

bool
nanogpt_model_calculate_loss(
    const struct nanogpt_model* model,
    const ggml_backend_t cpu_backend,
    struct ggml_cgraph* eval_graph,
    const float* targets,
    const int64_t targets_count,
    float* out_loss)
{
    const int64_t total_batched_size = model->params.n_ctx * model->params.n_batches;

    // make a new set of cpu allocators, context, graphs and tensors to do the 
    // final cross_entropy calcuation based on the computed logits and the
    // expected targets.
    ggml_gallocr_t cpu_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu_backend));
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
    ggml_backend_graph_compute(cpu_backend, cpu_gf);

    // get our results
    ggml_backend_tensor_get(e, out_loss, 0, sizeof(float));

    // free our allocations
    free(gpu_logits_data);
    ggml_free(cpu_ctx);
    ggml_gallocr_free(cpu_allocr);

    return true;
}