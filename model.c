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