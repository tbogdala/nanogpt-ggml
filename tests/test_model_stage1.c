#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include "unity.h"
#include "ggml-backend.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "../dataset.h" 
#include "../model.h"

void setUp(void) {}
void tearDown(void) {}

void test_model_stage1() {
    const int batch_count = 4;
    const int block_size = 8;
    //ggml_backend_t backend = NULL;

    // randomize the embeddings
    const unsigned int seed = 1337;
    if (seed > 0) {
        srand(seed);
    } else {
        srand(time(NULL));
    }

    // setup the dataset
    struct dataset_vocab vocab_data;
    int ret = dataset_vocab_from_file(&vocab_data, "data/shakes.txt");
   
    // tokenize the dataset into one big file
    TokenId* token_buffer;
    int64_t token_buffer_count;
    ret = dataset_vocab_tokenize_file(&vocab_data, "data/shakes.txt", &token_buffer, &token_buffer_count);
    TEST_ASSERT_EQUAL(0, ret);
    TEST_ASSERT_EQUAL(1115394, token_buffer_count);

    // prep the batch of data into inputs and targets
    int64_t total_batched_size = batch_count * block_size;
    TokenId input_ids[total_batched_size];
    TokenId target_ids[total_batched_size];
    dataset_build_input_and_target_arrays(token_buffer, token_buffer_count, batch_count, block_size, input_ids, target_ids);

    // print out the training data we got so far...
    char debug_array_decodes[total_batched_size + 1];
    dataset_vocab_decode_string(&vocab_data, input_ids, debug_array_decodes, total_batched_size);
    printf("Input batches: \"%s\"\n", debug_array_decodes);
    dataset_vocab_decode_string(&vocab_data, target_ids, debug_array_decodes, total_batched_size);
    printf("Target batches: \"%s\"\n\n", debug_array_decodes);

    /* ===================================================================== */

    // setup the model parameters
    struct nanogpt_model_hparams hparams;
    hparams.n_batches = batch_count;
    hparams.n_ctx = block_size;
    hparams.n_vocab = vocab_data.total_tokens;
    hparams.n_embd = hparams.n_vocab;

    // setup the nanogpt model
    struct nanogpt_model model;
    bool model_init_success = nanogpt_model_init(&model, hparams);
    if (!model_init_success) {
        fprintf(stderr, "%s: call to nanogpt_model_init() failed\n", __func__);
        return;
    }

    // setup an allocator
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    // build an evaluation graph
    struct ggml_cgraph* gf = nanogpt_model_build_eval_graph(&model);

    // do the necessary allocations so that we can pump our data in
    bool success = ggml_gallocr_alloc_graph(allocr, gf);
    if (!success) {
        fprintf(stderr, "%s: call to ggml_gallocr_alloc_graph failed!.\n", __func__);
    }

    /* ===================================================================== */

    // now that the memory for the buffers has been allocated, randomize the model data for initialization
    nanogpt_model_randomize(&model);

    struct ggml_tensor * tokens_input_tensor = ggml_graph_get_tensor(gf, "tokens_input");
    if (tokens_input_tensor == NULL) {
        fprintf(stderr, "%s: failed to get tokens_input tensor by name\n", __func__);
    }
        
    // make an array to use to zero out the initial targets
    int64_t targets_count = vocab_data.total_tokens * total_batched_size;
    float targets[targets_count];
    for (int i=0; i<targets_count; ++i) {
        targets[i] = 0.0f;
    }

    // populate targets and tokens_input tensors in batches
    for (int k=0; k<batch_count; k++) {
        struct ggml_tensor* tokens_input_k = ggml_view_1d(model.ctx,
                                                tokens_input_tensor,
                                                tokens_input_tensor->ne[0],
                                                k*tokens_input_tensor->nb[1]);

        int n_tokens = tokens_input_tensor->ne[0];
        TEST_ASSERT_EQUAL(block_size, n_tokens);

        const int base_batch_offset = vocab_data.total_tokens * block_size * k;

        // make targets a series of one-hot vectors for the desired tokens
        for (int i=0; i<n_tokens; ++i) {
            TokenId token = input_ids[k*n_tokens+i];
            TokenId next_token = target_ids[k*n_tokens+i];
            targets[base_batch_offset + i*hparams.n_vocab + next_token] = 1.0f;
            ggml_backend_tensor_set(tokens_input_k, &token, i*sizeof(TokenId), sizeof(TokenId));
        }
    }

    /* ===================================================================== */

    // setup the compute backend
    int n_threads = 1; 
    ggml_backend_t cpu_backend=NULL;
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
        cpu_backend = model.backend;
    }

#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend)) {
        ggml_backend_metal_set_n_cb(model.backend, n_threads);
    }
#endif

    // run the calculation for the expanded graph
    ggml_backend_graph_compute(model.backend, gf);

    // if we don't have a cpu backend yet, make one because not all
    // calculations we'll need to perform are supported on all backends.
    // for example, trying to calculate cross_entropy as coded above 
    // uses ops that are not supported on Metal.
    if (cpu_backend == NULL) {
        cpu_backend = ggml_backend_cpu_init();
    }

    float error_before_opt = -42.0f;
    success = nanogpt_model_calculate_loss(
        &model,
        cpu_backend,
        gf,
        targets,
        targets_count,
        &error_before_opt);

    //DEBUG: 4.338178 is the expected value with seed 1337 on my Macbook Air dev machine...
    printf("Error before optimizations: %f\n\n", error_before_opt / total_batched_size);

    // free up our allocations
    ggml_free(model.ctx);
    ggml_gallocr_free(allocr);
    ggml_backend_buffer_free(model.buffer_w);
    if (model.backend != cpu_backend) {
        ggml_backend_free(model.backend);
    }
    ggml_backend_free(cpu_backend);
}


int main() {
    UNITY_BEGIN();
    RUN_TEST(test_model_stage1);
    return UNITY_END();
}