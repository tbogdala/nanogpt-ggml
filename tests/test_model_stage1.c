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

void assert_shape_2d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1) {
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == ne1);
    GGML_ASSERT(tensor->ne[2] == 1);
    GGML_ASSERT(tensor->ne[3] == 1);
}


// build the input and target arrays. caller knows the batch size and the
// block size, so it is ASSUMED that `inputs` and `targets` are big enough
// to handle all the data.
void
build_input_and_target_arrays(
    TokenId* token_buffer,
    int64_t token_buffer_count,
    int batch_size,
    int block_size,
    TokenId* inputs,
    TokenId* targets)
{
    // TODO this is a super lame implementation that just gets
    // the input arrays, in order, from the start of the token buffer
    for (int i=0; i<batch_size*block_size; i++) {
        inputs[i] = token_buffer[i];
        targets[i] = token_buffer[i+1];
    }
}

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
    build_input_and_target_arrays(token_buffer, token_buffer_count, batch_count, block_size, input_ids, target_ids);

    // proint out the training data we got so far...
    char debug_array_decodes[total_batched_size + 1];
    dataset_vocab_decode_string(&vocab_data, input_ids, debug_array_decodes, total_batched_size);
    printf("Input batches: \"%s\"\n", debug_array_decodes);
    dataset_vocab_decode_string(&vocab_data, target_ids, debug_array_decodes, total_batched_size);
    printf("Target batches: \"%s\"\n\n", debug_array_decodes);

    // setup the model parameters
    struct nanogpt_model_hparams hparams;
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
    struct ggml_tensor* tokens_input = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, block_size, batch_count);
    ggml_set_name(tokens_input, "tokens_input");
    ggml_set_input(tokens_input);

    struct ggml_tensor* targets = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, vocab_data.total_tokens, block_size, batch_count);
    ggml_set_name(targets, "targets");
    ggml_set_input(targets);

    // pull the logits out of the embbedding table and mark it as our output
    struct ggml_tensor* tokens_input_1d = ggml_view_1d(ctx,
                                                tokens_input,
                                                tokens_input->ne[0] * tokens_input->ne[1],
                                                0);

    // pull the logits straight from the embeddings tensor of the model.
    // should be shape: [vocab_size, batch_size*block_size]
    struct ggml_tensor* logits = ggml_get_rows(ctx, model.embeddings, tokens_input_1d);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    assert_shape_2d(logits, vocab_data.total_tokens, total_batched_size);

    // expand the graph so we can allocate and compute it
    ggml_build_forward_expand(gf, targets); // targets isn't reference eslewhere, so explicitly expand here
    ggml_build_forward_expand(gf, logits);

    //printf("targets dims %lld %lld %lld %lld\n", targets->ne[0], targets->ne[1], targets->ne[2], targets->ne[3]);
    //printf("logits dims %lld %lld %lld %lld\n\n", logits->ne[0], logits->ne[1], logits->ne[2], logits->ne[3]);

    // do the necessary allocations
    bool success = ggml_gallocr_alloc_graph(allocr, gf);
    if (!success) {
        fprintf(stderr, "%s: call to ggml_gallocr_alloc_graph failed!.\n", __func__);
    }
    
    // now that the memory for the buffers has been allocated, randomize the model data for initialization
    nanogpt_model_randomize(&model);

    // make an array to use to zero out the initial targets
    int64_t target_count = vocab_data.total_tokens * total_batched_size;
    float zero_targets[target_count];
    for (int i=0; i<target_count; ++i) {
        zero_targets[i] = 0.0f;
    }

    // we could just use the tensors above, but this tests getting things by name for later use
    struct ggml_tensor * targets_tensor = ggml_graph_get_tensor(gf, "targets");
    struct ggml_tensor * tokens_input_tensor = ggml_graph_get_tensor(gf, "tokens_input");
    if (targets_tensor == NULL || tokens_input_tensor == NULL) {
        fprintf(stderr, "%s: failed to get targets and tokens_input tensors by name\n", __func__);
    }
        
    // populate targets and tokens_input tensors in batches
    for (int k=0; k<batch_count; k++) {
        struct ggml_tensor* tokens_input_k = ggml_view_1d(ctx,
                                                tokens_input_tensor,
                                                tokens_input->ne[0],
                                                k*tokens_input->nb[1]);
        struct ggml_tensor* targets_k    = ggml_view_2d(ctx,
                                                targets_tensor,
                                                targets->ne[0],
                                                targets->ne[1],
                                                targets->nb[1],
                                                k*targets->nb[2]);

        int n_tokens = tokens_input->ne[0];
        TEST_ASSERT_EQUAL(block_size, n_tokens);
        int n_vocab = targets->ne[0];
        TEST_ASSERT_EQUAL(vocab_data.total_tokens, n_vocab);

        // initialize all probabilities in the targets to -1.0f and then set
        // the corresponding embedding for the next_token that is expected to 1.0f
        ggml_backend_tensor_set(targets_k, zero_targets, 0, vocab_data.total_tokens*block_size * sizeof(float));
        for (int i=0; i<n_tokens; ++i) {
            TokenId token = input_ids[k*n_tokens+i];
            TokenId next_token = target_ids[k*n_tokens+i];
            float one_hot_float = 1.0f;
            ggml_backend_tensor_set(targets_k, &one_hot_float, (i*n_vocab + next_token)*sizeof(float), sizeof(float));
            ggml_backend_tensor_set(tokens_input_k, &token, i*sizeof(TokenId), sizeof(TokenId));
        }
    }

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


    // ************************************************


    // if we don't have a cpu backend yet, make one because not all
    // calculations we'll need to perform are supported on all backends.
    // for example, trying to calculate cross_entropy as coded above 
    // uses ops that are not supported on Metal.
    if (cpu_backend == NULL) {
        cpu_backend = ggml_backend_cpu_init();
    }

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
        vocab_data.total_tokens, 
        block_size * batch_count);
    ggml_set_param(cpu_ctx, cpu_targets);

    struct ggml_tensor* cpu_logits = ggml_new_tensor_2d(
        cpu_ctx, 
        GGML_TYPE_F32, 
        vocab_data.total_tokens, 
        block_size * batch_count);
    ggml_set_param(cpu_ctx, cpu_logits);

    // this is the final calculation we want so mark the resulting tensor
    // as an 'output' tensor.
    struct ggml_tensor * e = ggml_cross_entropy_loss(cpu_ctx, cpu_targets, cpu_logits);
    ggml_set_output(e);

    // expand the graph on the cpu for our cross entropy calculation
    ggml_build_forward_expand(cpu_gf, e);

    // do the allocations necessary on the cpu backend for this calculation
    success = ggml_gallocr_alloc_graph(cpu_allocr, cpu_gf);
    if (!success) {
        fprintf(stderr, "%s: call to ggml_gallocr_alloc_graph for cpu_gf failed!\n", __func__);
    }

    // get a handle to the previous graph's 'logits' tensor
    struct ggml_tensor * logits_tensor = ggml_graph_get_tensor(gf, "logits");
    if (logits_tensor == NULL) {
        fprintf(stderr, "%s: failed to get logits tensor by name.\n", __func__);    
    }

    // pull the data from the backend tensor and place it in a cpu tensor
    int64_t total_targets_count = vocab_data.total_tokens * total_batched_size;
    float* gpu_targets_data = (float*) calloc(total_targets_count, sizeof(float));
    float* gpu_logits_data = (float*) calloc(total_targets_count, sizeof(float));
    ggml_backend_tensor_get(targets_tensor, gpu_targets_data, 0, total_targets_count*sizeof(float));
    ggml_backend_tensor_set(cpu_targets, gpu_targets_data, 0, total_targets_count*sizeof(float));
    ggml_backend_tensor_get(logits_tensor, gpu_logits_data, 0, total_targets_count*sizeof(float));
    ggml_backend_tensor_set(cpu_logits, gpu_logits_data, 0, total_targets_count*sizeof(float));
    
    // run the computation for the cross entropy loss on the cpu
    ggml_backend_graph_compute(cpu_backend, cpu_gf);

    // get our results
    float error_before_opt;
    ggml_backend_tensor_get(e, &error_before_opt, 0, ggml_nbytes(e));
    printf("Error before optimizations: %f\n\n", error_before_opt / total_batched_size);
    

    // DEBUG printouts
    puts("We're gonna write out some data as sanity checks to make sure we're getting what we think we should...");
    puts("Input TokenIds:");
    for (int i=0; i<total_batched_size; ++i) {
        printf("%d ", input_ids[i]);
    }
    puts("");
    puts("Target TokenIds:");
    for (int i=0; i<total_batched_size; ++i) {
        printf("%d ", target_ids[i]);
    }
    puts("");

    // the `cpu_logits` and `cpu_targets` tensors won't necessarily show the right values unless
    // it's changed to an output parameter, so we're just gonna yoink
    // the data from the `gpu_data` buffer since it last held the logits...
    int dbg_logits_index = 1;
    printf("\nThe targets at index %d should show a one-hot matrix equal to the id in the same spot in Target TokenIds above.", dbg_logits_index);
    puts("Additionally, at this point logits are pulled straight from the embedding table, so the logits");
    puts("should match the expected embedding matrix row for the matching input id at that index...\n");
    printf("Targets at index %d:\n", dbg_logits_index);
    for (int i=0; i<vocab_data.total_tokens; ++i) {
        if (i!=0 && i%10 == 0) {
            puts("");
        }
        printf("%f ", gpu_targets_data[vocab_data.total_tokens*dbg_logits_index + i]);
    }
    puts("");
    printf("Logits at index %d:\n", dbg_logits_index);
    for (int i=0; i<vocab_data.total_tokens; ++i) {
        if (i!=0 && i%10 == 0) {
            puts("");
        }
        printf("%f ", gpu_logits_data[vocab_data.total_tokens*dbg_logits_index + i]);
    }
    puts("");
    int dbg_emb_index = 47;
    printf("Embedding at vocab index %d:\n", dbg_emb_index);
    for (int i=0; i<vocab_data.total_tokens; ++i) {
        if (i!=0 && i%10 == 0) {
            puts("");
        }
        printf("%f ", ggml_get_f32_nd(model.embeddings, i, dbg_emb_index, 0, 0));
    }
    puts("");
    
    // free up our allocations
    free(gpu_targets_data);
    free(gpu_logits_data);

    ggml_free(model.ctx);
    ggml_free(cpu_ctx);
    ggml_gallocr_free(allocr);
    ggml_gallocr_free(cpu_allocr);
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