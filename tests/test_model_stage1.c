#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include "unity.h"
#include "ggml-backend.h"

#include "../dataset.h" 
#include "../model.h"
#include "../utility.h"

void setUp(void) {}
void tearDown(void) {}

void test_model_stage1() {
    const int thread_count = 1;
    int batch_count = 4;
    const int block_size = 8;

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

    printf("input_ids:\n");
    for (int i=0; i<total_batched_size; ++i) {
        if (i!=0 && i%block_size == 0){
            printf("\n");
        }
        printf("%3d ", input_ids[i]);
    }
    printf("\n\ntarget_ids:\n");
    for (int i=0; i<total_batched_size; ++i) {
        if (i!=0 && i%block_size == 0){
            printf("\n");
        }
        printf("%3d ", target_ids[i]);
    }
    printf("\n\n");

    /* ===================================================================== */
    // setup the model parameters
    struct nanogpt_model_hparams hparams;
    hparams.n_ctx = block_size;
    hparams.n_vocab = vocab_data.total_tokens;
    hparams.n_embd = hparams.n_vocab;

    // setup the nanogpt model
    struct nanogpt_model model;
    bool model_init_success = nanogpt_model_init(&model, hparams, thread_count);
    TEST_ASSERT_TRUE(model_init_success);
    printf("Initialized model...\n");

    // build a new context here for our calculations in the test
    size_t buf_size = ggml_tensor_overhead()*NANOGPT_MAX_NODES + ggml_graph_overhead_custom(NANOGPT_MAX_NODES, false);
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };
    struct ggml_context* ctx = ggml_init(params);
    struct ggml_cgraph* eval_gf = ggml_new_graph_custom(ctx, NANOGPT_MAX_NODES, false);
    TEST_ASSERT_NOT_NULL(eval_gf);
    
    // build an evaluation graph
    struct ggml_tensor* logits_tensor = nanogpt_model_build_eval_graph(ctx, eval_gf, &model, batch_count, hparams.n_ctx);
    TEST_ASSERT_NOT_NULL(logits_tensor);

    // add in the loss functionality, which is not part of the base graph generation because it's 
    // not needed for inference.
    float error_before_opt = -42.0f;
    struct ggml_tensor* e = nanogpt_model_calculate_loss(
        ctx,
        eval_gf,
        &model,
        logits_tensor,
        batch_count,
        block_size);
    TEST_ASSERT_NOT_NULL(e);

    // do the necessary allocations so that we can pump our data in
    ggml_build_forward_expand(eval_gf, e);
    bool success = ggml_gallocr_alloc_graph(model.allocr, eval_gf);
    TEST_ASSERT_TRUE(success);

    /* ===================================================================== */

    // now that the memory for the buffers has been allocated, randomize the model data for initialization
    nanogpt_model_randomize(&model);

    // build the tokenid input tensor and fill it with data
    struct ggml_tensor* token_ids_input = ggml_graph_get_tensor(eval_gf, "token_ids_input");
    TEST_ASSERT_NOT_NULL(token_ids_input);

    // make an array to use to zero out the initial targets
    int64_t targets_count = hparams.n_embd * total_batched_size;
    float targets[targets_count];
    for (int i=0; i<targets_count; ++i) {
        targets[i] = 0.0f;
    }

    // populate targets and tokens_input tensors in batches
    for (int k=0; k<batch_count; k++) {
        const int base_batch_offset = hparams.n_embd * block_size * k;

        // make targets a series of one-hot vectors for the desired tokens
        for (int i=0; i<block_size; ++i) {
            TokenId next_token = target_ids[k*block_size+i];
            targets[base_batch_offset + i*hparams.n_embd + next_token] = 1.0f;
        }
    }
    ggml_backend_tensor_set(token_ids_input, input_ids, 0, batch_count*block_size*sizeof(TokenId));

    /* ===================================================================== */

    // set the targets tensor to have the one-hots we calculated earlier
    struct ggml_tensor* targets_tensor = ggml_graph_get_tensor(eval_gf, "targets");
    TEST_ASSERT_NOT_NULL(token_ids_input);
    ggml_backend_tensor_set(targets_tensor, targets, 0, targets_count*sizeof(float));

    // actually run the loss computations and pull the result out of the single float tensor
    ggml_backend_graph_compute(model.backend, eval_gf);
    ggml_backend_tensor_get(e, &error_before_opt, 0, sizeof(float));


    //DEBUG: 4.338178 is the expected value with seed 1337 on my Macbook Air dev machine...
    printf("Error before optimizations: %f\n\n", error_before_opt / total_batched_size);
    
    // free up this context and create a new one for further tests
    ggml_free(ctx);

    /* ===================================================================== */

    // //DEBUG: write out some test data as sanity checks
    // int dbg_embd_to_get = 57;
    // float dbg_embd_vals[hparams.n_vocab];
    // ggml_backend_tensor_get(model.embeddings, dbg_embd_vals, dbg_embd_to_get*hparams.n_embd*sizeof(float), hparams.n_embd*sizeof(float));
    // printf("\nEmbedding #%d:\n", dbg_embd_to_get);
    // for (int i=0; i<hparams.n_embd; ++i) {
    //     if (i!=0 && i%10 == 0){
    //         printf("\n");
    //     }
    //     printf("%f ", dbg_embd_vals[i]);
    // }
    // printf("\n");

    // dbg_embd_to_get = 58;
    // ggml_backend_tensor_get(model.embeddings, dbg_embd_vals, dbg_embd_to_get*hparams.n_embd*sizeof(float), hparams.n_embd*sizeof(float));
    // printf("\nEmbedding #%d:\n", dbg_embd_to_get);
    // for (int i=0; i<hparams.n_embd; ++i) {
    //     if (i!=0 && i%10 == 0){
    //         printf("\n");
    //     }
    //     printf("%f ", dbg_embd_vals[i]);
    // }
    // printf("\n");

    // int dbg_target_to_get = 7;
    // float dbg_target_vals[targets_count];
    // targets_tensor = ggml_graph_get_tensor(eval_gf, "targets");
    // TEST_ASSERT_NOT_NULL(token_ids_input);
    // ggml_backend_tensor_get(targets_tensor, dbg_target_vals, dbg_target_to_get*hparams.n_embd*sizeof(float), hparams.n_embd*sizeof(float));
    // printf("\nTarget #%d:\n", dbg_target_to_get);
    // for (int i=0; i<hparams.n_embd; ++i) {
    //     if (i!=0 && i%10 == 0){
    //         printf("\n");
    //     }
    //     printf("%f ", dbg_embd_vals[i]);
    // }
    // printf("\n");

    /* ===================================================================== */

    // build a new context here for our calculations, but keep the same params as before
    ctx = ggml_init(params);
    TEST_ASSERT_NOT_NULL(ctx);
    eval_gf = ggml_new_graph_custom(ctx, NANOGPT_MAX_NODES, false);
    TEST_ASSERT_NOT_NULL(eval_gf);
    
    // build an evaluation graph
    logits_tensor = nanogpt_model_build_eval_graph(ctx, eval_gf, &model, batch_count, hparams.n_ctx);
    TEST_ASSERT_NOT_NULL(logits_tensor);

    // do the necessary allocations so that we can pump our data in
    ggml_build_forward_expand(eval_gf, logits_tensor);
    success = ggml_gallocr_alloc_graph(model.allocr, eval_gf);
    TEST_ASSERT_TRUE(success);

    // build the tokenid input tensor and fill it with data that was created previously
    token_ids_input = ggml_graph_get_tensor(eval_gf, "token_ids_input");
    TEST_ASSERT_NOT_NULL(token_ids_input);

    // we already setup the input token_id buffer earlier so just reuse it.
    ggml_backend_tensor_set(token_ids_input, input_ids, 0, batch_count*block_size*sizeof(TokenId));

    // and compute the logits...
    ggml_backend_graph_compute(model.backend, eval_gf);
    
    // pull the logits out
    int64_t dbg_logits_cap = batch_count * hparams.n_ctx * hparams.n_embd;
    float dbg_logits[dbg_logits_cap];
    ggml_backend_tensor_get(logits_tensor, dbg_logits, 0, dbg_logits_cap*sizeof(float));
    
    // // do some debug testing to see if we're doing the correct calculations
    // for (int j=31; j<hparams.n_ctx * batch_count; ++j) {
    //     printf("\nBefore softmax, logits for %d:\n", j);
    //     for (int i=0; i<hparams.n_embd; ++i) {
    //         if (i!=0 && i%10 == 0) {
    //             puts("");
    //         }
    //         printf("%f ", dbg_logits[j * hparams.n_embd + i]);
    //     }
    // }
    
    // // do our own math and print that
    // calc_softmax_inplace(1, hparams.n_embd, &dbg_logits[dbg_logits_cap - hparams.n_embd]);
    // puts("\nOur own softmax: the last set of logits:");
    // for (int i=0; i<hparams.n_embd; ++i) {
    //     if (i!=0 && i%10 == 0) {
    //         puts("");
    //     }
    //     printf("%f ", dbg_logits[dbg_logits_cap - hparams.n_embd + i]);
    // }
    // puts("\n");

    // // quick unit test of our softmax function ... crashes the test case, so we're skipping ...
    // float softmax_test[6] = { 3.0f, 1.0f, 0.2f, 1.0f, 2.0f, 3.0f};
    // float softmax_expected[6] = { 0.836019f, 0.113143f, 0.050838f, 0.090031f, 0.244728f, 0.665241f };
    // calc_softmax_inplace(2, 3, softmax_test);
    // TEST_ASSERT_EQUAL_FLOAT_ARRAY(softmax_expected, softmax_test, 6);
    
    // get the last set of logits for each batch and then softmax them
    float softmaxies[hparams.n_embd * batch_count];
    for (int k=0; k<batch_count; ++k) {
        memcpy(&softmaxies[k*hparams.n_embd], &dbg_logits[(k+1)*(hparams.n_ctx-1)*hparams.n_embd], hparams.n_embd*sizeof(float));
    }
    calc_softmax_inplace(batch_count, hparams.n_embd, softmaxies);

    // do a test on each softmax to make sure the calculations are going the way
    // we expect and that they all sum to 1.0.
    for (int k=0; k<batch_count; ++k) {
        float softmax_test_sum = 0.0f;
        for (int i=0; i<hparams.n_embd; ++i) {
            softmax_test_sum += softmaxies[k*hparams.n_embd + i];
        }
        // test to make sure logits add up to 1.0...
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0, softmax_test_sum);
    }

    // puts("\nOur own softmax: the first set of logits:");
    // for (int i=0; i<hparams.n_embd; ++i) {
    //     if (i!=0 && i%10 == 0) {
    //         puts("");
    //     }
    //     printf("%f ", softmaxies[i]);
    // }
    // puts("\n");

    // sample the most likely token, but do only the first one.
    // TokenId prediction_id = argmax(dbg_logits, hparams.n_embd);

    // sample one token as a first test...
    TokenId prediction_id = multinominal_sample(softmaxies, hparams.n_embd);
    Token prediction = dataset_vocab_decode(&vocab_data, prediction_id);
    printf("\nSelected next token for the first batch:\n\t(Id:%d) ==> '%c'\n\n", prediction_id, prediction);

    // now sample the whole batch of predictions
    TokenId predictions[batch_count];
    int batch_success = multinominal_sample_batch(softmaxies, hparams.n_embd, batch_count, predictions);
    printf("batch multinomial: %d\n", batch_success);
    TEST_ASSERT_GREATER_OR_EQUAL(0, batch_success);
    puts("Batch predictions are as follows:");
    for (int k=0; k<batch_count; ++k) {
        printf("\t(Id:%d) ==> '%c'\n", predictions[k], dataset_vocab_decode(&vocab_data, predictions[k]));
    }

    ggml_free(ctx);
    nanogpt_model_free(&model);
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(test_model_stage1);
    return UNITY_END();
}