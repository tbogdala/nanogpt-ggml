#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include "unity.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "../dataset.h" 
#include "../model.h"
#include "../utility.h"

void setUp(void) {}
void tearDown(void) {}

void test_model_stage1() {
    const int thread_count = 1;
    int batch_count = 4;
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
    {
        // setup the model parameters
        struct nanogpt_model_hparams hparams;
        hparams.n_ctx = block_size;
        hparams.n_vocab = vocab_data.total_tokens;
        hparams.n_embd = hparams.n_vocab;

        // setup the nanogpt model
        struct nanogpt_model model;
        bool model_init_success = nanogpt_model_init(&model, hparams, thread_count, batch_count);
        if (!model_init_success) {
            fprintf(stderr, "%s: call to nanogpt_model_init() failed\n", __func__);
            return;
        }

        // setup an allocator
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        // build an evaluation graph
        struct ggml_cgraph* gf = nanogpt_model_build_eval_graph(&model, batch_count);

        // do the necessary allocations so that we can pump our data in
        bool success = ggml_gallocr_alloc_graph(allocr, gf);
        if (!success) {
            fprintf(stderr, "%s: call to ggml_gallocr_alloc_graph failed!.\n", __func__);
        }

        /* ===================================================================== */

        // now that the memory for the buffers has been allocated, randomize the model data for initialization
        nanogpt_model_randomize(&model);

        // make an array to use to zero out the initial targets
        int64_t targets_count = vocab_data.total_tokens * total_batched_size;
        float targets[targets_count];
        for (int i=0; i<targets_count; ++i) {
            targets[i] = 0.0f;
        }

        // populate targets and tokens_input tensors in batches
        for (int k=0; k<batch_count; k++) {
            struct ggml_tensor* tokens_input_k = ggml_view_1d(model.ctx,
                                                    model.token_ids_input,
                                                    model.token_ids_input->ne[0],
                                                    k*model.token_ids_input->nb[1]);

            int n_tokens = model.token_ids_input->ne[0];
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

        // run the calculation for the expanded graph
        ggml_backend_graph_compute(model.backend, gf);

        float error_before_opt = -42.0f;
        success = nanogpt_model_calculate_loss(
            &model,
            batch_count,
            gf,
            targets,
            targets_count,
            &error_before_opt);
        TEST_ASSERT_EQUAL(true, success);

        //DEBUG: 4.338178 is the expected value with seed 1337 on my Macbook Air dev machine...
        printf("Error before optimizations: %f\n\n", error_before_opt / total_batched_size);


        /* ===================================================================== */

        // logits should be [65 32] at this point for a batch count of 4 and a block size of 8.
        // create an array that can hold the results and initialize it to a recognizable pattern
        // to help detect uninitialization if inspected.
        int64_t last_logits_cap = batch_count * hparams.n_embd;
        float last_logits[last_logits_cap];
        for (int i=0; i<last_logits_cap; ++i) {
            last_logits[i] = -42.42f;
        }
        success = nanogpt_model_get_last_logits(
            &model,
            batch_count,
            gf,
            last_logits,
            last_logits_cap);
        TEST_ASSERT_EQUAL(true, success);

        // NOTE: some debugging info ...
        int dbg_batch_idx = 3;
        // puts("\nBefore softmax, the last set of logits:");
        // for (int i=0; i<hparams.n_embd; ++i) {
        //     if (i!=0 && i%10 == 0) {
        //         puts("");
        //     }
        //     printf("%f ", last_logits[dbg_batch_idx * hparams.n_embd + i]);
        // }
        // puts("\n");

        // get the  tensors again manually and check what we got back against
        // some of the raw tensors data.
        struct ggml_tensor * logits_tensor = ggml_graph_get_tensor(gf, "logits");
        TEST_ASSERT_NOT_NULL(logits_tensor);
        int debug_logits_count = batch_count * hparams.n_embd * hparams.n_ctx;
        float all_logits[debug_logits_count];
        ggml_backend_tensor_get(
                logits_tensor, 
                all_logits, 
                0, 
                batch_count * hparams.n_embd * hparams.n_ctx * sizeof(float));
        TEST_ASSERT_EQUAL_FLOAT(all_logits[debug_logits_count-1], last_logits[last_logits_cap-1]);
        TEST_ASSERT_EQUAL_FLOAT(all_logits[debug_logits_count-2], last_logits[last_logits_cap-2]);
        TEST_ASSERT_EQUAL_FLOAT(all_logits[debug_logits_count-3], last_logits[last_logits_cap-3]);

        // quick unit test of our softmax function
        float softmax_test[6] = { 3.0f, 1.0f, 0.2f, 1.0f, 2.0f, 3.0f};
        float softmax_expected[6] = { 0.836019f, 0.113143f, 0.050838f, 0.090031f, 0.244728f, 0.665241f };
        calc_softmax_inplace(2, 3, softmax_test);
        TEST_ASSERT_EQUAL_FLOAT_ARRAY(softmax_expected, softmax_test, 6);
        
        // softmax the logits
        calc_softmax_inplace(batch_count, hparams.n_embd, last_logits);
        float softmax_test_sum = 0.0f;
        for (int i=0; i<hparams.n_embd; ++i) {
            softmax_test_sum += last_logits[dbg_batch_idx * hparams.n_embd + i];
        }
        // test to make sure logits add up to 1.0...
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0, softmax_test_sum);

        //NOTE: some debugging info ...
        // puts("\nAfter softmax, the last set of logits:");
        // for (int i=0; i<hparams.n_embd; ++i) {
        //     if (i!=0 && i%10 == 0) {
        //         puts("");
        //     }
        //     printf("%f ", last_logits[dbg_batch_idx * hparams.n_embd + i]);
        // }
        // puts("\n");

        // sample the most likely token, but do only the first one.
        //TokenId prediction_id = argmax(last_logits, hparams.n_embd);
        TokenId prediction_id = multinominal_sample(last_logits, hparams.n_embd);
        
        Token prediction = dataset_vocab_decode(&vocab_data, prediction_id);
        printf("\nSelected next token for the first batch:\n\t(Id:%d) ==> '%c'\n\n", prediction_id, prediction);

        // now test the whole batch of predictions
        TokenId predictions[batch_count];
        int batch_success = multinominal_sample_batch(last_logits, hparams.n_embd, batch_count, predictions);
        TEST_ASSERT_GREATER_OR_EQUAL(0, batch_success);
        puts("Batch predictions are as follows:");
        for (int k=0; k<batch_count; ++k) {
            printf("\t(Id:%d) ==> '%c'\n", predictions[k], dataset_vocab_decode(&vocab_data, predictions[k]));
        }

        printf("\nReloading model ...\n\n");

        ggml_gallocr_free(allocr);
        nanogpt_model_free(&model);
    }

    /* ===================================================================== */

    ggml_time_init();
    
    // setup a new nanogpt model with a batch size of 1 for just simple text prediction
    struct nanogpt_model_hparams hparams;
    hparams.n_ctx = block_size;
    hparams.n_vocab = vocab_data.total_tokens;
    hparams.n_embd = hparams.n_vocab;
    batch_count = 1;

    struct nanogpt_model model;
    bool model_init_success = nanogpt_model_init(&model, hparams, thread_count, batch_count);
    if (!model_init_success) {
        fprintf(stderr, "%s: call to nanogpt_model_init() failed\n", __func__);
        return;
    }

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    struct ggml_cgraph* gf = nanogpt_model_build_eval_graph(&model, batch_count);
    bool success = ggml_gallocr_alloc_graph(allocr, gf);
    if (!success) {
        fprintf(stderr, "%s: call to ggml_gallocr_alloc_graph failed!.\n", __func__);
    }

    // now that the memory for the buffers has been allocated, randomize the model data for initialization
    nanogpt_model_randomize(&model);

    TokenId prompt[block_size];
    const char* prompt_str = "Hello...";
    success = dataset_vocab_encode_string(
        &vocab_data,
        prompt_str,
        prompt,
        block_size);
    TEST_ASSERT_EQUAL(true, success);

    puts("\nInput tokens:");
    for (int d=0; d<block_size; d++) {
        if (d != 0 && d % block_size == 0) {
            puts("");
        }
        printf("%d ", prompt[d]);
    }
    puts("\n");

    const int num_to_predict = 128;
    const int64_t t_predict_start_us = ggml_time_us();
    TokenId output_tokens[num_to_predict];
    success = nanogpt_model_predict_batch(
        &model,
        &vocab_data,
        1, // batch_count
        num_to_predict,
        allocr,
        prompt,
        block_size,
        output_tokens,
        num_to_predict);
    TEST_ASSERT_EQUAL(true, success);
    const int64_t t_predict_end_us = ggml_time_us();
    const float t_predict_ms = (t_predict_end_us - t_predict_start_us)/1000.0f;

    char predicted_str[num_to_predict + 1];
    memset(predicted_str, 0, (num_to_predict+1) * sizeof(char));
    dataset_vocab_decode_string(&vocab_data, output_tokens, predicted_str, num_to_predict);
    TEST_ASSERT_EQUAL(num_to_predict, strlen(predicted_str));

    printf("Final prompt + prediction:\n%s%s\n\n", prompt_str, predicted_str);
    printf("Predicting %d tokens ==>\n\ttotal time = %8.2f ms / %.2f ms per token / %.3f tokens per second\n\n", 
        num_to_predict, 
        t_predict_ms, 
        t_predict_ms/num_to_predict,
        1000.0f/(t_predict_ms/num_to_predict));

    // free up our allocations
    ggml_gallocr_free(allocr);
    nanogpt_model_free(&model);
}


int main() {
    UNITY_BEGIN();
    RUN_TEST(test_model_stage1);
    return UNITY_END();
}