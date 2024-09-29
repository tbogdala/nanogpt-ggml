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

void test_model_stage1_prediction() {
    ggml_time_init();

    int thread_count = 1;
    int batch_count = 1;
    int block_size = 8;

    // randomize the embeddings
    unsigned int seed = 1337;
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


    // setup a new nanogpt model with a batch size of 1 for just simple text prediction
    struct nanogpt_model_hparams hparams;
    hparams.n_ctx = block_size;
    hparams.n_vocab = vocab_data.total_tokens;
    hparams.n_embd = hparams.n_vocab;
    
    // setup the nanogpt model
    struct nanogpt_model model;
    bool success = nanogpt_model_init(&model, hparams, thread_count);
    TEST_ASSERT_TRUE(success);

    // now that the memory for the buffers has been allocated, randomize the model data for initialization
    nanogpt_model_randomize(&model);

    TokenId prompt[block_size];
    const char* prompt_str = "\n";
    success = dataset_vocab_encode_string(
        &vocab_data,
        prompt_str,
        prompt,
        block_size);
    TEST_ASSERT_TRUE(success);

    const int64_t t_predict_start_us = ggml_time_us();

    // establish our test sizes and create a token buffer
    const int prompt_len = strlen(prompt_str);
    const int num_to_predict =  1000;
    const int predict_buf_size = prompt_len + num_to_predict;
    TokenId predict_buffer[predict_buf_size];

    // initialize the token buffer with our prompt
    memset(predict_buffer, 0, sizeof(TokenId) * predict_buf_size);
    for (int i=0; i<strlen(prompt_str); ++i) {
        predict_buffer[i] = prompt[i];
    }

    success = nanogpt_model_predict_batch(
        &model,
        &vocab_data,
        batch_count,
        predict_buf_size,
        prompt_len,
        predict_buffer);
    TEST_ASSERT_TRUE(success);

    const int64_t t_predict_end_us = ggml_time_us();
    const float t_predict_ms = (t_predict_end_us - t_predict_start_us)/1000.0f;

    // decode all of the tokens in the buffer and print out our generated text
    char predicted_str[predict_buf_size + 1];
    memset(predicted_str, 0, (predict_buf_size+1) * sizeof(char));
    dataset_vocab_decode_string(&vocab_data, predict_buffer, predicted_str, predict_buf_size);
    TEST_ASSERT_EQUAL(predict_buf_size, strlen(predicted_str));

    printf("Final prompt + prediction:\n%s\n\n", predicted_str);
    printf("Predicting %d tokens ==>\n\ttotal time = %8.2f ms / %.2f ms per token / %.3f tokens per second\n\n", 
        num_to_predict, 
        t_predict_ms, 
        t_predict_ms/num_to_predict,
        1000.0f/(t_predict_ms/num_to_predict));

    // free up our allocations
    nanogpt_model_free(&model);
}


int main() {
    UNITY_BEGIN();
    RUN_TEST(test_model_stage1_prediction);    
    return UNITY_END();
}