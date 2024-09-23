#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "unity.h"
#include "../dataset.h" 

void setUp(void) {}
void tearDown(void) {}


void test_dataset() {
    // parse the test dataset, which should be predictable and spot
    // check some of the results.
    struct dataset_vocab vocab_data;
    int ret = dataset_vocab_from_file(&vocab_data, "data/shakes.txt");
    TEST_ASSERT_EQUAL(0, ret);
    TEST_ASSERT_EQUAL(65, vocab_data.total_tokens);
    TEST_ASSERT_EQUAL(47, vocab_data.ttoi['i']);
    TEST_ASSERT_EQUAL('i', vocab_data.itot[47]);

    // check to make sure data not found in the dataset is still unset
    TEST_ASSERT_EQUAL(-1, vocab_data.ttoi['*']);
    TEST_ASSERT_EQUAL('*', vocab_data.itot[100]);

    // check the encode and decode functions
    TEST_ASSERT_EQUAL(32, dataset_vocab_encode(&vocab_data, 'T'));
    TEST_ASSERT_EQUAL(64, dataset_vocab_encode(&vocab_data, 'z'));
    TEST_ASSERT_EQUAL('y', dataset_vocab_decode(&vocab_data, 63));
    TEST_ASSERT_EQUAL(' ', dataset_vocab_decode(&vocab_data, 1));
    
    // print the results and track the total
    printf("\n\nFound %d tokens:\n", vocab_data.total_tokens);

    // print the tokens out for visibility
    for (int i=0; i<vocab_data.total_tokens; ++i) {
        Token tok = vocab_data.itot[i];
        // here's a more thorough printout, but we'll keep it easy instead
        // printf("\ttoken %d: %c\n", i, tok);
        printf("%c", tok);
    }
    puts("\n");

    // encode the whole dataset into TokenIds...
    TokenId* token_buffer;
    int64_t token_buffer_count;
    ret = dataset_vocab_tokenize_file(&vocab_data, "data/shakes.txt", &token_buffer, &token_buffer_count);
    TEST_ASSERT_EQUAL(0, ret);
    TEST_ASSERT_EQUAL(1115394, token_buffer_count);

    // decode a little chunk at the start and ensure that we get the 
    // string that we expect back from the token ids...
    const int roundtrip_len = 24;
    char roundtripped_text[roundtrip_len+1];
    printf("TokenId buffer length for the whole dataset: %lld\n\nFirst %d:\n[", token_buffer_count, roundtrip_len);
    for (int u=0; u<roundtrip_len; ++u) {
        printf("%d ", token_buffer[u]);
        roundtripped_text[u] = dataset_vocab_decode(&vocab_data, token_buffer[u]);
    }
    puts("]\n");
    roundtripped_text[roundtrip_len] = 0;
    printf("Decoded:\n%s\n\n", roundtripped_text);
    TEST_ASSERT_EQUAL_STRING("First Citizen:\nBefore we", roundtripped_text);

    // make sure to deallocate the buffer created for the token ids
    free(token_buffer);
}


int main() {
    UNITY_BEGIN();
    RUN_TEST(test_dataset);
    return UNITY_END();
}