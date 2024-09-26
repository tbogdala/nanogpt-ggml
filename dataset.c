#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include "dataset.h"


void dataset_vocab_init(struct dataset_vocab* vocab_data)
{
    memset(vocab_data->itot, '*', MAX_TOKEN_VALUES * sizeof(Token));
    memset(vocab_data->ttoi, -1, MAX_TOKEN_VALUES * sizeof(TokenId));
    vocab_data->total_tokens = 0;
}

int dataset_vocab_from_file(struct dataset_vocab* vocab_data, const char* dataset_filepath) 
{
    char ch;
    FILE* dataset_file = NULL;

    // open the dataset text file for reading
    dataset_file = fopen(dataset_filepath, "r");
    if (dataset_file == NULL) {
        printf("Could not open the dataset file: %s\n", dataset_filepath);
        return 1;
    }

    // reinitialize the token vocabulary data
    dataset_vocab_init(vocab_data);

    // setup our results array where each token value that is found
    // in the dataset will be set to non-zero.
    char* token_presence = (char*) calloc(MAX_TOKEN_VALUES, sizeof(Token)); 

    // read the file character by character and build up the token presence array
    while ((ch = fgetc(dataset_file)) != EOF) {
        // use the ASCII character as the token index to see if it's been counted
        if (token_presence[ch] == 0) {
            // we found a new token, so set its presence in the array
            token_presence[ch] = 1;
        }
    }
    fclose(dataset_file);
    
    int total_tokens_found = 0;
    for (int i=0; i<MAX_TOKEN_VALUES; ++i) {
       if (token_presence[i] > 0) {
            // 'i' here is the ASCII value of the character
            vocab_data->itot[total_tokens_found] = i;
            vocab_data->ttoi[i] = total_tokens_found;
            total_tokens_found += 1;
       }
    }
    vocab_data->total_tokens = total_tokens_found;

    free(token_presence);
    return 0;
}

TokenId dataset_vocab_encode(const struct dataset_vocab* vocab_data, Token token)
{
    return vocab_data->ttoi[token];
}

Token dataset_vocab_decode(const struct dataset_vocab* vocab_data, TokenId id)
{
    return vocab_data->itot[id];
}

bool 
dataset_vocab_encode_string(
    const struct dataset_vocab* vocab_data, 
    const char* input_string,
    TokenId* token_id_buffer,
    int64_t token_id_buffer_size) 
{
    size_t len = strlen(input_string);
    if (len > token_id_buffer_size) {
        return false;
    }

    // clear the buffer
    memset(token_id_buffer, 0, token_id_buffer_size*sizeof(TokenId));

    for (int i=0; i<len; ++i) {
        token_id_buffer[i] = vocab_data->ttoi[input_string[i]];
    }    

    return true;
}

void 
dataset_vocab_decode_string(
    const struct dataset_vocab* vocab_data, 
    const TokenId* token_id_buffer,
    char* output_buffer,
    int64_t count)
{
    for (int i=0; i<count; i++) {
        output_buffer[i] = vocab_data->itot[token_id_buffer[i]];
    }
    output_buffer[count] = 0;
}

int
dataset_vocab_tokenize_file(
    const struct dataset_vocab* vocab_data,
    const char* filepath,
    TokenId** token_id_buffer,
    int64_t* token_id_buffer_size)
{
    FILE* dataset_file = NULL;

    // open the dataset text file for reading
    dataset_file = fopen(filepath, "r");
    if (dataset_file == NULL) {
        printf("Could not open the dataset file: %s\n", filepath);
        return 1;
    }

    // get the size of the dataset file in characters
    fseek(dataset_file, 0, SEEK_END);
    int64_t dataset_size = ftell(dataset_file);
    fseek(dataset_file, 0, SEEK_SET);

    // allocate the buffer
    *token_id_buffer = (TokenId*) calloc(dataset_size, sizeof(TokenId));
    *token_id_buffer_size = dataset_size;

    // now tokenize the whole dataset
    // for now, this is a character-level tokenization process
    char ch;
    int ret_value = 0;
    int64_t i = 0;
    while ((ch = fgetc(dataset_file)) != EOF) {
        if (i >= *token_id_buffer_size) {
            printf("ERROR: exceeded length of token id buffer at i=%lld\n", i);
            ret_value = 2;
            break;
        }
        TokenId id = dataset_vocab_encode(vocab_data, ch);
        (*token_id_buffer)[i++] = id;
    }

    fclose(dataset_file);
    return ret_value;
}

void
dataset_build_input_and_target_arrays(
    TokenId* token_buffer,
    int64_t token_buffer_count,
    int batch_size,
    int block_size,
    TokenId* inputs,
    TokenId* targets)
{
    // we subract one here to account for the lookahead while making
    // the `targets` buffer.
    int64_t max_index = token_buffer_count - block_size - 1;

    for (int b=0; b<batch_size; ++b) {
        int64_t random_offset = rand() % max_index;
        for (int i=0; i<block_size; ++i) {
            inputs[b*block_size + i] = token_buffer[random_offset+i];
            targets[b*block_size + i] = token_buffer[random_offset+i+1];
        }
    }
}