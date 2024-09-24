// make an alias for the token datatype for readability
typedef char Token;
typedef int32_t TokenId;

// Supporting only ASCII which has 256 characters
#define MAX_TOKEN_VALUES 256 

struct dataset_vocab {
    Token itot[MAX_TOKEN_VALUES];
    TokenId ttoi[MAX_TOKEN_VALUES];
    TokenId total_tokens;
};

// initialize the structure with default values
void 
dataset_vocab_init(
    struct dataset_vocab* vocab_data);

// populates the `vocab_data` structure supplied by reading the `dataset_filepath`
// speicifed file. builds the vocabulary for tokenizing and detokenizing.
// returns non-zero on error.
int 
dataset_vocab_from_file(
    struct dataset_vocab* vocab_data, 
    const char* dataset_filepath);

// for a given `token` return its TokenId. if the returned TokenId is negative,
// then a matching token for the `token` was not found.
TokenId 
dataset_vocab_encode(
    const struct dataset_vocab* vocab_data, 
    Token token);

// for a given token 'id' return its Token.
Token 
dataset_vocab_decode(
    const struct dataset_vocab* vocab_data, 
    TokenId id);

// for a given buffer of TokenIds, populate a null terminated char string.
// NOTE: it's the caller's responsibility to make sure (`count` + 1), for NULL terminator,
// does not overflow either `token_id_buffer` or `output_buffer`.
void 
dataset_vocab_decode_string(
    const struct dataset_vocab* vocab_data, 
    const TokenId* token_id_buffer,
    char* output_buffer,
    int64_t count);


// for a given file, specified with `filepath`, load the file, read in all
// of the text, allocate a buffer and populate it with the token ids.
// this buffer is then assigned to `token_id_buffer` and the size allocated
// to it is placed in `token_id_buffer_size`.
//
// NOTE: the caller is responsible for calling `free()` on `*token_id_buffer`
// when the caller is finished with it.
int
dataset_vocab_tokenize_file(
    const struct dataset_vocab* vocab_data,
    const char* filepath,
    TokenId** token_id_buffer,
    int64_t* token_id_buffer_size);


// build the input and target arrays. caller knows the batch size and the
// block size, so it is ASSUMED that `inputs` and `targets` are big enough
// to handle all the data.
void
dataset_build_input_and_target_arrays(
    TokenId* token_buffer,
    int64_t token_buffer_count,
    int batch_size,
    int block_size,
    TokenId* inputs,
    TokenId* targets);