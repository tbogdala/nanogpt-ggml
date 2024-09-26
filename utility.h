#pragma once 

#include "dataset.h"

// calculates softmax for all of the `batches` of `logits` which
// should each be `embd_size` floats wide. this is done in-place
// and the values in `logits` will be modified.
void
calc_softmax_inplace(
    int batches,
    int embd_size,
    float* logits);

// returns the index of the highest probability in `probabilities`.
// `probabilities_count` should be the number of floats pointed 
// to by `probabilities`.
int 
argmax(
    const float* probabilities, 
    int probabilities_count);

// perform multinomial sampling on a given probability distribution by
// selecting an index based on the given probabilities. the chosen index
// is returned as the matching TokenId. returns -1 on error.
TokenId 
multinominal_sample(
    const float* probabilities, 
    int probabilities_count);

// repeatedly calls `multinominal_sample` with each 'batch' of probabilites
// from `probabilities`, where each 'batch' is `probabilities_count` in size.
// does this for a total of `batch_count` and writes the TokenId to 
// `out_preidictions` which MUST be able to hold at least `batch_count` TokenId values.
int
multinominal_sample_batch(
    const float* probabilities, 
    int probabilities_count,
    int batch_count,
    TokenId* out_predictions);