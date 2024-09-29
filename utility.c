#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "dataset.h"
#include "utility.h"

void
calc_softmax_inplace(
    int batches,
    int embd_size,
    float* logits)
{
    for (int k=0; k<batches; ++k) {
        int offset = k*embd_size;
        float sum = 0.0;

        // Calculate the exponential of each logit and sum them up
        for (int i=0; i<embd_size; ++i) {
            float calc = exp(logits[offset + i]);
            logits[offset + i] = calc;
            sum += calc;
        }

        // Normalize the output probabilities by dividing each exponential value by their sum
        for (int i=0; i<embd_size; ++i) {
            logits[offset + i] /= sum;
        }
    }
}

int 
argmax(
    const float* probabilities, 
    int probabilities_count) 
{
    int max_index = 0;
    float max_value = probabilities[0];

    for (int i = 1; i < probabilities_count; i++) {
        if (probabilities[i] > max_value) {
            max_index = i;
            max_value = probabilities[i];
        }
    }

    return max_index;
}

TokenId 
multinominal_sample(
    const float* probabilities, 
    int probabilities_count)
{
    float random_value = (float)rand() / RAND_MAX;
    float cumulative_sum = 0.0f;
    for (int i=0; i<probabilities_count; ++i) {
        cumulative_sum += probabilities[i];
        if (random_value <= cumulative_sum) {
            return i;
        }
    }
    
    return -1;
}

int
multinominal_sample_batch(
    const float* probabilities, 
    int probabilities_count,
    int batch_count,
    TokenId* out_predictions)
{
    for (int k=0; k<batch_count; ++k) {
        const float* probs_batch = &probabilities[k*probabilities_count];
        TokenId prediction = multinominal_sample(probs_batch, probabilities_count);
        if (prediction < 0) {
            return -1;
        } 
        out_predictions[k] = prediction;
    }

    return batch_count;
}