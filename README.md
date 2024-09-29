# Nanogpt-ggml

Building GPT2 from scratch, in code, with Karpathy's guidance and using C and [GGML](https://github.com/ggerganov/ggml).

The idea is to build a transformer using GGML starting with character level tokenization on 
a sample dataset (Shakespeare text) and building up from there. Eventually, commits that
represent certain progress points will be listed here for anyone else that wants to follow along.

The source video: https://www.youtube.com/watch?v=kCc8FmEb1nY

Platform support: Only tested on CPU MacOS builds for now.


## Current Progress For This Commit

Still in the bigram language model phase... and current up to 34m 50s in the source youtube video.
The model is still untrained, but random garbage can be predicted based on the randomized
initialization of the embedding tensor.

All the code that tests the initial logits prediction is in `test_model_stage1.c`. 

GGML does cross_entropy a little different and requires tensors of the same dimensions, 
so a one-hot table is made for the target token id to be used for the loss calculations.


## Building the source

Generically, you can compile like this which will automatically detect Metal. Standard
config flags to compile features into GGML can be used and CUDA support should
theoretically work that way, but is untested.

```bash
cmake -B build -DGGML_METAL=Off
cmake --build build --config Release
./build/bin/test_model_stage1
```

**NOTE**: Metal has to be disabled because it doesn't support the functions
necessary to enable loss calculations for the training process.

As of right now, only the CPU backend is being tested ...


## Commit History Markers

Once I start piling up commits, this section will house a list of commits
that follow the progression of Karpathy's video.


## Licensed

Released under the MIT license. See the `LICENSE` file for further details.


## Notes:

* `data/shakes.txt` was downloaded from `https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt`
* Dataset is not split into 'eval' and 'training' sets