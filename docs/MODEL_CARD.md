# TesseraLM-Nano model card

TesseraLM-Nano is a very small decoder-only Transformer for education and local experimentation. It is not an instruction-tuned assistant and is not suitable for production or high-stakes use.

## Architecture
- Decoder-only Transformer with three pre-normalized blocks.
- Causal multi-head self-attention.
- GELU feed-forward network.
- Learned absolute position embeddings.
- LayerNorm and residual connections.
- Tied token embedding and output projection weights.
- Byte-level BPE tokenizer trained by this repository.

## Training
The initial research experiment used 101 optimizer steps on a small project-authored demonstration corpus. Repository clones can reproduce a fresh training run using examples/seed_corpus.txt; this run validates the pipeline, not broad language competence.

## Intended use
Study tokenization, next-token prediction, neural-network training, checkpoint loading, and local CPU inference.

## Limitations
Tiny training corpus, small parameter count, short training run, no instruction tuning, no preference optimization, no formal safety evaluation, and no task benchmark evaluation. Generations may be incoherent, repetitive, or factually wrong.

## Data and licensing
The included seed corpus is authored for this project. Source code and corpus are MIT licensed. Do not mix third-party datasets into a training run unless their license permits your intended use.
