# Architecture

TesseraLM is a compact decoder-only Transformer.

```text
UTF-8 text
  -> byte-level BPE tokenizer
  -> token IDs + learned position embeddings
  -> repeated Transformer blocks
       LayerNorm -> causal multi-head self-attention -> residual
       LayerNorm -> GELU feed-forward network -> residual
  -> final LayerNorm
  -> tied vocabulary projection
  -> next-token logits
  -> temperature / top-k / top-p sampling
  -> generated text
```

The attention implementation uses PyTorch scaled dot-product attention when available, with a causal-mask fallback. The model is intentionally small and uses learned absolute position embeddings. The UI is a local browser client; Python loads the checkpoint and executes inference.
