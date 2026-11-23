# TesseraLM

**A small, experimental decoder-only Transformer with a local ChatGPT-style browser interface.**

TesseraLM contains its own Transformer architecture, byte-level BPE tokenizer, CPU training pipeline, checkpoint loader, autoregressive sampler, and local browser chat UI. It does not call ChatGPT, Claude, Gemini, Ollama, or a hosted inference API.

> **Important model limitation:** TesseraLM-Nano is a tiny research model, not a capable general-purpose assistant. The initial experiment trained 661,120 parameters for 101 optimizer steps on the included 7.6 KB project-authored corpus. It has not been instruction-tuned or broadly evaluated, so its responses may be incoherent, repetitive, or incorrect.

## Run the ChatGPT-style HTML interface (Windows)

1. Install 64-bit Python 3.11–3.14.
2. Double-click install.bat once. It creates a virtual environment and installs CPU PyTorch.
3. Double-click start.bat. If there is no checkpoint yet, it trains a fresh Nano checkpoint from the included sample corpus, then starts the local model server and opens the browser.
4. The chat opens at http://127.0.0.1:8765. If it does not open automatically, enter that address in your browser.

The HTML page is the chat client; the Python process loads the actual neural-network weights and generates tokens. The page alone cannot run the model. Prompts remain on your computer because the server binds to loopback by default.

Features: conversation history, new chat, search and delete chats, streamed generation, stop, regenerate, copy, Markdown/code display, export to Markdown, temperature and sampling controls, and live model status. On mobile, the history sidebar becomes a slide-out panel.

## Train

Run train.bat, or use:

```bash
python -m training.train --data examples/seed_corpus.txt --config configs/nano.json --steps 101 --output checkpoints/nano.pt
```

The model and tokenizer are written to checkpoints/nano.pt and checkpoints/nano.tokenizer.json. Keep those files together. The included run verifies the training/inference pipeline; it is not intended to demonstrate broad language ability.

## Test

Install the development requirements and run:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Architecture

- Decoder-only Transformer with causal multi-head self-attention.
- Pre-normalized residual blocks, GELU feed-forward layers, learned position embeddings.
- Tied token embedding and vocabulary projection.
- Trainable byte-level BPE tokenizer.
- Temperature, top-k, top-p, and repetition-penalty sampling.
- Optional dynamic INT8 quantization helper for CPU Linear layers.

## Hardware and limitations

CPU inference is supported. On a 4 GB RAM computer, PyTorch itself can consume substantial memory; close other applications and use short generations. A tiny model can be trained for pipeline experiments on a CPU, but meaningful multi-billion-parameter pretraining requires much more compute, data, and memory.

The bundled seed corpus is original project-authored demonstration text. Any additional training dataset must be reviewed for licensing and provenance before use. Do not use the model for medical, legal, financial, or other high-stakes decisions.

## Project layout

- model/ — Transformer implementation
- tokenizer/ — byte-level BPE tokenizer
- training/ — tokenizer training and model training loop
- inference/ — checkpoint loading and token generation
- quantization/ — dynamic INT8 helper
- ui/ — local HTML chat and loopback server
- configs/ — Nano architecture preset
- examples/ — project-authored sample training corpus
- tests/ and docs/ — smoke tests and documentation

## License

MIT. See LICENSE.
