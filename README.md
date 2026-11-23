# TesseraLM

**An experimental decoder-only Transformer that trains and runs locally.**

TesseraLM is an open-source research project with its own tokenizer, Transformer implementation, CPU training pipeline, checkpoint loader, text generator, and browser chat interface. It does not call ChatGPT, Claude, Gemini, Ollama, or a hosted inference API.

> **Model status:** TesseraLM-Nano is a very small experimental language model, not a capable general-purpose assistant. It was trained for 101 optimizer steps on the included 7.6 KB demonstration corpus. Expect weak, repetitive, or nonsensical generations.

## Browser chat

Run the local server and open the ChatGPT-style interface:

1. Install Python 3.11 or newer.
2. Run `install.bat`.
3. Run `start.bat`.
4. Your browser opens at **http://127.0.0.1:8765**.

The browser talks to the Python inference server on loopback. The HTML page alone does not contain the neural network; the Python process loads the actual model checkpoint and performs generation.

Features include conversation history in browser storage, streaming responses, stop generation, regeneration, copying, Markdown/code formatting, export to Markdown, adjustable sampling settings, and a local model status indicator.

## Train from scratch

Run `train.bat`. The default Nano configuration is intended for small CPU experiments. Training writes a checkpoint under `checkpoints/`. For larger experiments, use a machine with more RAM and CPU/GPU capacity.

## Project layout

- `model/` — decoder-only Transformer and model configuration
- `tokenizer/` — trainable byte-pair tokenizer
- `training/` — dataset preparation and training loop
- `inference/` — checkpoint loading and autoregressive generation
- `quantization/` — experimental dynamic INT8 helper
- `ui/` — local HTTP/SSE server and browser chat interface
- `configs/` — model-size presets
- `examples/` — tiny demonstration training corpus
- `tests/` — automated tests

## Hardware

CPU inference is supported. On very low-memory systems, close other applications and use short responses. Training quality and speed are strongly limited by the tiny corpus and small number of training steps. The included 4 GB preset is experimental and may still be slow on older Celeron-class processors.

## Verification

Run:

```bash
python -m pytest -q
```

Run the browser server without opening a browser:

```bash
python -m ui.server --no-browser
```

Health information is available at `/health`; streamed chat generation uses `POST /api/chat`.

## Model and dataset disclosure

- Architecture: decoder-only Transformer with causal self-attention and learned positional embeddings.
- Tokenizer: project-local BPE implementation.
- Training data: `examples/seed_corpus.txt`, a short project-authored demonstration corpus.
- Training steps: 101 optimizer steps in the packaged experiment.
- Intended use: education and experimentation, not production deployment or factual assistance.

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [docs/TRAINING.md](docs/TRAINING.md).

## License

MIT. See [LICENSE](LICENSE).
