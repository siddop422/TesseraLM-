# Training

1. Install Python 3.10–3.13 (64-bit) and run install.bat.
2. Run train.bat to train the default Nano experiment.
3. The trainer writes checkpoints/nano.pt and its paired checkpoints/nano.tokenizer.json.
4. Run start.bat to load that checkpoint and open the local browser chat.

Manual command:

```bash
python -m training.train --data examples/seed_corpus.txt --config configs/nano.json --steps 101 --output checkpoints/nano.pt
```

The sample corpus is intentionally short and authored for this project. The experiment validates the pipeline but does not make a capable general-purpose chatbot. For meaningful language quality, use a substantially larger properly licensed corpus, more compute, instruction tuning, and held-out evaluation.
