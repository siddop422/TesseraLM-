from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
import torch
from model.gpt import TesseraGPT, count_parameters
from tokenizer.bpe import BPETokenizer


def _load_text(path: str) -> str:
    p = Path(path)
    files = sorted(p.rglob("*.txt")) if p.is_dir() else [p]
    chunks = [f.read_text(encoding="utf-8", errors="replace") for f in files]
    text = "\n\n".join(chunks).replace("\x00", " ").strip()
    seen, unique = set(), []
    for part in text.split("\n\n"):
        part = part.strip()
        if part and part not in seen:
            seen.add(part)
            unique.append(part)
    text = "\n\n".join(unique)
    if len(text) < 200:
        raise ValueError("Training corpus must contain at least 200 unique characters")
    return text


def train(data: str, config_path: str, steps: int, output: str):
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    text = _load_text(data)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out.with_suffix(".tokenizer.json")
    tokenizer = BPETokenizer.train(text, int(cfg.get("target_vocab_size", 400)))
    tokenizer.save(tokenizer_path)
    ids = tokenizer.encode(text)
    split = max(2, int(len(ids) * 0.95))
    train_ids, val_ids = ids[:split], ids[split:]
    if len(train_ids) < 130:
        raise ValueError("Corpus is too small for the selected sequence length")
    cfg["vocab_size"] = tokenizer.vocab_size
    cfg["name"] = cfg.get("name", "TesseraLM-Nano")
    model = TesseraGPT(**{k: cfg[k] for k in ("vocab_size", "d_model", "n_layers", "n_heads", "max_seq_len", "dropout")})
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    seq_len = min(int(cfg["max_seq_len"]), len(train_ids) - 2)
    batch_size = int(cfg.get("batch_size", 8))
    best = float("inf")
    started = time.time()
    print(f"Training {cfg['name']} | {count_parameters(model):,} parameters | {len(ids):,} tokens | CPU")
    for step in range(1, steps + 1):
        model.train()
        starts = [random.randint(0, len(train_ids) - seq_len - 2) for _ in range(batch_size)]
        x = torch.tensor([train_ids[i:i+seq_len] for i in starts], dtype=torch.long)
        y = torch.tensor([train_ids[i+1:i+seq_len+1] for i in starts], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 20 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                source = val_ids if len(val_ids) > seq_len + 1 else train_ids
                start_i = random.randint(0, len(source) - seq_len - 2)
                vx = torch.tensor([source[start_i:start_i+seq_len]], dtype=torch.long)
                vy = torch.tensor([source[start_i+1:start_i+seq_len+1]], dtype=torch.long)
                _, val_loss = model(vx, vy)
            best = min(best, float(val_loss))
            print(f"step {step}/{steps} | train {float(loss):.4f} | validation {float(val_loss):.4f} | {time.time()-started:.1f}s")
    checkpoint = {
        "format": "tesseralm-checkpoint-v1", "model_config": cfg,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "tokenizer_path": tokenizer_path.name,
        "training": {"steps": steps, "step": steps, "seed": 42, "dataset": data,
                     "dataset_characters": len(text), "token_count": len(ids),
                     "best_validation_loss": best, "device": "cpu"}
    }
    torch.save(checkpoint, out)
    print(f"Saved {out} ({out.stat().st_size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Train TesseraLM-Nano from text")
    parser.add_argument("--data", default="examples/seed_corpus.txt")
    parser.add_argument("--config", default="configs/nano.json")
    parser.add_argument("--steps", type=int, default=101)
    parser.add_argument("--output", default="checkpoints/nano.pt")
    args = parser.parse_args()
    train(args.data, args.config, args.steps, args.output)


if __name__ == "__main__":
    main()
