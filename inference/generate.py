from __future__ import annotations
from pathlib import Path
import torch
from model.gpt import TesseraGPT
from tokenizer.bpe import BPETokenizer


def load_model(checkpoint_path: str | Path, device: str | None = None):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Train a model first.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("format") != "tesseralm-checkpoint-v1":
        raise ValueError("Unsupported checkpoint format")
    cfg = ckpt["model_config"]
    model = TesseraGPT(**{k: cfg[k] for k in ("vocab_size", "d_model", "n_layers", "n_heads", "max_seq_len", "dropout")})
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    tok_path = path.parent / ckpt.get("tokenizer_path", path.with_suffix(".tokenizer.json").name)
    tokenizer = BPETokenizer.load(tok_path)
    return model, tokenizer, ckpt, device


def sample_next(logits: torch.Tensor, temperature: float = 0.85, top_k: int = 40,
                top_p: float = 0.9, repetition_penalty: float = 1.08,
                recent: list[int] | None = None) -> int:
    logits = logits.float().clone()
    if recent and repetition_penalty > 1:
        for token in set(recent[-64:]):
            if 0 <= token < logits.numel():
                logits[token] = logits[token] / repetition_penalty if logits[token] > 0 else logits[token] * repetition_penalty
    logits = logits / max(temperature, 1e-4)
    if top_k > 0:
        threshold = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits[logits < threshold] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    if 0 < top_p < 1:
        sorted_probs, indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cumulative - sorted_probs > top_p] = 0
        probs = torch.zeros_like(probs).scatter_(0, indices, sorted_probs)
        probs = probs / probs.sum().clamp_min(1e-12)
    return int(torch.multinomial(probs, 1).item())


@torch.inference_mode()
def generate_tokens(model, prompt_ids: list[int], max_new_tokens: int = 100,
                    temperature: float = 0.85, top_k: int = 40,
                    top_p: float = 0.9, repetition_penalty: float = 1.08):
    device = next(model.parameters()).device
    ids = list(prompt_ids) or [1]
    for _ in range(max_new_tokens):
        context = ids[-model.max_seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits, _ = model(x)
        nxt = sample_next(logits[0, -1], temperature, top_k, top_p, repetition_penalty, ids)
        ids.append(nxt)
        yield nxt
        if nxt == 2:
            break


def generate_text(model, tokenizer, prompt: str, **kwargs) -> str:
    prompt_ids = tokenizer.encode(prompt)
    return tokenizer.decode(list(generate_tokens(model, prompt_ids, **kwargs)))
