"""A small trainable byte-level BPE tokenizer with deterministic merges."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path


class BPETokenizer:
    PAD, BOS, EOS, UNK = 0, 1, 2, 3
    BASE = 4

    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = list(merges or [])
        self._rebuild_bytes()

    @property
    def vocab_size(self) -> int:
        return self.BASE + 256 + len(self.merges)

    def _rebuild_bytes(self):
        self.token_bytes = [b"", b"", b"", b""] + [bytes([i]) for i in range(256)]
        for left, right in self.merges:
            if left >= len(self.token_bytes) or right >= len(self.token_bytes):
                raise ValueError("Invalid BPE merge reference")
            self.token_bytes.append(self.token_bytes[left] + self.token_bytes[right])

    @classmethod
    def train(cls, text: str, vocab_size: int = 512, min_frequency: int = 2):
        if vocab_size < cls.BASE + 256:
            raise ValueError("vocab_size must be at least 260")
        sequences = [[b + cls.BASE for b in text.encode("utf-8")]]
        merges = []
        while cls.BASE + 256 + len(merges) < vocab_size:
            counts = Counter()
            for seq in sequences:
                counts.update(zip(seq, seq[1:]))
            if not counts:
                break
            pair, frequency = max(counts.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]))
            if frequency < min_frequency:
                break
            new_id = cls.BASE + 256 + len(merges)
            merges.append(pair)
            replaced = []
            for seq in sequences:
                out, i = [], 0
                while i < len(seq):
                    if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                replaced.append(out)
            sequences = replaced
        return cls(merges)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [b + self.BASE for b in text.encode("utf-8")]
        for rank, (left, right) in enumerate(self.merges):
            new_id = self.BASE + 256 + rank
            out, i = [], 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == (left, right):
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        if add_bos:
            ids.insert(0, self.BOS)
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: list[int]) -> str:
        payload = b"".join(self.token_bytes[int(idx)] for idx in ids if 0 <= int(idx) < len(self.token_bytes))
        return payload.decode("utf-8", errors="replace")

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = {"format": "tesseralm-byte-bpe-v1",
               "special_tokens": {"pad": 0, "bos": 1, "eos": 2, "unk": 3},
               "merges": self.merges, "vocab_size": self.vocab_size}
        path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("format") != "tesseralm-byte-bpe-v1":
            raise ValueError("Unsupported tokenizer format")
        return cls([tuple(map(int, pair)) for pair in raw["merges"]])
