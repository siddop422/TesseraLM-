import torch
from model.gpt import TesseraGPT, count_parameters
from tokenizer.bpe import BPETokenizer

def test_tokenizer_round_trip():
    text = 'Hello, TesseraLM! नमस्ते'
    tokenizer = BPETokenizer.train(text * 4, vocab_size=280)
    assert tokenizer.decode(tokenizer.encode(text)) == text

def test_model_forward_backward():
    model = TesseraGPT(vocab_size=280, d_model=32, n_layers=1, n_heads=4, max_seq_len=16, dropout=0.0)
    x = torch.randint(0, 280, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 280)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert count_parameters(model) > 0
