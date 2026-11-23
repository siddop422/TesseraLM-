import torch
from model.gpt import TesseraGPT
from tokenizer.bpe import BPETokenizer
from inference.generate import load_model, generate_tokens

def test_checkpoint_roundtrip(tmp_path):
    tok = BPETokenizer.train('this is a small corpus. ' * 20, vocab_size=280)
    config = {'vocab_size': tok.vocab_size, 'd_model': 32, 'n_layers': 2, 'n_heads': 4, 'max_seq_len': 16, 'dropout': 0.0}
    model = TesseraGPT(**config)
    tok.save(tmp_path / 'mini.tokenizer.json')
    checkpoint = tmp_path / 'mini.pt'
    torch.save({'format':'tesseralm-checkpoint-v1','model_config':config,'model_state_dict':model.state_dict(),'tokenizer_path':'mini.tokenizer.json','training':{'steps':0}}, checkpoint)
    loaded, loaded_tok, meta, device = load_model(checkpoint, 'cpu')
    assert loaded_tok.decode(loaded_tok.encode('small test')) == 'small test'
    assert meta['format'] == 'tesseralm-checkpoint-v1'
    assert device == 'cpu'
    generated = list(generate_tokens(loaded, loaded_tok.encode('tiny'), max_new_tokens=2, temperature=0.8))
    assert 0 < len(generated) <= 2
    assert all(0 <= token < tok.vocab_size for token in generated)
