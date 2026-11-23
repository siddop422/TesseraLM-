from __future__ import annotations
import argparse
from inference.generate import load_model, generate_text
from model.gpt import count_parameters

def main():
    parser = argparse.ArgumentParser(description='Generate a short sample from a TesseraLM checkpoint')
    parser.add_argument('--checkpoint', default='checkpoints/nano.pt')
    parser.add_argument('--prompt', default='A language model learns')
    parser.add_argument('--max-new-tokens', type=int, default=80)
    parser.add_argument('--temperature', type=float, default=0.85)
    args = parser.parse_args()
    model, tokenizer, metadata, device = load_model(args.checkpoint)
    print(f"{metadata.get('model_config', {}).get('name', 'TesseraLM')} | {count_parameters(model):,} parameters | {device}")
    print(generate_text(model, tokenizer, args.prompt, max_new_tokens=max(1, min(200, args.max_new_tokens)), temperature=args.temperature))

if __name__ == '__main__':
    main()
