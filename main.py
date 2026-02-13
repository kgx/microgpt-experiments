"""
Inference: load a saved model and generate text from an optional prompt.
Run: python main.py --input "your prompt here"
Use quotes around the prompt if it contains spaces.
"""

import argparse
import random

from model import gpt, load_model, softmax


def main():
    parser = argparse.ArgumentParser(
        description="Run micro GPT inference with an optional prompt.",
        epilog='Use quotes for prompts with spaces: python main.py --input "hello world"',
    )
    parser.add_argument(
        "--input", "-i",
        default="",
        help="Prompt string to condition generation (default: empty)",
    )
    parser.add_argument(
        "--model", "-m",
        default="model.json",
        help="Path to saved model (default: model.json)",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=20,
        help="Number of samples (default: 20, like original)",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.5,
        help="Sampling temperature, higher = more random (default: 0.5, like original)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible output (default: 42, like original)",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Do not set random seed (different output each run)",
    )
    args = parser.parse_args()

    if not args.no_seed:
        random.seed(args.seed)

    data = load_model(args.model)
    uchars = data["uchars"]
    BOS = data["BOS"]
    vocab_size = data["vocab_size"]
    state_dict = data["state_dict"]
    n_layer = data["n_layer"]
    n_head = data["n_head"]
    head_dim = data["head_dim"]
    block_size = data["block_size"]

    # Tokenize prompt: only chars in vocab; unknown chars are skipped
    prompt = args.input.strip()
    if prompt:
        prompt_ids = [uchars.index(ch) for ch in prompt if ch in uchars]
        context = [BOS] + prompt_ids
    else:
        context = [BOS]

    print("--- inference ---")
    for sample_idx in range(args.samples):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        n_context = min(len(context), block_size)
        # Fill KV cache: run forward for context positions 0 .. n_context-2
        for pos_id in range(n_context - 1):
            gpt(context[pos_id], pos_id, keys, values, state_dict, n_layer, n_head, head_dim, block_size)
        token_id = context[n_context - 1]
        pos_id = n_context - 1
        generated = []
        for _ in range(block_size - n_context):
            logits = gpt(token_id, pos_id, keys, values, state_dict, n_layer, n_head, head_dim, block_size)
            probs = softmax([l / args.temperature for l in logits])
            token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if token_id == BOS:
                break
            generated.append(uchars[token_id])
            pos_id += 1
        out = (prompt if prompt else "") + "".join(generated)
        print(f"sample {sample_idx+1:2d}: {out}")


if __name__ == "__main__":
    main()
