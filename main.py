"""
Inference: load a saved model and generate text from an optional prompt.
Model can be selected by path (--model) or by scenario (--scenario).
Fails immediately if the model file does not exist.

Run: python main.py --scenario names
     python main.py --scenario names -i "George"
     python main.py --model models/alice/model.json -i "Alice"
"""

import argparse
import os
import random
import sys

from config import load_config, resolve_model_path
from model import gpt, load_model, softmax


def resolve_model_path_from_args(args) -> str:
    """Resolve model path from --model or --scenario. Exits with error if file missing."""
    if args.model:
        path = args.model
    else:
        scenario = args.scenario or "names"
        try:
            load_config(scenario)  # ensure scenario exists
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        path = resolve_model_path(scenario)

    if not os.path.isfile(path):
        hint = (
            f"Train first with: python train.py --scenario {args.scenario or 'names'}"
            if not args.model
            else "Ensure the model file exists or train with: python train.py --scenario <name>"
        )
        print(f"Error: Model not found: {path}\n{hint}", file=sys.stderr)
        sys.exit(1)
    return path


def generate_one_run(
    prompt: str,
    data: dict,
    args: argparse.Namespace,
    keys: list,
    values: list,
) -> str:
    """Run one generation from prompt. Returns prompt + generated text. Mutates keys/values (KV cache)."""
    uchars = data["uchars"]
    BOS = data["BOS"]
    vocab_size = data["vocab_size"]
    state_dict = data["state_dict"]
    n_layer = data["n_layer"]
    n_head = data["n_head"]
    head_dim = data["head_dim"]
    block_size = data["block_size"]

    if prompt:
        prompt_ids = [uchars.index(ch) for ch in prompt if ch in uchars]
        context = [BOS] + prompt_ids
    else:
        context = [BOS]

    n_context = min(len(context), block_size)
    for pos_id in range(n_context - 1):
        gpt(context[pos_id], pos_id, keys, values, state_dict, n_layer, n_head, head_dim, block_size)

    token_id = context[n_context - 1]
    pos_id = n_context - 1
    generated = []
    max_gen = block_size - n_context
    if args.max_tokens is not None:
        max_gen = min(max_gen, args.max_tokens)
    for _ in range(max_gen):
        logits = gpt(token_id, pos_id, keys, values, state_dict, n_layer, n_head, head_dim, block_size)
        probs = softmax([l / args.temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            if not args.no_stop_at_bos:
                break
            continue  # skip appending BOS (no char for it); keep generating
        generated.append(uchars[token_id])
        pos_id += 1
    return (prompt if prompt else "") + "".join(generated)


def main():
    parser = argparse.ArgumentParser(
        description="Run micro GPT inference.",
        epilog='Use --scenario names or --model path. Use quotes for prompts with spaces.',
    )
    parser.add_argument("--input", "-i", default="", help="Prompt (default: none, generate from BOS)")
    parser.add_argument("--model", "-m", help="Path to saved model (e.g. models/names/model.json)")
    parser.add_argument("--scenario", "-s", help="Scenario name → models/<scenario>/model.json (default: names if no --model)")
    parser.add_argument("--samples", "-n", type=int, default=20, help="Number of samples")
    parser.add_argument("--temperature", "-t", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (reproducible)")
    parser.add_argument("--no-seed", action="store_true", help="No seed (different each run)")
    parser.add_argument("--no-stop-at-bos", action="store_true", help="Do not stop on BOS; generate until context full or --max-tokens")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max new tokens per sample (default: until BOS or block_size)")
    parser.add_argument("--chain", action="store_true", help="Chain segments: repeatedly use last block_size-1 chars as prompt until --max-chars reached")
    parser.add_argument("--max-chars", type=int, default=2000, help="Target total output length when using --chain (default: 2000)")
    args = parser.parse_args()

    if args.chain and args.max_chars <= 0:
        print("Error: --max-chars must be positive when using --chain", file=sys.stderr)
        sys.exit(1)
    if not args.model and not args.scenario:
        args.scenario = "names"

    model_path = resolve_model_path_from_args(args)

    if not args.no_seed:
        random.seed(args.seed)

    data = load_model(model_path)
    uchars = data["uchars"]
    BOS = data["BOS"]
    vocab_size = data["vocab_size"]
    state_dict = data["state_dict"]
    n_layer = data["n_layer"]
    n_head = data["n_head"]
    head_dim = data["head_dim"]
    block_size = data["block_size"]

    prompt = args.input.strip()
    print("--- inference ---")

    if args.chain:
        # Chain mode: repeatedly use last (block_size-1) chars as prompt until total length >= max_chars.
        chain_args = argparse.Namespace(
            **{**vars(args), "no_stop_at_bos": True, "max_tokens": None}
        )
        for sample_idx in range(args.samples):
            story = prompt
            current = prompt
            while len(story) < args.max_chars:
                segment_prompt = current[-(block_size - 1) :] if len(current) >= block_size - 1 else current
                keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
                output = generate_one_run(segment_prompt, data, chain_args, keys, values)
                generated_part = output[len(segment_prompt) :]
                if not generated_part:
                    break
                story += generated_part
                current = output[-(block_size - 1) :]
            print(f"sample {sample_idx+1:2d}: {story[: args.max_chars]}")
    else:
        for sample_idx in range(args.samples):
            keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
            out = generate_one_run(prompt, data, args, keys, values)
            print(f"sample {sample_idx+1:2d}: {out}")


if __name__ == "__main__":
    main()
