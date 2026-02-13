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
    args = parser.parse_args()

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
    if prompt:
        prompt_ids = [uchars.index(ch) for ch in prompt if ch in uchars]
        context = [BOS] + prompt_ids
    else:
        context = [BOS]

    print("--- inference ---")
    for sample_idx in range(args.samples):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        n_context = min(len(context), block_size)

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
