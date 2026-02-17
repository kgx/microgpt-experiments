"""
Inference: load a saved model and generate text from an optional prompt.
Model can be selected by path (--model) or by scenario (--scenario).
Fails immediately if the model file does not exist.

Run: python main.py --scenario names
     python main.py --scenario names -i "George"
     python main.py --model models/alice/model.json -i "Alice"
"""

import argparse
import math
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


def softmax_numeric(logits: list) -> list:
    """Stable softmax over raw floats; returns list of probabilities (for inference sampling)."""
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def sample_with_options(
    logits: list,
    temperature: float,
    repetition_penalty: float,
    repetition_window: int,
    sequence_ids: list,
    top_k: int,
    vocab_size: int,
) -> int:
    """Apply temperature, repetition penalty, top-k, then sample one token id. logits: list of Value or list of float."""
    logits_data = [getattr(l, "data", l) for l in logits]
    # Temperature
    if temperature != 1.0:
        logits_data = [x / temperature for x in logits_data]
    # Repetition penalty: reduce logits for tokens that appeared recently
    if repetition_penalty > 1.0 and sequence_ids:
        window_ids = sequence_ids[-repetition_window:]
        for tid in window_ids:
            if 0 <= tid < len(logits_data):
                # Penalize: divide logit so probability drops (avoid log(0))
                if logits_data[tid] > 0:
                    logits_data[tid] /= repetition_penalty
                else:
                    logits_data[tid] *= repetition_penalty
    # Top-k: keep only the k largest logits
    if top_k > 0 and top_k < vocab_size:
        k = min(top_k, vocab_size)
        threshold = sorted(logits_data, reverse=True)[k - 1]
        logits_data = [x if x >= threshold else -1e10 for x in logits_data]
    probs = softmax_numeric(logits_data)
    return random.choices(range(vocab_size), weights=probs)[0]


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
    n_layer = data["n_layer"]
    n_head = data["n_head"]
    head_dim = data["head_dim"]
    block_size = data["block_size"]
    use_pytorch = data.get("backend") == "pytorch"

    if prompt:
        prompt_ids = [uchars.index(ch) for ch in prompt if ch in uchars]
        context = [BOS] + prompt_ids
    else:
        context = [BOS]

    n_context = min(len(context), block_size)
    if use_pytorch:
        from backends.pytorch_backend import gpt_step_pytorch
        for pos_id in range(n_context - 1):
            gpt_step_pytorch(context[pos_id], pos_id, keys, values, data)
    else:
        state_dict = data["state_dict"]
        for pos_id in range(n_context - 1):
            gpt(context[pos_id], pos_id, keys, values, state_dict, n_layer, n_head, head_dim, block_size)

    token_id = context[n_context - 1]
    pos_id = n_context - 1
    sequence_ids = list(context[:n_context])
    generated = []
    max_gen = block_size - n_context
    if args.max_tokens is not None:
        max_gen = min(max_gen, args.max_tokens)
    rep_penalty = getattr(args, "repetition_penalty", 1.0)
    rep_window = getattr(args, "repetition_window", 32)
    top_k = getattr(args, "top_k", 0)
    for _ in range(max_gen):
        if use_pytorch:
            from backends.pytorch_backend import gpt_step_pytorch
            logits = gpt_step_pytorch(token_id, pos_id, keys, values, data)
        else:
            state_dict = data["state_dict"]
            logits = gpt(token_id, pos_id, keys, values, state_dict, n_layer, n_head, head_dim, block_size)
        token_id = sample_with_options(
            logits,
            args.temperature,
            rep_penalty,
            rep_window,
            sequence_ids,
            top_k,
            vocab_size,
        )
        sequence_ids.append(token_id)
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
        epilog="Use --scenario names or --model path. For a semi-coherent short story, use --chain --max-chars N (loops by using last block as prompt).",
    )
    parser.add_argument("--input", "-i", default="", help="Prompt (default: none, generate from BOS)")
    parser.add_argument("--model", "-m", help="Path to saved model (e.g. models/names/model.json)")
    parser.add_argument("--scenario", "-s", help="Scenario name → models/<scenario>/model.json (default: names if no --model)")
    parser.add_argument("--samples", "-n", type=int, default=20, help="Number of samples")
    parser.add_argument("--temperature", "-t", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="Penalize recent tokens (default 1.2; 1.0 = off)")
    parser.add_argument("--repetition-window", type=int, default=32, help="Window size for repetition penalty (default 32)")
    parser.add_argument("--top-k", type=int, default=40, help="Sample only from top-k tokens (default 40; 0 = off)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (reproducible)")
    parser.add_argument("--no-seed", action="store_true", help="No seed (different each run)")
    parser.add_argument("--no-stop-at-bos", action="store_true", help="Do not stop on BOS; generate until context full or --max-tokens")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max new tokens per sample (default: until BOS or block_size)")
    parser.add_argument("--chain", action="store_true", help="Chain segments: repeatedly extend by using last N chars as prompt until --max-chars reached")
    parser.add_argument("--max-chars", type=int, default=2000, help="Target total output length when using --chain (default: 2000)")
    parser.add_argument("--chain-context", type=int, default=None, help="Chars to keep as next prompt in --chain (default: block_size//2 so each segment can generate many tokens)")
    parser.add_argument("--backend", "-b", default="python", choices=("python", "pytorch"), help="Inference backend (default: python; pytorch for GPU)")
    parser.add_argument("--device", help="Device for pytorch backend (e.g. cpu, cuda, mps; default: cuda if available else cpu)")
    args = parser.parse_args()

    if args.chain and args.max_chars <= 0:
        print("Error: --max-chars must be positive when using --chain", file=sys.stderr)
        sys.exit(1)
    if not args.model and not args.scenario:
        args.scenario = "names"

    model_path = resolve_model_path_from_args(args)

    if not args.no_seed:
        random.seed(args.seed)

    if args.backend == "pytorch":
        try:
            from backends.pytorch_backend import load_model_pytorch
            data = load_model_pytorch(model_path, device=args.device)
        except ImportError:
            print("Warning: PyTorch not available (pip install microgpt[pytorch]), falling back to python backend.", file=sys.stderr)
            data = load_model(model_path)
    else:
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
        # Chain mode: repeatedly use last N chars as prompt so we have room to generate (block_size - N - 1 new tokens per segment).
        # Default N = block_size//2 so each segment adds many tokens; use --chain-context to override.
        context_len = args.chain_context if args.chain_context is not None else block_size // 2
        context_len = min(context_len, block_size - 1)
        chain_args = argparse.Namespace(
            **{**vars(args), "no_stop_at_bos": True, "max_tokens": None}
        )
        for sample_idx in range(args.samples):
            story = prompt
            current = prompt
            while len(story) < args.max_chars:
                segment_prompt = current[-context_len:] if len(current) >= context_len else current
                keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
                output = generate_one_run(segment_prompt, data, chain_args, keys, values)
                generated_part = output[len(segment_prompt):]
                if not generated_part:
                    break
                story += generated_part
                current = output  # next segment_prompt = current[-context_len:]
            print(f"sample {sample_idx+1:2d}: {story[: args.max_chars]}")
    else:
        for sample_idx in range(args.samples):
            keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
            out = generate_one_run(prompt, data, args, keys, values)
            print(f"sample {sample_idx+1:2d}: {out}")


if __name__ == "__main__":
    main()
