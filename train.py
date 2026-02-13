"""
Train micro GPT by scenario. Loads config from configs/<scenario>.json,
resolves data path, runs training, saves to models/<scenario>/model.json.

Run: python train.py --scenario names
     python train.py --scenario literary
     python train.py --config configs/custom.json --output models/custom/model.json
"""

import argparse
import math
import os
import random

from config import load_config, resolve_data_path, resolve_model_path
from model import (
    gpt,
    init_state_dict,
    save_model,
    softmax,
)


def load_docs(config: dict) -> list[str]:
    """Load documents from data config. Handles 'lines' (one doc per line) and 'chunked' (fixed-size chunks)."""
    data_cfg = config["data"]
    path = resolve_data_path(config)
    fmt = data_cfg.get("format", "lines")

    if fmt == "lines":
        if not os.path.isfile(path):
            url = data_cfg.get("download_url")
            if url:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                import urllib.request
                urllib.request.urlretrieve(url, path)
            else:
                raise FileNotFoundError(f"Data file not found: {path}")
        with open(path) as f:
            docs = [line.strip() for line in f.read().strip().split("\n") if line.strip()]
        random.shuffle(docs)
        return docs

    if fmt == "chunked":
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Data file not found: {path}\n"
                "For the alice scenario, run: python scripts/download_alice.py"
            )
        with open(path) as f:
            text = f.read()
        # Normalize: single spaces, strip
        text = " ".join(text.split())
        chunk_size = data_cfg.get("chunk_size", 128)
        overlap = data_cfg.get("chunk_overlap", chunk_size // 2)
        step = max(1, chunk_size - overlap)
        docs = [text[i : i + chunk_size] for i in range(0, len(text) - chunk_size + 1, step)]
        docs = [d for d in docs if len(d.strip()) >= chunk_size // 2]
        random.shuffle(docs)
        return docs

    raise ValueError(f"Unknown data format: {fmt}")


def run_training(config: dict, output_path: str) -> None:
    """Run one training run from config and save to output_path."""
    random.seed(42)
    docs = load_docs(config)
    print(f"num docs: {len(docs)}")

    model_cfg = config["model"]
    train_cfg = config["training"]
    n_embd = model_cfg["n_embd"]
    n_head = model_cfg["n_head"]
    n_layer = model_cfg["n_layer"]
    block_size = model_cfg["block_size"]
    learning_rate = train_cfg["learning_rate"]
    beta1 = train_cfg["beta1"]
    beta2 = train_cfg["beta2"]
    eps_adam = train_cfg["eps_adam"]
    num_steps = train_cfg["num_steps"]

    uchars = sorted(set("".join(docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    head_dim = n_embd // n_head
    print(f"vocab size: {vocab_size}")

    state_dict, params = init_state_dict(vocab_size, n_embd, n_layer, block_size)
    print(f"num params: {len(params)}")

    m = [0.0] * len(params)
    v = [0.0] * len(params)

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc if ch in uchars] + [BOS]
        if len(tokens) < 2:
            continue
        n = min(block_size, len(tokens) - 1)

        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values, state_dict, n_layer, n_head, head_dim, block_size)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        loss = (1 / n) * sum(losses)

        loss.backward()

        lr_t = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_model(output_path, state_dict, uchars, n_embd, n_head, n_layer, block_size)
    print(f"Saved model to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train micro GPT by scenario.",
        epilog="Scenarios: names (default), alice. For alice, run scripts/download_alice.py first.",
    )
    parser.add_argument("--scenario", "-s", default="names", help="Scenario name (configs/<scenario>.json)")
    parser.add_argument("--config", "-c", help="Path to config JSON (overrides --scenario)")
    parser.add_argument("--output", "-o", help="Output model path (default: models/<scenario>/model.json)")
    args = parser.parse_args()

    if args.config:
        if not os.path.isfile(args.config):
            print(f"Error: Config file not found: {args.config}")
            exit(1)
        with open(args.config) as f:
            import json
            config = json.load(f)
        scenario_name = config.get("name", "custom")
        output_path = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", scenario_name, "model.json")
    else:
        try:
            config = load_config(args.scenario)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            exit(1)
        output_path = args.output or resolve_model_path(args.scenario)

    run_training(config, output_path)


if __name__ == "__main__":
    main()
