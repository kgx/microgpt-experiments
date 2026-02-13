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
import time

from config import load_config, resolve_data_path, resolve_model_path
from model import save_model
from backends import get_backend


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


def _format_eta(seconds: float) -> str:
    """Format seconds as human-readable ETA (e.g. 26h 23m or 1m 30s)."""
    if seconds <= 0 or not math.isfinite(seconds):
        return "?"
    s = int(round(seconds))
    if s >= 3600:
        h, rest = divmod(s, 3600)
        m = rest // 60
        return f"{h}h {m}m"
    if s >= 60:
        m, sec = divmod(s, 60)
        return f"{m}m {sec}s"
    return f"{s}s"


def run_training(
    config: dict,
    output_path: str,
    show_eta: bool = True,
    show_timing: bool = False,
    max_steps: int | None = None,
    save: bool = True,
    backend_name: str | None = None,
    backend_kwargs: dict | None = None,
    grad_accum_steps: int = 1,
) -> None:
    """Run one training run from config and save to output_path.
    If max_steps is set, run at most that many steps. If save is False, do not write the model (for benchmarking).
    backend_name overrides config["training"]["backend"] when set (e.g. from --backend).
    backend_kwargs: passed to backend.create_state() (e.g. {"dtype": "float32"} for numpy).
    grad_accum_steps: accumulate gradients over this many docs per optimizer step (reduces steps, can speed up).
    """
    random.seed(42)
    docs = load_docs(config)
    print(f"num docs: {len(docs)}")

    model_cfg = config["model"]
    train_cfg = config["training"]
    n_embd = model_cfg["n_embd"]
    n_head = model_cfg["n_head"]
    n_layer = model_cfg["n_layer"]
    block_size = model_cfg["block_size"]
    num_steps = train_cfg["num_steps"]

    uchars = sorted(set("".join(docs)))
    vocab_size = len(uchars) + 1
    print(f"vocab size: {vocab_size}")

    backend = get_backend(backend_name or train_cfg.get("backend", "python"))
    state = backend.create_state(config, uchars, **(backend_kwargs or {}))
    params = state["params"]
    num_params = sum(p.size for p in params) if params and hasattr(params[0], "size") else len(params)
    print(f"num params: {num_params}")

    start_time = time.perf_counter()
    min_steps_for_eta = 2
    steps_limit = num_steps if max_steps is None else min(num_steps, max_steps)
    backend_impl = backend_name or train_cfg.get("backend", "python")
    accum = max(1, grad_accum_steps) if backend_impl in ("numpy", "numpy_batched") else 1

    for step in range(steps_limit):
        step_loss_sum = 0.0
        step_timing = {"forward": 0.0, "backward": 0.0, "optimizer": 0.0}
        for sub in range(accum):
            doc = docs[(step * accum + sub) % len(docs)]
            loss, timing = backend.run_one_step(
                step * accum + sub,
                doc,
                state,
                uchars,
                zero_grad=(sub == 0),
                do_optimizer=(sub == accum - 1),
                grad_accum_count=accum,
            )
            step_loss_sum += loss
            step_timing["forward"] += timing["forward"]
            step_timing["backward"] += timing["backward"]
            step_timing["optimizer"] += timing["optimizer"]
        avg_loss = step_loss_sum / accum

        if (step + 1) % 100 == 0 or step == 0:
            elapsed = time.perf_counter() - start_time
            steps_done = step + 1
            steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
            remaining = steps_limit - steps_done
            eta_str = _format_eta(remaining / steps_per_sec) if (show_eta and steps_done >= min_steps_for_eta and steps_per_sec > 0) else ""
            line = f"step {steps_done:4d} / {steps_limit:4d} | loss {avg_loss:.4f}"
            if steps_per_sec > 0:
                line += f" | {steps_per_sec:.4f} step/s"
            if eta_str:
                line += f" | ETA {eta_str}"
            if show_timing:
                t_forward, t_backward, t_optimizer = step_timing["forward"], step_timing["backward"], step_timing["optimizer"]
                line += f" | fwd {t_forward:.2f}s bwd {t_backward:.2f}s opt {t_optimizer:.2f}s"
            print(line)

    elapsed_total = time.perf_counter() - start_time
    if not save:
        steps_per_sec = steps_limit / elapsed_total if elapsed_total > 0 else 0
        full_eta = _format_eta((num_steps - steps_limit) / steps_per_sec) if steps_per_sec > 0 and num_steps > steps_limit else "N/A"
        print(f"Benchmark: {steps_limit} steps in {elapsed_total:.2f}s ({steps_per_sec:.4f} step/s). Full run ({num_steps} steps) would take ~{full_eta}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    state_dict_export = backend.weights_for_export(state)
    save_model(output_path, state_dict_export, uchars, n_embd, n_head, n_layer, block_size)
    print(f"Saved model to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train micro GPT by scenario.",
        epilog="Scenarios: names (default), alice. For alice, run scripts/download_alice.py first.",
    )
    parser.add_argument("--scenario", "-s", default="names", help="Scenario name (configs/<scenario>.json)")
    parser.add_argument("--config", "-c", help="Path to config JSON (overrides --scenario)")
    parser.add_argument("--output", "-o", help="Output model path (default: models/<scenario>/model.json)")
    parser.add_argument("--no-eta", action="store_true", help="Do not show ETA in progress")
    parser.add_argument("--timing", action="store_true", help="Show per-step breakdown (forward/backward/optimizer)")
    parser.add_argument("--backend", "-b", help="Training backend (default: python, or config training.backend)")
    parser.add_argument("--dtype", choices=("float32", "float64"), help="NumPy backend only: float32 for faster CPU (default: float64)")
    parser.add_argument("--grad-accum", type=int, default=1, metavar="N", help="Accumulate gradients over N docs per optimizer step (numpy / numpy_batched; default 1)")
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

    backend_kwargs = {}
    if args.dtype:
        backend_kwargs["dtype"] = args.dtype
    run_training(
        config,
        output_path,
        show_eta=not args.no_eta,
        show_timing=args.timing,
        backend_name=args.backend,
        backend_kwargs=backend_kwargs if backend_kwargs else None,
        grad_accum_steps=max(1, args.grad_accum),
    )


if __name__ == "__main__":
    main()
