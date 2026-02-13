"""
Run a short training benchmark to measure steps/sec and estimate full-run time.
Uses the same code path as train.py but runs only N steps and does not save the model.

Examples:
  python scripts/benchmark_train.py --scenario alice --steps 5
  python scripts/benchmark_train.py --scenario alice --steps 5 --backend numpy
"""

import argparse
import os
import sys

# Project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from train import run_training
from config import load_config, resolve_model_path


def main():
    parser = argparse.ArgumentParser(description="Benchmark training for a scenario (no model saved).")
    parser.add_argument("--scenario", "-s", default="alice", help="Scenario name (default: alice)")
    parser.add_argument("--steps", "-n", type=int, default=5, help="Number of steps to run (default: 5)")
    parser.add_argument("--backend", "-b", default="python", help="Backend: python or numpy (default: python)")
    args = parser.parse_args()

    try:
        config = load_config(args.scenario)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    output_path = resolve_model_path(args.scenario)
    run_training(
        config,
        output_path,
        show_eta=True,
        show_timing=True,
        max_steps=args.steps,
        save=False,
        backend_name=args.backend,
    )


if __name__ == "__main__":
    main()
