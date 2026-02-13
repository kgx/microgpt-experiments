# MicroGPT experiments

A minimal GPT in pure Python - no PyTorch, no dependencies. I built this to mess around and actually understand how the thing works.

**Architecture:** [docs/architecture.md](docs/architecture.md)

**Credit:** Based on [Andrej Karpathy's microgpt gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). What an amazing contribution.

---

## Layout

- **configs/** — Scenario configs (e.g. `names`, `alice`). Each defines data path, model size, and training settings.
- **data/** — Training data (not versioned). Populated by training or download scripts.
- **models/** — Trained model checkpoints (not versioned). One per scenario: `models/<scenario>/model.json`.
- **scripts/** — Helpers (e.g. download Alice's Adventures in Wonderland for the alice scenario).

## Scenarios

| Scenario   | Data                    | Use case                          |
|-----------|-------------------------|------------------------------------|
| **names** | One name per line       | Original micro GPT; name-like output. |
| **alice** | Alice's Adventures in Wonderland (chunked) | First step toward language-style text. |

## Commands

**Train**

```bash
# Names (downloads data if missing; saves to models/names/model.json)
python train.py --scenario names

# Alice (requires data first: run scripts/download_alice.py)
python scripts/download_alice.py
python train.py --scenario alice

# Optional: faster training with NumPy backend (pip install microgpt[numpy] or uv sync --extra numpy)
python train.py --scenario alice --backend numpy

# Further speed on CPU: float32, gradient accumulation, batched backend
python train.py --scenario alice_v2 --backend numpy --dtype float32
python train.py --scenario alice_v2 --backend numpy --dtype float32 --grad-accum 4
python train.py --scenario alice_v2 --backend numpy_batched   # batched-over-positions (often faster)
```

**Inference**

```bash
# By scenario (fails if model not trained)
python main.py --scenario names
python main.py --scenario names -i "George"
python main.py --scenario alice -i "Alice"

# By explicit model path
python main.py --model models/names/model.json -i "A"

# Longer / continuous generation (e.g. for prose models trained with trailing_bos: false)
python main.py --scenario alice_v2 -i "Alice" --no-stop-at-bos --max-tokens 100
python main.py --scenario alice_v2 -i "Alice" --chain --max-chars 500 -n 1
```

Inference options: `--no-stop-at-bos` (do not stop on BOS; use with length-based stopping), `--max-tokens N` (cap new tokens per sample), `--chain` (repeatedly extend by using the last `block_size-1` chars as the next prompt until `--max-chars` is reached).

If the model file does not exist, inference exits immediately with an error (fail-fast).

**Tests**

```bash
pytest tests/ -v
```

(Use `python -m pytest tests/ -v` or `.venv/bin/pytest tests/ -v` if needed.)

**Benchmarking alice (with NumPy + Numba)**

Install the optional backend (numpy, numba, scipy), then run a short benchmark. Use `--backend numpy` for the fast path.

```bash
# One-time: install optional deps (numpy, numba, scipy)
uv sync --extra numpy

# Benchmark: a few steps, no model saved (first step may be slow due to JIT)
python scripts/benchmark_train.py --scenario alice --steps 5 --backend numpy
```

Compare with the default Python backend:

```bash
python scripts/benchmark_train.py --scenario alice --steps 2 --backend python
```

The script prints steps/sec and estimated time for a full 2000-step run. The NumPy+Numba backend is typically an order of magnitude faster than pure Python (e.g. ~12 min vs ~7 h for alice).

**Verify Numba JIT is active**

```bash
python scripts/verify_numba_jit.py
```

You should see `JIT verification: numba JIT active (... compiled)`. To print the same check on the first training step, run with `MICROGPT_VERIFY_JIT=1`.

**Further acceleration (numpy backend, before GPU)**

- **`--dtype float32`** — Use single precision. Often ~1.5–2× faster on CPU (less memory bandwidth, better SIMD). Config: `training.dtype: "float32"` in your scenario JSON.
- **`--grad-accum N`** — Accumulate gradients over N docs per optimizer step. Fewer optimizer steps for the same number of docs; can reduce total time when optimizer is a noticeable fraction of step cost. Only supported with the numpy backend.
- **`--backend numpy_batched`** — Same as `numpy` but with batched-over-positions forward/backward (one sequence-wide pass). Same math, typically faster; tested for numerical equivalence with `numpy`.

**Using more cores**

- The fused attention kernel uses Numba’s `parallel=True` and `prange` over heads, so it can use multiple CPU cores. No extra config is needed.
- NumPy’s matrix ops use BLAS when available; thread count is often controlled by `OPENBLAS_NUM_THREADS` or `OMP_NUM_THREADS` (e.g. `export OMP_NUM_THREADS=4`). Setting this too high can slow things down due to overhead.

**Equivalence**

The numpy backend is tested for numerical equivalence with the python backend (same init, same steps, losses within tolerance). Run: `pytest tests/test_backends.py -v`.

## Alice scenario

The **alice** scenario uses *Alice's Adventures in Wonderland* (Project Gutenberg). Data is chunked into 128-character segments (with overlap). Each chunk is tokenized as `[BOS, ...chars..., BOS]`; see [docs/architecture.md](docs/architecture.md) for how **BOS** (start/end-of-sequence token) works and why it is a better fit for document-style data (e.g. names) than for chunked prose. Download the text once:

```bash
python scripts/download_alice.py
```

Then train with a larger context and a slightly bigger model (see `configs/alice.json`).

**alice_v2** (`configs/alice_v2.json`) uses the same data but with `trailing_bos: false` so the model learns continuation across chunks, and 5000 steps for better coherence. Train with `python train.py --scenario alice_v2` (or `--backend numpy` for speed). At inference use `--no-stop-at-bos` and optionally `--chain --max-chars N` for longer output.
