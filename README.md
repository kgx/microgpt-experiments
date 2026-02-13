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
```

**Inference**

```bash
# By scenario (fails if model not trained)
python main.py --scenario names
python main.py --scenario names -i "George"
python main.py --scenario alice -i "Alice"

# By explicit model path
python main.py --model models/names/model.json -i "A"
```

If the model file does not exist, inference exits immediately with an error (fail-fast).

**Tests**

```bash
pytest tests/ -v
```

(Use `python -m pytest tests/ -v` or `.venv/bin/pytest tests/ -v` if needed.)

## Alice scenario

The **alice** scenario uses *Alice's Adventures in Wonderland* (Project Gutenberg). Download the text once:

```bash
python scripts/download_alice.py
```

Then train with a larger context and a slightly bigger model (see `configs/alice.json`).
