"""Tests for training backends: equivalence of python vs numpy (when numpy is available)."""

import random

import pytest

from backends import get_backend
from model import state_dict_to_json


def _minimal_config():
    """Small config for fast equivalence check."""
    return {
        "name": "test",
        "data": {"path": "data/names/input.txt", "format": "lines"},
        "model": {"n_embd": 8, "n_head": 2, "n_layer": 1, "block_size": 8},
        "training": {
            "learning_rate": 0.01,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps_adam": 1e-8,
            "num_steps": 10,
        },
    }


@pytest.fixture
def config_and_docs():
    from train import load_docs
    config = _minimal_config()
    random.seed(42)
    docs = load_docs(config)
    # Use short docs so each step is fast
    docs = [d for d in docs if 2 <= len(d) <= 6][:20]
    return config, docs


def test_python_backend_runs(config_and_docs):
    """Python backend runs a few steps without error."""
    config, docs = config_and_docs
    uchars = sorted(set("".join(docs)))
    backend = get_backend("python")
    state = backend.create_state(config, uchars)
    for step in range(3):
        doc = docs[step % len(docs)]
        loss, _ = backend.run_one_step(step, doc, state, uchars)
    assert loss >= 0


def test_numpy_backend_available():
    """Numpy backend is registered when numpy is installed."""
    try:
        get_backend("numpy")
    except KeyError:
        pytest.skip("numpy backend not available (pip install microgpt[numpy])")


def test_numpy_backend_equivalence(config_and_docs):
    """NumPy backend (if available) matches Python backend loss for same init and steps."""
    numpy = pytest.importorskip("numpy")
    config, docs = config_and_docs
    uchars = sorted(set("".join(docs)))
    num_steps = 3

    # Python: deterministic init, export weights before any steps
    random.seed(42)
    py_backend = get_backend("python")
    py_state = py_backend.create_state(config, uchars)
    init_weights = state_dict_to_json(py_state["state_dict"])

    py_losses = []
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        loss, _ = py_backend.run_one_step(step, doc, py_state, uchars)
        py_losses.append(loss)

    # NumPy: same initial weights (frozen copy), same docs
    np_backend = get_backend("numpy")
    np_state = np_backend.create_state(config, uchars, init_from=init_weights)
    np_losses = []
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        loss, _ = np_backend.run_one_step(step, doc, np_state, uchars)
        np_losses.append(loss)

    for step, (py_l, np_l) in enumerate(zip(py_losses, np_losses)):
        assert numpy.isclose(py_l, np_l, rtol=1e-2, atol=1e-4), (
            f"Step {step}: python loss {py_l:.6f} vs numpy loss {np_l:.6f}"
        )


def test_numpy_batched_matches_sequential(config_and_docs):
    """numpy_batched backend matches numpy backend (same init, same steps)."""
    numpy = pytest.importorskip("numpy")
    config, docs = config_and_docs
    uchars = sorted(set("".join(docs)))
    num_steps = 3

    # Same initial weights for both (from one numpy init, then export)
    np_backend = get_backend("numpy")
    state_init = np_backend.create_state(config, uchars)
    init_from = {k: v.tolist() for k, v in state_init["state_dict"].items()}

    state_seq = np_backend.create_state(config, uchars, init_from=init_from)
    batch_backend = get_backend("numpy_batched")
    state_batch = batch_backend.create_state(config, uchars, init_from=init_from)
    seq_losses = []
    batch_losses = []
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        loss_s, _ = np_backend.run_one_step(step, doc, state_seq, uchars)
        loss_b, _ = batch_backend.run_one_step(step, doc, state_batch, uchars)
        seq_losses.append(loss_s)
        batch_losses.append(loss_b)

    for step, (seq_l, batch_l) in enumerate(zip(seq_losses, batch_losses)):
        assert numpy.isclose(seq_l, batch_l, rtol=1e-4, atol=1e-3), (
            f"Step {step}: sequential {seq_l:.6f} vs batched {batch_l:.6f}"
        )


def test_pytorch_backend_available():
    """PyTorch backend is registered when torch is installed."""
    try:
        get_backend("pytorch")
    except KeyError:
        pytest.skip("pytorch backend not available (pip install microgpt[pytorch])")


def test_pytorch_backend_equivalence(config_and_docs):
    """PyTorch backend (if available) matches Python backend loss for same init and steps."""
    pytest.importorskip("torch")
    config, docs = config_and_docs
    uchars = sorted(set("".join(docs)))
    num_steps = 3

    random.seed(42)
    py_backend = get_backend("python")
    py_state = py_backend.create_state(config, uchars)
    init_weights = state_dict_to_json(py_state["state_dict"])

    py_losses = []
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        loss, _ = py_backend.run_one_step(step, doc, py_state, uchars)
        py_losses.append(loss)

    try:
        pt_backend = get_backend("pytorch")
    except KeyError:
        pytest.skip("pytorch backend not available (pip install microgpt[pytorch])")
    pt_state = pt_backend.create_state(
        config, uchars, init_from=init_weights, device="cpu", dtype="float64"
    )
    pt_losses = []
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        loss, _ = pt_backend.run_one_step(step, doc, pt_state, uchars)
        pt_losses.append(loss)

    for step, (py_l, pt_l) in enumerate(zip(py_losses, pt_losses)):
        assert abs(py_l - pt_l) < 1e-4 or abs(py_l - pt_l) / (abs(py_l) + 1e-8) < 1e-2, (
            f"Step {step}: python loss {py_l:.6f} vs pytorch loss {pt_l:.6f}"
        )


def test_pytorch_backend_smoke(config_and_docs):
    """PyTorch backend run_one_step and weights_for_export run without error."""
    try:
        backend = get_backend("pytorch")
    except KeyError:
        pytest.skip("pytorch backend not available (pip install microgpt[pytorch])")
    config, docs = config_and_docs
    uchars = sorted(set("".join(docs)))
    state = backend.create_state(config, uchars, device="cpu")
    doc = docs[0] if docs else "x"
    loss, timing = backend.run_one_step(0, doc, state, uchars)
    assert loss >= 0
    exported = backend.weights_for_export(state)
    assert "wte" in exported
    assert isinstance(exported["wte"], list)
