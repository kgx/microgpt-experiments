"""Tests for inference: smoke test, fail-fast on missing model."""

import os
import subprocess
import sys

import pytest

# Default model path for names scenario (relative to project root)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES_MODEL = os.path.join(ROOT, "models", "names", "model.json")


def test_inference_fail_fast_missing_model():
    """Inference with non-existent model exits with non-zero and prints error."""
    result = subprocess.run(
        [sys.executable, "main.py", "--model", "/nonexistent/model.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr.lower() or "Error" in result.stderr


def test_inference_fail_fast_missing_scenario_model():
    """Inference with --scenario when model not trained yet exits with non-zero."""
    result = subprocess.run(
        [sys.executable, "main.py", "--scenario", "alice"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    # alice model likely not trained; should fail
    assert result.returncode != 0
    assert "not found" in result.stderr.lower() or "Error" in result.stderr


def test_inference_smoke():
    """If names model exists, run one sample and check output is a string."""
    if not os.path.isfile(NAMES_MODEL):
        pytest.skip(f"Model not found: {NAMES_MODEL}. Run: python train.py --scenario names")
    result = subprocess.run(
        [sys.executable, "main.py", "--scenario", "names", "-i", "A", "--samples", "1", "--seed", "42"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "inference" in result.stdout
    assert "sample" in result.stdout or "A" in result.stdout


def test_inference_pytorch_smoke():
    """If names model exists and PyTorch is installed, run one sample with --backend pytorch."""
    if not os.path.isfile(NAMES_MODEL):
        pytest.skip(f"Model not found: {NAMES_MODEL}. Run: python train.py --scenario names")
    try:
        from backends import get_backend
        get_backend("pytorch")
    except (KeyError, ImportError):
        pytest.skip("pytorch backend not available (pip install microgpt[pytorch])")
    result = subprocess.run(
        [sys.executable, "main.py", "--scenario", "names", "--backend", "pytorch", "-i", "A", "--samples", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "inference" in result.stdout
    assert "sample" in result.stdout
