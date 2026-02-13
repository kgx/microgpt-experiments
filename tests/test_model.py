"""Tests for model: tokenizer, forward pass, save/load."""

import os
import tempfile

import pytest

from model import (
    gpt,
    init_state_dict,
    load_model,
    save_model,
)


def test_tokenizer_roundtrip():
    """Build vocab from docs; encode string to ids; decode ids to string (excluding BOS)."""
    docs = ["ab", "ac", "bc"]
    uchars = sorted(set("".join(docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    assert vocab_size == 4  # a, b, c + BOS
    s = "abc"
    ids = [uchars.index(ch) for ch in s]
    decoded = "".join(uchars[i] for i in ids)
    assert decoded == s


def test_forward_one_step():
    """One forward step returns logits of length vocab_size."""
    vocab_size = 10
    n_embd = 8
    n_head = 2
    n_layer = 1
    block_size = 16
    state_dict, _ = init_state_dict(vocab_size, n_embd, n_layer, block_size)
    head_dim = n_embd // n_head
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    logits = gpt(0, 0, keys, values, state_dict, n_layer, n_head, head_dim, block_size)
    assert len(logits) == vocab_size


def test_save_load_roundtrip():
    """Save model to temp file, load back; vocab and config match."""
    vocab_size = 6
    n_embd = 4
    n_head = 2
    n_layer = 1
    block_size = 8
    uchars = ["a", "b", "c", "d", "e"]
    state_dict, _ = init_state_dict(vocab_size, n_embd, n_layer, block_size)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        save_model(path, state_dict, uchars, n_embd, n_head, n_layer, block_size)
        data = load_model(path)
        assert data["uchars"] == uchars
        assert data["vocab_size"] == vocab_size
        assert data["n_embd"] == n_embd
        assert data["n_layer"] == n_layer
        assert data["block_size"] == block_size
        assert len(data["state_dict"]["wte"]) == vocab_size
    finally:
        os.unlink(path)
