"""Sequential (position-by-position) backend implementation when Numba is not installed."""

import math

import numpy as np


def _attention_scores_numba(q: np.ndarray, k: np.ndarray, scale: float) -> np.ndarray:
    return (k @ q) * scale


def _attention_out_numba(weights: np.ndarray, v: np.ndarray) -> np.ndarray:
    return weights @ v


def attention_all_heads_fused(
    q: np.ndarray, keys: np.ndarray, values: np.ndarray, head_dim: int, n_head: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = keys.shape[0]
    n_embd = q.shape[0]
    x_attn = np.zeros(n_embd)
    weights = np.zeros((n_head, T))
    logits = np.zeros((n_head, T))
    scale = 1.0 / math.sqrt(head_dim)
    for h in range(n_head):
        hs = h * head_dim
        q_h = q[hs : hs + head_dim]
        k_h = keys[:, hs : hs + head_dim]
        v_h = values[:, hs : hs + head_dim]
        logits[h, :] = (k_h @ q_h) * scale
        # stable softmax (inline to avoid circular import from shared)
        l = logits[h, :]
        exps = np.exp(l - l.max())
        weights[h, :] = exps / exps.sum()
        x_attn[hs : hs + head_dim] = weights[h, :] @ v_h
    return x_attn, weights, logits
