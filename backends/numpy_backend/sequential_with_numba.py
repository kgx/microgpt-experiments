"""Sequential (position-by-position) backend implementation when Numba is available."""

import math

import numpy as np
from numba import jit, prange


@jit(nopython=True)
def _attention_scores_numba(q: np.ndarray, k: np.ndarray, scale: float) -> np.ndarray:
    T = k.shape[0]
    out = np.empty(T)
    for t in range(T):
        out[t] = np.dot(q, k[t]) * scale
    return out


@jit(nopython=True)
def _attention_out_numba(weights: np.ndarray, v: np.ndarray) -> np.ndarray:
    T, d = v.shape
    out = np.zeros(d)
    for t in range(T):
        for j in range(d):
            out[j] += weights[t] * v[t, j]
    return out


@jit(nopython=True, cache=True, parallel=True)
def attention_all_heads_fused(
    q: np.ndarray, keys: np.ndarray, values: np.ndarray, head_dim: int, n_head: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = keys.shape[0]
    n_embd = q.shape[0]
    x_attn = np.zeros(n_embd)
    weights = np.zeros((n_head, T))
    logits = np.zeros((n_head, T))
    scale = 1.0 / math.sqrt(head_dim)
    for h in prange(n_head):
        hs = h * head_dim
        for t in range(T):
            logits[h, t] = 0.0
            for j in range(head_dim):
                logits[h, t] += q[hs + j] * keys[t, hs + j]
            logits[h, t] *= scale
        max_l = logits[h, 0]
        for t in range(1, T):
            if logits[h, t] > max_l:
                max_l = logits[h, t]
        for t in range(T):
            weights[h, t] = math.exp(logits[h, t] - max_l)
        s = 0.0
        for t in range(T):
            s += weights[h, t]
        for t in range(T):
            weights[h, t] /= s
        for j in range(head_dim):
            for t in range(T):
                x_attn[hs + j] += weights[h, t] * values[t, hs + j]
    return x_attn, weights, logits
