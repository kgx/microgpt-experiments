"""
Shared NumPy backend helpers (functions only): scalar ops, batched ops, state, attention.
"""

import math

import numpy as np

try:
    from numba import jit
    from numba import prange  # noqa: F401
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    prange = range

    def jit(*args, **kwargs):
        def dec(f):
            return f
        return dec if not args else dec(*args)


# --- Scalar ops ---

def linear_fwd(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """out = w @ x. x (nin,), w (nout, nin) -> out (nout,)."""
    return w @ x


def rmsnorm_fwd(x: np.ndarray, eps: float = 1e-5) -> tuple[np.ndarray, float]:
    """RMSNorm: scale so RMS is 1. Returns (out, scale)."""
    ms = (x * x).mean() + eps
    scale = ms ** -0.5
    return x * scale, scale


def softmax_fwd(logits: np.ndarray) -> np.ndarray:
    """Stable softmax."""
    m = logits.max()
    exps = np.exp(logits - m)
    return exps / exps.sum()


def relu2_fwd(x: np.ndarray) -> np.ndarray:
    """ReLU(x)^2."""
    return np.maximum(0, x) ** 2


def linear_bwd(grad_out: np.ndarray, x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dL/dw and dL/dx for out = w @ x."""
    grad_w = np.outer(grad_out, x)
    grad_x = w.T @ grad_out
    return grad_w, grad_x


def rmsnorm_bwd(grad_out: np.ndarray, x: np.ndarray, scale: float, eps: float = 1e-5) -> np.ndarray:
    """dL/dx for y = x * scale, scale = (mean(x^2)+eps)^-0.5."""
    n = x.size
    grad_x = scale * (grad_out - (scale * scale / n) * x * (x @ grad_out))
    return grad_x


def softmax_nll_bwd(probs: np.ndarray, target: int) -> np.ndarray:
    """Gradient of -log(probs[target]) w.r.t. logits (softmax backward)."""
    grad = probs.copy()
    grad[target] -= 1.0
    return grad


def relu2_bwd(grad_out: np.ndarray, x: np.ndarray) -> np.ndarray:
    """d/dx ReLU(x)^2 = 2*ReLU(x)*(x>0)."""
    relu_x = np.maximum(0, x)
    return (2.0 * relu_x * (x > 0)) * grad_out


# --- Batched ops ---

def rmsnorm_fwd_batched(x: np.ndarray, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """x (T, d) -> (T, d), scale (T,) so each row has RMS 1."""
    ms = (x * x).mean(axis=1) + eps
    scale = ms ** -0.5
    return x * scale[:, np.newaxis], scale


def rmsnorm_bwd_batched(
    grad_out: np.ndarray, x: np.ndarray, scale: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """grad_out (T, d), x (T, d), scale (T,) -> grad_x (T, d)."""
    n = x.shape[1]
    x_dot_dout = (x * grad_out).sum(axis=1)
    grad_x = scale[:, np.newaxis] * (
        grad_out - (scale * scale / n)[:, np.newaxis] * x * x_dot_dout[:, np.newaxis]
    )
    return grad_x


def linear_fwd_batched(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x (T, nin), w (nout, nin) -> out (T, nout)."""
    return x @ w.T


def linear_bwd_batched(
    grad_out: np.ndarray, x: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """grad_out (T, nout), x (T, nin), w (nout, nin) -> grad_w (nout, nin), grad_x (T, nin)."""
    grad_w = grad_out.T @ x
    grad_x = grad_out @ w
    return grad_w, grad_x


def causal_attention_batched_fwd(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, head_dim: int, n_head: int
) -> tuple[np.ndarray, np.ndarray]:
    """Q,K,V (T, n_embd). Causal attention over positions. Returns (out (T, n_embd), weights (T, T) causal)."""
    T = q.shape[0]
    scale = 1.0 / math.sqrt(head_dim)
    q_h = q.reshape(T, n_head, head_dim)
    k_h = k.reshape(T, n_head, head_dim)
    v_h = v.reshape(T, n_head, head_dim)
    scores = np.einsum("ihd,jhd->ihj", q_h, k_h) * scale
    causal_mask = np.triu(np.full((T, T), -np.inf), k=1)
    scores = scores + causal_mask[:, np.newaxis, :]
    weights = np.exp(scores - scores.max(axis=2, keepdims=True))
    weights = weights / weights.sum(axis=2, keepdims=True)
    out = np.einsum("ihj,jhd->ihd", weights, v_h)
    return out.reshape(T, -1), weights


def causal_attention_batched_bwd(
    grad_out: np.ndarray,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
    head_dim: int,
    n_head: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """grad_out (T, n_embd). Backward of causal attention. Returns grad_q, grad_k, grad_v (T, n_embd)."""
    T = q.shape[0]
    scale = 1.0 / math.sqrt(head_dim)
    grad_out_h = grad_out.reshape(T, n_head, head_dim)
    q_h = q.reshape(T, n_head, head_dim)
    k_h = k.reshape(T, n_head, head_dim)
    v_h = v.reshape(T, n_head, head_dim)
    grad_weights = np.einsum("ihd,jhd->ihj", grad_out_h, v_h)
    grad_scores = weights * (grad_weights - (weights * grad_weights).sum(axis=2, keepdims=True))
    grad_q = scale * np.einsum("ihj,jhd->ihd", grad_scores, k_h)
    grad_k = scale * np.einsum("ihj,ihd->jhd", grad_scores, q_h)
    grad_v = np.einsum("ihj,ihd->jhd", weights, grad_out_h)
    return grad_q.reshape(T, -1), grad_k.reshape(T, -1), grad_v.reshape(T, -1)


# --- State ---

def ordered_state_keys(n_layer: int) -> list[str]:
    keys = ["wte", "wpe", "lm_head"]
    for i in range(n_layer):
        keys.extend([
            f"layer{i}.attn_wq",
            f"layer{i}.attn_wk",
            f"layer{i}.attn_wv",
            f"layer{i}.attn_wo",
            f"layer{i}.mlp_fc1",
            f"layer{i}.mlp_fc2",
        ])
    return keys


def state_dict_shapes(
    vocab_size: int, n_embd: int, n_layer: int, block_size: int
) -> dict[str, tuple[int, ...]]:
    shapes = {
        "wte": (vocab_size, n_embd),
        "wpe": (block_size, n_embd),
        "lm_head": (vocab_size, n_embd),
    }
    for i in range(n_layer):
        shapes[f"layer{i}.attn_wq"] = (n_embd, n_embd)
        shapes[f"layer{i}.attn_wk"] = (n_embd, n_embd)
        shapes[f"layer{i}.attn_wv"] = (n_embd, n_embd)
        shapes[f"layer{i}.attn_wo"] = (n_embd, n_embd)
        shapes[f"layer{i}.mlp_fc1"] = (4 * n_embd, n_embd)
        shapes[f"layer{i}.mlp_fc2"] = (n_embd, 4 * n_embd)
    return shapes


# --- Attention (sequential / Numba) ---

if _HAS_NUMBA:

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

else:

    def _attention_scores_numba(q: np.ndarray, k: np.ndarray, scale: float) -> np.ndarray:
        return (k @ q) * scale

    def _attention_out_numba(weights: np.ndarray, v: np.ndarray) -> np.ndarray:
        return weights @ v


def verify_numba_jit() -> tuple[bool, str]:
    """Verify that Numba JIT is active and compiled. Returns (ok, message)."""
    if not _HAS_NUMBA:
        return False, "numba not installed"
    q = np.ones(8, dtype=np.float64)
    k = np.ones((4, 8), dtype=np.float64)
    w = np.ones(4, dtype=np.float64)
    v = np.ones((4, 8), dtype=np.float64)
    _attention_scores_numba(q, k, 1.0 / math.sqrt(8))
    _attention_out_numba(w, v)
    has_scores = getattr(_attention_scores_numba, "signatures", None) is not None and len(getattr(_attention_scores_numba, "signatures", [])) > 0
    has_out = getattr(_attention_out_numba, "signatures", None) is not None and len(getattr(_attention_out_numba, "signatures", [])) > 0
    if has_scores and has_out:
        return True, "numba JIT active (attention_scores and attention_out compiled)"
    return False, "numba present but JIT not compiled (signatures empty)"


if _HAS_NUMBA:

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

else:

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
            weights[h, :] = softmax_fwd(logits[h, :])
            x_attn[hs : hs + head_dim] = weights[h, :] @ v_h
        return x_attn, weights, logits


def attention_fwd(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, head_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal attention: logits = q @ k.T / sqrt(d), weights = softmax(logits), out = weights @ v."""
    scale = 1.0 / math.sqrt(head_dim)
    logits = _attention_scores_numba(q, k, scale)
    weights = softmax_fwd(logits)
    out = _attention_out_numba(weights, v)
    return out, weights, logits


def attention_bwd(
    grad_out: np.ndarray,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
    logits: np.ndarray,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward of attention: dout w.r.t. q, k, v."""
    scale = 1.0 / math.sqrt(head_dim)
    T = k.shape[0]
    grad_weights = np.zeros(T)
    for t in range(T):
        grad_weights[t] = np.dot(grad_out, v[t])
    grad_logits = weights * (grad_weights - np.dot(weights, grad_weights))
    grad_q = scale * (k.T @ grad_logits)
    grad_k = scale * np.outer(grad_logits, q)
    grad_v = np.outer(weights, grad_out)
    return grad_q, grad_k, grad_v
