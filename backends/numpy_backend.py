"""
NumPy (and optional Numba) training backend. Same math as the Python backend
but with vectorized ops and manual backward for speed. Requires numpy; uses
numba for hot loops if available.
"""

import math
import os
import time

import numpy as np

try:
    from numba import jit
    from numba import prange  # noqa: F401 - used in fused kernel

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    prange = range  # fallback

    def jit(*args, **kwargs):
        def dec(f):
            return f

        return dec if not args else dec(*args)


# -----------------------------------------------------------------------------
# Forward helpers (NumPy)
# -----------------------------------------------------------------------------


def _linear_fwd(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """out = w @ x. x (nin,), w (nout, nin) -> out (nout,)."""
    return w @ x


def _rmsnorm_fwd(x: np.ndarray, eps: float = 1e-5) -> tuple[np.ndarray, float]:
    """RMSNorm: scale so RMS is 1. Returns (out, scale)."""
    ms = (x * x).mean() + eps
    scale = ms ** -0.5
    return x * scale, scale


def _softmax_fwd(logits: np.ndarray) -> np.ndarray:
    """Stable softmax."""
    m = logits.max()
    exps = np.exp(logits - m)
    return exps / exps.sum()


def _relu2_fwd(x: np.ndarray) -> np.ndarray:
    """ReLU(x)^2."""
    return np.maximum(0, x) ** 2


# -----------------------------------------------------------------------------
# Backward helpers
# -----------------------------------------------------------------------------


def _linear_bwd(grad_out: np.ndarray, x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dL/dw and dL/dx for out = w @ x."""
    grad_w = np.outer(grad_out, x)
    grad_x = w.T @ grad_out
    return grad_w, grad_x


def _rmsnorm_bwd(grad_out: np.ndarray, x: np.ndarray, scale: float, eps: float = 1e-5) -> np.ndarray:
    """dL/dx for y = x * scale, scale = (mean(x^2)+eps)^-0.5."""
    n = x.size
    # dx = scale * (dout - (scale^2/n) * x * (x . dout))
    grad_x = scale * (grad_out - (scale * scale / n) * x * (x @ grad_out))
    return grad_x


def _softmax_nll_bwd(probs: np.ndarray, target: int) -> np.ndarray:
    """Gradient of -log(probs[target]) w.r.t. logits (softmax backward)."""
    grad = probs.copy()
    grad[target] -= 1.0
    return grad


def _relu2_bwd(grad_out: np.ndarray, x: np.ndarray) -> np.ndarray:
    """d/dx ReLU(x)^2 = 2*ReLU(x)*(x>0)."""
    relu_x = np.maximum(0, x)
    return (2.0 * relu_x * (x > 0)) * grad_out


# -----------------------------------------------------------------------------
# Batched (sequence) helpers for optional batched-over-positions path
# -----------------------------------------------------------------------------

def _rmsnorm_fwd_batched(x: np.ndarray, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """x (T, d) -> (T, d), scale (T,) so each row has RMS 1."""
    ms = (x * x).mean(axis=1) + eps
    scale = ms ** -0.5
    return x * scale[:, np.newaxis], scale


def _rmsnorm_bwd_batched(
    grad_out: np.ndarray, x: np.ndarray, scale: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """grad_out (T, d), x (T, d), scale (T,) -> grad_x (T, d)."""
    n = x.shape[1]
    # per row: grad_x = scale * (grad_out - (scale^2/n) * x * (x . grad_out))
    x_dot_dout = (x * grad_out).sum(axis=1)
    grad_x = scale[:, np.newaxis] * (
        grad_out - (scale * scale / n)[:, np.newaxis] * x * x_dot_dout[:, np.newaxis]
    )
    return grad_x


def _linear_fwd_batched(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x (T, nin), w (nout, nin) -> out (T, nout)."""
    return x @ w.T


def _linear_bwd_batched(
    grad_out: np.ndarray, x: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """grad_out (T, nout), x (T, nin), w (nout, nin) -> grad_w (nout, nin), grad_x (T, nin)."""
    grad_w = grad_out.T @ x
    grad_x = grad_out @ w
    return grad_w, grad_x


def _causal_attention_batched_fwd(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, head_dim: int, n_head: int
) -> tuple[np.ndarray, np.ndarray]:
    """Q,K,V (T, n_embd). Causal attention over positions. Returns (out (T, n_embd), weights (T, T) causal)."""
    T = q.shape[0]
    scale = 1.0 / math.sqrt(head_dim)
    # (T, n_head, head_dim)
    q_h = q.reshape(T, n_head, head_dim)
    k_h = k.reshape(T, n_head, head_dim)
    v_h = v.reshape(T, n_head, head_dim)
    # scores (T, n_head, T): scores[i,h,j] = q[i,h] . k[j,h]
    scores = np.einsum("ihd,jhd->ihj", q_h, k_h) * scale
    causal_mask = np.triu(np.full((T, T), -np.inf), k=1)  # j > i -> -inf
    scores = scores + causal_mask[:, np.newaxis, :]  # (T, 1, T) broadcasts to (T, n_head, T)
    weights = np.exp(scores - scores.max(axis=2, keepdims=True))
    weights = weights / weights.sum(axis=2, keepdims=True)
    out = np.einsum("ihj,jhd->ihd", weights, v_h)
    return out.reshape(T, -1), weights


def _causal_attention_batched_bwd(
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
    # grad_weights: dL/d(weights)  -> (T, n_head, T)
    grad_weights = np.einsum("ihd,jhd->ihj", grad_out_h, v_h)
    # softmax backward on weights (over last dim, causal)
    grad_scores = weights * (grad_weights - (weights * grad_weights).sum(axis=2, keepdims=True))
    # scores[i,h,j] = scale * q[i,h].k[j,h]  (j<=i)
    grad_q = scale * np.einsum("ihj,jhd->ihd", grad_scores, k_h)
    grad_k = scale * np.einsum("ihj,ihd->jhd", grad_scores, q_h)
    grad_v = np.einsum("ihj,ihd->jhd", weights, grad_out_h)
    return grad_q.reshape(T, -1), grad_k.reshape(T, -1), grad_v.reshape(T, -1)


if _HAS_NUMBA:

    @jit(nopython=True)
    def _attention_scores_numba(q: np.ndarray, k: np.ndarray, scale: float) -> np.ndarray:
        """logits[t] = q . k[t] * scale. q (d,), k (T, d) -> (T,)."""
        T = k.shape[0]
        out = np.empty(T)
        for t in range(T):
            out[t] = np.dot(q, k[t]) * scale
        return out

    @jit(nopython=True)
    def _attention_out_numba(weights: np.ndarray, v: np.ndarray) -> np.ndarray:
        """out[j] = sum_t weights[t] * v[t,j]. weights (T,), v (T,d) -> (d,)."""
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
    # Warmup: trigger compilation with small arrays
    q = np.ones(8, dtype=np.float64)
    k = np.ones((4, 8), dtype=np.float64)
    w = np.ones(4, dtype=np.float64)
    v = np.ones((4, 8), dtype=np.float64)
    _attention_scores_numba(q, k, 1.0 / math.sqrt(8))
    _attention_out_numba(w, v)
    # Check compiled (numba dispatcher has .signatures after first call)
    has_scores = getattr(_attention_scores_numba, "signatures", None) is not None and len(getattr(_attention_scores_numba, "signatures", [])) > 0
    has_out = getattr(_attention_out_numba, "signatures", None) is not None and len(getattr(_attention_out_numba, "signatures", [])) > 0
    if has_scores and has_out:
        return True, "numba JIT active (attention_scores and attention_out compiled)"
    return False, "numba present but JIT not compiled (signatures empty)"


if _HAS_NUMBA:

    @jit(nopython=True, cache=True, parallel=True)
    def _attention_all_heads_fused(
        q: np.ndarray, keys: np.ndarray, values: np.ndarray, head_dim: int, n_head: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All heads in one kernel: q (n_embd,), keys/values (T, n_embd). Returns (x_attn, weights (n_head,T), logits (n_head,T)). Uses prange over heads for multi-core."""
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

    def _attention_all_heads_fused(
        q: np.ndarray, keys: np.ndarray, values: np.ndarray, head_dim: int, n_head: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback: same math in NumPy (no JIT)."""
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
            weights[h, :] = _softmax_fwd(logits[h, :])
            x_attn[hs : hs + head_dim] = weights[h, :] @ v_h
        return x_attn, weights, logits


def _attention_fwd(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, head_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal attention: logits = q @ k.T / sqrt(d), weights = softmax(logits), out = weights @ v."""
    scale = 1.0 / math.sqrt(head_dim)
    logits = _attention_scores_numba(q, k, scale)
    weights = _softmax_fwd(logits)
    out = _attention_out_numba(weights, v)
    return out, weights, logits


def _attention_bwd(
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
    d = q.shape[0]
    # grad_weights[t] = grad_out . v[t]
    grad_weights = np.zeros(T)
    for t in range(T):
        grad_weights[t] = np.dot(grad_out, v[t])
    # softmax backward on logits
    grad_logits = weights * (grad_weights - np.dot(weights, grad_weights))
    # logits[t] = scale * (q . k[t]), so grad_q += scale * k[t] * grad_logits[t], grad_k[t] += scale * q * grad_logits[t]
    grad_q = scale * (k.T @ grad_logits)
    grad_k = scale * np.outer(grad_logits, q)
    grad_v = np.outer(weights, grad_out)
    return grad_q, grad_k, grad_v


# -----------------------------------------------------------------------------
# State dict key order (must match Python backend)
# -----------------------------------------------------------------------------


def _ordered_state_keys(n_layer: int) -> list[str]:
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


def _state_dict_shapes(vocab_size: int, n_embd: int, n_layer: int, block_size: int) -> dict[str, tuple[int, ...]]:
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


_JIT_VERIFY_DONE = False


class NumPyBackendBase:
    """Shared base for NumPy backends: state creation, run_one_step loop, optimizer, export."""

    def create_state(
        self,
        config: dict,
        uchars: list[str],
        *,
        seed: int | None = 42,
        init_from: dict[str, list[list[float]]] | None = None,
        dtype: type | str | None = None,
    ):
        """Build model state. If init_from is provided (e.g. from Python backend), use those weights.
        dtype: np.float32 for faster CPU training, or np.float64 (default). Can be string 'float32'/'float64'.
        """
        model_cfg = config["model"]
        train_cfg = config["training"]
        data_cfg = config.get("data", {})
        n_embd = model_cfg["n_embd"]
        n_head = model_cfg["n_head"]
        n_layer = model_cfg["n_layer"]
        block_size = model_cfg["block_size"]
        vocab_size = len(uchars) + 1
        head_dim = n_embd // n_head
        trailing_bos = data_cfg.get("trailing_bos", True)

        if dtype is None:
            dtype = train_cfg.get("dtype", "float64")
        if isinstance(dtype, str):
            dtype = getattr(np, dtype, np.float64)

        shapes = _state_dict_shapes(vocab_size, n_embd, n_layer, block_size)
        state_dict = {}
        if init_from:
            for k, shape in shapes.items():
                arr = np.array(init_from[k], dtype=dtype)
                assert arr.shape == shape, f"{k}: {arr.shape} vs {shape}"
                state_dict[k] = arr.copy()
        else:
            rng = np.random.default_rng(seed)
            for k, shape in shapes.items():
                std = 0.02
                if "attn_wo" in k or "mlp_fc2" in k:
                    std = 0.0
                state_dict[k] = rng.normal(0, std, size=shape).astype(dtype)

        grad_dict = {k: np.zeros_like(v) for k, v in state_dict.items()}
        m_dict = {k: np.zeros_like(v) for k, v in state_dict.items()}
        v_dict = {k: np.zeros_like(v) for k, v in state_dict.items()}

        ordered_keys = _ordered_state_keys(n_layer)
        params = [state_dict[k] for k in ordered_keys]

        return {
            "state_dict": state_dict,
            "grad_dict": grad_dict,
            "m_dict": m_dict,
            "v_dict": v_dict,
            "ordered_keys": ordered_keys,
            "params": params,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "block_size": block_size,
            "head_dim": head_dim,
            "vocab_size": vocab_size,
            "learning_rate": train_cfg["learning_rate"],
            "beta1": train_cfg["beta1"],
            "beta2": train_cfg["beta2"],
            "eps_adam": train_cfg["eps_adam"],
            "num_steps": train_cfg["num_steps"],
            "trailing_bos": trailing_bos,
            "dtype": dtype,
        }

    def run_one_step(
        self,
        step: int,
        doc: str,
        state: dict,
        uchars: list[str],
        *,
        zero_grad: bool = True,
        do_optimizer: bool = True,
        grad_accum_count: int = 1,
    ) -> tuple[float, dict]:
        """One training step: forward, backward, and optionally optimizer. Updates state in place.
        zero_grad: clear gradient buffers before forward (set False when accumulating).
        do_optimizer: run Adam after backward. When True and grad_accum_count>1, grads are scaled by 1/grad_accum_count.
        """
        BOS = len(uchars)
        state_dict = state["state_dict"]
        grad_dict = state["grad_dict"]
        m_dict = state["m_dict"]
        v_dict = state["v_dict"]
        n_layer = state["n_layer"]
        n_head = state["n_head"]
        head_dim = state["head_dim"]
        block_size = state["block_size"]
        vocab_size = state["vocab_size"]
        lr = state["learning_rate"]
        beta1, beta2 = state["beta1"], state["beta2"]
        eps = state["eps_adam"]
        num_steps = state["num_steps"]
        trailing_bos = state.get("trailing_bos", True)

        char_ids = [uchars.index(ch) for ch in doc if ch in uchars]
        tokens = [BOS] + char_ids + ([BOS] if trailing_bos else [])
        if len(tokens) < 2:
            if zero_grad:
                for g in grad_dict.values():
                    g.fill(0)
            return 0.0, {"forward": 0, "backward": 0, "optimizer": 0}

        global _JIT_VERIFY_DONE
        if not _JIT_VERIFY_DONE and os.environ.get("MICROGPT_VERIFY_JIT") == "1":
            ok, msg = verify_numba_jit()
            print(f"[numpy backend] JIT check: {msg}")
            _JIT_VERIFY_DONE = True

        n = min(block_size, len(tokens) - 1)

        if zero_grad:
            for g in grad_dict.values():
                g.fill(0)

        t0 = time.perf_counter()
        loss, cache = self._forward(state, tokens, n)
        t_forward = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._backward(state, tokens, n, cache, loss, grad_dict)
        t_backward = time.perf_counter() - t0

        t0 = time.perf_counter()
        if do_optimizer:
            if grad_accum_count > 1:
                scale = 1.0 / grad_accum_count
                for g in grad_dict.values():
                    g *= scale
            lr_t = lr * 0.5 * (1 + math.cos(math.pi * step / num_steps))
            for k in state["ordered_keys"]:
                m = m_dict[k]
                v = v_dict[k]
                g = grad_dict[k]
                m[:] = beta1 * m + (1 - beta1) * g
                v[:] = beta2 * v + (1 - beta2) * (g * g)
                m_hat = m / (1 - beta1 ** (step + 1))
                v_hat = v / (1 - beta2 ** (step + 1))
                state_dict[k][:] -= lr_t * m_hat / (np.sqrt(v_hat) + eps)
        t_optimizer = time.perf_counter() - t0

        return float(loss), {"forward": t_forward, "backward": t_backward, "optimizer": t_optimizer}

    def _forward(self, state: dict, tokens: list[int], n: int) -> tuple[float, dict]:
        """Forward over n positions; return mean loss and cache for backward."""
        state_dict = state["state_dict"]
        n_layer = state["n_layer"]
        n_head = state["n_head"]
        head_dim = state["head_dim"]
        cache = {
            "layers": [
                {
                    "keys": [], "values": [],
                    "x_attn": [], "x_in_attn": [], "x_before_rms_attn": [], "x_after_attn": [],
                    "rms_scale_attn": [], "rms_scale_mlp": [],
                    "q": [], "k": [], "v": [], "attn_weights": [], "attn_logits": [],
                    "relu2_pre": [],
                }
                for _ in range(n_layer)
            ],
            "x_final": [],
            "x_emb": [],
            "rms_scale_emb": [],
            "x_before_rms_emb": [],
        }
        losses = []

        for pos in range(n):
            token_id, target_id = tokens[pos], tokens[pos + 1]
            # Embed
            x_emb_raw = state_dict["wte"][token_id] + state_dict["wpe"][pos]
            cache["x_before_rms_emb"].append(x_emb_raw.copy())
            x, rms_scale_emb = _rmsnorm_fwd(x_emb_raw)
            cache["x_emb"].append(x.copy())
            cache["rms_scale_emb"].append(rms_scale_emb)

            for li in range(n_layer):
                x_res = x.copy()
                cache["layers"][li]["x_before_rms_attn"].append(x_res.copy())
                x, rms_scale_attn = _rmsnorm_fwd(x)
                cache["layers"][li]["x_in_attn"].append(x.copy())
                q = _linear_fwd(x, state_dict[f"layer{li}.attn_wq"])
                k = _linear_fwd(x, state_dict[f"layer{li}.attn_wk"])
                v = _linear_fwd(x, state_dict[f"layer{li}.attn_wv"])
                cache["layers"][li]["keys"].append(k.copy())
                cache["layers"][li]["values"].append(v.copy())
                cache["layers"][li]["rms_scale_attn"].append(rms_scale_attn)

                # Fused kernel: all heads in one call (keys/values as (T, n_embd))
                T = len(cache["layers"][li]["keys"])
                dt = state.get("dtype", np.float64)
                keys_arr = np.empty((T, state["n_embd"]), dtype=dt)
                values_arr = np.empty((T, state["n_embd"]), dtype=dt)
                for t in range(T):
                    keys_arr[t] = cache["layers"][li]["keys"][t]
                    values_arr[t] = cache["layers"][li]["values"][t]
                x_attn, weights_all, logits_all = _attention_all_heads_fused(q, keys_arr, values_arr, head_dim, n_head)
                for h in range(n_head):
                    hs = h * head_dim
                    cache["layers"][li]["q"].append(q[hs : hs + head_dim].copy())
                    cache["layers"][li]["k"].append(keys_arr[:, hs : hs + head_dim].copy())
                    cache["layers"][li]["v"].append(values_arr[:, hs : hs + head_dim].copy())
                    cache["layers"][li]["attn_weights"].append(weights_all[h].copy())
                    cache["layers"][li]["attn_logits"].append(logits_all[h].copy())

                cache["layers"][li]["x_attn"].append(x_attn.copy())
                x = _linear_fwd(x_attn, state_dict[f"layer{li}.attn_wo"]) + x_res
                cache["layers"][li]["x_after_attn"].append(x.copy())

                # MLP
                x_res = x.copy()
                x, rms_scale_mlp = _rmsnorm_fwd(x)
                cache["layers"][li]["rms_scale_mlp"].append(rms_scale_mlp)
                x = _linear_fwd(x, state_dict[f"layer{li}.mlp_fc1"])
                cache["layers"][li]["relu2_pre"].append(x.copy())
                x = _relu2_fwd(x)
                x = _linear_fwd(x, state_dict[f"layer{li}.mlp_fc2"]) + x_res

            cache["x_final"].append(x.copy())
            logits = _linear_fwd(x, state_dict["lm_head"])
            probs = _softmax_fwd(logits)
            loss_t = -math.log(probs[target_id] + 1e-15)
            losses.append((loss_t, probs, target_id, logits))

        mean_loss = sum(l[0] for l in losses) / n
        cache["losses"] = losses
        cache["n"] = n
        return mean_loss, cache

    def _forward_batched(self, state: dict, tokens: list[int], n: int) -> tuple[float, dict]:
        """Batched forward over n positions in one pass. Returns mean loss and cache for _backward_batched."""
        state_dict = state["state_dict"]
        n_layer = state["n_layer"]
        n_head = state["n_head"]
        head_dim = state["head_dim"]
        n_embd = state["n_embd"]
        dtype = state.get("dtype", np.float64)
        T = n

        token_ids = np.array([tokens[i] for i in range(T)], dtype=np.intp)
        target_ids = np.array([tokens[i + 1] for i in range(T)], dtype=np.intp)

        x = state_dict["wte"][token_ids] + state_dict["wpe"][:T]
        cache = {"x_before_rms_emb": x.copy(), "layers": []}
        x, rms_scale_emb = _rmsnorm_fwd_batched(x)
        cache["x_emb"] = x.copy()
        cache["rms_scale_emb"] = rms_scale_emb.copy()

        for li in range(n_layer):
            x_res = x.copy()
            cache_li = {"x_before_rms_attn": x_res.copy()}
            x, rms_scale_attn = _rmsnorm_fwd_batched(x)
            cache_li["x_in_attn"] = x.copy()
            cache_li["rms_scale_attn"] = rms_scale_attn.copy()
            q = _linear_fwd_batched(x, state_dict[f"layer{li}.attn_wq"])
            k = _linear_fwd_batched(x, state_dict[f"layer{li}.attn_wk"])
            v = _linear_fwd_batched(x, state_dict[f"layer{li}.attn_wv"])
            cache_li["q"], cache_li["k"], cache_li["v"] = q.copy(), k.copy(), v.copy()
            x_attn, attn_weights = _causal_attention_batched_fwd(q, k, v, head_dim, n_head)
            cache_li["attn_weights"] = attn_weights.copy()
            cache_li["x_attn"] = x_attn.copy()
            x = _linear_fwd_batched(x_attn, state_dict[f"layer{li}.attn_wo"]) + x_res
            cache_li["x_after_attn"] = x.copy()
            x_res = x.copy()
            x, rms_scale_mlp = _rmsnorm_fwd_batched(x)
            cache_li["rms_scale_mlp"] = rms_scale_mlp.copy()
            x = _linear_fwd_batched(x, state_dict[f"layer{li}.mlp_fc1"])
            cache_li["relu2_pre"] = x.copy()
            x = _relu2_fwd(x)
            x = _linear_fwd_batched(x, state_dict[f"layer{li}.mlp_fc2"]) + x_res
            cache["layers"].append(cache_li)

        cache["x_final"] = x.copy()
        logits = _linear_fwd_batched(x, state_dict["lm_head"])
        logits_max = logits.max(axis=1, keepdims=True)
        exps = np.exp(logits - logits_max)
        probs = exps / exps.sum(axis=1, keepdims=True)
        cache["logits"] = logits
        cache["probs"] = probs
        cache["target_ids"] = target_ids
        loss_per_pos = -np.log(probs[np.arange(T), target_ids] + 1e-15)
        mean_loss = float(loss_per_pos.mean())
        cache["n"] = n
        return mean_loss, cache

    def _backward_batched(
        self,
        state: dict,
        tokens: list[int],
        n: int,
        cache: dict,
        loss: float,
        grad_dict: dict,
    ) -> None:
        """Backward for batched forward. Accumulates into grad_dict. (loss unused, for signature compatibility.)"""
        state_dict = state["state_dict"]
        n_layer = state["n_layer"]
        n_head = state["n_head"]
        head_dim = state["head_dim"]
        n_embd = state["n_embd"]
        T = n
        dtype = state.get("dtype", np.float64)
        scale = 1.0 / T
        probs = cache["probs"]
        target_ids = cache["target_ids"]
        grad_logits = probs.copy()
        grad_logits[np.arange(T), target_ids] -= 1.0
        grad_logits *= scale

        grad_x = grad_logits @ state_dict["lm_head"]
        grad_dict["lm_head"] += grad_logits.T @ cache["x_final"]

        for li in range(n_layer - 1, -1, -1):
            c = cache["layers"][li]
            grad_mlp_out = grad_x.copy()
            grad_fc2_w, grad_fc2_in = _linear_bwd_batched(
                grad_mlp_out, c["relu2_pre"], state_dict[f"layer{li}.mlp_fc2"]
            )
            grad_dict[f"layer{li}.mlp_fc2"] += grad_fc2_w
            grad_relu2 = _relu2_bwd(grad_fc2_in, c["relu2_pre"])
            x_after_rms = c["x_after_attn"] * c["rms_scale_mlp"][:, np.newaxis]
            grad_fc1_w, grad_fc1_in = _linear_bwd_batched(
                grad_relu2, x_after_rms, state_dict[f"layer{li}.mlp_fc1"]
            )
            grad_dict[f"layer{li}.mlp_fc1"] += grad_fc1_w
            grad_x_after_rms = _rmsnorm_bwd_batched(
                grad_fc1_in, c["x_after_attn"], c["rms_scale_mlp"]
            )
            grad_x_after_attn = grad_x_after_rms + grad_mlp_out

            grad_attn_wo_out = grad_x_after_attn.copy()
            grad_attn_wo_w, grad_attn_in = _linear_bwd_batched(
                grad_attn_wo_out, c["x_attn"], state_dict[f"layer{li}.attn_wo"]
            )
            grad_dict[f"layer{li}.attn_wo"] += grad_attn_wo_w
            grad_q, grad_k, grad_v = _causal_attention_batched_bwd(
                grad_attn_in, c["q"], c["k"], c["v"], c["attn_weights"], head_dim, n_head
            )
            grad_x_in_attn = grad_q @ state_dict[f"layer{li}.attn_wq"] + grad_k @ state_dict[f"layer{li}.attn_wk"] + grad_v @ state_dict[f"layer{li}.attn_wv"]
            grad_dict[f"layer{li}.attn_wq"] += grad_q.T @ c["x_in_attn"]
            grad_dict[f"layer{li}.attn_wk"] += grad_k.T @ c["x_in_attn"]
            grad_dict[f"layer{li}.attn_wv"] += grad_v.T @ c["x_in_attn"]
            grad_x_before_rms_attn = _rmsnorm_bwd_batched(
                grad_x_in_attn, c["x_before_rms_attn"], c["rms_scale_attn"]
            )
            grad_x = grad_x_after_attn + grad_x_before_rms_attn

        grad_emb = _rmsnorm_bwd_batched(
            grad_x, cache["x_before_rms_emb"], cache["rms_scale_emb"]
        )
        token_ids = np.array([tokens[i] for i in range(T)], dtype=np.intp)
        for i in range(T):
            grad_dict["wte"][token_ids[i]] += grad_emb[i]
        grad_dict["wpe"][:T] += grad_emb

    def _backward(self, state: dict, tokens: list[int], n: int, cache: dict, loss: float, grad_dict: dict) -> None:
        """Backward: accumulate gradients into grad_dict."""
        state_dict = state["state_dict"]
        n_layer = state["n_layer"]
        n_head = state["n_head"]
        head_dim = state["head_dim"]
        n_embd = state["n_embd"]
        losses = cache["losses"]
        scale = 1.0 / n

        for pos in range(n - 1, -1, -1):
            _, probs, target_id, _ = losses[pos]
            grad_logits = _softmax_nll_bwd(probs, target_id) * scale

            x_final = cache["x_final"][pos]
            grad_x = state_dict["lm_head"].T @ grad_logits
            grad_dict["lm_head"] += np.outer(grad_logits, x_final)

            for li in range(n_layer - 1, -1, -1):
                x_after_attn = cache["layers"][li]["x_after_attn"][pos]
                rms_scale_mlp = cache["layers"][li]["rms_scale_mlp"][pos]
                relu2_pre = cache["layers"][li]["relu2_pre"][pos]
                w_fc2 = state_dict[f"layer{li}.mlp_fc2"]
                w_fc1 = state_dict[f"layer{li}.mlp_fc1"]

                # MLP backward: grad_x is at output of layer
                grad_mlp_out = grad_x.copy()
                grad_fc2_w, grad_fc2_in = _linear_bwd(grad_mlp_out, relu2_pre, w_fc2)
                grad_dict[f"layer{li}.mlp_fc2"] += grad_fc2_w
                grad_relu2 = _relu2_bwd(grad_fc2_in, relu2_pre)
                x_after_rms_mlp = x_after_attn * rms_scale_mlp
                grad_fc1_w, grad_fc1_in = _linear_bwd(grad_relu2, x_after_rms_mlp, w_fc1)
                grad_dict[f"layer{li}.mlp_fc1"] += grad_fc1_w
                grad_x_after_rms = _rmsnorm_bwd(grad_fc1_in, x_after_attn, rms_scale_mlp)
                grad_x_after_attn = grad_x_after_rms + grad_mlp_out  # residual

                # Attention backward
                grad_attn_wo_out = grad_x_after_attn.copy()
                grad_attn_wo_w, grad_attn_in = _linear_bwd(
                    grad_attn_wo_out, cache["layers"][li]["x_attn"][pos], state_dict[f"layer{li}.attn_wo"]
                )
                grad_dict[f"layer{li}.attn_wo"] += grad_attn_wo_w

                grad_x_in_attn = np.zeros(n_embd, dtype=state.get("dtype", np.float64))
                for h in range(n_head):
                    hs = h * head_dim
                    g_o = grad_attn_in[hs : hs + head_dim]
                    # Cache stores per (position, head): indices pos*n_head + h
                    idx = pos * n_head + h
                    q_h = cache["layers"][li]["q"][idx]
                    k_h = cache["layers"][li]["k"][idx]
                    v_h = cache["layers"][li]["v"][idx]
                    w = cache["layers"][li]["attn_weights"][idx]
                    logits_h = cache["layers"][li]["attn_logits"][idx]
                    g_q, g_k, g_v = _attention_bwd(g_o, q_h, k_h, v_h, w, logits_h, head_dim)
                    grad_x_in_attn += state_dict[f"layer{li}.attn_wq"][hs : hs + head_dim, :].T @ g_q
                    grad_dict[f"layer{li}.attn_wq"][hs : hs + head_dim, :] += np.outer(g_q, cache["layers"][li]["x_in_attn"][pos])
                    for t in range(pos + 1):
                        x_t = cache["layers"][li]["x_in_attn"][t]
                        grad_dict[f"layer{li}.attn_wk"][hs : hs + head_dim, :] += np.outer(g_k[t], x_t)
                        grad_dict[f"layer{li}.attn_wv"][hs : hs + head_dim, :] += np.outer(g_v[t], x_t)
                    grad_x_in_attn += state_dict[f"layer{li}.attn_wk"][hs : hs + head_dim, :].T @ g_k[pos]
                    grad_x_in_attn += state_dict[f"layer{li}.attn_wv"][hs : hs + head_dim, :].T @ g_v[pos]

                grad_x_before_rms_attn = _rmsnorm_bwd(
                    grad_x_in_attn, cache["layers"][li]["x_before_rms_attn"][pos], cache["layers"][li]["rms_scale_attn"][pos]
                )
                grad_x = grad_x_after_attn + grad_x_before_rms_attn  # residual + from rmsnorm

            grad_emb = _rmsnorm_bwd(grad_x, cache["x_before_rms_emb"][pos], cache["rms_scale_emb"][pos])
            grad_dict["wte"][tokens[pos]] += grad_emb
            grad_dict["wpe"][pos] += grad_emb

    def weights_for_export(self, state: dict) -> dict:
        """Return state_dict as list-of-lists for JSON."""
        return {k: v.tolist() for k, v in state["state_dict"].items()}


class NumPyBackend(NumPyBackendBase):
    """Sequential position-by-position forward/backward. Uses Numba for attention when available."""


class NumPyBatchedBackend(NumPyBackendBase):
    """Batched-over-positions forward/backward (one sequence pass). Same math, often faster."""

    def _forward(self, state: dict, tokens: list[int], n: int) -> tuple[float, dict]:
        return self._forward_batched(state, tokens, n)

    def _backward(
        self,
        state: dict,
        tokens: list[int],
        n: int,
        cache: dict,
        loss: float,
        grad_dict: dict,
    ) -> None:
        self._backward_batched(state, tokens, n, cache, loss, grad_dict)
