"""
NumPy (and optional Numba) training backend. Same math as the Python backend
but with vectorized ops and manual backward for speed. Requires numpy; uses
numba for hot loops if available.
"""

import math
import time

import numpy as np

try:
    from numba import jit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

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


class NumPyBackend:
    """Backend using NumPy arrays and manual backward. Optional Numba for hot loops."""

    def create_state(
        self,
        config: dict,
        uchars: list[str],
        *,
        seed: int | None = 42,
        init_from: dict[str, list[list[float]]] | None = None,
    ):
        """Build model state. If init_from is provided (e.g. from Python backend), use those weights."""
        model_cfg = config["model"]
        train_cfg = config["training"]
        n_embd = model_cfg["n_embd"]
        n_head = model_cfg["n_head"]
        n_layer = model_cfg["n_layer"]
        block_size = model_cfg["block_size"]
        vocab_size = len(uchars) + 1
        head_dim = n_embd // n_head

        shapes = _state_dict_shapes(vocab_size, n_embd, n_layer, block_size)
        state_dict = {}
        if init_from:
            for k, shape in shapes.items():
                arr = np.array(init_from[k], dtype=np.float64)
                assert arr.shape == shape, f"{k}: {arr.shape} vs {shape}"
                state_dict[k] = arr.copy()
        else:
            rng = np.random.default_rng(seed)
            for k, shape in shapes.items():
                std = 0.02
                if "attn_wo" in k or "mlp_fc2" in k:
                    std = 0.0
                state_dict[k] = rng.normal(0, std, size=shape).astype(np.float64)

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
        }

    def run_one_step(self, step: int, doc: str, state: dict, uchars: list[str]) -> tuple[float, dict]:
        """One training step: forward, backward, optimizer. Updates state in place."""
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

        tokens = [BOS] + [uchars.index(ch) for ch in doc if ch in uchars] + [BOS]
        if len(tokens) < 2:
            for g in grad_dict.values():
                g.fill(0)
            return 0.0, {"forward": 0, "backward": 0, "optimizer": 0}

        n = min(block_size, len(tokens) - 1)

        # Zero grads
        for g in grad_dict.values():
            g.fill(0)

        t0 = time.perf_counter()
        loss, cache = self._forward(state, tokens, n)
        t_forward = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._backward(state, tokens, n, cache, loss, grad_dict)
        t_backward = time.perf_counter() - t0

        t0 = time.perf_counter()
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

                x_attn = np.zeros(state["n_embd"], dtype=np.float64)
                for h in range(n_head):
                    hs = h * head_dim
                    q_h = q[hs : hs + head_dim]
                    k_h = np.array([cache["layers"][li]["keys"][t][hs : hs + head_dim] for t in range(len(cache["layers"][li]["keys"]))])
                    v_h = np.array([cache["layers"][li]["values"][t][hs : hs + head_dim] for t in range(len(cache["layers"][li]["values"]))])
                    out_h, weights, logits = _attention_fwd(q_h, k_h, v_h, head_dim)
                    cache["layers"][li]["q"].append(q_h.copy())
                    cache["layers"][li]["k"].append(k_h.copy())
                    cache["layers"][li]["v"].append(v_h.copy())
                    cache["layers"][li]["attn_weights"].append(weights.copy())
                    cache["layers"][li]["attn_logits"].append(logits.copy())
                    x_attn[hs : hs + head_dim] = out_h

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

                grad_x_in_attn = np.zeros(n_embd, dtype=np.float64)
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
