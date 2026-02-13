"""Sequential (position-by-position) NumPy backend."""

import math

import numpy as np

from . import shared
from .numpy_base import NumPyBackendBase


class NumPyBackend(NumPyBackendBase):
    """Sequential position-by-position forward/backward. Uses Numba for attention when available."""

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
            x_emb_raw = state_dict["wte"][token_id] + state_dict["wpe"][pos]
            cache["x_before_rms_emb"].append(x_emb_raw.copy())
            x, rms_scale_emb = shared.rmsnorm_fwd(x_emb_raw)
            cache["x_emb"].append(x.copy())
            cache["rms_scale_emb"].append(rms_scale_emb)

            for li in range(n_layer):
                x_res = x.copy()
                cache["layers"][li]["x_before_rms_attn"].append(x_res.copy())
                x, rms_scale_attn = shared.rmsnorm_fwd(x)
                cache["layers"][li]["x_in_attn"].append(x.copy())
                q = shared.linear_fwd(x, state_dict[f"layer{li}.attn_wq"])
                k = shared.linear_fwd(x, state_dict[f"layer{li}.attn_wk"])
                v = shared.linear_fwd(x, state_dict[f"layer{li}.attn_wv"])
                cache["layers"][li]["keys"].append(k.copy())
                cache["layers"][li]["values"].append(v.copy())
                cache["layers"][li]["rms_scale_attn"].append(rms_scale_attn)

                T = len(cache["layers"][li]["keys"])
                dt = state.get("dtype", np.float64)
                keys_arr = np.empty((T, state["n_embd"]), dtype=dt)
                values_arr = np.empty((T, state["n_embd"]), dtype=dt)
                for t in range(T):
                    keys_arr[t] = cache["layers"][li]["keys"][t]
                    values_arr[t] = cache["layers"][li]["values"][t]
                x_attn, weights_all, logits_all = shared.attention_all_heads_fused(
                    q, keys_arr, values_arr, head_dim, n_head
                )
                for h in range(n_head):
                    hs = h * head_dim
                    cache["layers"][li]["q"].append(q[hs : hs + head_dim].copy())
                    cache["layers"][li]["k"].append(keys_arr[:, hs : hs + head_dim].copy())
                    cache["layers"][li]["v"].append(values_arr[:, hs : hs + head_dim].copy())
                    cache["layers"][li]["attn_weights"].append(weights_all[h].copy())
                    cache["layers"][li]["attn_logits"].append(logits_all[h].copy())

                cache["layers"][li]["x_attn"].append(x_attn.copy())
                x = shared.linear_fwd(x_attn, state_dict[f"layer{li}.attn_wo"]) + x_res
                cache["layers"][li]["x_after_attn"].append(x.copy())

                x_res = x.copy()
                x, rms_scale_mlp = shared.rmsnorm_fwd(x)
                cache["layers"][li]["rms_scale_mlp"].append(rms_scale_mlp)
                x = shared.linear_fwd(x, state_dict[f"layer{li}.mlp_fc1"])
                cache["layers"][li]["relu2_pre"].append(x.copy())
                x = shared.relu2_fwd(x)
                x = shared.linear_fwd(x, state_dict[f"layer{li}.mlp_fc2"]) + x_res

            cache["x_final"].append(x.copy())
            logits = shared.linear_fwd(x, state_dict["lm_head"])
            probs = shared.softmax_fwd(logits)
            loss_t = -math.log(probs[target_id] + 1e-15)
            losses.append((loss_t, probs, target_id, logits))

        mean_loss = sum(l[0] for l in losses) / n
        cache["losses"] = losses
        cache["n"] = n
        return mean_loss, cache

    def _backward(
        self,
        state: dict,
        tokens: list[int],
        n: int,
        cache: dict,
        loss: float,
        grad_dict: dict,
    ) -> None:
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
            grad_logits = shared.softmax_nll_bwd(probs, target_id) * scale

            grad_x = state_dict["lm_head"].T @ grad_logits
            grad_dict["lm_head"] += np.outer(grad_logits, cache["x_final"][pos])

            for li in range(n_layer - 1, -1, -1):
                x_after_attn = cache["layers"][li]["x_after_attn"][pos]
                rms_scale_mlp = cache["layers"][li]["rms_scale_mlp"][pos]
                relu2_pre = cache["layers"][li]["relu2_pre"][pos]
                w_fc2 = state_dict[f"layer{li}.mlp_fc2"]
                w_fc1 = state_dict[f"layer{li}.mlp_fc1"]

                grad_mlp_out = grad_x.copy()
                grad_fc2_w, grad_fc2_in = shared.linear_bwd(
                    grad_mlp_out, relu2_pre, w_fc2
                )
                grad_dict[f"layer{li}.mlp_fc2"] += grad_fc2_w
                grad_relu2 = shared.relu2_bwd(grad_fc2_in, relu2_pre)
                x_after_rms_mlp = x_after_attn * rms_scale_mlp
                grad_fc1_w, grad_fc1_in = shared.linear_bwd(
                    grad_relu2, x_after_rms_mlp, w_fc1
                )
                grad_dict[f"layer{li}.mlp_fc1"] += grad_fc1_w
                grad_x_after_rms = shared.rmsnorm_bwd(
                    grad_fc1_in, x_after_attn, rms_scale_mlp
                )
                grad_x_after_attn = grad_x_after_rms + grad_mlp_out

                grad_attn_wo_out = grad_x_after_attn.copy()
                grad_attn_wo_w, grad_attn_in = shared.linear_bwd(
                    grad_attn_wo_out,
                    cache["layers"][li]["x_attn"][pos],
                    state_dict[f"layer{li}.attn_wo"],
                )
                grad_dict[f"layer{li}.attn_wo"] += grad_attn_wo_w

                grad_x_in_attn = np.zeros(n_embd, dtype=state.get("dtype", np.float64))
                for h in range(n_head):
                    hs = h * head_dim
                    g_o = grad_attn_in[hs : hs + head_dim]
                    idx = pos * n_head + h
                    q_h = cache["layers"][li]["q"][idx]
                    k_h = cache["layers"][li]["k"][idx]
                    v_h = cache["layers"][li]["v"][idx]
                    w = cache["layers"][li]["attn_weights"][idx]
                    logits_h = cache["layers"][li]["attn_logits"][idx]
                    g_q, g_k, g_v = shared.attention_bwd(
                        g_o, q_h, k_h, v_h, w, logits_h, head_dim
                    )
                    grad_x_in_attn += state_dict[f"layer{li}.attn_wq"][hs : hs + head_dim, :].T @ g_q
                    grad_dict[f"layer{li}.attn_wq"][hs : hs + head_dim, :] += np.outer(
                        g_q, cache["layers"][li]["x_in_attn"][pos]
                    )
                    for t in range(pos + 1):
                        x_t = cache["layers"][li]["x_in_attn"][t]
                        grad_dict[f"layer{li}.attn_wk"][hs : hs + head_dim, :] += np.outer(
                            g_k[t], x_t
                        )
                        grad_dict[f"layer{li}.attn_wv"][hs : hs + head_dim, :] += np.outer(
                            g_v[t], x_t
                        )
                    grad_x_in_attn += state_dict[f"layer{li}.attn_wk"][hs : hs + head_dim, :].T @ g_k[pos]
                    grad_x_in_attn += state_dict[f"layer{li}.attn_wv"][hs : hs + head_dim, :].T @ g_v[pos]

                grad_x_before_rms_attn = shared.rmsnorm_bwd(
                    grad_x_in_attn,
                    cache["layers"][li]["x_before_rms_attn"][pos],
                    cache["layers"][li]["rms_scale_attn"][pos],
                )
                grad_x = grad_x_after_attn + grad_x_before_rms_attn

            grad_emb = shared.rmsnorm_bwd(
                grad_x,
                cache["x_before_rms_emb"][pos],
                cache["rms_scale_emb"][pos],
            )
            grad_dict["wte"][tokens[pos]] += grad_emb
            grad_dict["wpe"][pos] += grad_emb
