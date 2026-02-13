"""Batched-over-positions NumPy backend. All batched-specific forward/backward logic lives here."""

import numpy as np

from . import shared
from .numpy_base import NumPyBackendBase


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

    def _forward_batched(self, state: dict, tokens: list[int], n: int) -> tuple[float, dict]:
        """Batched forward over n positions in one pass."""
        state_dict = state["state_dict"]
        n_layer = state["n_layer"]
        n_head = state["n_head"]
        head_dim = state["head_dim"]
        n_embd = state["n_embd"]
        T = n

        token_ids = np.array([tokens[i] for i in range(T)], dtype=np.intp)
        target_ids = np.array([tokens[i + 1] for i in range(T)], dtype=np.intp)

        x = state_dict["wte"][token_ids] + state_dict["wpe"][:T]
        cache = {"x_before_rms_emb": x.copy(), "layers": []}
        x, rms_scale_emb = shared.rmsnorm_fwd_batched(x)
        cache["x_emb"] = x.copy()
        cache["rms_scale_emb"] = rms_scale_emb.copy()

        for li in range(n_layer):
            x_res = x.copy()
            cache_li = {"x_before_rms_attn": x_res.copy()}
            x, rms_scale_attn = shared.rmsnorm_fwd_batched(x)
            cache_li["x_in_attn"] = x.copy()
            cache_li["rms_scale_attn"] = rms_scale_attn.copy()
            q = shared.linear_fwd_batched(x, state_dict[f"layer{li}.attn_wq"])
            k = shared.linear_fwd_batched(x, state_dict[f"layer{li}.attn_wk"])
            v = shared.linear_fwd_batched(x, state_dict[f"layer{li}.attn_wv"])
            cache_li["q"], cache_li["k"], cache_li["v"] = q.copy(), k.copy(), v.copy()
            x_attn, attn_weights = shared.causal_attention_batched_fwd(
                q, k, v, head_dim, n_head
            )
            cache_li["attn_weights"] = attn_weights.copy()
            cache_li["x_attn"] = x_attn.copy()
            x = shared.linear_fwd_batched(x_attn, state_dict[f"layer{li}.attn_wo"]) + x_res
            cache_li["x_after_attn"] = x.copy()
            x_res = x.copy()
            x, rms_scale_mlp = shared.rmsnorm_fwd_batched(x)
            cache_li["rms_scale_mlp"] = rms_scale_mlp.copy()
            x = shared.linear_fwd_batched(x, state_dict[f"layer{li}.mlp_fc1"])
            cache_li["relu2_pre"] = x.copy()
            x = shared.relu2_fwd(x)
            x = shared.linear_fwd_batched(x, state_dict[f"layer{li}.mlp_fc2"]) + x_res
            cache["layers"].append(cache_li)

        cache["x_final"] = x.copy()
        logits = shared.linear_fwd_batched(x, state_dict["lm_head"])
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
        """Backward for batched forward. (loss unused, for signature compatibility.)"""
        state_dict = state["state_dict"]
        n_layer = state["n_layer"]
        n_head = state["n_head"]
        head_dim = state["head_dim"]
        T = n
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
            grad_fc2_w, grad_fc2_in = shared.linear_bwd_batched(
                grad_mlp_out, c["relu2_pre"], state_dict[f"layer{li}.mlp_fc2"]
            )
            grad_dict[f"layer{li}.mlp_fc2"] += grad_fc2_w
            grad_relu2 = shared.relu2_bwd(grad_fc2_in, c["relu2_pre"])
            x_after_rms = c["x_after_attn"] * c["rms_scale_mlp"][:, np.newaxis]
            grad_fc1_w, grad_fc1_in = shared.linear_bwd_batched(
                grad_relu2, x_after_rms, state_dict[f"layer{li}.mlp_fc1"]
            )
            grad_dict[f"layer{li}.mlp_fc1"] += grad_fc1_w
            grad_x_after_rms = shared.rmsnorm_bwd_batched(
                grad_fc1_in, c["x_after_attn"], c["rms_scale_mlp"]
            )
            grad_x_after_attn = grad_x_after_rms + grad_mlp_out

            grad_attn_wo_out = grad_x_after_attn.copy()
            grad_attn_wo_w, grad_attn_in = shared.linear_bwd_batched(
                grad_attn_wo_out, c["x_attn"], state_dict[f"layer{li}.attn_wo"]
            )
            grad_dict[f"layer{li}.attn_wo"] += grad_attn_wo_w
            grad_q, grad_k, grad_v = shared.causal_attention_batched_bwd(
                grad_attn_in, c["q"], c["k"], c["v"], c["attn_weights"], head_dim, n_head
            )
            grad_x_in_attn = (
                grad_q @ state_dict[f"layer{li}.attn_wq"]
                + grad_k @ state_dict[f"layer{li}.attn_wk"]
                + grad_v @ state_dict[f"layer{li}.attn_wv"]
            )
            grad_dict[f"layer{li}.attn_wq"] += grad_q.T @ c["x_in_attn"]
            grad_dict[f"layer{li}.attn_wk"] += grad_k.T @ c["x_in_attn"]
            grad_dict[f"layer{li}.attn_wv"] += grad_v.T @ c["x_in_attn"]
            grad_x_before_rms_attn = shared.rmsnorm_bwd_batched(
                grad_x_in_attn, c["x_before_rms_attn"], c["rms_scale_attn"]
            )
            grad_x = grad_x_after_attn + grad_x_before_rms_attn

        grad_emb = shared.rmsnorm_bwd_batched(
            grad_x, cache["x_before_rms_emb"], cache["rms_scale_emb"]
        )
        token_ids = np.array([tokens[i] for i in range(T)], dtype=np.intp)
        for i in range(T):
            grad_dict["wte"][token_ids[i]] += grad_emb[i]
        grad_dict["wpe"][:T] += grad_emb
