"""
Pure-Python (Value-based) training backend. Uses model.gpt, model.softmax, and
custom autograd from model.py.
"""

import math
import time

from model import (
    gpt,
    init_state_dict,
    softmax,
    state_dict_to_json,
)


class PythonBackend:
    """Backend that uses the default Value-based model and autograd."""

    def create_state(self, config: dict, uchars: list[str], **kwargs):
        """Build model state and optimizer state. Returns a state dict for use in run_one_step and weights_for_export."""
        model_cfg = config["model"]
        train_cfg = config["training"]
        data_cfg = config.get("data", {})
        n_embd = model_cfg["n_embd"]
        n_head = model_cfg["n_head"]
        n_layer = model_cfg["n_layer"]
        block_size = model_cfg["block_size"]
        vocab_size = len(uchars) + 1
        # For chunked format, trailing_bos=false trains continuation across chunks (no BOS at chunk end).
        trailing_bos = data_cfg.get("trailing_bos", True)

        state_dict, params = init_state_dict(vocab_size, n_embd, n_layer, block_size)
        m = [0.0] * len(params)
        v = [0.0] * len(params)

        return {
            "state_dict": state_dict,
            "params": params,
            "m": m,
            "v": v,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "block_size": block_size,
            "head_dim": n_embd // n_head,
            "learning_rate": train_cfg["learning_rate"],
            "beta1": train_cfg["beta1"],
            "beta2": train_cfg["beta2"],
            "eps_adam": train_cfg["eps_adam"],
            "num_steps": train_cfg["num_steps"],
            "trailing_bos": trailing_bos,
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
        """
        Run one training step (forward, backward, optionally optimizer). Updates state in place.
        Returns (loss_value, timing_dict) with keys "forward", "backward", "optimizer".
        zero_grad/do_optimizer/grad_accum_count: see numpy_backend.run_one_step.
        """
        BOS = len(uchars)
        state_dict = state["state_dict"]
        params = state["params"]
        m, v = state["m"], state["v"]
        n_layer = state["n_layer"]
        n_head = state["n_head"]
        head_dim = state["head_dim"]
        block_size = state["block_size"]
        learning_rate = state["learning_rate"]
        beta1, beta2 = state["beta1"], state["beta2"]
        eps_adam = state["eps_adam"]
        num_steps = state["num_steps"]
        trailing_bos = state.get("trailing_bos", True)

        char_ids = [uchars.index(ch) for ch in doc if ch in uchars]
        tokens = [BOS] + char_ids + ([BOS] if trailing_bos else [])
        if len(tokens) < 2:
            if zero_grad:
                for p in params:
                    p.grad = 0
            return 0.0, {"forward": 0, "backward": 0, "optimizer": 0}

        n = min(block_size, len(tokens) - 1)
        if zero_grad:
            for p in params:
                p.grad = 0
        keys = [[] for _ in range(n_layer)]
        values = [[] for _ in range(n_layer)]
        losses = []

        t0 = time.perf_counter()
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values, state_dict, n_layer, n_head, head_dim, block_size)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        loss = (1 / n) * sum(losses)
        t_forward = time.perf_counter() - t0

        t0 = time.perf_counter()
        loss.backward()
        t_backward = time.perf_counter() - t0

        t0 = time.perf_counter()
        if do_optimizer:
            scale = 1.0 / grad_accum_count if grad_accum_count > 1 else 1.0
            lr_t = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
            for i, p in enumerate(params):
                g = p.grad * scale
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** (step + 1))
                v_hat = v[i] / (1 - beta2 ** (step + 1))
                p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            for p in params:
                p.grad = 0
        t_optimizer = time.perf_counter() - t0

        return loss.data, {"forward": t_forward, "backward": t_backward, "optimizer": t_optimizer}

    def weights_for_export(self, state: dict) -> dict:
        """Return state_dict in JSON-serializable form (list-of-lists of floats)."""
        return state_dict_to_json(state["state_dict"])
