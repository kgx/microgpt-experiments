"""NumPy backend base class: state creation, run_one_step loop, optimizer, export."""

import math
import os
import time

import numpy as np

from abc import ABC, abstractmethod

from . import shared


_JIT_VERIFY_DONE = False


class NumPyBackendBase(ABC):
    """Shared base for NumPy backends: state creation, run_one_step loop, optimizer, export.
    Subclasses must implement _forward and _backward."""

    def create_state(
        self,
        config: dict,
        uchars: list[str],
        *,
        seed: int | None = 42,
        init_from: dict[str, list[list[float]]] | None = None,
        dtype: type | str | None = None,
    ):
        """Build model state. If init_from is provided (e.g. from Python backend), use those weights."""
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

        shapes = shared.state_dict_shapes(vocab_size, n_embd, n_layer, block_size)
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

        ordered_keys = shared.ordered_state_keys(n_layer)
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
        """One training step: forward, backward, and optionally optimizer."""
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
            ok, msg = shared.verify_numba_jit()
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

    @abstractmethod
    def _forward(self, state: dict, tokens: list[int], n: int) -> tuple[float, dict]:
        """Forward over n positions; return mean loss and cache for backward."""
        ...

    @abstractmethod
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
        ...

    def weights_for_export(self, state: dict) -> dict:
        """Return state_dict as list-of-lists for JSON."""
        return {k: v.tolist() for k, v in state["state_dict"].items()}
