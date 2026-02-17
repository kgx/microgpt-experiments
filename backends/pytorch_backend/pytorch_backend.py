"""PyTorch training backend: batched forward with autograd and Adam."""

import math
import time

import torch

from . import shared
from .forward import forward_batched


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


class PyTorchBackend:
    """Training backend using PyTorch with batched forward and autograd."""

    def create_state(
        self,
        config: dict,
        uchars: list[str],
        *,
        seed: int | None = 42,
        init_from: dict[str, list[list[float]]] | None = None,
        dtype: str | torch.dtype | None = None,
        device: str | torch.device | None = None,
    ) -> dict:
        """Build model state. If init_from is provided, copy those weights."""
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

        dev = _resolve_device(device)
        if dtype is None:
            dtype = train_cfg.get("dtype", "float32")
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, torch.float32)

        shapes = shared.state_dict_shapes(vocab_size, n_embd, n_layer, block_size)
        state_dict = {}
        if init_from:
            for k, shape in shapes.items():
                t = torch.tensor(init_from[k], dtype=dtype, device=dev)
                assert t.shape == torch.Size(shape), f"{k}: {t.shape} vs {shape}"
                state_dict[k] = torch.nn.Parameter(t)
        else:
            # Use CPU generator to avoid CUDA/MPS library load issues; init on CPU then move to device
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)
            for k, shape in shapes.items():
                std = 0.02
                if "attn_wo" in k or "mlp_fc2" in k:
                    std = 0.0
                t = torch.empty(shape, dtype=dtype, device="cpu")
                torch.nn.init.normal_(t, 0, std, generator=generator)
                state_dict[k] = torch.nn.Parameter(t.to(dev))

        ordered_keys = shared.ordered_state_keys(n_layer)
        params = [state_dict[k] for k in ordered_keys]
        optimizer = torch.optim.Adam(
            params,
            lr=train_cfg["learning_rate"],
            betas=(train_cfg["beta1"], train_cfg["beta2"]),
            eps=train_cfg["eps_adam"],
        )

        return {
            "state_dict": state_dict,
            "optimizer": optimizer,
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
            "device": dev,
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
        """One training step: forward, backward, optionally optimizer."""
        BOS = len(uchars)
        state_dict = state["state_dict"]
        optimizer = state["optimizer"]
        n_layer = state["n_layer"]
        n_head = state["n_head"]
        head_dim = state["head_dim"]
        n_embd = state["n_embd"]
        block_size = state["block_size"]
        vocab_size = state["vocab_size"]
        num_steps = state["num_steps"]
        learning_rate = state["learning_rate"]
        trailing_bos = state.get("trailing_bos", True)
        dev = state["device"]

        char_ids = [uchars.index(ch) for ch in doc if ch in uchars]
        tokens = [BOS] + char_ids + ([BOS] if trailing_bos else [])
        if len(tokens) < 2:
            if zero_grad:
                optimizer.zero_grad(set_to_none=True)
            return 0.0, {"forward": 0.0, "backward": 0.0, "optimizer": 0.0}

        n = min(block_size, len(tokens) - 1)
        token_ids = torch.tensor([tokens[i] for i in range(n)], dtype=torch.long, device=dev)
        target_ids = torch.tensor([tokens[i + 1] for i in range(n)], dtype=torch.long, device=dev)

        if zero_grad:
            optimizer.zero_grad(set_to_none=True)

        t0 = time.perf_counter()
        loss = forward_batched(
            state_dict, token_ids, target_ids, n_embd, n_head, n_layer, head_dim
        )
        t_forward = time.perf_counter() - t0

        t0 = time.perf_counter()
        loss.backward()
        t_backward = time.perf_counter() - t0

        t0 = time.perf_counter()
        if do_optimizer:
            if grad_accum_count > 1:
                for p in state["params"]:
                    if p.grad is not None:
                        p.grad.div_(grad_accum_count)
            lr_t = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
            for g in optimizer.param_groups:
                g["lr"] = lr_t
            optimizer.step()
        t_optimizer = time.perf_counter() - t0

        return float(loss.item()), {
            "forward": t_forward,
            "backward": t_backward,
            "optimizer": t_optimizer,
        }

    def weights_for_export(self, state: dict) -> dict[str, list[list[float]]]:
        """Return state_dict as list-of-lists for JSON."""
        return {
            k: v.detach().cpu().tolist()
            for k, v in state["state_dict"].items()
        }
