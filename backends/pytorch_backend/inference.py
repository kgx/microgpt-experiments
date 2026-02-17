"""PyTorch inference: load model from JSON and single-step forward."""

import json

import torch

from .forward import forward_step
from .pytorch_backend import _resolve_device


def load_model_pytorch(
    path: str,
    device: str | torch.device | None = None,
) -> dict:
    """
    Load model from JSON into PyTorch tensors on the given device.
    Returns same structure as model.load_model(): uchars, BOS, vocab_size, n_embd,
    n_head, n_layer, block_size, head_dim, state_dict (torch tensors).
    """
    dev = _resolve_device(device)
    with open(path) as f:
        payload = json.load(f)
    uchars = payload["uchars"]
    BOS = len(uchars)
    vocab_size = payload["vocab_size"]
    n_embd = payload["n_embd"]
    n_head = payload["n_head"]
    n_layer = payload["n_layer"]
    block_size = payload["block_size"]
    head_dim = n_embd // n_head

    state_dict = {}
    for k, mat in payload["state_dict"].items():
        t = torch.tensor(mat, dtype=torch.float32, device=dev)
        state_dict[k] = t

    return {
        "uchars": uchars,
        "BOS": BOS,
        "vocab_size": vocab_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "block_size": block_size,
        "head_dim": head_dim,
        "state_dict": state_dict,
        "backend": "pytorch",
    }


def gpt_step_pytorch(
    token_id: int,
    pos_id: int,
    keys: list[list[torch.Tensor]],
    values: list[list[torch.Tensor]],
    data: dict,
) -> list[float]:
    """
    One forward step for inference. Mutates keys and values.
    Returns logits as a list of floats (for sample_with_options).
    """
    with torch.no_grad():
        logits = forward_step(
            token_id,
            pos_id,
            keys,
            values,
            data["state_dict"],
            data["n_layer"],
            data["n_head"],
            data["head_dim"],
            next(iter(data["state_dict"].values())).device,
        )
    return logits.cpu().tolist()
