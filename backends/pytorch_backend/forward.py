"""
Shared PyTorch forward: batched (training) and single-step (inference).
Same math as numpy_batched / model.gpt.
"""

import math

import torch
import torch.nn.functional as F


def _rmsnorm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """x (*, d) -> same shape. Scale so RMS per last dim is 1."""
    ms = (x * x).mean(dim=-1, keepdim=True) + eps
    scale = ms ** -0.5
    return x * scale


def _rmsnorm_with_scale(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """x (*, d) -> (out, scale) where scale is over last dim."""
    ms = (x * x).mean(dim=-1, keepdim=True) + eps
    scale = ms ** -0.5
    return x * scale, scale.squeeze(-1)


def _relu2(x: torch.Tensor) -> torch.Tensor:
    """ReLU(x)^2."""
    return F.relu(x) ** 2


def _causal_attention_batched(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head_dim: int, n_head: int
) -> torch.Tensor:
    """Q, K, V (T, n_embd). Causal multi-head attention. Returns out (T, n_embd)."""
    T = q.shape[0]
    scale = 1.0 / math.sqrt(head_dim)
    q_h = q.view(T, n_head, head_dim)
    k_h = k.view(T, n_head, head_dim)
    v_h = v.view(T, n_head, head_dim)
    # scores (T, n_head, T)
    scores = torch.einsum("ihd,jhd->ihj", q_h, k_h) * scale
    causal_mask = torch.triu(
        torch.full((T, T), float("-inf"), device=q.device, dtype=q.dtype), diagonal=1
    )
    scores = scores + causal_mask.unsqueeze(1)
    weights = F.softmax(scores, dim=-1)
    out = torch.einsum("ihj,jhd->ihd", weights, v_h)
    return out.reshape(T, -1)


def forward_batched(
    state_dict: dict[str, torch.Tensor],
    token_ids: torch.Tensor,
    target_ids: torch.Tensor,
    n_embd: int,
    n_head: int,
    n_layer: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Batched forward over T positions. token_ids (T,), target_ids (T,).
    Returns scalar loss (mean cross-entropy over positions).
    """
    T = token_ids.shape[0]
    wte = state_dict["wte"]
    wpe = state_dict["wpe"]
    x = wte[token_ids] + wpe[:T]
    x = _rmsnorm(x)

    for li in range(n_layer):
        x_res = x
        x = _rmsnorm(x)
        wq = state_dict[f"layer{li}.attn_wq"]
        wk = state_dict[f"layer{li}.attn_wk"]
        wv = state_dict[f"layer{li}.attn_wv"]
        wo = state_dict[f"layer{li}.attn_wo"]
        q = x @ wq.T
        k = x @ wk.T
        v = x @ wv.T
        x_attn = _causal_attention_batched(q, k, v, head_dim, n_head)
        x = x_attn @ wo.T + x_res
        x_res = x
        x = _rmsnorm(x)
        w1 = state_dict[f"layer{li}.mlp_fc1"]
        w2 = state_dict[f"layer{li}.mlp_fc2"]
        x = _relu2(x @ w1.T) @ w2.T + x_res

    lm_head = state_dict["lm_head"]
    logits = x @ lm_head.T
    loss = F.cross_entropy(logits, target_ids, reduction="mean")
    return loss


def _causal_attention_single_query(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head_dim: int, n_head: int
) -> torch.Tensor:
    """q (1, n_embd), k (T, n_embd), v (T, n_embd). Causal: one query attends to all T. Returns (1, n_embd)."""
    scale = 1.0 / math.sqrt(head_dim)
    q_h = q.view(1, n_head, head_dim)
    k_h = k.view(-1, n_head, head_dim)
    v_h = v.view(-1, n_head, head_dim)
    scores = torch.einsum("ihd,jhd->ihj", q_h, k_h) * scale
    weights = F.softmax(scores, dim=-1)
    out = torch.einsum("ihj,jhd->ihd", weights, v_h)
    return out.reshape(1, -1)


def forward_step(
    token_id: int,
    pos_id: int,
    keys: list[list[torch.Tensor]],
    values: list[list[torch.Tensor]],
    state_dict: dict[str, torch.Tensor],
    n_layer: int,
    n_head: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Single-step forward for inference. Mutates keys and values (appends new K, V per layer).
    Returns logits tensor of shape (vocab_size,).
    """
    wte = state_dict["wte"]
    wpe = state_dict["wpe"]
    # x: (1, n_embd)
    x = (wte[token_id : token_id + 1] + wpe[pos_id : pos_id + 1])
    x = _rmsnorm(x)

    for li in range(n_layer):
        x_res = x
        x = _rmsnorm(x)
        wq = state_dict[f"layer{li}.attn_wq"]
        wk = state_dict[f"layer{li}.attn_wk"]
        wv = state_dict[f"layer{li}.attn_wv"]
        wo = state_dict[f"layer{li}.attn_wo"]
        q = x @ wq.T
        k = x @ wk.T
        v = x @ wv.T
        keys[li].append(k)
        values[li].append(v)
        k_cat = torch.cat(keys[li], dim=0)
        v_cat = torch.cat(values[li], dim=0)
        x_attn = _causal_attention_single_query(q, k_cat, v_cat, head_dim, n_head)
        x = x_attn @ wo.T + x_res
        x_res = x
        x = _rmsnorm(x)
        w1 = state_dict[f"layer{li}.mlp_fc1"]
        w2 = state_dict[f"layer{li}.mlp_fc2"]
        x = _relu2(x @ w1.T) @ w2.T + x_res

    lm_head = state_dict["lm_head"]
    logits = (x @ lm_head.T).squeeze(0)
    return logits
