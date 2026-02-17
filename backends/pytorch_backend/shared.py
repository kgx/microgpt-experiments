"""
Shared PyTorch backend: state dict shapes and ordered keys (mirror of numpy_backend/shared.py).
"""


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
