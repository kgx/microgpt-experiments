"""
Shared GPT model: autograd Value, forward pass, and save/load.
Used by both train.py and inference (main.py).
"""

import json
import math
import random

# --- Autograd ---
class Value:
    """Stores a single scalar value and its gradient, as a node in a computation graph."""

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def gpt(token_id, pos_id, keys, values, state_dict, n_layer, n_head, head_dim, block_size):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() ** 2 for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits


# --- Persistence ---
def state_dict_to_json(state_dict):
    """Convert state_dict of Value matrices to JSON-serializable list-of-lists of floats."""
    out = {}
    for k, mat in state_dict.items():
        out[k] = [[p.data for p in row] for row in mat]
    return out


def state_dict_from_json(data, Value_class=Value):
    """Build state_dict of Value from loaded list-of-lists of floats."""
    out = {}
    for k, mat in data.items():
        out[k] = [[Value_class(v) for v in row] for row in mat]
    return out


def save_model(path, state_dict, uchars, n_embd, n_head, n_layer, block_size):
    vocab_size = len(uchars) + 1
    payload = {
        "uchars": list(uchars),
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "block_size": block_size,
        "vocab_size": vocab_size,
        "state_dict": state_dict_to_json(state_dict),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=0)


def load_model(path, Value_class=Value):
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
    state_dict = state_dict_from_json(payload["state_dict"], Value_class=Value_class)
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
    }


def make_matrix(nout, nin, std=0.02):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


def init_state_dict(vocab_size, n_embd, n_layer, block_size):
    """Initialize state_dict with Value parameters. Returns (state_dict, flat params list)."""
    state_dict = {
        "wte": make_matrix(vocab_size, n_embd),
        "wpe": make_matrix(block_size, n_embd),
        "lm_head": make_matrix(vocab_size, n_embd),
    }
    for i in range(n_layer):
        state_dict[f"layer{i}.attn_wq"] = make_matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wk"] = make_matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wv"] = make_matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wo"] = make_matrix(n_embd, n_embd, std=0)
        state_dict[f"layer{i}.mlp_fc1"] = make_matrix(4 * n_embd, n_embd)
        state_dict[f"layer{i}.mlp_fc2"] = make_matrix(n_embd, 4 * n_embd, std=0)
    params = [p for mat in state_dict.values() for row in mat for p in row]
    return state_dict, params
