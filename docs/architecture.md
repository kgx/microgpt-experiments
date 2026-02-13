# MicroGPT architecture

A minimal, dependency-free GPT in pure Python: character-level language model with a single transformer block, custom autograd, and Adam training.

---

## Overview

- **Input**: List of documents (e.g. one name per line in `input.txt`).
- **Tokenization**: Each character is a token; vocab = unique chars + one special **BOS** (Beginning of Sequence) token. Token IDs: `0 .. len(uchars)-1` for chars, `len(uchars)` for BOS.
- **Model**: GPT-style decoder-only transformer. One token + one position in → **logits** over the next token. Training learns to predict the next token; inference samples from those logits autoregressively.
- **Training**: Cross-entropy loss over next-token predictions, backprop via a small autograd engine, Adam optimizer with cosine LR decay.
- **Inference**: Load saved weights, optionally condition on a prompt, then sample one token at a time (with temperature) until BOS or max length.

---

## Data and tokenizer

- **Documents**: `docs` = list of strings (e.g. names). Each doc is a sequence of characters.
- **Vocabulary**: `uchars` = sorted unique characters in the corpus. Token ID for char `c` = `uchars.index(c)`. **BOS** token ID = `len(uchars)`; used to mark start/end of a sequence.
- **vocab_size** = `len(uchars) + 1`.
- **Sequence format**: `[BOS, t_0, t_1, ..., t_k, BOS]`. Training and inference both use BOS at the start; generation stops when the model samples BOS.

---

## Model layout (GPT-style decoder)

Hyperparameters (in code): `n_embd=16`, `n_head=4`, `n_layer=1`, `block_size=8`, `head_dim = n_embd // n_head = 4`.

### 1. Embedding

- **Token embedding** `wte`: shape `(vocab_size, n_embd)`. Lookup by `token_id` → vector.
- **Position embedding** `wpe`: shape `(block_size, n_embd)`. Lookup by `pos_id` → vector.
- **Combined**: `x = wte[token_id] + wpe[pos_id]`, then **RMSNorm**(x).

### 2. Transformer block (repeated `n_layer` times)

Each layer has:

- **Causal self-attention**
  - RMSNorm → project to Q, K, V (each `n_embd`).
  - Current position’s K, V are appended to **KV cache** (keys/values per layer).
  - Split Q, K, V into `n_head` heads (each dim `head_dim`).
  - For each head: `attn_logits[t] = (Q · K[t]) / sqrt(head_dim)` for all positions `t` so far (causal: only past and present).
  - Softmax over those logits → attention weights; then `out = weighted sum of V`.
  - Concat heads, project with `attn_wo`, add residual.
- **MLP**
  - RMSNorm → linear `4*n_embd` → ReLU² → linear `n_embd` → add residual.

(No biases; ReLU² is a simple stand-in for GeLU.)

### 3. Unembed

- After all layers: linear project hidden state to **vocab_size** logits (language-model head `lm_head`).
- One forward call: input `(token_id, pos_id)` + current KV cache → **logits** over next token.

---

## Forward pass (one step)

`gpt(token_id, pos_id, keys, values, state_dict, ...)`:

1. Embed token and position; add and RMSNorm.
2. For each layer:
   - Compute Q, K, V; append K, V to `keys[li]`, `values[li]`.
   - Multi-head attention over **all positions stored in the cache** (causal: current position can attend to itself and all previous).
   - Add residual; then RMSNorm → MLP → add residual.
3. Linear to vocab_size → **logits**.

So the model is **stateful in the sense of the KV cache**: each call appends one (K, V) per layer. Training runs the forward step at positions 0, 1, 2, … for one document; inference runs it for context positions then for each generated token.

---

## Training

- **Per step**: Pick one document; tokenize to `[BOS, ...chars..., BOS]`; take the first `block_size` positions (or less if doc is short).
- **Loss**: For each position `pos_id` in `0..n-1`, run `gpt(tokens[pos_id], pos_id, ...)` → logits; softmax → probs; add `-log(probs[target])` where `target = tokens[pos_id+1]`. Average over positions = cross-entropy for next-token prediction.
- **Backprop**: `loss.backward()` over the shared autograd graph (all ops use `Value`); gradients are accumulated on every parameter.
- **Optimizer**: Adam (with beta1, beta2, eps); learning rate uses cosine decay from `learning_rate` to 0 over `num_steps`. Then zero grads and repeat.

Training produces one JSON file: tokenizer (`uchars`), config (n_embd, n_head, n_layer, block_size, vocab_size), and all weight matrices as floats.

---

## Inference

- **Load**: Read JSON; rebuild `state_dict` with `Value` wrappers (so the same `gpt` and `softmax` work); get `uchars`, BOS, block_size, etc.
- **Context**: Optional prompt string → tokenize to ids (skip chars not in vocab); context = `[BOS] + prompt_ids`, capped at `block_size`.
- **KV fill**: Run `gpt(context[pos], pos, ...)` for `pos = 0 .. n_context-2` with no sampling, so the KV cache is filled for the prompt.
- **Generation**: Start from `token_id = context[n_context-1]`, `pos_id = n_context-1`. Loop: run `gpt` → logits; scale by **temperature** (divide logits by T); softmax → sample next token; append to output; stop on BOS or when reaching `block_size`. Temperature > 0: higher = more random; lower = more greedy.
- **Samples**: Repeat the above `--samples` times (each time with a fresh KV cache). With a fixed `--seed`, the run is reproducible.

---

## File roles

| File | Role |
|------|------|
| **model.py** | Autograd `Value`, `linear` / `softmax` / `rmsnorm`, `gpt` forward, save/load model and tokenizer. |
| **train.py** | Load data, build tokenizer, init weights, training loop (loss + backward + Adam), save model. |
| **main.py** | Load model, parse CLI (prompt, temperature, seed, samples), run inference (KV fill + autoregressive sampling). |

---

## Design notes

- **No PyTorch**: All tensors are scalars (`Value`); matrices are lists of lists. Backprop is explicit over this graph.
- **Causal attention**: Implemented by the KV cache: at position `pos`, the cache has exactly positions `0..pos`, so attention is only over past and current.
- **BOS**: Used as start token and as end-of-sequence signal when generating; the model learns when to emit BOS to stop.
