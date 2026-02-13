"""
Train the micro GPT on input.txt and persist the model to disk.
Run: python train.py [--output model.json]
"""

import argparse
import math
import os
import random

from model import (
    gpt,
    init_state_dict,
    save_model,
    softmax,
)

random.seed(42)

# Hyperparameters (match original micro GPT)
n_embd = 16
n_head = 4
n_layer = 1
block_size = 8
learning_rate = 1e-2
beta1, beta2, eps_adam = 0.9, 0.95, 1e-8
num_steps = 500


def main():
    parser = argparse.ArgumentParser(description="Train micro GPT and save model.")
    parser.add_argument("--output", "-o", default="model.json", help="Path to save model")
    parser.add_argument("--input", "-i", default="input.txt", help="Training data (one doc per line)")
    args = parser.parse_args()

    # Data: one document per line; download names.txt if no input.txt
    if not os.path.exists(args.input):
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt",
            args.input,
        )
    docs = [l.strip() for l in open(args.input).read().strip().split("\n") if l.strip()]
    random.shuffle(docs)
    print(f"num docs: {len(docs)}")

    # Tokenizer: unique chars → ids 0..vocab_size-2; BOS = len(uchars), so vocab_size = len(uchars)+1
    uchars = sorted(set("".join(docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    head_dim = n_embd // n_head
    print(f"vocab size: {vocab_size}")

    state_dict, params = init_state_dict(vocab_size, n_embd, n_layer, block_size)
    print(f"num params: {len(params)}")

    # Adam buffers (first and second moment)
    m = [0.0] * len(params)
    v = [0.0] * len(params)

    for step in range(num_steps):
        # One document per step; format: BOS + char ids + BOS
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)  # number of (input, target) pairs we use

        # Forward: at each position predict next token; accumulate cross-entropy loss
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values, state_dict, n_layer, n_head, head_dim, block_size)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()  # cross-entropy for one position
            losses.append(loss_t)
        loss = (1 / n) * sum(losses)

        loss.backward()

        # Adam update with cosine learning rate decay
        lr_t = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0

        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

    save_model(args.output, state_dict, uchars, n_embd, n_head, n_layer, block_size)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
