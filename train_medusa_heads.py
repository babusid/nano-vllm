"""
train_medusa_heads.py — Self-supervised training of MEDUSA-1 draft heads.

MEDUSA-1 keeps the base model frozen and trains only the K lightweight draft
heads.  Each head i is trained to predict the token at offset i+1 from every
position in the training corpus, using the base model's hidden states.

Because the base model never changes, its hidden states for a given text are
fixed.  This means you can cache the hidden states and train heads extremely
quickly — typical training time is < 1 GPU-hour on a small model.

Usage
-----
# Minimal (uses ShareGPT-style text from a local JSONL file):
MODEL_PATH=~/models/Qwen3-0.6B python train_medusa_heads.py \
    --data data.jsonl \
    --num-heads 4 \
    --epochs 2 \
    --output ~/models/Qwen3-0.6B/medusa_heads.safetensors

# With custom settings:
python train_medusa_heads.py \
    --model ~/models/Qwen3-1.7B \
    --data data.jsonl \
    --num-heads 4 \
    --num-layers 1 \
    --lr 1e-3 \
    --batch-size 4 \
    --max-seq-len 2048 \
    --epochs 3 \
    --output ~/models/Qwen3-1.7B/medusa_heads.safetensors

Data format
-----------
The --data file should be a JSONL file where each line is a JSON object with
a "text" field.  Any ShareGPT / Alpaca / instruction-tuning dataset works.
If no data file is provided a small synthetic dataset is generated for testing.

Checkpoint format
-----------------
The saved safetensors file uses keys compatible with load_medusa_heads():
    heads.{i}.blocks.{j}.linear.weight
    heads.{i}.blocks.{j}.linear.bias
    heads.{i}.lm_head.weight   (optional, skipped — tied to base model)
"""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from nanovllm.speculative.medusa import MedusaHeads, load_medusa_heads


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Tokenises a JSONL corpus and yields fixed-length token windows."""

    def __init__(self, texts: list[str], tokenizer, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: list[list[int]] = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True)
            # Slide non-overlapping windows
            for start in range(0, len(ids) - 1, max_seq_len):
                chunk = ids[start: start + max_seq_len + 1]
                if len(chunk) >= 2:
                    self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch):
    """Pad to longest sequence in the batch."""
    max_len = max(t.size(0) for t in batch)
    padded = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
    for i, t in enumerate(batch):
        padded[i, :t.size(0)] = t
    return padded


def synthetic_texts(n: int = 200) -> list[str]:
    """Generate trivial synthetic text for smoke-testing without real data."""
    words = ["the", "cat", "sat", "on", "a", "mat", "dog", "ran", "fast", "slow"]
    return [" ".join(random.choices(words, k=200)) for _ in range(n)]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load base model (frozen) ----------------------------------------
    print(f"[train] Loading base model from {args.model} …")
    from transformers import AutoConfig, AutoModelForCausalLM
    hf_config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    vocab_size = hf_config.vocab_size
    hidden_size = hf_config.hidden_size

    # ---- Build MEDUSA heads ----------------------------------------------
    heads = MedusaHeads(
        num_heads=args.num_heads,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=args.num_layers,
    ).to(dtype=torch.bfloat16, device=device)

    # Optionally resume from an existing checkpoint
    if args.resume and os.path.isfile(args.resume):
        load_medusa_heads(heads, args.model, args.resume)

    # Weight-tie lm_head to base model (so heads can exploit the same embedding)
    base_lm_head = base_model.lm_head
    for head in heads.heads:
        head.lm_head.weight = base_lm_head.weight  # shared, not a copy

    optimizer = torch.optim.AdamW(
        [p for p in heads.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.0,
    )

    # ---- Data ------------------------------------------------------------
    if args.data and os.path.isfile(args.data):
        print(f"[train] Loading data from {args.data} …")
        texts = []
        with open(args.data) as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj.get("text", obj.get("content", "")))
    else:
        print("[train] No data file provided — using synthetic text for testing.")
        texts = synthetic_texts(500)

    dataset = TextDataset(texts, tokenizer, max_seq_len=args.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    print(f"[train] Dataset: {len(dataset)} examples, {len(loader)} batches/epoch")

    # ---- Training --------------------------------------------------------
    for epoch in range(args.epochs):
        heads.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, tokens in enumerate(loader):
            tokens = tokens.to(device)  # [B, T+1]
            input_ids = tokens[:, :-1]  # [B, T]  — model input
            B, T = input_ids.shape

            # Forward through frozen base model → hidden states
            with torch.no_grad():
                outputs = base_model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                # Last layer hidden states: [B, T, H]
                hidden = outputs.hidden_states[-1].to(torch.bfloat16)

            # Compute loss for each head
            # Head i should predict tokens[:, i+1 : T+i+1]
            loss = torch.tensor(0.0, device=device)
            for i, head in enumerate(heads.heads):
                target_offset = i + 1  # head i predicts token at position t + (i+1)
                if target_offset >= T:
                    continue
                # Use hidden states at positions 0..T-target_offset-1
                h = hidden[:, : T - target_offset, :]  # [B, T-offset, H]
                logits = head(h)                         # [B, T-offset, V]
                targets = tokens[:, target_offset: T]   # [B, T-offset]
                loss = loss + F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    targets.reshape(-1),
                    ignore_index=0,  # ignore padding
                )

            loss = loss / args.num_heads
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 50 == 0:
                avg = total_loss / num_batches
                print(
                    f"  Epoch {epoch + 1}/{args.epochs}  "
                    f"batch {batch_idx + 1}/{len(loader)}  "
                    f"loss={avg:.4f}"
                )

        avg_loss = total_loss / max(num_batches, 1)
        print(f"[train] Epoch {epoch + 1} complete — avg loss = {avg_loss:.4f}")

    # ---- Save ------------------------------------------------------------
    output_path = args.output or os.path.join(args.model, "medusa_heads.safetensors")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Save only the head parameters (NOT the tied lm_head.weight — that lives
    # in the base model checkpoint already)
    state_dict = {}
    for name, param in heads.named_parameters():
        if "lm_head.weight" not in name:
            state_dict[name] = param.detach().cpu()

    from safetensors.torch import save_file
    save_file(state_dict, output_path)
    print(f"[train] Saved MEDUSA heads → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train MEDUSA-1 draft heads")
    p.add_argument("--model", default=os.path.expanduser(
        os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-0.6B/")),
        help="Path to the base model directory")
    p.add_argument("--data", default="",
                   help="Path to JSONL training data (each line: {\"text\": ...})")
    p.add_argument("--num-heads", type=int, default=4,
                   help="Number of MEDUSA draft heads to train (default: 4)")
    p.add_argument("--num-layers", type=int, default=1,
                   help="Number of ResBlock layers per head (default: 1)")
    p.add_argument("--max-seq-len", type=int, default=2048,
                   help="Maximum sequence length for training windows (default: 2048)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--output", default="",
                   help="Output path for saved heads (default: <model>/medusa_heads.safetensors)")
    p.add_argument("--resume", default="",
                   help="Path to an existing heads checkpoint to resume from")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)
