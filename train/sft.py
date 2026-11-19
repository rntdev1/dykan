"""LoRA instruction tuning for Gemma 4 E2B/E4B (text only).

    python -m train.sft --model google/gemma-4-E2B-it --data outputs/pilot.jsonl --out checkpoints/pilot-e2b --merge

A plain PyTorch loop, so the loss mask and batching are explicit:
  * loss only on what the model writes (train/masking.py);
  * batches are built to a token budget, since tool episodes are ~10x longer
    than exam items;
  * embeddings, the vision and audio encoders and everything outside the LoRA
    adapters stay frozen;
  * no gradient checkpointing (it triggers a use_cache bug on E2B/E4B).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from train.masking import supervised_spans, token_mask


def encode(tokenizer, row: dict, max_len: int) -> dict | None:
    # Rows marked "thinking" carry the working in the thinking channel and are rendered with thinking switched on.
    rendered = tokenizer.apply_chat_template(row["messages"], tools=row.get("tools") or None, tokenize=False, add_generation_prompt=False,
                                             enable_thinking=bool(row.get("thinking")))
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if len(ids) > max_len:
        return None
    mask = token_mask(enc["offset_mapping"], supervised_spans(rendered))
    if not any(mask):
        return None
    return {"ids": ids, "labels": [t if m else -100 for t, m in zip(ids, mask)], "source": row.get("source", "")}


def encode_text(tokenizer, row: dict, max_len: int) -> list[dict]:
    """A plain-text document ({"text": ...}, continued pretraining): cut into windows of
    max_len tokens, each starting with <bos>, with the loss on every token after it.
    A short last window is dropped unless it is the only one."""
    ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
    body = max_len - 1
    out = []
    for start in range(0, len(ids), body):
        piece = ids[start:start + body]
        if out and len(piece) < body // 4:
            break
        out.append({"ids": [tokenizer.bos_token_id] + piece, "labels": [-100] + piece, "source": row.get("source", "text")})
    return out


def encode_rows(tokenizer, rows: list[dict], max_len: int) -> list[dict]:
    examples = []
    for row in rows:
        if "text" in row:
            examples += encode_text(tokenizer, row, max_len)
        else:
            example = encode(tokenizer, row, max_len)
            if example is not None:
                examples.append(example)
    return examples


def make_batches(examples: list[dict], token_budget: int, rng: random.Random) -> list[list[dict]]:
    """Length-sorted batches under a padded-token budget, in shuffled order."""
    ordered = sorted(examples, key=lambda e: len(e["ids"]))
    batches, current, longest = [], [], 0
    for ex in ordered:
        longest_if_added = max(longest, len(ex["ids"]))
        if current and longest_if_added * (len(current) + 1) > token_budget:
            batches.append(current)
            current, longest_if_added = [], len(ex["ids"])
        current.append(ex)
        longest = longest_if_added
    if current:
        batches.append(current)
    rng.shuffle(batches)
    return batches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True, nargs="+", help="JSONL files: chat rows {messages, tools} and/or plain-text rows {text}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--token-budget", type=int, default=8000, help="padded tokens per batch")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--attn", default=None, help="attention implementation passed to from_pretrained (eager, sdpa)")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", action="store_true", help="also save merged full weights for eval")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer

    rows = []
    for path in args.data:  # several files are mixed: chat examples and plain-text documents
        with path.open(encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    rows = rows[: args.limit] if args.limit else rows
    examples = encode_rows(tokenizer, rows, args.max_len)
    total_tokens = sum(len(e["ids"]) for e in examples)
    supervised = sum(sum(l != -100 for l in e["labels"]) for e in examples)
    print(f"examples {len(examples)}/{len(rows)} | tokens {total_tokens:,} | supervised {supervised:,} ({supervised / total_tokens:.1%})", flush=True)

    # Only the language model's attention is switched; the audio tower supports eager only.
    extra = {"attn_implementation": {"text_config": args.attn, "vision_config": "eager", "audio_config": "eager", "": "eager"}} if args.attn else {}
    model = AutoModelForMultimodalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0}, **extra)
    print("attention:", getattr(model.config.get_text_config(), "_attn_implementation", None), flush=True)
    # Adapters on every attention and MLP projection of the language model only.
    targets = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=args.dropout, bias="none", task_type="CAUSAL_LM", target_modules=targets))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    outside_lm = [n for n, _ in trainable if any(k in n for k in ("vision", "audio", "embed"))]
    print(f"trainable tensors {len(trainable)} | params {sum(p.numel() for _, p in trainable):,} | outside language model: {len(outside_lm)}", flush=True)
    print("sample:", [n for n, _ in trainable[:3]], flush=True)
    assert not outside_lm, outside_lm[:5]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    device = model.device

    def collate(batch: list[dict]) -> dict:
        width = max(len(e["ids"]) for e in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        attention = torch.zeros((len(batch), width), dtype=torch.long)
        for i, e in enumerate(batch):
            n = len(e["ids"])
            ids[i, :n] = torch.tensor(e["ids"])
            labels[i, :n] = torch.tensor(e["labels"])
            attention[i, :n] = 1
        return {"input_ids": ids.to(device), "attention_mask": attention.to(device), "labels": labels.to(device)}

    core = model.base_model.model  # the Gemma4ForConditionalGeneration under the PEFT wrapper
    softcap = core.config.get_text_config().final_logit_softcapping

    def backward_on_supervised(inputs: dict, chunk: int = 256) -> float:
        """Cross-entropy on supervised positions only, backpropagated; returns the loss.

        The output layer has 262k entries per position. Computing it for every
        token (as model(labels=...) does) needs several GB per thousand tokens, and
        even for supervised positions alone a long answer does not fit. The output
        layer is frozen, so the loss is taken in chunks: each chunk's gradient with
        respect to the hidden states is collected and its logits freed, and one
        backward pass through the model follows."""
        hidden = core.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
        targets_next = inputs["labels"][:, 1:]
        keep = targets_next != -100
        states, targets = hidden[:, :-1][keep], targets_next[keep]
        leaf = states.detach().requires_grad_(True)
        total = 0.0
        for start in range(0, len(targets), chunk):
            logits = core.lm_head(leaf[start:start + chunk])
            if softcap is not None:
                logits = torch.tanh(logits / softcap) * softcap
            part = torch.nn.functional.cross_entropy(logits.float(), targets[start:start + chunk], reduction="sum") / len(targets)
            part.backward()
            total += part.item()
        states.backward(leaf.grad)
        return total

    epochs_full = math.ceil(args.epochs)
    schedule = []
    for _ in range(epochs_full):
        schedule += make_batches(examples, args.token_budget, rng)
    steps = max(1, int(len(schedule) * args.epochs / epochs_full))
    schedule = schedule[:steps]
    warmup = max(10, steps // 30)

    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0.0)
    lr_at = lambda s: (s + 1) / warmup if s < warmup else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "train_log.jsonl").open("w", encoding="utf-8")
    model.train()
    started, seen_tokens, window, skipped = time.time(), 0, [], 0
    for step, batch in enumerate(schedule, 1):
        inputs = collate(batch)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = backward_on_supervised(inputs)
            if not math.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
        except torch.OutOfMemoryError:
            # One oversized batch must not kill the run: drop it and carry on.
            del inputs
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            skipped += 1
            scheduler.step()
            print(json.dumps({"step": step, "skipped_oom": skipped, "batch": len(batch), "width": max(len(e["ids"]) for e in batch)}), flush=True)
            if skipped > 20 and skipped > 0.02 * step:
                raise RuntimeError(f"{skipped} of {step} batches did not fit in memory; lower --token-budget")
            continue
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        seen_tokens += int(inputs["attention_mask"].sum())
        window.append(loss)
        if step == 1 or step % 20 == 0 or step == steps:
            elapsed = time.time() - started
            record = {"step": step, "steps": steps, "loss": round(sum(window) / len(window), 4), "lr": scheduler.get_last_lr()[0],
                      "tok_per_s": round(seen_tokens / elapsed), "minutes": round(elapsed / 60, 1),
                      "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1), "skipped_oom": skipped}
            window = []
            if step % 500 == 0:
                model.save_pretrained(args.out / "adapter")
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
        if step == steps // 2:  # a snapshot at the halfway point (end of epoch 1 of 2), to compare with the final one
            model.save_pretrained(args.out / "adapter-half")

    model.save_pretrained(args.out / "adapter")
    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(args.out / "merged", safe_serialization=True)
        processor.save_pretrained(args.out / "merged")
        print(f"merged weights saved to {args.out / 'merged'}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
