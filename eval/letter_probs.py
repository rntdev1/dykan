"""Diagnostic: the probability a model gives each answer letter on single-choice exam tasks.

    python -m eval.letter_probs --model google/gemma-4-E2B-it [--adapter DIR] --zno all.jsonl.gz [--zno-split dev]

Greedy answers hide whether a model that keeps answering one letter is confidently
wrong or merely has a tilt on top of a flat distribution. This reads the
distribution itself, right after "Відповідь:", and reports plain accuracy, how
peaked the distributions are, the average probability of each letter (the tilt),
and accuracy after that tilt is divided out. If the last number is clearly higher
than the first, the knowledge is there and only a letter bias gets in the way.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from data.sources.zno_eval import load
from data.split import carve_dev, split_exam_tasks
from eval import zno

LETTERS = "АБВГД"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--zno", type=Path, required=True)
    ap.add_argument("--zno-split", choices=["test", "dev"], default="test")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch

    from eval.run_hf import Runner

    runner = Runner(args.model, None, args.adapter)
    tok = runner.processor.tokenizer
    ids = [tok("Відповідь: " + letter, add_special_tokens=False)["input_ids"][-1] for letter in LETTERS]
    assert len(set(ids)) == len(LETTERS), ids

    train, test, _ = split_exam_tasks(list(load(args.zno)))
    tasks = [t for t in (test if args.zno_split == "test" else carve_dev(train)[1])
             if zno.eligible(t) and t.question_type == "single_choice" and t.correct[0] in LETTERS]
    tasks = tasks[: args.limit] if args.limit else tasks
    rows = []
    for task in tasks:
        prompt = zno.build_prompt(task) + " Пояснення не потрібні."
        text = runner.processor.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False) + "Відповідь:"
        inputs = tok(text, add_special_tokens=False, return_tensors="pt").to(runner.model.device)
        with torch.inference_mode():
            logits = runner.model(**inputs).logits[0, -1].float()
        allowed = [i for i, letter in enumerate(LETTERS) if any(label == letter for label, _ in task.options)]
        probs = torch.softmax(logits[[ids[i] for i in allowed]], dim=-1).tolist()
        rows.append({"subject": task.subject, "gold": LETTERS.index(task.correct[0]), "allowed": allowed, "probs": probs,
                     "letter_mass": torch.softmax(logits, dim=-1)[ids].sum().item()})

    def accuracy(prior: list[float] | None) -> float:
        hits = 0
        for row in rows:
            scores = [p / (prior[i] if prior else 1.0) for i, p in zip(row["allowed"], row["probs"])]
            hits += row["allowed"][scores.index(max(scores))] == row["gold"]
        return hits / len(rows)

    mean = [0.0] * len(LETTERS)
    for row in rows:
        for i, p in zip(row["allowed"], row["probs"]):
            mean[i] += p / len(rows)
    pred = Counter(LETTERS[row["allowed"][row["probs"].index(max(row["probs"]))]] for row in rows)
    entropy = sum(-sum(p * math.log(p) for p in row["probs"] if p > 0) / math.log(len(row["probs"])) for row in rows) / len(rows)
    out = {
        "tasks": len(rows), "accuracy": round(accuracy(None), 4), "accuracy_tilt_removed": round(accuracy(mean), 4),
        "mean_prob_of_gold": round(sum(row["probs"][row["allowed"].index(row["gold"])] for row in rows) / len(rows), 4),
        "mean_top_prob": round(sum(max(row["probs"]) for row in rows) / len(rows), 4), "relative_entropy": round(entropy, 4),
        "mean_prob_per_letter": {letter: round(m, 3) for letter, m in zip(LETTERS, mean)}, "predicted": dict(pred.most_common()),
        "gold": dict(Counter(LETTERS[row["gold"]] for row in rows).most_common()),
        "mass_on_letters": round(sum(row["letter_mass"] for row in rows) / len(rows), 4),
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
