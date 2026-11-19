"""Rank candidate documents by how close they are in meaning to school material.

    python -m data.corpus.semantic --in outputs/corpus/wiki.jsonl --school outputs/train/dykan-v1.jsonl --out outputs/corpus/wiki.ranked.jsonl

Word overlap (relevance.py) could not tell a physics article from a biography that
happens to share common words. Here every candidate's opening is embedded with a
small multilingual model (intfloat/multilingual-e5-small, MIT licence) and
compared with a set of "school" queries: the syllabus topics, real exam questions
and the questions behind the generated explanations and tasks. The score is the
mean similarity to the closest few queries. Output: the same rows with a "sem"
field, best first, and a .tsv with score, words and the title or URL.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data.teacher.syllabus import SYLLABUS

MODEL = "intfloat/multilingual-e5-small"


def school_queries(path: Path, per_source: int = 2500, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    queries = [f"{subject}, {grade} клас: {topic}" for subject, items in SYLLABUS.items() for grade, topic in items]
    by_source: dict[str, list[str]] = {"exam_key": [], "explain": [], "tasks": []}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["source"] in by_source:
                user = next(m["content"] for m in row["messages"] if m["role"] == "user")
                # Exam prompts start with a fixed header and end with format instructions; keep the task itself.
                by_source[row["source"]].append(user.split("\n\n")[1] if row["source"] == "exam_key" and user.count("\n\n") >= 2 else user)
    for items in by_source.values():
        queries += rng.sample(items, min(per_source, len(items)))
    return queries


def head(text: str, words: int = 220) -> str:
    return " ".join(text.split()[:words])


class Embedder:
    def __init__(self, device: str = "cpu", batch: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.device, self.batch = torch, device, batch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModel.from_pretrained(MODEL).to(device).eval()

    def __call__(self, texts: list[str], prefix: str, log: str | None = None):
        torch, out = self.torch, []
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths together: less padding
        with torch.inference_mode():
            for start in range(0, len(order), self.batch):
                chunk = [prefix + texts[i] for i in order[start:start + self.batch]]
                enc = self.tokenizer(chunk, padding=True, truncation=True, max_length=384, return_tensors="pt").to(self.device)
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                out.append(torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1).cpu())
                if log and (start // self.batch) % 200 == 0:
                    print(f"{log}: {start:,}/{len(texts):,}", flush=True)
        stacked = torch.cat(out)
        result = torch.empty_like(stacked)
        result[torch.tensor(order)] = stacked
        return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--school", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top", type=int, default=5, help="score = mean similarity to this many closest queries")
    args = ap.parse_args()

    import torch

    rows = [json.loads(line) for line in args.src.open(encoding="utf-8")]
    embed = Embedder(args.device)
    queries = embed(school_queries(args.school), "query: ", log="queries")
    docs = embed([(row.get("title", "") + ". " if row.get("title") else "") + head(row["text"]) for row in rows], "passage: ", log="documents")
    scores = torch.cat([(docs[i:i + 4096] @ queries.T).topk(args.top, dim=1).values.mean(1) for i in range(0, len(rows), 4096)]).tolist()
    for row, s in zip(rows, scores):
        row["sem"] = round(s, 4)
    rows.sort(key=lambda r: -r["sem"])
    with args.out.open("w", encoding="utf-8") as f, args.out.with_suffix(".tsv").open("w", encoding="utf-8") as tsv:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            label = row.get("title") or row.get("url", "")[:110] + " | " + row["text"][:90].replace("\n", " ").replace("\t", " ")
            tsv.write(f"{row['sem']:.4f}\t{len(row['text'].split())}\t{label}\n")
    print(f"ranked {len(rows):,} documents", flush=True)


if __name__ == "__main__":
    main()
