"""Held-out split for exam data.

Two rules keep the eval honest:
  1. Year split: exam forms from HOLDOUT_FROM_YEAR onward are eval-only.
  2. Near-duplicate filter: any training text that closely matches a protected
     eval item is dropped, because exam items are reused across years and leak
     through web and third-party datasets.

`leaks()` works on plain strings, so the same filter is applied to every training
source (teacher output, web text, other datasets), not only exam tasks.

    uv run python -m data.split path/to/all.jsonl.gz
"""

from __future__ import annotations

import hashlib
import re
import sys
from collections import Counter, defaultdict
from typing import Iterable, Sequence

from data.sources.zno_eval import ExamTask, load

HOLDOUT_FROM_YEAR = 2020
SHINGLE_WORDS = 3
JACCARD_LIMIT = 0.6

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text.replace("’", "'").replace("ʼ", "'"))]


def shingles(text: str, n: int = SHINGLE_WORDS) -> frozenset[tuple[str, ...]]:
    words = normalize(text)
    if len(words) < n:
        return frozenset({tuple(words)}) if words else frozenset()
    return frozenset(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


class ProtectedSet:
    """Index of eval items; answers "does this text leak one of them?"."""

    def __init__(self, texts: Iterable[str]):
        self._items: list[frozenset] = []
        self._index: dict[tuple, list[int]] = defaultdict(list)
        for text in texts:
            sh = shingles(text)
            if not sh:
                continue
            idx = len(self._items)
            self._items.append(sh)
            for s in sh:
                self._index[s].append(idx)

    def __len__(self) -> int:
        return len(self._items)

    def leaks(self, text: str, limit: float = JACCARD_LIMIT) -> bool:
        """True if `text` is a near-duplicate of, or contains, a protected item."""
        sh = shingles(text)
        if not sh:
            return False
        shared: Counter[int] = Counter()
        for s in sh:
            for idx in self._index.get(s, ()):
                shared[idx] += 1
        for idx, inter in shared.items():
            item = self._items[idx]
            jaccard = inter / (len(sh) + len(item) - inter)
            # Containment catches an eval question quoted inside a longer document.
            containment = inter / len(item)
            if jaccard >= limit or (len(item) >= 8 and containment >= 0.9):
                return True
        return False


def split_exam_tasks(
    tasks: Sequence[ExamTask], holdout_from: int = HOLDOUT_FROM_YEAR
) -> tuple[list[ExamTask], list[ExamTask], list[ExamTask]]:
    """Returns (train, eval, dropped_as_leak)."""
    eval_tasks = [t for t in tasks if t.year >= holdout_from]
    protected = ProtectedSet(t.prompt_text() for t in eval_tasks)
    train, dropped = [], []
    for t in tasks:
        if t.year >= holdout_from:
            continue
        (dropped if protected.leaks(t.prompt_text()) else train).append(t)
    return train, eval_tasks, dropped


def carve_dev(train: Sequence[ExamTask], one_in: int = 15) -> tuple[list[ExamTask], list[ExamTask]]:
    """Returns (train, dev): a fixed slice of the pre-holdout tasks set aside for
    choosing hyperparameters, so that the held-out years are only used to report.
    Training tasks that near-duplicate a dev task are dropped from training."""
    dev = [t for t in train if int(hashlib.sha1(t.uid.encode()).hexdigest(), 16) % one_in == 0]
    dev_uids = {t.uid for t in dev}
    protected = ProtectedSet(t.prompt_text() for t in dev)
    return [t for t in train if t.uid not in dev_uids and not protected.leaks(t.prompt_text())], dev


def main() -> None:
    tasks = list(load(sys.argv[1]))
    train, eval_tasks, dropped = split_exam_tasks(tasks)
    print(f"tasks {len(tasks)} | train {len(train)} | eval {len(eval_tasks)} | dropped as leaks {len(dropped)}")
    for name, group in (("train", train), ("eval", eval_tasks), ("dropped", dropped)):
        by_subject = Counter(t.subject for t in group)
        usable_text = sum(t.usable and not t.needs_image for t in group)
        print(f"  {name:8s} {dict(sorted(by_subject.items()))} | usable text-only: {usable_text}")


if __name__ == "__main__":
    main()
