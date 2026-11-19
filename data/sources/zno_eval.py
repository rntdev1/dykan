"""Loader for NLPForUA/zno-eval v1 (ZNO/NMT tasks 2007-2024 with answer keys).

The exam text belongs to UCEQA and is not openly licensed: tasks loaded here are
used for training and evaluation with attribution, and are never republished.

Download `data/tasks/all.jsonl.gz` from the dataset repo and pass its path.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class ExamTask:
    uid: str  # e.g. "ukrainian:zno:2013:session-2:task-001"
    exam_uid: str  # e.g. "ukrainian:zno:2013:session-2"
    subject: str  # ukrainian | history | geography | math
    exam_kind: str  # zno | nmt
    year: int
    question_type: str  # single_choice | matching | multi_select | open_text | sequence
    question: str
    options: tuple[tuple[str, str], ...]  # (label, text)
    correct: tuple[str, ...]
    needs_image: bool
    usable: bool  # passed the dataset maintainers' review with no open issue
    source: str = field(default="zno-eval-v1")

    def prompt_text(self) -> str:
        """Question plus options, as shown to a model or a teacher."""
        lines = [self.question.strip()]
        lines += [f"{label}. {text}" for label, text in self.options]
        return "\n".join(lines)


def _parse(row: dict) -> ExamTask:
    subject, kind, year, _session = row["exam_uid"].split(":")
    return ExamTask(
        uid=row["task_uid"],
        exam_uid=row["exam_uid"],
        subject=subject,
        exam_kind=kind,
        year=int(year),
        question_type=row["question_type"],
        question=row["question"],
        options=tuple((a["answer"], a["text"]) for a in row["answers"]),
        correct=tuple(row["correct_answer"]),
        needs_image=bool(row["with_photo"]),
        usable=row["benchmark_status"] == "eligible" and not row["has_known_issue"],
    )


def load(path: str | Path) -> Iterator[ExamTask]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield _parse(json.loads(line))
