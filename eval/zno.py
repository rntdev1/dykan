"""Held-out ZNO/NMT scoring: prompt building, answer parsing, accuracy.

Model-agnostic: `build_prompt` gives the user message, `is_correct` checks the
model's reply. Four question types are scored (single choice, matching,
sequence, multi-select); open-text maths answers are not scored yet.

Tasks that need an image are skipped here; they belong to the MMZNO eval.
"""

from __future__ import annotations

import re
from collections import defaultdict

from data.sources.zno_eval import ExamTask

SCORED_TYPES = ("single_choice", "matching", "sequence", "multi_select")
CYR = "АБВГДЕЖ"
# A model that answers with Latin letters means the option in that position.
LATIN_TO_CYR = dict(zip("ABCDEFG", CYR))
_LOOKALIKE = str.maketrans({"A": "А", "B": "В", "E": "Е"})

SUBJECT_UK = {"ukrainian": "українська мова та література", "history": "історія України", "geography": "географія", "math": "математика"}

FORMAT = {
    "single_choice": "Вибери один правильний варіант. Останнім рядком напиши «Відповідь: » і одну літеру.",
    "matching": "Добери до кожного пункту, позначеного цифрою, відповідник, позначений літерою. Останнім рядком напиши «Відповідь: » у форматі 1-А, 2-Б, 3-В, 4-Г.",
    "sequence": "Розташуй варіанти в правильній послідовності. Останнім рядком напиши «Відповідь: » і літери через кому в правильному порядку.",
    "multi_select": "Вибери всі правильні варіанти. Останнім рядком напиши «Відповідь: » і їхні номери через кому.",
}


def eligible(task: ExamTask) -> bool:
    return task.usable and not task.needs_image and task.question_type in SCORED_TYPES and bool(task.correct)


def build_prompt(task: ExamTask) -> str:
    return f"Завдання ЗНО/НМТ, предмет: {SUBJECT_UK.get(task.subject, task.subject)}.\n\n{task.prompt_text()}\n\n{FORMAT[task.question_type]}"


def _answer_part(reply: str) -> str:
    """Text after the last 'Відповідь:'; the whole reply if the marker is absent."""
    parts = re.split(r"відповідь\s*[:—-]", reply, flags=re.IGNORECASE)
    return parts[-1].strip() if len(parts) > 1 else reply.strip()


def _letters(text: str, allowed: str) -> list[str]:
    """Standalone option letters in order, Latin answers mapped by position."""
    out = []
    for token in re.findall(r"(?<![\w'’])([A-Za-zА-Яа-яЇїІіЄєҐґ])(?![\w'’])", text):
        ch = token.upper()
        if ch not in allowed:
            ch = LATIN_TO_CYR.get(ch, ch.translate(_LOOKALIKE))
        if ch in allowed:
            out.append(ch)
    return out


def parse_answer(task: ExamTask, reply: str) -> tuple[str, ...] | None:
    answer = _answer_part(reply)
    if task.question_type == "multi_select":
        digits = sorted(set(re.findall(r"\d", answer)))
        return tuple(digits) or None

    allowed = "".join(sorted({label for label, _ in task.options if not label.isdigit()})) or CYR[:5]
    if task.question_type == "single_choice":
        letters = _letters(answer, allowed)
        return (letters[0],) if letters else None
    if task.question_type == "sequence":
        letters = _letters(answer, allowed)
        return tuple(letters[: len(task.correct)]) if len(letters) >= len(task.correct) else None
    # matching: "1-А, 2-Д, ..." in any order; fall back to bare letters in row order
    pairs = {int(n): LATIN_TO_CYR.get(l.upper(), l.upper()) for n, l in re.findall(r"(\d)\s*[-–—:.)]?\s*([A-Za-zА-Яа-яЇїІіЄєҐґ])(?![\w'’])", answer)}
    rows = range(1, len(task.correct) + 1)
    if all(n in pairs for n in rows):
        return tuple(pairs[n] for n in rows)
    letters = _letters(answer, allowed)
    return tuple(letters[: len(task.correct)]) if len(letters) >= len(task.correct) else None


def is_correct(task: ExamTask, reply: str) -> bool:
    parsed = parse_answer(task, reply)
    if parsed is None:
        return False
    if task.question_type == "multi_select":
        return set(parsed) == set(task.correct)
    return parsed == task.correct


def summarize(rows: list[dict]) -> dict:
    """rows: [{"subject", "question_type", "correct": bool, "parsed": bool}]"""
    def rate(items, key="correct"):
        return round(sum(r[key] for r in items) / len(items), 4) if items else None

    by_subject, by_type = defaultdict(list), defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append(r)
        by_type[r["question_type"]].append(r)
    return {
        "tasks": len(rows),
        "accuracy": rate(rows),
        "unparsed_rate": round(1 - rate(rows, "parsed"), 4) if rows else None,
        "by_subject": {k: {"n": len(v), "accuracy": rate(v)} for k, v in sorted(by_subject.items())},
        "by_type": {k: {"n": len(v), "accuracy": rate(v)} for k, v in sorted(by_type.items())},
    }
