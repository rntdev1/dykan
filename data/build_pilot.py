"""Build the pilot training set: real pre-2020 exam tasks plus generated tool episodes.

    uv run python -m data.build_pilot path/to/all.jsonl.gz outputs/pilot.jsonl

Purpose: a cheap check that the two baselines move at all. It is not the real
dataset: exam answers are bare keys with no explanations (no teacher yet), and
tool episodes use templated wording.

Each line: {"source", "messages", "tools"} in the Gemma 4 chat-template format.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from data.sources.zno_eval import ExamTask, load
from data.split import split_exam_tasks
from data.tools.episodes import generate
from eval import zno

TOOL_EPISODES = 2600
TOOL_SEED = 777  # the eval set uses 20261114; worlds and wording differ


def gold_answer(task: ExamTask) -> str:
    if task.question_type == "matching":
        body = ", ".join(f"{n}-{letter}" for n, letter in enumerate(task.correct, 1))
    else:
        body = ", ".join(task.correct)
    return f"Відповідь: {body}"


def exam_examples(path: Path) -> list[dict]:
    train, _eval, _dropped = split_exam_tasks(list(load(path)))
    out = []
    for task in train:
        if not zno.eligible(task):
            continue
        prompt = zno.build_prompt(task) + " Пояснення не потрібні."
        assert zno.is_correct(task, gold_answer(task)), task.uid
        out.append({"source": f"zno:{task.subject}", "tools": None,
                    "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": gold_answer(task)}]})
    return out


def tool_examples(n: int = TOOL_EPISODES, seed: int = TOOL_SEED) -> list[dict]:
    return [{"source": f"tools:{e.family}", "tools": e.tools, "messages": e.messages()} for e in generate(n, seed=seed, train=True)]


def main() -> None:
    zno_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    rows = exam_examples(zno_path) + tool_examples()
    random.Random(0).shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for row in rows:
        key = row["source"].split(":")[0]
        counts[key] = counts.get(key, 0) + 1
    print(f"wrote {len(rows)} examples to {out_path}: {counts}")


if __name__ == "__main__":
    main()
