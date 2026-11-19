"""Assemble the training set from verified teacher output and real exam keys.

    uv run python -m data.build_dataset --zno all.jsonl.gz --teacher outputs/teacher --out outputs/train/dykan-v1.jsonl

Sources (only rows the generators marked "ok"):
  exam explanations   teacher solutions to pre-2020 exam tasks
  exam answer keys    every pre-2020 task in answer-only form (the eval protocol)
  tasks, explain, tutor, chat, tools (school tools), anytools (invented tools
  from other domains)   see data/teacher/recipes.py

Then, for every example:
  * drop it if any text repeats a held-out exam item (leak filter);
  * drop near-duplicates of an example already kept;
  * give it a system prompt the way the app will: tool episodes always have one
    (most with the short calendar), other examples have the Dykan prompt half
    of the time (half of those with unneeded tools declared) and none otherwise.

Each output line: {"source", "messages", "tools"}.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from data.build_pilot import gold_answer
from data.sources.zno_eval import load
from data.split import ProtectedSet, carve_dev, shingles, split_exam_tasks
from data.teacher.exam_solutions import fact_check_passed
from data.teacher.syllabus import UNRELIABLE_FOR_TEACHER
from data.tools.episodes import generate, system_prompt
from data.tools.rename import random_renaming
from data.tools.schemas import TOOLS
from eval import zno

_ISO_RE = re.compile(r"\((\d{4}-\d{2}-\d{2})\)")
RECIPES = ("tasks", "explain", "tutor", "chat", "tools", "anytools")
RENAMED_SHARE = 0.5
MAX_SAME_CONVERSATION, MAX_SAME_REQUEST = 3, 8


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[\w']+", text.lower()))


class Deduper:
    """Keeps an example only if it is not a near-duplicate of one already kept."""

    def __init__(self, limit: float = 0.6):
        self.limit, self.items, self.index = limit, [], {}

    def is_new(self, text: str) -> bool:
        sh = shingles(text)
        if not sh:
            return False
        shared: Counter[int] = Counter()
        for s in sh:
            for idx in self.index.get(s, ()):
                shared[idx] += 1
        for idx, inter in shared.items():
            if inter / (len(sh) + len(self.items[idx]) - inter) >= self.limit:
                return False
        idx = len(self.items)
        self.items.append(sh)
        for s in sh:
            self.index.setdefault(s, []).append(idx)
        return True


_REFERS_TO_OPTIONS = re.compile(r"\b(усі|всі|жоден|жодна|жодне|обидва|обидві|правильн\w+ відповід\w+|варіант\w*)\b|\b[АБВГД] (і|й|та) [АБВГД]\b", re.IGNORECASE)


def shuffled_options(task, rng: random.Random):
    """The same single-choice task with its options in another order, or None.

    An exam key supervises one letter. Seeing a task with the options reordered
    ties that letter to the content of the right option, not to its position."""
    labels = [label for label, _ in task.options]
    texts = [text for _, text in task.options]
    if (task.question_type != "single_choice" or len(task.correct) != 1 or task.correct[0] not in labels or len(set(texts)) < 3
            or any(label.isdigit() for label in labels) or any(_REFERS_TO_OPTIONS.search(t) for t in texts)):
        return None
    right = texts[labels.index(task.correct[0])]
    order = texts[:]
    for _ in range(5):
        rng.shuffle(order)
        if order != texts:
            break
    else:
        return None
    return replace(task, options=tuple(zip(labels, order)), correct=(labels[order.index(right)],))


def read_rows(folder: Path, prefix: str) -> list[dict]:
    """All rows of every shard file `<prefix>*.jsonl` under folder (searched recursively), de-duplicated by id."""
    rows: dict[str, dict] = {}
    for path in sorted(folder.rglob(f"{prefix}*.jsonl")):
        if "smoke" in path.name or "gate" in path.name:
            continue
        # Split on "\n" only (the text itself may hold other line separators), and
        # tolerate a cut-off last line in a file copied while it was being written.
        for line in path.read_bytes().split(b"\n"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rows.setdefault(row["id"], row)
    return list(rows.values())


def with_calendar(messages: list[dict], rng: random.Random, share: float = 0.7) -> list[dict]:
    """Rebuild a tool episode's system prompt, most of the time with the calendar."""
    found = _ISO_RE.search(messages[0]["content"])
    if messages[0]["role"] != "system" or not found:
        return messages
    today = date.fromisoformat(found.group(1))
    clock = f"{rng.randint(7, 21):02d}:{rng.choice(['00', '10', '15', '30', '45'])}"
    return [{"role": "system", "content": system_prompt(today, clock, calendar=rng.random() < share)}] + messages[1:]


_FINAL_RE = re.compile(r"\n\s*\**Відповідь\**\s*:\s*(?P<answer>[^\n]+)\s*$")
THINKING_SOURCES = ("tasks", "exam_explained")


def with_thinking(ex: dict) -> dict | None:
    """The same worked solution in Gemma's thinking form, or None if it has no clear final answer.

    The working goes into the thinking channel ("reasoning" on the assistant
    message) and the reply itself is the short final answer. The row is marked
    "thinking": the trainer then renders it with thinking switched on, so the
    model thinks only when the app asks for it."""
    last = ex["messages"][-1]
    found = _FINAL_RE.search(last["content"]) if last["role"] == "assistant" and isinstance(last.get("content"), str) else None
    if found is None or sum(m["role"] == "assistant" for m in ex["messages"]) != 1:
        return None
    working = last["content"][: found.start()].strip()
    if len(working) < 80:
        return None
    return {**ex, "thinking": True,
            "messages": ex["messages"][:-1] + [{"role": "assistant", "reasoning": working, "content": f"Відповідь: {found.group('answer').strip().strip('*')}"}]}


def with_persona(ex: dict, rng: random.Random, share: float = 0.5, tools_share: float = 0.5) -> dict:
    """Half of the ordinary examples get the Dykan system prompt, and half of those
    also get a random list of declared tools they do not need: the app always
    declares its tools, and the model has to answer plain questions directly."""
    if rng.random() >= share:
        return ex
    today = date(2026, 9, 1) + timedelta(days=rng.randint(0, 270))
    clock = f"{rng.randint(7, 21):02d}:{rng.choice(['00', '10', '15', '30', '45'])}"
    ex["messages"] = [{"role": "system", "content": system_prompt(today, clock, calendar=rng.random() < 0.5)}] + ex["messages"]
    if rng.random() < tools_share:
        ex["tools"] = rng.sample(TOOLS, k=rng.randint(3, len(TOOLS)))
    return ex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zno", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thinking-share", type=float, default=0.0, help="share of worked solutions recast with the working in the thinking channel")
    ap.add_argument("--shuffles", type=int, default=0, help="extra copies of each single-choice exam task with its options reordered")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    train_tasks, eval_tasks, _ = split_exam_tasks(list(load(args.zno)))
    train_tasks, dev_tasks = carve_dev(train_tasks)
    protected = ProtectedSet(t.prompt_text() for t in eval_tasks + dev_tasks)
    by_uid = {t.uid: t for t in train_tasks}
    examples: list[dict] = []
    dropped: Counter[str] = Counter()

    # Real exam tasks: answer keys for all, explanations where the teacher's was verified.
    for task in train_tasks:
        if zno.eligible(task):
            prompt = zno.build_prompt(task) + " Пояснення не потрібні."
            examples.append({"source": "exam_key", "tools": None, "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": gold_answer(task)}]})
            for _ in range(args.shuffles):
                variant = shuffled_options(task, rng)
                if variant is not None:
                    examples.append({"source": "exam_key", "tools": None, "messages": [
                        {"role": "user", "content": zno.build_prompt(variant) + " Пояснення не потрібні."}, {"role": "assistant", "content": gold_answer(variant)}]})
    for row in read_rows(args.teacher, "exam"):
        task = by_uid.get(row["id"])
        ok = row.get("agree", 0) >= 2 and row.get("solution") and fact_check_passed(row.get("fact_check", ""))
        if task is None or not ok or not zno.is_correct(task, row["solution"]):
            dropped["exam_explained:not_verified"] += 1
            continue
        examples.append({"source": "exam_explained", "tools": None,
                         "messages": [{"role": "user", "content": zno.build_prompt(task)}, {"role": "assistant", "content": row["solution"]}]})

    for recipe in RECIPES:
        for row in read_rows(args.teacher, recipe):
            if row.get("verdict") != "ok" or not row.get("messages"):
                dropped[f"{recipe}:{row.get('verdict', 'error')}"] += 1
                continue
            if row.get("subject") in UNRELIABLE_FOR_TEACHER:
                dropped[f"{recipe}:unreliable_subject"] += 1
                continue
            examples.append({"source": recipe, "tools": row.get("tools"), "messages": row["messages"]})

    # A share of the worked solutions is recast for thinking mode (in place, not as extra copies).
    for i, ex in enumerate(examples):
        if ex["source"] in THINKING_SOURCES and rng.random() < args.thinking_share:
            examples[i] = with_thinking(ex) or ex

    rng.shuffle(examples)
    dedupe = Deduper()
    tool_copies: Counter[str] = Counter()
    eval_requests = {_norm(e.user) for e in generate(130, seed=20261114)}  # the tool eval set (eval/run_hf.py default)
    kept: list[dict] = []
    for ex in examples:
        text = "\n".join((m.get("reasoning", "") + "\n" + m["content"]).strip() for m in ex["messages"] if m["role"] != "system" and isinstance(m.get("content"), str))
        if ex["source"] != "exam_key":
            if protected.leaks(text):
                dropped[f"{ex['source']}:leaks_eval"] += 1
                continue
            # Tool episodes of one task type share wording by design and differ in the school
            # world behind them (ids, dates, tool results), which is what the model has to
            # learn; near-duplicate filtering removed most of the chained task types. So they
            # are limited by count instead: a few copies of one conversation (they differ only
            # in the declared tools) and of one request (they differ in the world).
            if ex["source"] == "tools":
                request = _norm(ex["messages"][1]["content"])
                if request in eval_requests:
                    dropped["tools:same_as_eval_request"] += 1
                    continue
                conversation = json.dumps(ex["messages"][1:], ensure_ascii=False, sort_keys=True)
                tool_copies[conversation] += 1
                tool_copies["request:" + request] += 1
                is_new = tool_copies[conversation] <= MAX_SAME_CONVERSATION and tool_copies["request:" + request] <= MAX_SAME_REQUEST
            else:
                is_new = dedupe.is_new(text)
            if not is_new:
                dropped[f"{ex['source']}:duplicate"] += 1
                continue
        if ex["source"] == "tools":
            ex["messages"] = with_calendar(ex["messages"], rng)
            if rng.random() < RENAMED_SHARE:  # other tool and parameter names, so the declarations have to be read
                skin = random_renaming(rng)
                ex["messages"], ex["tools"] = skin.messages(ex["messages"]), skin.declarations(ex["tools"])
        elif ex["source"] != "anytools":  # those come with their own tools and system prompt
            ex = with_persona(ex, rng)
        kept.append(ex)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    stats = {"examples": len(kept), "thinking": sum(bool(ex.get("thinking")) for ex in kept), "by_source": dict(Counter(ex["source"] for ex in kept).most_common()), "dropped": dict(dropped.most_common())}
    args.out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
