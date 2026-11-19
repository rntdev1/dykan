"""Teacher-written explanations for real pre-2020 exam tasks, kept only when verified.

    python -m data.teacher.exam_solutions --zno all.jsonl.gz --model google/gemma-4-31B-it --out outputs/teacher/exam.jsonl [--limit 200]

The teacher is NOT shown the official key. It solves each task several times; a
solution is accepted only if
  * at least MIN_AGREE of ATTEMPTS independent attempts reach the official key
    (a task the teacher gets right only sometimes is one it does not know),
  * the text is clean Ukrainian (no loops, no foreign-script bursts, no Russian),
  * it does not repeat a held-out eval item,
  * a separate fact-check pass finds no factual error in the explanation. The
    teacher check showed that a right final answer does not guarantee this:
    accepted explanations contained wrong authors, dates and terms.

`--limit` with a small number is the teacher gate: it reports the acceptance rate
and throughput, and the accepted texts are read by hand before bulk generation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from data.filters import analyze, russian_letter_share
from data.sources.zno_eval import ExamTask, load
from data.split import HOLDOUT_FROM_YEAR, ProtectedSet, split_exam_tasks
from data.teacher.client import Teacher, run_jobs
from eval import zno

ATTEMPTS = 3
MIN_AGREE = 2

SYSTEM = (
    "Ти — досвідчений український учитель, який готує учнів до ЗНО та НМТ. "
    "Розв'яжи завдання й поясни розв'язання так, щоб учень зрозумів: яке правило, факт чи метод застосовуємо "
    "і чому правильний саме цей варіант. "
    "Пиши лише те, у чому впевнений. Не додавай дат, імен, назв творів, термінів чи прикладів, яких немає в умові, "
    "якщо без них можна обійтися; не розбирай кожен неправильний варіант, якщо не знаєш напевно, чому він хибний. "
    "Пиши коротко (зазвичай 2–6 речень, для математики — покрокове розв'язання), грамотною українською мовою, без русизмів. "
    "Не згадуй, що ти модель чи вчитель, просто пояснюй. Останнім рядком напиши відповідь у форматі, вказаному в завданні."
)

CHECK_SYSTEM = (
    "Ти — прискіпливий редактор навчальних матеріалів. Тобі дають завдання, правильну відповідь і пояснення. "
    "Перевір пояснення на фактичні помилки: неправильні дати, імена, авторство чи назви творів, терміни, означення, "
    "формулювання правил, обчислення, а також твердження, що суперечать умові чи правильній відповіді. Стиль не оцінюй. "
    "Якщо сумніваєшся в якомусь твердженні, вважай його помилкою. "
    "Перший рядок відповіді — рівно «ПОМИЛОК НЕМАЄ» або «Є ПОМИЛКИ». Якщо помилки є, далі коротко перелічи їх."
)


def fact_check_passed(reply: str) -> bool:
    first = reply.strip().splitlines()[0].strip(" .*«»\"").upper() if reply.strip() else ""
    # The teacher sometimes writes the phrase with a different case ending.
    return first in {"ПОМИЛОК НЕМАЄ", "ПОМИЛКИ НЕМАЄ", "НЕМАЄ ПОМИЛОК"}


def judge(task: ExamTask, text: str, protected: ProtectedSet) -> str:
    """'ok' or the reason the solution is rejected."""
    if not text.strip():
        return "empty"
    if not zno.is_correct(task, text):
        return "wrong_answer" if zno.parse_answer(task, text) is not None else "no_answer"
    if analyze(text).degenerate:
        return "degenerate"
    if russian_letter_share(text) > 0:
        return "russian"
    if len(text.split()) < 12:
        return "too_short"
    if protected.leaks(text):
        return "leaks_eval"
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zno", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", default="{}", help='extra request fields as JSON, e.g. {"chat_template_kwargs": {"enable_thinking": false}}')
    args = ap.parse_args()

    tasks = list(load(args.zno))
    train, eval_tasks, _ = split_exam_tasks(tasks)
    assert all(t.year < HOLDOUT_FROM_YEAR for t in train)
    protected = ProtectedSet(t.prompt_text() for t in eval_tasks)
    pool = [t for t in train if zno.eligible(t)]
    random.Random(args.seed).shuffle(pool)
    pool = pool[: args.limit] if args.limit else pool
    by_uid = {t.uid: t for t in pool}
    teacher = Teacher(args.model, args.base_url, max_tokens=args.max_tokens, extra=json.loads(args.extra))

    def work(job: dict) -> dict:
        task = by_uid[job["id"]]
        prompt = zno.build_prompt(task)
        tokens_total, verdicts, good = 0, [], []
        for _ in range(ATTEMPTS):
            text, tokens = teacher.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
            tokens_total += tokens
            verdict = judge(task, text, protected)
            verdicts.append(verdict)
            if verdict == "ok":
                good.append(text.strip())
        row = {"id": task.uid, "subject": task.subject, "question_type": task.question_type, "gold": list(task.correct),
               "prompt": prompt, "attempt_verdicts": verdicts, "agree": len(good), "teacher": args.model}
        if len(good) < MIN_AGREE:
            row["verdict"] = "inconsistent" if good else Counter(verdicts).most_common(1)[0][0]
            row["solution"] = good[0] if good else ""
        else:
            solution = min(good, key=len)  # the shortest correct attempt makes the fewest extra claims
            check, tokens = teacher.chat([
                {"role": "system", "content": CHECK_SYSTEM},
                {"role": "user", "content": f"{prompt}\n\nПравильна відповідь: {', '.join(task.correct)}\n\nПояснення для перевірки:\n{solution}"},
            ], temperature=0.0, max_tokens=300)
            tokens_total += tokens
            row.update(solution=solution, fact_check=check.strip(), verdict="ok" if fact_check_passed(check) else "failed_fact_check")
        row["completion_tokens"] = tokens_total
        return row

    stats = run_jobs(({"id": t.uid} for t in pool), work, args.out, workers=args.workers)

    rows = [json.loads(line) for line in args.out.read_text(encoding="utf-8").splitlines() if line.strip()]
    verdicts = Counter(r.get("verdict", "error") for r in rows)
    by_subject: dict[str, list[bool]] = {}
    for r in rows:
        by_subject.setdefault(r.get("subject", "?"), []).append(r.get("verdict") == "ok")
    summary = {
        "teacher": args.model, "tasks": len(rows), "accepted": verdicts.get("ok", 0),
        "acceptance_rate": round(verdicts.get("ok", 0) / len(rows), 4) if rows else None,
        "verdicts": dict(verdicts.most_common()),
        "acceptance_by_subject": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_subject.items())},
        "run": stats,
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
