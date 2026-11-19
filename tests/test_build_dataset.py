import random

from data.build_dataset import Deduper, with_calendar, with_persona
from data.tools.episodes import generate


def test_deduper_drops_near_copies_only():
    d = Deduper()
    a = "Знайдіть площу прямокутника зі сторонами 7 см і 12 см та запишіть відповідь у квадратних сантиметрах."
    assert d.is_new(a)
    assert not d.is_new(a)
    assert not d.is_new(a.replace("запишіть", "напишіть"))
    assert d.is_new("Обчисліть периметр квадрата зі стороною 9 см і поясніть, яку формулу ви використали для цього.")
    assert not d.is_new("")


def test_calendar_is_added_to_tool_episodes_with_the_right_dates():
    episode = next(e for e in generate(40, seed=9, train=True) if e.family == "homework_for_day")
    seen = set()
    for seed in range(30):
        out = with_calendar(episode.messages(), random.Random(seed))
        assert out[0]["role"] == "system" and episode.today.isoformat() in out[0]["content"] and out[1:] == episode.messages()[1:]
        seen.add("Найближчі дні" in out[0]["content"])
        if "Найближчі дні" in out[0]["content"]:
            assert (episode.today.replace(day=episode.today.day) .isoformat()) in out[0]["content"]
    assert seen == {True, False}


def test_persona_prompt_is_optional_for_plain_examples():
    base = [{"role": "user", "content": "Поясни дроби."}, {"role": "assistant", "content": "Дріб — це частина цілого."}]
    results = [with_persona({"source": "explain", "tools": None, "messages": list(base)}, random.Random(s)) for s in range(40)]
    plain = [r for r in results if len(r["messages"]) == 2]
    prompted = [r for r in results if len(r["messages"]) == 3]
    assert plain and prompted and len(plain) + len(prompted) == 40
    assert all(r["messages"][0]["role"] == "system" and "Дикань" in r["messages"][0]["content"] for r in prompted)
    assert all(r["messages"][-2:] == base for r in results)
    # Tools are only ever declared together with the system prompt, and not always.
    assert all(r["tools"] is None for r in plain)
    assert any(r["tools"] for r in prompted) and any(r["tools"] is None for r in prompted)


def test_shuffled_options_keeps_the_right_answer():
    from data.build_dataset import shuffled_options
    from data.sources.zno_eval import ExamTask

    def task(options, correct=("Б",), kind="single_choice"):
        return ExamTask("u", "e", "history", "zno", 2015, kind, "Коли?", tuple(options), correct, False, True)

    base = task([("А", "1648"), ("Б", "1654"), ("В", "1709"), ("Г", "1991")])
    seen = set()
    for seed in range(20):
        variant = shuffled_options(base, random.Random(seed))
        assert variant is not None and [label for label, _ in variant.options] == ["А", "Б", "В", "Г"]
        assert dict(variant.options)[variant.correct[0]] == "1654" and variant.options != base.options
        seen.add(variant.correct[0])
    assert len(seen) > 1
    assert shuffled_options(task([("А", "1648"), ("Б", "1654"), ("В", "усі відповіді правильні")]), random.Random(0)) is None
    assert shuffled_options(task([("А", "x"), ("Б", "y"), ("В", "А і Б")]), random.Random(0)) is None
    assert shuffled_options(task([("А", "x"), ("Б", "y"), ("В", "z")], kind="matching"), random.Random(0)) is None


def test_thinking_rows_put_the_working_in_the_thinking_channel():
    from pathlib import Path

    import pytest

    from data.build_dataset import with_thinking
    from train.masking import supervised_spans

    working = "1. За теоремою косинусів c² = a² + b² − 2ab·cos γ.\n2. Підставимо: c² = 25 + 64 − 40 = 49.\n3. Отже, c = 7."
    ex = {"source": "tasks", "tools": None, "messages": [{"role": "user", "content": "Знайди третю сторону."}, {"role": "assistant", "content": working + "\n\nВідповідь: 7 см"}]}
    row = with_thinking(ex)
    assert row["thinking"] is True and row["messages"][-1] == {"role": "assistant", "reasoning": working, "content": "Відповідь: 7 см"}
    assert "thinking" not in ex and ex["messages"][-1]["content"].endswith("7 см")  # the original is not changed
    assert with_thinking({**ex, "messages": [ex["messages"][0], {"role": "assistant", "content": "Просто текст без підсумку."}]}) is None
    assert with_thinking({**ex, "messages": [ex["messages"][0], {"role": "assistant", "content": "Коротко.\n\nВідповідь: 7"}]}) is None

    sandbox = pytest.importorskip("jinja2.sandbox")
    template = sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
        (Path(__file__).parent.parent / "configs" / "gemma4_chat_template_4adf157.jinja").read_text(encoding="utf-8"))
    rendered = template.render(messages=row["messages"], enable_thinking=True, add_generation_prompt=False, bos_token="<bos>")
    assert "<|think|>" in rendered and f"<|channel>thought\n{working}\n<channel|>Відповідь: 7 см" in rendered
    # The loss covers the thinking and the answer, not the question.
    supervised = "".join(rendered[a:b] for a, b in supervised_spans(rendered))
    assert working in supervised and "Відповідь: 7 см" in supervised and "Знайди третю сторону" not in supervised
    plain = template.render(messages=ex["messages"], enable_thinking=False, add_generation_prompt=False, bos_token="<bos>")
    assert "<|think|>" not in plain and "<|channel>" not in plain
