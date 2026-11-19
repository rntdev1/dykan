from pathlib import Path

import pytest

from data.sources.zno_eval import ExamTask
from data.tools.episodes import generate
from eval.gemma_calls import parse_tool_calls, strip_markup
from eval.zno import build_prompt, eligible, is_correct, parse_answer

TEMPLATE = Path(__file__).parent.parent / "configs" / "gemma4_chat_template_4adf157.jinja"


def task(qtype, options, correct, **kw) -> ExamTask:
    return ExamTask(uid="u", exam_uid="history:zno:2021:main", subject="history", exam_kind="zno", year=2021, question_type=qtype,
                    question="Питання?", options=tuple(options), correct=tuple(correct), needs_image=kw.get("needs_image", False), usable=True)


ABCD = [("А", "перше"), ("Б", "друге"), ("В", "третє"), ("Г", "четверте")]
MATCH = [(str(i), f"пункт {i}") for i in range(1, 5)] + [(l, f"варіант {l}") for l in "АБВГД"]
MULTI = [(str(i), f"твердження {i}") for i in range(1, 8)]


# ---------- exam answers ----------

@pytest.mark.parametrize("reply, expected", [
    ("Відповідь: Б", ("Б",)),
    ("Б", ("Б",)),
    ("Правильний варіант — В, бо так.\nВідповідь: В.", ("В",)),
    ("**Відповідь:** Г", ("Г",)),
    ("Відповідь: B", ("Б",)),  # Latin B means the second option
    ("Answer: C\nВідповідь: C", ("В",)),
    ("Я не знаю.", None),
    ("", None),
])
def test_single_choice(reply, expected):
    assert parse_answer(task("single_choice", ABCD, ["Б"]), reply) == expected


def test_single_choice_ignores_letters_inside_words():
    t = task("single_choice", ABCD, ["Г"])
    assert parse_answer(t, "Відповідь: Г") == ("Г",)
    assert is_correct(t, "А це неправильно, бо В суперечить умові.\nВідповідь: Г")
    assert not is_correct(t, "Відповідь: А")


def test_matching():
    t = task("matching", MATCH, ["А", "Д", "Б", "Г"])
    assert is_correct(t, "Відповідь: 1-А, 2-Д, 3-Б, 4-Г")
    assert is_correct(t, "Відповідь: 4 — Г; 1 — А; 3 — Б; 2 — Д")
    assert is_correct(t, "Відповідь: А, Д, Б, Г")
    assert not is_correct(t, "Відповідь: 1-А, 2-Д, 3-Г, 4-Б")
    assert not is_correct(t, "Відповідь: 1-А, 2-Д")


def test_sequence_and_multi_select():
    seq = task("sequence", ABCD, ["В", "А", "Г", "Б"])
    assert is_correct(seq, "Відповідь: В, А, Г, Б") and is_correct(seq, "Відповідь: ВАГБ".replace("ВАГБ", "В А Г Б"))
    assert not is_correct(seq, "Відповідь: А, Б, В, Г")
    multi = task("multi_select", MULTI, ["2", "4", "7"])
    assert is_correct(multi, "Відповідь: 7, 2, 4") and not is_correct(multi, "Відповідь: 2, 4") and not is_correct(multi, "Відповідь: 2, 4, 6, 7")


def test_eligibility_and_prompt():
    assert eligible(task("single_choice", ABCD, ["А"]))
    assert not eligible(task("single_choice", ABCD, ["А"], needs_image=True))
    assert not eligible(task("open_text", [], ["12"]))
    prompt = build_prompt(task("single_choice", ABCD, ["А"]))
    assert "історія України" in prompt and "А. перше" in prompt and "Відповідь:" in prompt


# ---------- Gemma tool-call syntax ----------

def test_parse_simple_and_nested_calls():
    text = ('<|tool_call>call:list_homework{due_date:<|"|>2026-11-17<|"|>,status:<|"|>todo<|"|>}<tool_call|>'
            '<|tool_call>call:get_last_grades{limit:5,subject:<|"|>Хімія<|"|>}<tool_call|>'
            '<|tool_call>call:x{flag:true,none:null,list:[1,2.5,<|"|>a, b<|"|>],obj:{k:{deep:false}}}<tool_call|>'
            '<|tool_call>call:list_scheduled_tasks{}<tool_call|>')
    assert parse_tool_calls(text) == [
        {"name": "list_homework", "arguments": {"due_date": "2026-11-17", "status": "todo"}},
        {"name": "get_last_grades", "arguments": {"limit": 5, "subject": "Хімія"}},
        {"name": "x", "arguments": {"flag": True, "none": None, "list": [1, 2.5, "a, b"], "obj": {"k": {"deep": False}}}},
        {"name": "list_scheduled_tasks", "arguments": {}},
    ]


def test_strings_may_contain_syntax_characters():
    text = '<|tool_call>call:send_message{recipient_id:<|"|>t1<|"|>,text:<|"|>Добрий день! {жарт}, [1]: так, «ні»\nрядок<|"|>}<tool_call|>'
    assert parse_tool_calls(text)[0]["arguments"]["text"] == "Добрий день! {жарт}, [1]: так, «ні»\nрядок"


def test_malformed_call_is_reported_not_raised():
    assert parse_tool_calls('<|tool_call>call:send_message{recipient_id:<|"|>t1') == [{"name": "send_message", "arguments": None}]
    assert parse_tool_calls("Просто текст без викликів.") == []


def test_strip_markup():
    text = '<|channel>thought\nдумаю<channel|><|tool_call>call:a{}<tool_call|><|tool_response>response:a{ok:true}<tool_response|>Готово.<turn|>'
    assert strip_markup(text) == "Готово."


def test_round_trip_through_the_pinned_chat_template():
    """Render gold episodes with Google's template, parse the calls back, compare."""
    sandbox = pytest.importorskip("jinja2.sandbox")
    env = sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(TEMPLATE.read_text(encoding="utf-8"))
    for episode in generate(130, seed=5):
        rendered = template.render(messages=episode.messages(), tools=episode.tools, bos_token="<bos>", add_generation_prompt=False)
        model_side = rendered.split("<|turn>model\n", 1)[1] if episode.gold else ""
        assert parse_tool_calls(model_side) == episode.gold, episode.family
        assert strip_markup(rendered.split("<|turn>model\n", 1)[1]) == episode.final.strip()
