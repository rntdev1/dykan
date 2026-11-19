import random
from pathlib import Path

import pytest

from data.build_pilot import gold_answer, tool_examples
from data.sources.zno_eval import ExamTask
from eval import zno
from train.masking import supervised_spans, token_mask
from train.sft import make_batches

TEMPLATE = Path(__file__).parent.parent / "configs" / "gemma4_chat_template_4adf157.jinja"


def covered(text: str) -> str:
    return "".join(text[a:b] for a, b in supervised_spans(text))


def test_plain_turn():
    text = "<bos><|turn>user\nПитання?<turn|>\n<|turn>model\nВідповідь: Б<turn|>\n"
    assert covered(text) == "Відповідь: Б<turn|>"


def test_tool_turn_masks_the_response_but_keeps_the_handover_token():
    text = ('<bos><|turn>system\nсистема<|tool>declaration:a{}<tool|><turn|>\n<|turn>user\nзроби<turn|>\n<|turn>model\n'
            '<|tool_call>call:a{x:1}<tool_call|><|tool_response>response:a{secret:<|"|>не вчити<|"|>}<tool_response|>'
            '<|tool_call>call:b{}<tool_call|><|tool_response>response:b{ok:true}<tool_response|>Готово.<turn|>\n')
    assert covered(text) == "<|tool_call>call:a{x:1}<tool_call|><|tool_response><|tool_call>call:b{}<tool_call|><|tool_response>Готово.<turn|>"


def test_multi_turn_and_no_model_turn():
    text = "<|turn>user\nа<turn|>\n<|turn>model\nб<turn|>\n<|turn>user\nв<turn|>\n<|turn>model\nг<turn|>\n"
    assert covered(text) == "б<turn|>г<turn|>"
    assert supervised_spans("<|turn>user\nа<turn|>\n") == []


def test_token_mask_follows_spans():
    text = "<|turn>user\nа<turn|>\n<|turn>model\nбв<turn|>\n"
    offsets = [(i, i + 1) for i in range(len(text))]  # one token per character
    mask = token_mask(offsets, supervised_spans(text))
    assert "".join(c for c, m in zip(text, mask) if m) == "бв<turn|>"
    assert token_mask([(0, 0), (3, 4)], [(0, 10)]) == [False, True]  # zero-width tokens carry no loss


def test_real_episodes_supervise_only_model_output():
    sandbox = pytest.importorskip("jinja2.sandbox")
    template = sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(TEMPLATE.read_text(encoding="utf-8"))
    for row in tool_examples(n=52, seed=3):
        rendered = template.render(messages=row["messages"], tools=row["tools"], bos_token="<bos>", add_generation_prompt=False)
        kept = covered(rendered)
        assert "declaration:" not in kept and "response:" not in kept and "Дикань" not in kept
        assert kept.endswith("<turn|>") and row["messages"][-1]["content"] in kept
        assert kept.count("<|tool_call>") == rendered.count("<|tool_call>") == kept.count("<|tool_response>")


def test_batches_respect_the_token_budget():
    rng = random.Random(0)
    examples = [{"ids": [0] * rng.choice([40, 60, 300, 2500, 3900])} for _ in range(500)]
    batches = make_batches(examples, 12000, random.Random(1))
    assert sum(len(b) for b in batches) == 500
    for b in batches:
        assert len(b) == 1 or max(len(e["ids"]) for e in b) * len(b) <= 12000


def test_gold_answers_parse_as_correct():
    def task(qtype, options, correct):
        return ExamTask(uid="u", exam_uid="history:zno:2015:main", subject="history", exam_kind="zno", year=2015, question_type=qtype,
                        question="Питання?", options=tuple(options), correct=tuple(correct), needs_image=False, usable=True)
    letters = [(l, l) for l in "АБВГД"]
    cases = [task("single_choice", letters, ["Г"]), task("sequence", letters[:4], ["В", "А", "Г", "Б"]),
             task("matching", [(str(i), "x") for i in range(1, 5)] + letters, ["А", "Д", "Б", "Г"]),
             task("multi_select", [(str(i), "x") for i in range(1, 8)], ["2", "4", "7"])]
    for t in cases:
        assert zno.is_correct(t, gold_answer(t)), t.question_type
    assert gold_answer(cases[2]) == "Відповідь: 1-А, 2-Д, 3-Б, 4-Г"


def test_plain_text_rows_are_cut_into_windows():
    from train.sft import encode_text

    class Tok:
        bos_token_id = 2

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(10, 10 + len(text.split())))}

    windows = encode_text(Tok(), {"text": " ".join(["w"] * 25), "source": "wiki"}, max_len=11)
    assert [len(w["ids"]) for w in windows] == [11, 11, 6]  # 10 + 10 + 5 tokens, each after <bos>
    assert all(w["ids"][0] == 2 and w["labels"][0] == -100 and w["labels"][1:] == w["ids"][1:] for w in windows)
    assert windows[0]["ids"][1:] + windows[1]["ids"][1:] + windows[2]["ids"][1:] == list(range(10, 35))
    # A tiny tail is dropped, a tiny whole document is kept.
    assert len(encode_text(Tok(), {"text": " ".join(["w"] * 21)}, max_len=11)) == 2
    assert len(encode_text(Tok(), {"text": "w"}, max_len=11)) == 1
