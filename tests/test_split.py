from data.sources.zno_eval import ExamTask
from data.split import ProtectedSet, split_exam_tasks


def task(uid: str, year: int, question: str, options=("кропива", "периметр", "підошва", "однаковий")) -> ExamTask:
    return ExamTask(
        uid=uid,
        exam_uid=f"ukrainian:zno:{year}:main",
        subject="ukrainian",
        exam_kind="zno",
        year=year,
        question_type="single_choice",
        question=question,
        options=tuple(zip("АБВГ", options)),
        correct=("А",),
        needs_image=False,
        usable=True,
    )


Q_EVAL = "Укажіть рядок, у якому всі слова пишуться з апострофом після префікса, що закінчується на приголосний"
Q_OTHER = "Визначте, у якому столітті було засновано Києво-Могилянську академію та хто був її фундатором"


def test_year_split():
    tasks = [task("a", 2015, Q_OTHER), task("b", 2019, Q_OTHER + " саме"), task("c", 2020, Q_EVAL), task("d", 2024, Q_EVAL + " завжди")]
    train, ev, dropped = split_exam_tasks(tasks)
    assert {t.uid for t in ev} == {"c", "d"}
    assert {t.uid for t in train} | {t.uid for t in dropped} == {"a", "b"}


def test_repeated_item_is_dropped_from_train():
    old_copy = task("old", 2012, Q_EVAL)  # same item reused in a held-out year
    train, ev, dropped = split_exam_tasks([old_copy, task("new", 2021, Q_EVAL), task("ok", 2012, Q_OTHER, ("XV", "XVI", "XVII", "XVIII"))])
    assert [t.uid for t in dropped] == ["old"]
    assert [t.uid for t in train] == ["ok"]


def test_reworded_copy_is_caught():
    protected = ProtectedSet([task("e", 2022, Q_EVAL).prompt_text()])
    reworded = task("x", 2010, Q_EVAL.replace("Укажіть", "Позначте")).prompt_text()
    assert protected.leaks(reworded)


def test_quote_inside_longer_document_is_caught():
    item = task("e", 2022, Q_EVAL).prompt_text()
    protected = ProtectedSet([item])
    doc = "Розбір завдань НМТ для учнів одинадцятого класу. " * 5 + item + " Правильна відповідь А, бо так вимагає правило."
    assert protected.leaks(doc)


def test_unrelated_text_passes():
    protected = ProtectedSet([task("e", 2022, Q_EVAL).prompt_text()])
    assert not protected.leaks(task("y", 2010, Q_OTHER, ("XV", "XVI", "XVII", "XVIII")).prompt_text())
    assert not protected.leaks("")
