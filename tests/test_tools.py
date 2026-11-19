from datetime import date, timedelta

from data.tools.schemas import TOOLS, TOOLS_BY_NAME
from data.tools.sim import SchoolWorld

TODAY = date(2026, 11, 16)  # a Monday


def test_schemas_are_well_formed():
    assert len(TOOLS) == len(TOOLS_BY_NAME) == 12
    for fn in TOOLS_BY_NAME.values():
        assert fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert set(params["required"]) <= set(params["properties"])
        for spec in params["properties"].values():
            assert spec["type"] in {"string", "integer", "boolean"} and spec["description"]


def test_every_tool_has_an_implementation():
    world = SchoolWorld(0, TODAY)
    for name in TOOLS_BY_NAME:
        assert hasattr(world, f"_{name}")


def test_world_is_deterministic_per_seed():
    a, b, c = SchoolWorld(7, TODAY), SchoolWorld(7, TODAY), SchoolWorld(8, TODAY)
    assert a.homework == b.homework and a.grades == b.grades and a.timetable == b.timetable
    assert a.homework != c.homework


def test_homework_is_due_on_a_day_with_that_lesson():
    for seed in range(20):
        world = SchoolWorld(seed, TODAY)
        for hw in world.homework:
            lessons = world.timetable.get(date.fromisoformat(hw["due_date"]).weekday(), [])
            assert hw["subject"] in {x["subject"] for x in lessons}


def test_six_pm_homework_digest_episode():
    """The reference task: a daily 18:00 job, and what that job does when it runs."""
    world = SchoolWorld(3, TODAY)
    task = world.call("create_scheduled_task", {
        "time": "18:00", "recurrence": "daily",
        "instruction": "Зібрати невиконані домашні завдання на завтра і надіслати сповіщення, що прочитати й підготувати.",
    })
    assert task["id"] == "task1" and world.scheduled[0]["recurrence"] == "daily"

    tomorrow = (TODAY + timedelta(days=1)).isoformat()
    todo = world.call("list_homework", {"due_date": tomorrow, "status": "todo"})["homework"]
    assert all(h["due_date"] == tomorrow and not h["done"] for h in todo)
    details = [world.call("get_homework_details", {"homework_id": h["id"]}) for h in todo]
    text = "\n".join(f"{d['subject']}: {d['details']}" for d in details) or "На завтра все виконано."
    assert world.call("send_notification", {"title": "Домашнє на завтра", "text": text}) == {"shown": True}
    assert world.notifications[-1]["text"] == text


def test_message_to_a_teacher_found_by_subject():
    world = SchoolWorld(1, TODAY)
    subject = world.subjects[0]
    found = world.call("list_recipients", {"role": "teacher", "query": subject})["recipients"]
    assert len(found) == 1 and found[0]["subject"] == subject
    sent = world.call("send_message", {"recipient_id": found[0]["id"], "text": "Добрий день! Я завтра буду відсутній."})
    assert sent["sent"] and world.messages == [{"recipient_id": found[0]["id"], "text": "Добрий день! Я завтра буду відсутній."}]


def test_mark_done_changes_state():
    world = SchoolWorld(2, TODAY)
    hw = next(h for h in world.homework if not h["done"])
    assert world.call("mark_homework_done", {"homework_id": hw["id"]}) == {"id": hw["id"], "done": True}
    assert hw["id"] not in {h["id"] for h in world.call("list_homework", {"status": "todo"})["homework"]}


def test_grades_analytics_and_schedule():
    world = SchoolWorld(4, TODAY)
    grades = world.call("get_last_grades", {"limit": 5})["grades"]
    assert len(grades) == 5 and all(1 <= g["grade"] <= 12 for g in grades)
    assert grades == sorted(grades, key=lambda g: g["date"], reverse=True)
    stats = world.call("get_analytics", {"period": "month"})
    assert len(stats["subjects"]) == len(world.subjects) and 0 <= stats["homework_done_share"] <= 1
    assert 5 <= len(world.call("get_schedule", {"date": "2026-11-17"})["lessons"]) <= 8
    assert world.call("get_schedule", {"date": "2026-11-21"})["lessons"] == []  # Saturday


def test_errors_are_returned_not_raised():
    world = SchoolWorld(5, TODAY)
    bad_calls = [
        ("no_such_tool", {}),
        ("get_homework_details", {}),
        ("get_homework_details", {"homework_id": "hw999"}),
        ("get_schedule", {"date": "завтра"}),
        ("get_schedule", {"date": "2026-02-30"}),
        ("list_homework", {"status": "pending"}),
        ("list_homework", {"subject": "Астрономія"}),
        ("get_last_grades", {"limit": "5"}),
        ("send_message", {"recipient_id": "t1", "text": "Привіт", "urgent": True}),
        ("create_scheduled_task", {"time": "6pm", "recurrence": "daily", "instruction": "x"}),
        ("create_scheduled_task", {"time": "18:00", "recurrence": "once", "instruction": "x"}),
        ("create_scheduled_task", {"time": "18:00", "recurrence": "once", "instruction": "x", "date": "2026-11-01"}),
        ("cancel_scheduled_task", {"task_id": "task9"}),
    ]
    for name, args in bad_calls:
        assert "error" in world.call(name, args), (name, args)
    assert len(world.calls) == len(bad_calls)
    assert not world.messages and not world.scheduled


def test_renaming_round_trips_and_keeps_episodes_solvable():
    import random

    from data.tools.episodes import generate
    from data.tools.rename import HELDOUT, HELDOUT_PARAM_NAMES, HELDOUT_TOOL_NAMES, TRAIN_PARAM_NAMES, TRAIN_TOOL_NAMES, random_renaming
    from data.tools.schemas import TOOLS, TOOLS_BY_NAME

    # The held-out names are really held out, and every tool and parameter has one.
    train_names = {n for names in TRAIN_TOOL_NAMES.values() for n in names} | set(TRAIN_TOOL_NAMES)
    assert not train_names & set(HELDOUT_TOOL_NAMES.values()) and set(HELDOUT_TOOL_NAMES) == set(TOOLS_BY_NAME)
    train_params = {n for names in TRAIN_PARAM_NAMES.values() for n in names} | set(TRAIN_PARAM_NAMES)
    assert not train_params & set(HELDOUT_PARAM_NAMES.values())
    all_params = {k for t in TOOLS for k in t["function"]["parameters"]["properties"]}
    assert all_params == set(HELDOUT_PARAM_NAMES) == set(TRAIN_PARAM_NAMES)

    for skin in [HELDOUT] + [random_renaming(random.Random(s)) for s in range(30)]:
        declared = skin.declarations(TOOLS)
        for tool in declared:
            props = tool["function"]["parameters"]["properties"]
            assert set(tool["function"]["parameters"]["required"]) <= set(props)
        assert len({t["function"]["name"] for t in declared}) == len(TOOLS)
        for original, renamed in zip(TOOLS, declared):  # no two parameters merged into one
            assert len(renamed["function"]["parameters"]["properties"]) == len(original["function"]["parameters"]["properties"])
        for episode in generate(26, seed=5):
            world = episode.make_world()
            for message in skin.messages(episode.messages()):
                for call in message.get("tool_calls", []):
                    fn = call["function"]
                    assert fn["name"] in {t["function"]["name"] for t in declared}
                    result = world.call(*skin.call_back(fn["name"], fn["arguments"]))
                    assert "error" not in result or episode.family == "unknown_subject"

    # Names that were renamed away are rejected, not silently accepted.
    world = generate(1, seed=5)[0].make_world()
    assert "error" in world.call(*HELDOUT.call_back("list_homework", {}))
    assert "error" in world.call(*HELDOUT.call_back("hw_query", {"status": "todo"}))
    assert "error" not in world.call(*HELDOUT.call_back("hw_query", {"progress": "todo"}))
    described = HELDOUT.declarations(TOOLS)
    text = " ".join(t["function"]["description"] + " ".join(p.get("description", "") for p in t["function"]["parameters"]["properties"].values()) for t in described)
    assert not any(name in text for name in TOOLS_BY_NAME) and "cadence=once" in text and "done — виконані" in text


def test_chat_app_finds_image_paths_in_a_message(tmp_path):
    from demo.chat import split_images

    photo = tmp_path / "my task.JPG"
    photo.write_bytes(b"x")
    escaped = str(photo).replace(" ", "\\ ")
    assert split_images(f"що тут написано? {escaped}") == ("що тут написано?", [str(photo)])
    assert split_images(f"'{photo}' розв'яжи") == (f"'{photo}' розв'яжи", [])  # an apostrophe in the text: left as typed
    assert split_images(f'"{photo}" поясни') == ("поясни", [str(photo)])
    assert split_images("файл notes.png не існує") == ("файл notes.png не існує", [])
    assert split_images("звичайне питання") == ("звичайне питання", [])


def test_chat_app_separates_thinking_from_the_answer():
    from demo.chat import split_thought

    assert split_thought("Відповідь: 7") == ("", "Відповідь: 7")
    assert split_thought("<|channel>thought\n1. c² = 49") == ("1. c² = 49", None)
    assert split_thought("<|channel>thought\n1. c² = 49\n<channel|>Відповідь: 7") == ("1. c² = 49", "Відповідь: 7")
    thought, rest = split_thought("<|channel>thought\nтреба список<channel|><|tool_call>call:list_homework{}<tool_call|>")
    assert thought == "треба список" and rest.startswith("<|tool_call>")
