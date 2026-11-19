from data.teacher.recipes import answers_match, explain_jobs, final_answer, reword_keeps_facts, slots, split_sections, task_jobs
from data.teacher.syllabus import SYLLABUS, topics

GOOD = """### ЗАВДАННЯ
Знайдіть 15% від 240.

### РОЗВ'ЯЗАННЯ
15% — це 0,15. Множимо: 240 · 0,15 = 36.

### ВІДПОВІДЬ
36"""


def test_split_sections():
    parts = split_sections(GOOD)
    assert parts == {"task": "Знайдіть 15% від 240.", "solution": "15% — це 0,15. Множимо: 240 · 0,15 = 36.", "answer": "36"}
    assert split_sections(GOOD.replace("РОЗВ'ЯЗАННЯ", "РОЗВ’ЯЗАННЯ"))["answer"] == "36"
    assert split_sections(GOOD.replace("### ВІДПОВІДЬ\n36", "")) is None
    assert split_sections(GOOD.replace("### ЗАВДАННЯ", "### РОЗВ'ЯЗАННЯ")) is None
    assert split_sections("Просто текст") is None


def test_final_answer():
    assert final_answer("Міркую.\nВідповідь: 36 км") == "36 км"
    assert final_answer("Спершу відповідь: ні.\nПотім думаю.\nВідповідь: Б") == "Б"
    assert final_answer("42") == "42"


def test_answers_match():
    assert answers_match("number", "36", "36,0 км") and answers_match("number", "0,5", "0.5")
    assert answers_match("number", "-3", "x = −3")
    assert not answers_match("number", "36", "63") and not answers_match("number", "36", "не знаю")
    assert answers_match("choice", "Б", "Б) друге") and not answers_match("choice", "Б", "В")
    assert not answers_match("choice", "Б", "Було б добре")  # letters inside words do not count
    assert answers_match("short", "Іван Франко", "Це Іван Франко.") and answers_match("short", "1648", "1648 рік")
    assert not answers_match("short", "Іван Франко", "Леся Українка") and not answers_match("short", "", "")


def test_reword_must_keep_every_fact():
    original = "Постав нагадування щодня о 18: підсумуй домашку на завтра з предмета «Історія України»."
    assert "«історія україни»" in slots(original) and "18" in slots(original) and "завтра" in slots(original)
    assert reword_keeps_facts(original, "Слухай, зроби щодня о 18 зведення по «Історія України» на завтра, ок?")
    assert not reword_keeps_facts(original, "Зроби щодня о 19 зведення по «Історія України» на завтра")
    assert not reword_keeps_facts(original, "Зроби щодня о 18 зведення з історії на завтра")
    assert not reword_keeps_facts(original, "Зроби о 18 зведення по «Історія України» на завтра")


def test_jobs_cover_the_syllabus_and_are_deterministic():
    all_topics = topics()
    assert len(all_topics) > 150 and len(SYLLABUS) == 13
    assert len(topics(include_unreliable=True)) > len(all_topics) and not any(s == "Українська мова" for s, _, _ in all_topics)
    jobs = task_jobs(len(all_topics) + 50, seed=1)
    assert {(j["subject"], j["topic"]) for j in jobs[: len(all_topics)]} == {(s, t) for s, _, t in all_topics}
    assert jobs == task_jobs(len(all_topics) + 50, seed=1) and len({j["id"] for j in jobs}) == len(jobs)
    assert all(j["kind"] in {"choice", "number", "short"} for j in jobs)
    assert any(j["kind"] == "number" for j in jobs if j["subject"] == "Фізика")
    assert not any(j["kind"] == "number" for j in jobs if j["subject"] == "Історія України")
    assert all("{topic}" in j["ask"] for j in explain_jobs(20, 1))


def test_section_headings_tolerate_small_variations():
    assert split_sections(GOOD.replace("### ", "**").replace("ЗАВДАННЯ", "ЗАВДАННЯ**").replace("РОЗВ'ЯЗАННЯ", "РОЗВ'ЯЗАННЯ**").replace("ВІДПОВІДЬ", "ВІДПОВІДЬ**"))["answer"] == "36"
    assert split_sections(GOOD.replace("### ЗАВДАННЯ", "## Завдання:"))["task"].startswith("Знайдіть")


def test_short_answers_with_units_or_extra_words():
    assert answers_match("short", "4", "4°C") and answers_match("short", "11 см", "11")
    assert answers_match("short", "Франко", "Іван Франко") and not answers_match("short", "Іван Франко", "Іван Котляревський")
    assert not answers_match("short", "4", "5 °C")


def test_parse_dialogue():
    from data.teacher.recipes import chat_jobs, parse_dialogue, tutor_jobs

    text = "УЧЕНЬ: Не розумію, як знайти 15% від 240.\nДИКАНЬ: Що означає один відсоток?\n\n**УЧЕНЬ:** Соту частину.\n**ДИКАНЬ:** Так. Тоді 240 · 0,15 = 36."
    msgs = parse_dialogue(text)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "Не розумію, як знайти 15% від 240." and msgs[3]["content"] == "Так. Тоді 240 · 0,15 = 36."
    assert parse_dialogue("УЧЕНЬ: а\nДИКАНЬ: б") is None  # one exchange is not a tutoring dialogue
    assert parse_dialogue("ДИКАНЬ: а\nУЧЕНЬ: б\nДИКАНЬ: в\nУЧЕНЬ: г") is None
    assert parse_dialogue("УЧЕНЬ: а\nУЧЕНЬ: б\nДИКАНЬ: в\nДИКАНЬ: г") is None
    assert parse_dialogue("УЧЕНЬ: а\nДИКАНЬ: б\nУЧЕНЬ: в") is None
    assert len({j["id"] for j in tutor_jobs(50, 1) + chat_jobs(50, 1)}) == 100


def test_anytool_episode_checks():
    from data.teacher.recipes import check_anytool_episode, check_arguments, parse_json_object

    tools = [
        {"name": "books.find", "description": "Шукає книжки.", "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Назва або автор."}, "limit": {"type": "integer", "description": "Скільки показати."}}, "required": ["query"]}},
        {"name": "books.reserve", "description": "Бронює книжку.", "parameters": {"type": "object", "properties": {
            "book_id": {"type": "string", "description": "Ідентифікатор книжки."}}, "required": ["book_id"]}},
    ]
    find = {"name": "books.find", "arguments": {"query": "Кобзар"}, "result": {"books": [{"id": "b17", "title": "Кобзар", "free": True}]}}
    reserve = {"name": "books.reserve", "arguments": {"book_id": "b17"}, "result": {"reserved": True}}
    good = {"tools": tools, "user": "Забронюй мені «Кобзар».", "steps": [find, reserve], "final": "Забронював «Кобзар»."}
    assert check_anytool_episode(good, "chain") is None
    assert check_anytool_episode({**good, "steps": [find]}, "single") is None
    assert check_anytool_episode(good, "single") == "wrong_step_count"
    # The second call must use what the first one returned.
    assert check_anytool_episode({**good, "steps": [find, {**reserve, "arguments": {"book_id": "b99"}}]}, "chain") == "chain_does_not_use_results"
    assert check_anytool_episode({**good, "steps": [{**find, "arguments": {"title": "Кобзар"}}]}, "single") == "arguments_do_not_fit_schema"
    assert check_anytool_episode({**good, "steps": [{**find, "name": "books.search"}]}, "single") == "bad_step"
    assert check_anytool_episode({**good, "steps": [find]}, "error") == "no_error_in_result"
    assert check_anytool_episode({**good, "steps": [{**find, "result": {"error": "Нічого не знайдено"}}]}, "error") is None
    assert check_anytool_episode({**good, "steps": []}, "none") is None
    assert check_anytool_episode({**good, "steps": [], "final": "Виклич books.find."}, "none") == "tool_name_in_text"

    schema = tools[0]["parameters"]
    assert check_arguments(schema, {"query": "x", "limit": 3}) is None
    assert check_arguments(schema, {"limit": 3}) == "missing required query"
    assert check_arguments(schema, {"query": "x", "limit": "3"}) == "wrong type for limit"
    assert check_arguments(schema, {"query": "x", "limit": True}) == "wrong type for limit"
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1} and parse_json_object("not json") is None
