from collections import Counter

from data.tools.episodes import FAMILIES, generate
from eval.toolcall import score_calls, summarize

EPISODES = generate(260, seed=1)


def test_all_families_are_generated():
    counts = Counter(e.family for e in EPISODES)
    assert set(counts) == {f.__name__ for f in FAMILIES}
    assert min(counts.values()) >= 10


def test_generation_is_deterministic():
    again = generate(260, seed=1)
    assert [(e.family, e.seed, e.user, e.gold) for e in again] == [(e.family, e.seed, e.user, e.gold) for e in EPISODES]
    assert [e.user for e in generate(40, seed=2)] != [e.user for e in EPISODES[:40]]


def test_gold_trajectory_always_succeeds():
    results = [score_calls(e, e.gold) for e in EPISODES]
    failed = [(e.family, r["checks"]) for e, r in zip(EPISODES, results) if not r["success"]]
    assert not failed, failed[:5]
    assert summarize(results)["task_completion"] == 1.0


def test_gold_calls_are_valid_except_the_deliberate_error():
    for e in EPISODES:
        r = score_calls(e, e.gold)
        assert r["invalid_calls"] == (1 if e.family == "unknown_subject" else 0), e.family


def test_doing_nothing_fails_tasks_that_need_action():
    needs_action = {"homework_for_day", "digest_now", "schedule_digest", "schedule_once", "mark_done", "message_teacher",
                    "last_grades", "analytics", "schedule_for_weekday", "cancel_task"}
    for e in EPISODES:
        assert score_calls(e, [])["success"] == (e.family not in needs_action), e.family


def test_unwanted_side_effects_fail():
    spam = {"name": "send_notification", "arguments": {"title": "x", "text": "y"}}
    for e in EPISODES:
        if e.family != "digest_now":
            assert not score_calls(e, e.gold + [spam])["success"], e.family


def test_wrong_target_fails():
    checked_mark_done = False
    for e in EPISODES:
        if e.family == "mark_done":
            world = e.make_world()
            other = next((h["id"] for h in world.homework if not h["done"] and h["id"] != e.gold[1]["arguments"]["homework_id"]), None)
            if other is None:  # this world has a single unfinished item
                continue
            checked_mark_done = True
            assert not score_calls(e, [e.gold[0], {"name": "mark_homework_done", "arguments": {"homework_id": other}}])["success"]
        if e.family == "schedule_digest":
            wrong = dict(e.gold[0]["arguments"], time="06:00")
            assert not score_calls(e, [{"name": "create_scheduled_task", "arguments": wrong}])["success"]
        if e.family == "homework_for_day":
            assert not score_calls(e, [{"name": "list_homework", "arguments": {"due_date": "2020-01-01"}}])["success"]
    assert checked_mark_done


def test_messages_have_the_template_shape():
    for e in EPISODES[:60]:
        msgs = e.messages()
        assert [m["role"] for m in msgs[:2]] == ["system", "user"] and msgs[-1]["role"] == "assistant" and msgs[-1]["content"]
        assert e.today.isoformat() in msgs[0]["content"]
        assert len(msgs) == len(e.gold) + 3
        for step in msgs[2:-1]:
            assert step["role"] == "assistant"
            assert isinstance(step["tool_calls"][0]["function"]["arguments"], dict)
            assert isinstance(step["tool_responses"][0]["response"], dict)
            assert step["tool_responses"][0]["name"] == step["tool_calls"][0]["function"]["name"]
