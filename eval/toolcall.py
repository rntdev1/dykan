"""Scoring for tool use: did the model reach the right end state?

An episode's `expect` has two parts:
  reads: calls that must have happened without error, matched on name and on a
         subset of arguments (extra arguments are allowed);
  state: the exact set of changes the task should leave behind
         homework_done   [homework ids newly marked done]
         messages        [{"recipient_id", "text_contains"?}]
         notifications   [{"text_contains"?}]
         scheduled       [{"time", "recurrence", "date"?, "weekday"?}]
         cancelled       [scheduled task ids removed]

Anything changed beyond `state` fails the episode: a wrong message sent or the
wrong homework ticked is worse than doing nothing.
"""

from __future__ import annotations

from data.tools.episodes import Episode
from data.tools.sim import SchoolWorld


def _contains_all(text: str, needles: list[str]) -> bool:
    low = text.lower()
    return all(n.lower() in low for n in needles)


def _match_each(expected: list[dict], actual: list[dict], matches) -> bool:
    """Every expected item pairs with a distinct actual item, and none is left over."""
    if len(expected) != len(actual):
        return False
    remaining = list(actual)
    for want in expected:
        hit = next((got for got in remaining if matches(want, got)), None)
        if hit is None:
            return False
        remaining.remove(hit)
    return True


def score_world(episode: Episode, world: SchoolWorld) -> dict:
    """Score a world after the model's calls were executed on `episode.make_world()`."""
    baseline = episode.make_world()
    expect_state = episode.expect.get("state", {})

    ok_calls = [c for c in world.calls if "error" not in c["result"]]
    reads_ok = all(
        any(c["name"] == want["name"] and all(c["arguments"].get(k) == v for k, v in want["arguments"].items()) for c in ok_calls)
        for want in episode.expect.get("reads", [])
    )

    before_done = {h["id"] for h in baseline.homework if h["done"]}
    after_done = {h["id"] for h in world.homework if h["done"]}
    homework_ok = (after_done - before_done) == set(expect_state.get("homework_done", [])) and not (before_done - after_done)

    messages_ok = _match_each(
        expect_state.get("messages", []), world.messages,
        lambda want, got: got["recipient_id"] == want["recipient_id"] and _contains_all(got["text"], want.get("text_contains", [])),
    )
    notifications_ok = _match_each(
        expect_state.get("notifications", []), world.notifications,
        lambda want, got: _contains_all(got["title"] + "\n" + got["text"], want.get("text_contains", [])),
    )

    before_ids = {t["id"] for t in baseline.scheduled}
    after_ids = {t["id"] for t in world.scheduled}
    added = [t for t in world.scheduled if t["id"] not in before_ids]
    scheduled_ok = _match_each(
        expect_state.get("scheduled", []), added,
        lambda want, got: all(got.get(k) == v for k, v in want.items()),
    )
    cancelled_ok = (before_ids - after_ids) == set(expect_state.get("cancelled", []))

    checks = {
        "reads": reads_ok, "homework": homework_ok, "messages": messages_ok,
        "notifications": notifications_ok, "scheduled": scheduled_ok, "cancelled": cancelled_ok,
    }
    return {
        "family": episode.family,
        "success": all(checks.values()),
        "checks": checks,
        "calls": len(world.calls),
        "invalid_calls": len(world.calls) - len(ok_calls),
    }


def score_calls(episode: Episode, calls: list[dict]) -> dict:
    """Convenience for a fixed list of calls (gold trajectories, unit tests)."""
    world = episode.make_world()
    for call in calls:
        world.call(call["name"], call.get("arguments") or {})
    return score_world(episode, world)


def summarize(results: list[dict]) -> dict:
    by_family: dict[str, list[bool]] = {}
    for r in results:
        by_family.setdefault(r["family"], []).append(r["success"])
    total_calls = sum(r["calls"] for r in results)
    return {
        "episodes": len(results),
        "task_completion": round(sum(r["success"] for r in results) / len(results), 4) if results else None,
        "invalid_call_rate": round(sum(r["invalid_calls"] for r in results) / total_calls, 4) if total_calls else 0.0,
        "by_family": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_family.items())},
    }
