"""Renamed variants of the example tools.

The app's real tools will not be called `list_homework`. So that the model reads
the declarations it is given instead of memorising one set of names, training
episodes are rewritten with other tool and parameter names (drawn at random per
episode from the alternatives below), and one set of names is held out: it
never appears in training and is used by `eval/run_hf.py --skin heldout` to
measure how well a model works with tools under names it has not seen.

A renaming covers the declarations (names, parameter keys, `required`, and
mentions of other tools inside descriptions), the calls and the tool responses.
Keys inside a tool's result are left as they are.
"""

from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass, field

TRAIN_TOOL_NAMES: dict[str, list[str]] = {
    "list_homework": ["listHomework", "homework_list", "diary_tasks", "get_assignments", "tasks.search"],
    "get_homework_details": ["getHomeworkDetails", "homework_info", "diary_task_info", "get_assignment", "tasks.get"],
    "mark_homework_done": ["markHomeworkDone", "homework_set_done", "diary_set_done", "complete_assignment", "tasks.setDone"],
    "list_recipients": ["listRecipients", "contacts_search", "find_contacts", "get_contacts", "contacts.list"],
    "send_message": ["sendMessage", "chat_send", "message_send", "write_message", "contacts.message"],
    "get_last_grades": ["getLastGrades", "grades_recent", "recent_marks", "fetch_grades", "grades.latest"],
    "get_analytics": ["getAnalytics", "progress_report", "study_stats", "fetch_statistics", "grades.stats"],
    "get_schedule": ["getSchedule", "timetable_day", "lessons_for_date", "fetch_timetable", "timetable.get"],
    "create_scheduled_task": ["createScheduledTask", "reminder_create", "add_reminder", "schedule_action", "reminders.create"],
    "list_scheduled_tasks": ["listScheduledTasks", "reminder_list", "show_reminders", "scheduled_actions", "reminders.list"],
    "cancel_scheduled_task": ["cancelScheduledTask", "reminder_delete", "remove_reminder", "unschedule_action", "reminders.delete"],
    "send_notification": ["sendNotification", "push_notify", "notify_user", "show_notification", "device.notify"],
}
TRAIN_PARAM_NAMES: dict[str, list[str]] = {
    "due_date": ["dueDate", "due", "day"], "status": ["state", "filter"], "subject": ["subjectName", "discipline"],
    "homework_id": ["homeworkId", "id", "assignment_id"], "done": ["isDone", "value"], "role": ["group", "type"],
    "query": ["search", "q"], "recipient_id": ["recipientId", "contact_id", "to"], "text": ["message", "content"],
    "limit": ["count", "n"], "period": ["range", "interval"], "date": ["day", "for_date"], "time": ["at", "clock"],
    "recurrence": ["repeat", "frequency"], "instruction": ["action", "prompt"], "weekday": ["dayOfWeek", "week_day"],
    "task_id": ["taskId", "reminder_id", "id"], "title": ["header", "caption"],
}

# Held out: none of these names may appear in the training alternatives above.
HELDOUT_TOOL_NAMES: dict[str, str] = {
    "list_homework": "hw_query", "get_homework_details": "hw_open", "mark_homework_done": "hw_tick",
    "list_recipients": "people_lookup", "send_message": "dm_post", "get_last_grades": "marks_feed",
    "get_analytics": "performance_summary", "get_schedule": "classes_on", "create_scheduled_task": "job_add",
    "list_scheduled_tasks": "job_index", "cancel_scheduled_task": "job_drop", "send_notification": "alert_push",
}
HELDOUT_PARAM_NAMES: dict[str, str] = {
    "due_date": "deadline", "status": "progress", "subject": "course", "homework_id": "hw", "done": "completed",
    "role": "kind", "query": "name_part", "recipient_id": "person", "text": "body", "limit": "max_items",
    "period": "span", "date": "on_date", "time": "hhmm", "recurrence": "cadence", "instruction": "task_text",
    "weekday": "dow", "task_id": "job", "title": "heading",
}


@dataclass
class Renaming:
    tools: dict[str, str] = field(default_factory=dict)  # original tool name -> new name
    params: dict[str, str] = field(default_factory=dict)  # original parameter name -> new name

    def __post_init__(self) -> None:
        self._back_tools = {new: old for old, new in self.tools.items()}
        self._back_params = {new: old for old, new in self.params.items()}

        def either(names) -> str:
            return "(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + ")"

        self._tool_mention = re.compile(r"(?<![\w.])" + either(self.tools) + r"(?!\w)") if self.tools else None
        # A parameter is renamed in prose only where it is clearly meant as one ("recurrence=once",
        # "параметр date"): "done" or "date" may also be an enum value or an ordinary word.
        self._param_mention = re.compile(r"(?<![\w.])" + either(self.params) + r"(?==)|(?<=параметр )" + either(self.params) + r"(?!\w)") if self.params else None

    def text(self, value: str) -> str:
        """Descriptions and error messages mention tools and parameters by name."""
        if self._tool_mention:
            value = self._tool_mention.sub(lambda m: self.tools[m.group(1)], value)
        if self._param_mention:
            value = self._param_mention.sub(lambda m: self.params[m.group(1) or m.group(2)], value)
        return value

    def call(self, name: str, arguments: dict) -> tuple[str, dict]:
        return self.tools.get(name, name), {self.params.get(k, k): v for k, v in arguments.items()}

    def call_back(self, name: str, arguments: dict) -> tuple[str, dict]:
        """A call the model made under the new names, as the simulator understands it.

        An original name that was renamed away is not declared any more, so it must
        fail like any unknown name would (a model may have memorised it): it is
        passed on with a "!" in front, which the simulator rejects."""
        def back(value: str, forward: dict, backward: dict) -> str:
            return backward[value] if value in backward else f"!{value}" if value in forward else value

        return back(name, self.tools, self._back_tools), {back(k, self.params, self._back_params): v for k, v in arguments.items()}

    def declarations(self, tools: list[dict]) -> list[dict]:
        out = []
        for tool in copy.deepcopy(tools):
            fn = tool["function"]
            fn["name"] = self.tools.get(fn["name"], fn["name"])
            fn["description"] = self.text(fn["description"])
            schema = fn["parameters"]
            schema["properties"] = {self.params.get(k, k): {**v, "description": self.text(v["description"])} if "description" in v else v
                                    for k, v in schema["properties"].items()}
            schema["required"] = [self.params.get(k, k) for k in schema.get("required", [])]
            out.append(tool)
        return out

    def response(self, result: dict) -> dict:
        return {**result, "error": self.text(result["error"])} if isinstance(result.get("error"), str) else result

    def messages(self, messages: list[dict]) -> list[dict]:
        out = []
        for message in copy.deepcopy(messages):
            for call in message.get("tool_calls", []):
                fn = call["function"]
                fn["name"], fn["arguments"] = self.call(fn["name"], fn["arguments"])
            for item in message.get("tool_responses", []):
                item["name"] = self.tools.get(item["name"], item["name"])
                item["response"] = self.response(item["response"])
            out.append(message)
        return out


HELDOUT = Renaming(HELDOUT_TOOL_NAMES, HELDOUT_PARAM_NAMES)


def random_renaming(rng: random.Random, tool_share: float = 0.8, param_share: float = 0.5) -> Renaming:
    """A different mix of names each time, so no single alternative set can be memorised either."""
    tools = {old: rng.choice(new) for old, new in TRAIN_TOOL_NAMES.items() if rng.random() < tool_share}
    params: dict[str, str] = {}
    for old, new in TRAIN_PARAM_NAMES.items():
        if rng.random() < param_share:
            params[old] = rng.choice(new)
    # Two parameters of one tool must not end up with the same name ("date" and "due_date" -> "day", ids -> "id").
    seen: dict[str, str] = {}
    for old in list(params):
        if params[old] in seen or params[old] in TRAIN_PARAM_NAMES and params[old] not in params:
            del params[old]
        else:
            seen[params[old]] = old
    return Renaming(tools, params)
