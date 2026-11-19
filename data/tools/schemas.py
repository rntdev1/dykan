"""Example tool set for Dykan, in the JSON-schema function format that
`apply_chat_template(..., tools=TOOLS)` accepts.

This is a stand-in until the app's real tools exist. Training does not memorise
it: episodes also use renamed and reworded variants and unrelated tools, so the
model learns to read whatever declarations it is given.

The app is expected to put the current date, time and weekday in the system
prompt; the model needs them to resolve "завтра" or "щодня о 18:00".
"""

from __future__ import annotations

DATE = {"type": "string", "description": "Дата у форматі РРРР-ММ-ДД, наприклад 2026-11-16."}


def _tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


TOOLS: list[dict] = [
    _tool(
        "list_homework",
        "Повертає список домашніх завдань учня. Можна відфільтрувати за датою здачі, статусом і предметом.",
        {
            "due_date": {**DATE, "description": "Показати лише завдання, які треба здати цього дня (РРРР-ММ-ДД)."},
            "status": {
                "type": "string",
                "enum": ["todo", "done", "all"],
                "description": "todo — невиконані, done — виконані, all — усі. Типово all.",
            },
            "subject": {"type": "string", "description": "Назва предмета, наприклад «Алгебра»."},
        },
    ),
    _tool(
        "get_homework_details",
        "Повертає повний опис одного домашнього завдання: що саме прочитати, які вправи виконати, що підготувати.",
        {"homework_id": {"type": "string", "description": "Ідентифікатор завдання зі списку list_homework."}},
        ["homework_id"],
    ),
    _tool(
        "mark_homework_done",
        "Позначає домашнє завдання як виконане або знімає цю позначку.",
        {
            "homework_id": {"type": "string", "description": "Ідентифікатор завдання."},
            "done": {"type": "boolean", "description": "true — виконано, false — не виконано. Типово true."},
        },
        ["homework_id"],
    ),
    _tool(
        "list_recipients",
        "Повертає людей, яким учень може написати: вчителів, батьків, однокласників.",
        {
            "role": {"type": "string", "enum": ["teacher", "parent", "classmate"], "description": "Показати лише людей із цією роллю."},
            "query": {"type": "string", "description": "Пошук за іменем або предметом, який викладає вчитель."},
        },
    ),
    _tool(
        "send_message",
        "Надсилає текстове повідомлення одному отримувачу.",
        {
            "recipient_id": {"type": "string", "description": "Ідентифікатор отримувача зі списку list_recipients."},
            "text": {"type": "string", "description": "Текст повідомлення."},
        },
        ["recipient_id", "text"],
    ),
    _tool(
        "get_last_grades",
        "Повертає останні оцінки учня за 12-бальною шкалою, від найновіших.",
        {
            "subject": {"type": "string", "description": "Назва предмета. Якщо не вказано — з усіх предметів."},
            "limit": {"type": "integer", "description": "Скільки оцінок повернути. Типово 10."},
        },
    ),
    _tool(
        "get_analytics",
        "Повертає статистику успішності за період: середній бал з кожного предмета, зміну порівняно з попереднім періодом і частку виконаних домашніх завдань.",
        {
            "period": {"type": "string", "enum": ["week", "month", "semester"], "description": "Період: тиждень, місяць або семестр."},
            "subject": {"type": "string", "description": "Назва предмета. Якщо не вказано — усі предмети."},
        },
        ["period"],
    ),
    _tool(
        "get_schedule",
        "Повертає розклад уроків на вказаний день: номер уроку, час, предмет, кабінет, учитель.",
        {"date": DATE},
        ["date"],
    ),
    _tool(
        "create_scheduled_task",
        "Створює заплановане завдання для асистента: у вказаний час він виконає інструкцію (наприклад, збере невиконані домашні завдання й надішле сповіщення).",
        {
            "time": {"type": "string", "description": "Час у форматі ГГ:ХХ, наприклад 18:00."},
            "recurrence": {
                "type": "string",
                "enum": ["once", "daily", "weekdays", "weekly"],
                "description": "once — один раз, daily — щодня, weekdays — з понеділка по п'ятницю, weekly — щотижня.",
            },
            "instruction": {"type": "string", "description": "Що асистент має зробити в цей час, своїми словами."},
            "date": {**DATE, "description": "Дата для recurrence=once (РРРР-ММ-ДД)."},
            "weekday": {
                "type": "string",
                "enum": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
                "description": "День тижня для recurrence=weekly.",
            },
        },
        ["time", "recurrence", "instruction"],
    ),
    _tool("list_scheduled_tasks", "Повертає всі заплановані завдання асистента.", {}),
    _tool(
        "cancel_scheduled_task",
        "Скасовує заплановане завдання.",
        {"task_id": {"type": "string", "description": "Ідентифікатор зі списку list_scheduled_tasks."}},
        ["task_id"],
    ),
    _tool(
        "send_notification",
        "Показує сповіщення на телефоні учня.",
        {
            "title": {"type": "string", "description": "Короткий заголовок."},
            "text": {"type": "string", "description": "Текст сповіщення."},
        },
        ["title", "text"],
    ),
]

TOOLS_BY_NAME: dict[str, dict] = {t["function"]["name"]: t["function"] for t in TOOLS}
