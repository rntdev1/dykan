"""A simulated school backend for the example tool set.

Two uses:
  * data generation: produces realistic, consistent tool results for training
    episodes, so the model never learns from made-up results that contradict
    each other;
  * evaluation: a model's calls are executed here and scored on the end state
    (was the right homework marked done, was the right message sent, ...).

Each world is built deterministically from a seed and a "today" date.
"""

from __future__ import annotations

import random
import re
from copy import deepcopy
from datetime import date, timedelta

from data.tools.schemas import TOOLS_BY_NAME

SUBJECTS = [
    "Алгебра", "Геометрія", "Українська мова", "Українська література", "Історія України",
    "Всесвітня історія", "Англійська мова", "Фізика", "Хімія", "Біологія", "Географія", "Інформатика",
]

HOMEWORK_TEMPLATES = {
    "Алгебра": [("Квадратні рівняння", "§{p}, вправи {a}–{b}. Розв'язати через дискримінант, записати перевірку."),
                ("Арифметична прогресія", "§{p}, № {a}, {b}, {c}. Повторити формулу суми перших n членів.")],
    "Геометрія": [("Теорема косинусів", "§{p}, задачі {a}–{b}. Виконати рисунки до кожної задачі."),
                  ("Площі многокутників", "§{p}, № {a}, {c}. Вивчити формули площі трапеції та ромба.")],
    "Українська мова": [("Складнопідрядні речення", "§{p}, вправа {a}. Виписати речення, накреслити схеми."),
                        ("Правопис апострофа", "Вправа {a} письмово, вправа {b} усно. Повторити правила на с. {c}.")],
    "Українська література": [("Тарас Шевченко. «Катерина»", "Прочитати поему повністю, с. {a}–{b}. Підготувати характеристику Катерини."),
                              ("Леся Українка. «Лісова пісня»", "Прочитати дію першу, с. {a}–{b}. Вивчити напам'ять монолог Мавки.")],
    "Історія України": [("Національно-визвольна війна", "§{p}, прочитати. Скласти хронологічну таблицю подій 1648–1657 рр."),
                        ("Українська революція 1917–1921", "§{p}–{q}. Відповісти письмово на запитання {a}–{b} після параграфа.")],
    "Всесвітня історія": [("Велика французька революція", "§{p}, прочитати. Підготувати повідомлення на 3 хвилини.")],
    "Англійська мова": [("Present Perfect", "Workbook p. {a}, ex. {b}–{c}. Вивчити неправильні дієслова, с. {p}."),
                        ("My future profession", "Написати твір на 120–150 слів. Вивчити нові слова, с. {a}.")],
    "Фізика": [("Закони Ньютона", "§{p}, вправа {a} (задачі 1–4). Повторити означення сили та маси."),
               ("Електромагнітна індукція", "§{p}–{q}, прочитати. Розв'язати задачі {a}, {b}.")],
    "Хімія": [("Окисно-відновні реакції", "§{p}, № {a}–{b}. Урівняти рівняння методом електронного балансу."),
              ("Вуглеводні", "§{p}, прочитати. Скласти формули ізомерів пентану.")],
    "Біологія": [("Фотосинтез", "§{p}, прочитати. Заповнити таблицю «Світлова і темнова фази»."),
                 ("Мітоз і мейоз", "§{p}–{q}. Замалювати фази мітозу в зошиті.")],
    "Географія": [("Клімат України", "§{p}, прочитати. Позначити на контурній карті кліматичні області."),],
    "Інформатика": [("Цикли в Python", "Написати програму, що обчислює суму чисел від 1 до n. Надіслати файл учителю."),],
}

TEACHER_NAMES = [
    "Олена Петрівна Коваленко", "Ірина Василівна Шевчук", "Андрій Миколайович Бондар", "Наталія Іванівна Мельник",
    "Сергій Олександрович Ткаченко", "Оксана Богданівна Кравець", "Тетяна Степанівна Гнатюк", "Віктор Павлович Лисенко",
    "Людмила Григорівна Савчук", "Юлія Андріївна Мороз", "Олег Романович Поліщук", "Марія Федорівна Руденко",
]
CLASSMATE_NAMES = ["Максим Гончар", "Софія Литвин", "Артем Захарченко", "Анна Дмитренко", "Дмитро Остапчук", "Вікторія Павленко"]
PARENT_NAMES = [("Мама", "Світлана"), ("Тато", "Володимир")]

LESSON_TIMES = ["08:30", "09:25", "10:20", "11:25", "12:20", "13:15", "14:10", "15:05"]
WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_JSON_TYPES = {"string": str, "integer": int, "boolean": bool, "number": (int, float)}


class ToolError(Exception):
    pass


class SchoolWorld:
    def __init__(self, seed: int = 0, today: date | None = None):
        self.rng = random.Random(seed)
        self.today = today or date(2026, 11, 16)
        self.subjects = self.rng.sample(SUBJECTS, k=self.rng.randint(8, len(SUBJECTS)))
        self.recipients = self._make_recipients()
        self.timetable = self._make_timetable()
        self.homework = self._make_homework()
        self.grades = self._make_grades()
        self.messages: list[dict] = []
        self.notifications: list[dict] = []
        self.scheduled: list[dict] = []
        self._next_task = 1
        self.calls: list[dict] = []  # log of every call, for scoring

    # ---------- world construction ----------

    def _make_recipients(self) -> list[dict]:
        names = self.rng.sample(TEACHER_NAMES, k=len(self.subjects))
        people = [
            {"id": f"t{i + 1}", "name": name, "role": "teacher", "subject": subject}
            for i, (name, subject) in enumerate(zip(names, self.subjects))
        ]
        people += [{"id": f"p{i + 1}", "name": f"{label} ({name})", "role": "parent"} for i, (label, name) in enumerate(PARENT_NAMES)]
        people += [{"id": f"c{i + 1}", "name": name, "role": "classmate"} for i, name in enumerate(self.rng.sample(CLASSMATE_NAMES, k=4))]
        return people

    def _teacher_of(self, subject: str) -> str:
        return next(p["name"] for p in self.recipients if p.get("subject") == subject)

    def _make_timetable(self) -> dict[int, list[dict]]:
        days = [self.rng.sample(self.subjects, k=min(len(self.subjects), self.rng.randint(5, 7))) for _ in range(5)]
        # Every subject is taught at least once a week, so its homework always has a due day.
        for subject in self.subjects:
            if not any(subject in day for day in days):
                min(days, key=len).append(subject)
        table = {}
        for weekday, lessons in enumerate(days):
            table[weekday] = [
                {"lesson": n + 1, "time": LESSON_TIMES[n], "subject": s, "room": str(self.rng.randint(101, 320)), "teacher": self._teacher_of(s)}
                for n, s in enumerate(lessons)
            ]
        return table

    def _next_lesson_day(self, subject: str, after: date) -> date:
        for step in range(1, 15):
            day = after + timedelta(days=step)
            if any(lesson["subject"] == subject for lesson in self.timetable.get(day.weekday(), [])):
                return day
        return after + timedelta(days=1)

    def _make_homework(self) -> list[dict]:
        items = []
        for n in range(self.rng.randint(8, 14)):
            subject = self.rng.choice(self.subjects)
            title, template = self.rng.choice(HOMEWORK_TEMPLATES[subject])
            a = self.rng.randint(20, 300)
            p = self.rng.randint(5, 40)
            details = template.format(p=p, q=p + 1, a=a, b=a + self.rng.randint(2, 6), c=a + self.rng.randint(7, 15))
            start = self.today - timedelta(days=self.rng.randint(0, 4))
            due = self._next_lesson_day(subject, start)
            items.append({
                "id": f"hw{n + 1}", "subject": subject, "title": title, "details": details,
                "due_date": due.isoformat(), "done": due < self.today or self.rng.random() < 0.3,
            })
        return sorted(items, key=lambda h: (h["due_date"], h["id"]))

    def _make_grades(self) -> list[dict]:
        level = {s: self.rng.randint(6, 11) for s in self.subjects}
        grades = []
        for back in range(1, 120):
            day = self.today - timedelta(days=back)
            if day.weekday() >= 5:
                continue
            for subject in self.rng.sample(self.subjects, k=self.rng.randint(0, 2)):
                grade = max(1, min(12, level[subject] + self.rng.randint(-2, 2)))
                kind = self.rng.choice(["усна відповідь", "самостійна робота", "контрольна робота", "домашнє завдання", "тест"])
                grades.append({"date": day.isoformat(), "subject": subject, "grade": grade, "type": kind})
        return grades  # newest first

    # ---------- calling ----------

    def call(self, name: str, arguments: dict | None = None) -> dict:
        """Execute one tool call. Never raises: errors come back as {"error": ...},
        which is what the model must learn to handle."""
        arguments = dict(arguments or {})
        entry = {"name": name, "arguments": deepcopy(arguments)}
        try:
            if name not in TOOLS_BY_NAME:
                raise ToolError(f"Невідомий інструмент: {name}")
            self._validate(name, arguments)
            result = getattr(self, f"_{name}")(**arguments)
        except ToolError as e:
            result = {"error": str(e)}
        entry["result"] = deepcopy(result)
        self.calls.append(entry)
        return result

    def _validate(self, name: str, arguments: dict) -> None:
        params = TOOLS_BY_NAME[name]["parameters"]
        for key in params["required"]:
            if key not in arguments:
                raise ToolError(f"Бракує обов'язкового параметра: {key}")
        for key, value in arguments.items():
            spec = params["properties"].get(key)
            if spec is None:
                raise ToolError(f"Невідомий параметр: {key}")
            expected = _JSON_TYPES[spec["type"]]
            if not isinstance(value, expected) or (spec["type"] != "boolean" and isinstance(value, bool)):
                raise ToolError(f"Параметр {key} має бути типу {spec['type']}")
            if "enum" in spec and value not in spec["enum"]:
                raise ToolError(f"Параметр {key} має бути одним із: {', '.join(spec['enum'])}")

    @staticmethod
    def _date(value: str) -> date:
        if not _DATE_RE.match(value):
            raise ToolError("Дата має бути у форматі РРРР-ММ-ДД")
        try:
            return date.fromisoformat(value)
        except ValueError:
            raise ToolError(f"Такої дати не існує: {value}")

    def _subject(self, subject: str) -> str:
        match = next((s for s in self.subjects if s.lower() == subject.strip().lower()), None)
        if match is None:
            raise ToolError(f"Предмет «{subject}» не знайдено. Доступні предмети: {', '.join(self.subjects)}")
        return match

    def _find(self, items: list[dict], item_id: str, what: str) -> dict:
        item = next((x for x in items if x["id"] == item_id), None)
        if item is None:
            raise ToolError(f"{what} з ідентифікатором {item_id} не знайдено")
        return item

    # ---------- tools ----------

    def _list_homework(self, due_date: str | None = None, status: str = "all", subject: str | None = None) -> dict:
        items = self.homework
        if due_date is not None:
            day = self._date(due_date).isoformat()
            items = [h for h in items if h["due_date"] == day]
        if subject is not None:
            name = self._subject(subject)
            items = [h for h in items if h["subject"] == name]
        if status != "all":
            items = [h for h in items if h["done"] == (status == "done")]
        return {"homework": [{k: h[k] for k in ("id", "subject", "title", "due_date", "done")} for h in items]}

    def _get_homework_details(self, homework_id: str) -> dict:
        return dict(self._find(self.homework, homework_id, "Завдання"))

    def _mark_homework_done(self, homework_id: str, done: bool = True) -> dict:
        item = self._find(self.homework, homework_id, "Завдання")
        item["done"] = done
        return {"id": item["id"], "done": done}

    def _list_recipients(self, role: str | None = None, query: str | None = None) -> dict:
        people = self.recipients
        if role is not None:
            people = [p for p in people if p["role"] == role]
        if query:
            q = query.strip().lower()
            people = [p for p in people if q in p["name"].lower() or q in p.get("subject", "").lower()]
        return {"recipients": people}

    def _send_message(self, recipient_id: str, text: str) -> dict:
        person = self._find(self.recipients, recipient_id, "Отримувача")
        if not text.strip():
            raise ToolError("Текст повідомлення порожній")
        self.messages.append({"recipient_id": recipient_id, "text": text})
        return {"sent": True, "recipient": person["name"]}

    def _get_last_grades(self, subject: str | None = None, limit: int = 10) -> dict:
        if limit < 1:
            raise ToolError("limit має бути додатним числом")
        grades = self.grades
        if subject is not None:
            name = self._subject(subject)
            grades = [g for g in grades if g["subject"] == name]
        return {"grades": grades[:limit]}

    def _get_analytics(self, period: str, subject: str | None = None) -> dict:
        days = {"week": 7, "month": 30, "semester": 110}[period]
        subjects = [self._subject(subject)] if subject is not None else self.subjects
        start = self.today - timedelta(days=days)
        previous = start - timedelta(days=days)

        def average(name: str, lo: date, hi: date) -> float | None:
            values = [g["grade"] for g in self.grades if g["subject"] == name and lo <= date.fromisoformat(g["date"]) < hi]
            return round(sum(values) / len(values), 1) if values else None

        rows = []
        for name in subjects:
            now, before = average(name, start, self.today), average(name, previous, start)
            rows.append({
                "subject": name, "average": now, "previous_average": before,
                "change": round(now - before, 1) if now is not None and before is not None else None,
            })
        relevant = [h for h in self.homework if h["subject"] in subjects]
        done_share = round(sum(h["done"] for h in relevant) / len(relevant), 2) if relevant else None
        return {"period": period, "subjects": rows, "homework_done_share": done_share}

    def _get_schedule(self, date: str) -> dict:  # noqa: A002 - parameter name is part of the tool schema
        day = self._date(date)
        return {"date": day.isoformat(), "weekday": WEEKDAYS[day.weekday()], "lessons": self.timetable.get(day.weekday(), [])}

    def _create_scheduled_task(self, time: str, recurrence: str, instruction: str, date: str | None = None, weekday: str | None = None) -> dict:  # noqa: A002
        if not _TIME_RE.match(time):
            raise ToolError("Час має бути у форматі ГГ:ХХ")
        if not instruction.strip():
            raise ToolError("Інструкція порожня")
        if recurrence == "once":
            if date is None:
                raise ToolError("Для recurrence=once потрібен параметр date")
            if self._date(date) < self.today:
                raise ToolError("Дата вже минула")
        if recurrence == "weekly" and weekday is None:
            raise ToolError("Для recurrence=weekly потрібен параметр weekday")
        task = {"id": f"task{self._next_task}", "time": time, "recurrence": recurrence, "instruction": instruction}
        if recurrence == "once":
            task["date"] = date
        if recurrence == "weekly":
            task["weekday"] = weekday
        self._next_task += 1
        self.scheduled.append(task)
        return dict(task)

    def _list_scheduled_tasks(self) -> dict:
        return {"tasks": [dict(t) for t in self.scheduled]}

    def _cancel_scheduled_task(self, task_id: str) -> dict:
        task = self._find(self.scheduled, task_id, "Заплановане завдання")
        self.scheduled.remove(task)
        return {"cancelled": task_id}

    def _send_notification(self, title: str, text: str) -> dict:
        if not text.strip():
            raise ToolError("Текст сповіщення порожній")
        self.notifications.append({"title": title, "text": text})
        return {"shown": True}
