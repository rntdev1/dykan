"""Multi-step assistant episodes over the example tool set.

Each episode is a user request in Ukrainian, a gold tool-call trajectory executed
against the simulated school world (so every tool result is real and consistent),
and a declarative expectation of the end state.

The same object serves two purposes:
  * eval: run a model on `system` + `user`, execute its calls on `make_world()`,
    then score the world with eval/toolcall.py;
  * training skeleton: `messages()` is a complete conversation in the format the
    Gemma 4 chat template accepts. Final answers here are plain templates; the
    teacher model later rewrites user requests and final answers for variety.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from data.tools.schemas import TOOLS
from data.tools.sim import SchoolWorld

WEEKDAY_NOM = ["понеділок", "вівторок", "середа", "четвер", "п'ятниця", "субота", "неділя"]
WEEKDAY_ACC = ["понеділок", "вівторок", "середу", "четвер", "п'ятницю", "суботу", "неділю"]
MONTH_GEN = ["січня", "лютого", "березня", "квітня", "травня", "червня", "липня", "серпня", "вересня", "жовтня", "листопада", "грудня"]


def calendar_lines(today: date, days: int = 7) -> str:
    """'tomorrow', 'the day after' and the following days with weekday and ISO date,
    so the model can look a date up instead of computing it (both stock models
    get weekday dates wrong, and E4B produced 31 November)."""
    labels = {1: "завтра", 2: "післязавтра"}
    parts = []
    for step in range(1, days + 1):
        day = today + timedelta(days=step)
        name = f"{labels[step]} — {WEEKDAY_NOM[day.weekday()]}" if step in labels else WEEKDAY_NOM[day.weekday()]
        parts.append(f"{name}, {day.day} {MONTH_GEN[day.month - 1]} ({day.isoformat()})")
    return "Найближчі дні: " + "; ".join(parts) + "."


def system_prompt(today: date, clock: str = "15:30", calendar: bool = False) -> str:
    text = (
        "Ти — Дикань, шкільний асистент. Відповідай українською, коротко й по суті. "
        "Користуйся інструментами, коли потрібні дані учня або дія; не вигадуй даних. "
        f"Сьогодні {WEEKDAY_NOM[today.weekday()]}, {today.day} {MONTH_GEN[today.month - 1]} {today.year} року "
        f"({today.isoformat()}), зараз {clock}."
    )
    return f"{text} {calendar_lines(today)}" if calendar else text


@dataclass
class Episode:
    family: str
    seed: int
    today: date
    user: str
    gold: list[dict]  # [{"name", "arguments"}] in order
    expect: dict  # see eval/toolcall.py
    final: str  # template final answer
    setup: list[dict] = field(default_factory=list)  # calls applied before the episode starts
    tools: list[dict] = field(default_factory=lambda: TOOLS)

    @property
    def system(self) -> str:
        return system_prompt(self.today)

    def make_world(self) -> SchoolWorld:
        world = SchoolWorld(self.seed, self.today)
        for call in self.setup:
            world.call(call["name"], call["arguments"])
        world.calls.clear()
        return world

    def messages(self) -> list[dict]:
        """Full gold conversation.

        Tool results ride on the assistant message as `tool_responses` (Gemma's
        native form). The template renders those as structured
        `response:name{key:value}` blocks; a separate role:tool message would have
        to carry a string and would be rendered as one opaque quoted value.
        """
        world = self.make_world()
        out = [{"role": "system", "content": self.system}, {"role": "user", "content": self.user}]
        for call in self.gold:
            result = world.call(call["name"], call["arguments"])
            out.append({
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}],
                "tool_responses": [{"name": call["name"], "response": result}],
            })
        out.append({"role": "assistant", "content": self.final})
        return out


def _call(name: str, **arguments) -> dict:
    return {"name": name, "arguments": arguments}


def _expect(reads: list[dict] | None = None, **state) -> dict:
    return {"reads": reads or [], "state": state}


def _pick(rng, train: bool, eval_options: list, train_options: list):
    """Training and eval use disjoint phrasings, so the tool eval measures the
    skill and not memorised wording. The eval branch must keep consuming the
    random stream exactly as before, or baselines stop being comparable."""
    return rng.choice(train_options if train else eval_options)


def _hw_line(hw: dict) -> str:
    return f"{hw['subject']} — {hw['title']}: {hw['details']}"


# ---------- episode families ----------
# Each takes (seed, today, rng) and returns an Episode, or None if this world
# cannot host the task (for example, nothing is due tomorrow).


def homework_for_day(seed, today, rng, train=False):
    world = SchoolWorld(seed, today)
    offset, phrase = rng.choice([(1, "на завтра"), (2, "на післязавтра"), (0, "на сьогодні")])
    day = (today + timedelta(days=offset)).isoformat()
    items = world.call("list_homework", {"due_date": day})["homework"]
    user = _pick(rng, train, [f"Що мені задали {phrase}?", f"Яка домашка {phrase}?", f"Покажи домашнє завдання {phrase}."],
                 [f"Що треба зробити {phrase}?", f"Нагадай, що задано {phrase}.", f"Є якісь завдання {phrase}?"])
    final = (
        f"Домашнє завдання {phrase}:\n" + "\n".join(f"• {h['subject']}: {h['title']}" + (" (виконано)" if h["done"] else "") for h in items)
        if items else f"{phrase.capitalize()} нічого не задано."
    )
    return Episode("homework_for_day", seed, today, user, [_call("list_homework", due_date=day)],
                   _expect([_call("list_homework", due_date=day)]), final)


def digest_now(seed, today, rng, train=False):
    world = SchoolWorld(seed, today)
    tomorrow = (today + timedelta(days=1)).isoformat()
    todo = world.call("list_homework", {"due_date": tomorrow, "status": "todo"})["homework"]
    if not todo:
        return None
    gold = [_call("list_homework", due_date=tomorrow, status="todo")]
    details = []
    for h in todo:
        gold.append(_call("get_homework_details", homework_id=h["id"]))
        details.append(world.call("get_homework_details", {"homework_id": h["id"]}))
    text = "\n".join(_hw_line(d) for d in details)
    gold.append(_call("send_notification", title="Домашнє на завтра", text=text))
    user = _pick(rng, train, [
        "Збери всю невиконану домашку на завтра — що саме прочитати й підготувати — і надішли мені сповіщенням на телефон.",
        "Підсумуй, що я ще не зробив на завтра, з подробицями, і кинь це мені сповіщенням.",
    ], [
        "Перевір, що в мене не зроблено на завтра, випиши, що читати й готувати, і надішли сповіщення.",
        "Зроби зведення невиконаних завдань на завтра з деталями та відправ його мені сповіщенням на телефон.",
    ])
    return Episode(
        "digest_now", seed, today, user, gold,
        _expect([_call("list_homework", due_date=tomorrow)], notifications=[{"text_contains": sorted({d["subject"] for d in details})}]),
        "Надіслав сповіщення з невиконаними завданнями на завтра:\n" + text,
    )


def schedule_digest(seed, today, rng, train=False):
    time = rng.choice(["18:00", "17:30", "19:00", "20:15", "07:00"])
    recurrence, phrase = rng.choice([("daily", "щодня"), ("daily", "кожного дня"), ("weekdays", "у будні"), ("weekdays", "з понеділка по п'ятницю")])
    hour = time.replace(":00", "") if time.endswith(":00") else time
    user = _pick(rng, train, [
        f"Постав нагадування {phrase} о {hour}: коли я приходжу зі школи, підсумуй невиконану домашку на завтра, що саме прочитати й підготувати, і надішли мені на телефон.",
        f"Хочу, щоб ти {phrase} о {hour} збирав мені невиконані домашні завдання на наступний день і надсилав сповіщення.",
    ], [
        f"Зроби так, щоб {phrase} о {hour} мені приходило сповіщення з невиконаною домашкою на завтра: що прочитати й що підготувати.",
        f"Налаштуй {phrase} о {hour} автоматичний підсумок: які завдання на завтра ще не зроблені, і надсилай його мені на телефон.",
    ])
    instruction = "Зібрати невиконані домашні завдання на завтра з подробицями (що прочитати й підготувати) і надіслати сповіщення на телефон."
    return Episode(
        "schedule_digest", seed, today, user,
        [_call("create_scheduled_task", time=time, recurrence=recurrence, instruction=instruction)],
        _expect(scheduled=[{"time": time, "recurrence": recurrence}]),
        f"Готово. {phrase.capitalize()} о {time} я збиратиму невиконані завдання на наступний день і надсилатиму сповіщення.",
    )


def schedule_once(seed, today, rng, train=False):
    offset, phrase = rng.choice([(1, "завтра"), (2, "післязавтра")])
    day = (today + timedelta(days=offset)).isoformat()
    time = rng.choice(["07:15", "16:00", "19:30"])
    what = rng.choice(["взяти спортивну форму", "здати гроші на екскурсію", "принести контурні карти"])
    return Episode(
        "schedule_once", seed, today,
        rng.choice([f"Постав нагадування на {phrase}, {time}: {what}.", f"{phrase.capitalize()} о {time} нагадай, будь ласка, {what}."]) if train
        else f"Нагадай мені {phrase} о {time} {what}.",
        [_call("create_scheduled_task", time=time, recurrence="once", date=day, instruction=f"Нагадати: {what}.")],
        _expect(scheduled=[{"time": time, "recurrence": "once", "date": day}]),
        f"Добре, нагадаю {phrase} о {time}: {what}.",
    )


def mark_done(seed, today, rng, train=False):
    world = SchoolWorld(seed, today)
    todo = world.call("list_homework", {"status": "todo"})["homework"]
    by_subject: dict[str, list] = {}
    for h in todo:
        by_subject.setdefault(h["subject"], []).append(h)
    unique = [items[0] for items in by_subject.values() if len(items) == 1]
    if not unique:
        return None
    hw = rng.choice(unique)
    subject = hw["subject"]
    user = _pick(rng, train, [f"Я зробив домашку з предмета «{subject}», познач.", f"Познач завдання з предмета «{subject}» як виконане."],
                 [f"{subject} — готово, відміть як зроблене.", f"Домашнє з предмета «{subject}» я вже виконав, зазнач це."])
    return Episode(
        "mark_done", seed, today, user,
        [_call("list_homework", subject=subject, status="todo"), _call("mark_homework_done", homework_id=hw["id"])],
        _expect(homework_done=[hw["id"]]),
        f"Позначив як виконане: {subject} — {hw['title']}.",
    )


def message_teacher(seed, today, rng, train=False):
    world = SchoolWorld(seed, today)
    teacher = rng.choice([p for p in world.recipients if p["role"] == "teacher"])
    reason = rng.choice(["завтра не прийду на урок, бо йду до лікаря", "не встигаю здати проєкт до п'ятниці й прошу ще два дні"])
    user = (rng.choice([f"Передай учителю, який веде предмет «{teacher['subject']}», що я {reason}.", f"Надішли повідомлення вчителю ({teacher['subject']}): я {reason}."])
            if train else f"Напиши вчителю з предмета «{teacher['subject']}», що я {reason}.")
    text = f"Добрий день! Повідомляю, що я {reason}."
    return Episode(
        "message_teacher", seed, today, user,
        [_call("list_recipients", role="teacher", query=teacher["subject"]), _call("send_message", recipient_id=teacher["id"], text=text)],
        _expect(messages=[{"recipient_id": teacher["id"]}]),
        f"Надіслав повідомлення: {teacher['name']}.",
    )


def last_grades(seed, today, rng, train=False):
    world = SchoolWorld(seed, today)
    subject = rng.choice(world.subjects)
    grades = world.call("get_last_grades", {"subject": subject, "limit": 5})["grades"]
    user = _pick(rng, train, [f"Які в мене останні оцінки з предмета «{subject}»?", f"Покажи останні п'ять оцінок: {subject}."],
                 [f"Що в мене з оцінками: {subject}?", f"Глянь мої останні оцінки з предмета «{subject}»."])
    final = (f"Останні оцінки, {subject}: " + ", ".join(f"{g['grade']} ({g['type']}, {g['date']})" for g in grades)) if grades else f"Оцінок з предмета «{subject}» ще немає."
    return Episode("last_grades", seed, today, user, [_call("get_last_grades", subject=subject, limit=5)],
                   _expect([_call("get_last_grades", subject=subject)]), final)


def analytics(seed, today, rng, train=False):
    world = SchoolWorld(seed, today)
    period, phrase = rng.choice([("week", "за тиждень"), ("month", "за місяць"), ("semester", "за семестр")])
    stats = world.call("get_analytics", {"period": period})
    rows = [r for r in stats["subjects"] if r["average"] is not None]
    user = _pick(rng, train, [f"Як у мене справи з навчанням {phrase}?", f"Покажи мою статистику {phrase}."],
                 [f"Як я вчуся {phrase}? Покажи середні бали.", f"Дай зведення успішності {phrase}."])
    final = f"Статистика {phrase}: " + "; ".join(f"{r['subject']} — {r['average']}" for r in rows) + f". Виконано домашніх завдань: {round((stats['homework_done_share'] or 0) * 100)}%."
    return Episode("analytics", seed, today, user, [_call("get_analytics", period=period)], _expect([_call("get_analytics", period=period)]), final)


def schedule_for_weekday(seed, today, rng, train=False):
    world = SchoolWorld(seed, today)
    step = rng.randint(1, 6)
    day = today + timedelta(days=step)
    lessons = world.call("get_schedule", {"date": day.isoformat()})["lessons"]
    phrase = "завтра" if step == 1 else f"у {WEEKDAY_ACC[day.weekday()]}"
    user = _pick(rng, train, [f"Які в мене уроки {phrase}?", f"Покажи розклад на {'завтра' if step == 1 else WEEKDAY_ACC[day.weekday()]}."],
                 [f"Що в мене {phrase} за розкладом?", f"Скільки уроків {phrase} і які саме?"])
    final = (f"Розклад {phrase}:\n" + "\n".join(f"{x['lesson']}. {x['time']} {x['subject']}, каб. {x['room']}" for x in lessons)) if lessons else f"{phrase.capitalize()} уроків немає."
    return Episode("schedule_for_weekday", seed, today, user, [_call("get_schedule", date=day.isoformat())],
                   _expect([_call("get_schedule", date=day.isoformat())]), final)


def cancel_task(seed, today, rng, train=False):
    setup = [
        _call("create_scheduled_task", time="18:00", recurrence="daily", instruction="Надіслати невиконану домашку на завтра."),
        _call("create_scheduled_task", time="07:00", recurrence="weekdays", instruction="Надіслати розклад на сьогодні."),
    ]
    target = rng.choice([0, 1])
    time = setup[target]["arguments"]["time"]
    return Episode(
        "cancel_task", seed, today,
        _pick(rng, train, [f"Скасуй нагадування о {time}.", f"Більше не треба нічого надсилати о {time}, прибери це."],
              [f"Видали нагадування, яке спрацьовує о {time}.", f"Нагадування о {time} мені вже не потрібне, скасуй його."]),
        [_call("list_scheduled_tasks"), _call("cancel_scheduled_task", task_id=f"task{target + 1}")],
        _expect(cancelled=[f"task{target + 1}"]),
        f"Скасував нагадування о {time}.", setup=setup,
    )


def no_tool_needed(seed, today, rng, train=False):
    user, final = _pick(rng, train, [
        ("Що таке дискримінант і для чого він потрібен?", "Дискримінант квадратного рівняння ax² + bx + c = 0 — це число D = b² − 4ac. Якщо D > 0, рівняння має два корені, якщо D = 0 — один, якщо D < 0 — дійсних коренів немає."),
        ("Як правильно: «на протязі дня» чи «протягом дня»?", "Правильно — «протягом дня». «На протязі» означає «на вітрі, що дме крізь кімнату»."),
        ("Привіт! Що ти вмієш?", "Привіт! Я можу пояснити тему чи задачу, показати домашні завдання, оцінки й розклад, написати вчителю або поставити нагадування."),
    ], [
        ("Чим відрізняється мітоз від мейозу?", "Мітоз дає дві клітини з таким самим набором хромосом, як у материнської; так ростуть і оновлюються тканини. Мейоз дає чотири клітини з удвічі меншим набором; так утворюються статеві клітини."),
        ("Як пишеться: «будь ласка» чи «будь-ласка»?", "Правильно — «будь ласка», двома словами без дефіса. У реченні це вставне слово, тому виділяється комами."),
        ("Скільки буде 15% від 240?", "15% від 240 — це 240 · 0,15 = 36."),
        ("Дякую, ти дуже допоміг!", "Радий допомогти! Звертайся, якщо буде ще щось потрібно."),
    ])
    return Episode("no_tool_needed", seed, today, user, [], _expect(), final)


def missing_detail(seed, today, rng, train=False):
    user, final = _pick(rng, train, [
        ("Надішли повідомлення, що я запізнюся.", "Кому надіслати повідомлення — вчителю, батькам чи комусь з однокласників?"),
        ("Постав мені нагадування про контрольну.", "На який день і о котрій годині поставити нагадування?"),
    ], [
        ("Напиши, що мене завтра не буде.", "Кому написати — класному керівнику, комусь із учителів чи батькам?"),
        ("Нагадай мені про репетитора.", "Коли нагадати — якого дня і о котрій годині?"),
        ("Познач домашку як зроблену.", "Яке саме завдання позначити — з якого предмета?"),
    ])
    return Episode("missing_detail", seed, today, user, [], _expect(), final)


def unknown_subject(seed, today, rng, train=False):
    world = SchoolWorld(seed, today)
    missing = next((s for s in ["Астрономія", "Креслення", "Латина"] if s not in world.subjects))
    return Episode(
        "unknown_subject", seed, today,
        f"Покажи оцінки з предмета «{missing}»." if train else f"Які в мене оцінки з предмета «{missing}»?",
        [_call("get_last_grades", subject=missing)], _expect(),
        f"Предмета «{missing}» немає у твоєму списку. Твої предмети: {', '.join(world.subjects)}.",
    )


FAMILIES = [
    homework_for_day, digest_now, schedule_digest, schedule_once, mark_done, message_teacher,
    last_grades, analytics, schedule_for_weekday, cancel_task, no_tool_needed, missing_detail, unknown_subject,
]


def _tool_subset(episode: Episode, rng: random.Random) -> list[dict]:
    """The tools this episode needs plus a random handful of others, shuffled:
    training sees many different tool lists, so it cannot rely on a fixed one."""
    needed = {c["name"] for c in episode.gold} | {c["name"] for c in episode.setup}
    others = [t for t in TOOLS if t["function"]["name"] not in needed]
    chosen = [t for t in TOOLS if t["function"]["name"] in needed] + rng.sample(others, k=rng.randint(2, len(others)))
    rng.shuffle(chosen)
    return chosen


def generate(n: int, seed: int = 0, start: date = date(2026, 9, 1), train: bool = False) -> list[Episode]:
    """n episodes, cycling through the families, each in its own school world and day.

    train=True switches to the training phrasings and gives each episode a
    random subset of the tools; the default is the eval set with all tools."""
    rng = random.Random(seed)
    episodes: list[Episode] = []
    world_seed = seed * 1_000_003
    while len(episodes) < n:
        family = FAMILIES[len(episodes) % len(FAMILIES)]
        world_seed += 1
        today = start + timedelta(days=rng.randint(0, 250))
        if today.weekday() >= 5:
            today += timedelta(days=7 - today.weekday())
        episode = family(world_seed, today, rng, train)
        if episode is not None:
            if train:
                episode.tools = _tool_subset(episode, random.Random(world_seed))
            episodes.append(episode)
    return episodes
