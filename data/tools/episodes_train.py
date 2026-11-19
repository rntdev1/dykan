"""Wide training-only episode generator.

`episodes.py` holds the evaluation episodes and a first, narrow set of training
phrasings. This module produces training episodes only, with far more variety
in what changes from one request to the next: times, days (named weekdays and
explicit dates), subjects in the forms pupils really use ("з алгебри"),
recipients of every role, kinds of reminders, and more task types (chains of
two tools, undoing an action, listing reminders, requests no tool can serve).

The evaluation set in `episodes.py` is not touched by anything here.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from data.tools.episodes import MONTH_GEN, WEEKDAY_ACC, Episode, _call, _expect, _tool_subset
from data.tools.sim import WEEKDAYS, SchoolWorld

SUBJECT_GEN = {
    "Алгебра": "з алгебри", "Геометрія": "з геометрії", "Українська мова": "з української мови",
    "Українська література": "з української літератури", "Історія України": "з історії України",
    "Всесвітня історія": "із всесвітньої історії", "Англійська мова": "з англійської мови", "Фізика": "з фізики",
    "Хімія": "з хімії", "Біологія": "з біології", "Географія": "з географії", "Інформатика": "з інформатики",
}
WEEKLY = {
    "mon": ["щопонеділка", "кожного понеділка"], "tue": ["щовівторка", "кожного вівторка"], "wed": ["щосереди", "кожної середи"],
    "thu": ["щочетверга", "кожного четверга"], "fri": ["щоп'ятниці", "кожної п'ятниці"], "sat": ["щосуботи", "кожної суботи"],
    "sun": ["щонеділі", "кожної неділі"],
}
CLASSMATE_DATIVE = {
    "Максим Гончар": "Максиму Гончару", "Софія Литвин": "Софії Литвин", "Артем Захарченко": "Артему Захарченку",
    "Анна Дмитренко": "Анні Дмитренко", "Дмитро Остапчук": "Дмитру Остапчуку", "Вікторія Павленко": "Вікторії Павленко",
}
LAST_N = {3: ["три останні оцінки", "3 останні оцінки"], 5: ["п'ять останніх оцінок", "5 останніх оцінок"], 10: ["десять останніх оцінок", "10 останніх оцінок"]}


def _subject(rng, subject: str) -> str:
    """The subject the way a pupil would mention it after "задано", "оцінки" and the like."""
    return rng.choice([SUBJECT_GEN[subject], SUBJECT_GEN[subject], f"з предмета «{subject}»"])


def _time(rng, lo: int = 6, hi: int = 22) -> tuple[str, str]:
    """(HH:MM, the phrase a pupil would write, with its preposition)."""
    hour, minute = rng.randint(lo, hi), rng.choice([0, 0, 0, 0, 30, 30, 15, 45, 10, 20, 40, 50, 5])
    value = f"{hour:02d}:{minute:02d}"
    prep = "об" if hour == 11 else "о"
    forms = [f"{prep} {value}", f"{prep} {hour}:{minute:02d}"]
    if minute == 0:
        forms.append(f"{prep} {hour}")
        if hour <= 11:
            forms.append(f"{prep} {hour} ранку")
        if hour >= 18:
            evening = hour - 12
            forms.append(f"{'об' if evening == 11 else 'о'} {evening} вечора")
    return value, rng.choice(forms)


def _day(rng, today: date, lo: int = 0, hi: int = 7) -> tuple[date, str, str]:
    """(date, "when" phrase such as «завтра» / «у п'ятницю», "for when" phrase such as «на завтра»)."""
    step = rng.randint(lo, hi)
    day = today + timedelta(days=step)
    explicit = f"{day.day} {MONTH_GEN[day.month - 1]}"
    if step == 0:
        return day, "сьогодні", "на сьогодні"
    if step == 1:
        return day, "завтра", "на завтра"
    if step == 2 and rng.random() < 0.6:
        return day, "післязавтра", "на післязавтра"
    if step <= 6 and rng.random() < 0.7:
        name = WEEKDAY_ACC[day.weekday()]
        return day, f"у {name}", f"на {name}"
    return day, explicit, f"на {explicit}"


# ---------- reading the diary ----------


def homework_for_day(seed, today, rng):
    world = SchoolWorld(seed, today)
    day, _, on = _day(rng, today, 0, 6)
    args = {"due_date": day.isoformat()}
    kind = rng.choice(["all", "all", "todo", "subject"])
    if kind == "todo":
        args["status"] = "todo"
        user = rng.choice([f"Що в мене ще не зроблено {on}?", f"Які завдання {on} я ще не виконав?", f"Що лишилося зробити {on}?",
                           f"Покажи тільки невиконану домашку {on}."])
    elif kind == "subject":
        subject = rng.choice(world.subjects)
        args["subject"] = subject
        user = rng.choice([f"Що задано {_subject(rng, subject)} {on}?", f"Є домашка {_subject(rng, subject)} {on}?",
                           f"{subject}: що треба зробити {on}?"])
    else:
        user = rng.choice([f"Що треба зробити {on}?", f"Нагадай, що задано {on}.", f"Є якісь завдання {on}?", f"Що по домашці {on}?",
                           f"Скинь список домашніх завдань {on}.", f"Багато задали {on}?", f"Домашка {on} — що там?"])
    return Episode("homework_for_day", seed, today, user, [_call("list_homework", **args)], _expect(), "")


def homework_details(seed, today, rng):
    world = SchoolWorld(seed, today)
    by_subject: dict[str, list] = {}
    for h in world.call("list_homework", {"status": "todo"})["homework"]:
        by_subject.setdefault(h["subject"], []).append(h)
    options = [(s, items) for s, items in by_subject.items() if len(items) <= 2]
    if not options:
        return None
    subject, items = rng.choice(options)
    s = _subject(rng, subject)
    user = rng.choice([f"Що саме треба зробити {s}? Розпиши детально.", f"Розкажи докладно, що задано {s}.", f"Які вправи й параграфи мені задали {s}?",
                       f"Що читати й що писати {s}?", f"Не пам'ятаю, що конкретно задали {s}. Глянь, будь ласка."])
    gold = [_call("list_homework", subject=subject, status="todo")] + [_call("get_homework_details", homework_id=h["id"]) for h in items]
    return Episode("homework_details", seed, today, user, gold, _expect(), "")


def day_overview(seed, today, rng):
    day, at, on = _day(rng, today, 0, 6)
    iso = day.isoformat()
    user = rng.choice([f"Які в мене уроки {at} і що на них задано?", f"Розклад і домашка {on}, будь ласка.", f"Що мене чекає {at}: уроки й завдання?",
                       f"Підготуй мені план {on}: які уроки та що треба було зробити.", f"Покажи розклад {on} разом із домашніми завданнями."])
    return Episode("day_overview", seed, today, user, [_call("get_schedule", date=iso), _call("list_homework", due_date=iso)], _expect(), "")


def schedule_for_day(seed, today, rng):
    day, at, on = _day(rng, today, 0, 7)
    user = rng.choice([f"Що в мене {at} за розкладом?", f"Скільки уроків {at} і які саме?", f"Який розклад {on}?", f"Які уроки {at}?",
                       f"О котрій {at} починається перший урок?", f"Коли {at} закінчуються уроки?", f"Скинь розклад {on}.", f"У якому кабінеті {at} перший урок?"])
    return Episode("schedule_for_weekday", seed, today, user, [_call("get_schedule", date=day.isoformat())], _expect(), "")


def last_grades(seed, today, rng):
    world = SchoolWorld(seed, today)
    args: dict = {}
    limit = rng.choice([None, None, 3, 5, 10])
    if rng.random() < 0.7:
        subject = rng.choice(world.subjects)
        args["subject"] = subject
        s = _subject(rng, subject)
        if limit:
            last = rng.choice(LAST_N[limit])
            user = rng.choice([f"Покажи {last} {s}.", f"Які в мене {last} {s}?", f"Мені потрібні {last} {s}."])
        else:
            user = rng.choice([f"Що в мене з оцінками {s}?", f"Глянь мої останні оцінки {s}.", f"Які в мене оцінки {s}?", f"Що я отримав {s} останнім часом?",
                               f"Мені вже поставили оцінку {s}?", f"{subject} — які останні оцінки?"])
    elif limit:
        last = rng.choice(LAST_N[limit])
        user = rng.choice([f"Покажи {last} з усіх предметів.", f"Які мої {last}?"])
    else:
        user = rng.choice(["Які в мене останні оцінки?", "Що мені поставили останнім часом?", "Покажи свіжі оцінки.", "Є нові оцінки?"])
    if limit:
        args["limit"] = limit
    return Episode("last_grades", seed, today, user, [_call("get_last_grades", **args)], _expect(), "")


def analytics(seed, today, rng):
    world = SchoolWorld(seed, today)
    period, phrase = rng.choice([("week", "за тиждень"), ("week", "за останній тиждень"), ("month", "за місяць"), ("month", "за останній місяць"),
                                 ("semester", "за семестр"), ("semester", "за цей семестр")])
    args = {"period": period}
    if rng.random() < 0.35:
        subject = rng.choice(world.subjects)
        args["subject"] = subject
        s = _subject(rng, subject)
        user = rng.choice([f"Який у мене середній бал {s} {phrase}?", f"Як я вчуся {s} {phrase}?", f"Покажи мою успішність {s} {phrase}.",
                           f"Я покращив оцінки {s} {phrase} чи ні?"])
    else:
        user = rng.choice([f"Як я вчуся {phrase}? Покажи середні бали.", f"Дай зведення успішності {phrase}.", f"Які в мене середні бали {phrase}?",
                           f"З яких предметів я {phrase} з'їхав, а з яких підтягнувся?", f"Покажи аналітику навчання {phrase}.",
                           f"Скільки домашніх завдань я виконав і які середні оцінки {phrase}?", f"Який мій найслабший предмет {phrase}?"])
    return Episode("analytics", seed, today, user, [_call("get_analytics", **args)], _expect(), "")


def unknown_subject(seed, today, rng):
    missing = rng.choice(["Астрономія", "Креслення", "Латина", "Економіка", "Правознавство", "Мистецтво", "Німецька мова", "Фізкультура",
                          "Основи здоров'я", "Трудове навчання"])
    kind = rng.choice(["grades", "grades", "homework", "analytics"])
    if kind == "grades":
        gold = [_call("get_last_grades", subject=missing)]
        user = rng.choice([f"Покажи оцінки з предмета «{missing}».", f"Що в мене з оцінками з предмета «{missing}»?", f"«{missing}» — які останні оцінки?",
                           f"Мені щось поставили з предмета «{missing}»?"])
    elif kind == "homework":
        gold = [_call("list_homework", subject=missing, status="todo")]
        user = rng.choice([f"Що задано з предмета «{missing}»?", f"Є невиконана домашка з предмета «{missing}»?", f"«{missing}»: що треба зробити?"])
    else:
        period, phrase = rng.choice([("month", "за місяць"), ("semester", "за семестр")])
        gold = [_call("get_analytics", period=period, subject=missing)]
        user = rng.choice([f"Який у мене середній бал з предмета «{missing}» {phrase}?", f"Покажи успішність з предмета «{missing}» {phrase}."])
    return Episode("unknown_subject", seed, today, user, gold, _expect(), "")


# ---------- acting ----------


def digest_now(seed, today, rng):
    world = SchoolWorld(seed, today)
    day, _, on = _day(rng, today, 0, 2)
    todo = world.call("list_homework", {"due_date": day.isoformat(), "status": "todo"})["homework"]
    if not todo:
        return None
    gold = [_call("list_homework", due_date=day.isoformat(), status="todo")]
    details = []
    for h in todo:
        gold.append(_call("get_homework_details", homework_id=h["id"]))
        details.append(world.call("get_homework_details", {"homework_id": h["id"]}))
    text = "\n".join(f"{d['subject']} — {d['title']}: {d['details']}" for d in details)
    gold.append(_call("send_notification", title=f"Домашнє {on}", text=text))
    user = rng.choice([
        f"Перевір, що в мене не зроблено {on}, випиши, що читати й готувати, і надішли сповіщення.",
        f"Зроби зведення невиконаних завдань {on} з деталями та відправ його мені сповіщенням на телефон.",
        f"Надішли мені на телефон список того, що ще треба зробити {on}, з подробицями.",
        f"Збери все невиконане {on} — параграфи, вправи — і кинь мені сповіщення.",
        f"Хочу сповіщення з усім, що лишилося зробити {on}. Тільки з деталями, що саме робити.",
        f"Скинь мені в сповіщення домашку {on}, яку я ще не зробив, з описом завдань.",
    ])
    return Episode("digest_now", seed, today, user, gold, _expect(), "")


SCHEDULED_KINDS = [
    ("digest", "Зібрати невиконані домашні завдання на завтра з подробицями (що прочитати й підготувати) і надіслати сповіщення на телефон.",
     ["мені приходило сповіщення з невиконаною домашкою на завтра: що прочитати й що підготувати",
      "ти збирав мені невиконані завдання на наступний день і надсилав на телефон",
      "я отримував підсумок: що на завтра ще не зроблено й що саме треба зробити"], "нагадування про домашку"),
    ("timetable", "Надіслати сповіщення з розкладом уроків на сьогодні.",
     ["мені приходив розклад уроків на сьогодні", "ти надсилав мені сповіщення з уроками на цей день",
      "я отримував на телефон розклад на день"], "нагадування з розкладом"),
    ("grades", "Надіслати сповіщення з останніми оцінками.",
     ["мені приходили мої останні оцінки", "ти надсилав мені сповіщення зі свіжими оцінками"], "нагадування з оцінками"),
    ("week", "Надіслати зведення успішності за тиждень: середні бали та частку виконаних домашніх завдань.",
     ["мені приходило зведення успішності за тиждень", "ти надсилав мені підсумок навчання за тиждень: середні бали й виконану домашку"],
     "тижневе зведення успішності"),
]


def _recurrence(rng, hour: int) -> tuple[dict, str]:
    kind = rng.choice(["daily", "daily", "weekdays", "weekdays", "weekly"])
    if kind == "daily":
        phrases = ["щодня", "кожного дня"] + (["щовечора"] if hour >= 17 else []) + (["щоранку"] if hour <= 10 else [])
        return {"recurrence": "daily"}, rng.choice(phrases)
    if kind == "weekdays":
        return {"recurrence": "weekdays"}, rng.choice(["у будні", "з понеділка по п'ятницю", "по буднях", "кожного буднього дня"])
    weekday = rng.choice(WEEKDAYS)
    return {"recurrence": "weekly", "weekday": weekday}, rng.choice(WEEKLY[weekday])


def schedule_recurring(seed, today, rng):
    time, at = _time(rng)
    recurrence, phrase = _recurrence(rng, int(time[:2]))
    _, instruction, wishes, _ = rng.choice(SCHEDULED_KINDS)
    wish = rng.choice(wishes)
    user = rng.choice([f"Зроби так, щоб {phrase} {at} {wish}.", f"Налаштуй, будь ласка: {phrase} {at} — щоб {wish}.", f"Хочу, щоб {phrase} {at} {wish}.",
                       f"Постав регулярне нагадування {phrase} {at}, щоб {wish}.", f"Можеш зробити, щоб {phrase} {at} {wish}?"])
    return Episode("schedule_digest", seed, today, user, [_call("create_scheduled_task", time=time, **recurrence, instruction=instruction)], _expect(), "")


REMIND_ABOUT = [
    "взяти спортивну форму", "здати гроші на екскурсію", "принести контурні карти", "взяти змінне взуття", "віднести книжки в бібліотеку",
    "зарядити планшет", "подзвонити бабусі", "взяти атлас і циркуль", "повторити вірш напам'ять", "покласти в рюкзак лабораторний зошит",
    "підписати щоденник у батьків", "записатися на гурток з робототехніки", "здати реферат", "принести довідку від лікаря", "взяти гроші на обід",
    "підготувати презентацію", "повторити неправильні дієслова", "зібрати речі на тренування", "вийти з дому раніше, бо перший урок — контрольна",
    "привітати Софію з днем народження", "взяти фарби й пензлі", "віддати Максимові зошит", "доробити проєкт з інформатики", "випити ліки",
]


def schedule_once(seed, today, rng):
    day, at_day, on_day = _day(rng, today, 1, 7)  # not today: the clock in the system prompt varies
    time, at = _time(rng)
    what = rng.choice(REMIND_ABOUT)
    user = rng.choice([f"Нагадай мені {at_day} {at} {what}.", f"Постав нагадування {on_day}, {at}: {what}.", f"{at_day.capitalize()} {at} нагадай, будь ласка, {what}.",
                       f"Мені треба {at_day} {what}. Нагадай {at}.", f"Не дай мені забути {what} — нагадай {at_day} {at}.",
                       f"Кинь мені нагадування {at_day} {at}: {what}."])
    return Episode("schedule_once", seed, today, user,
                   [_call("create_scheduled_task", time=time, recurrence="once", date=day.isoformat(), instruction=f"Нагадати: {what}.")], _expect(), "")


def _existing_tasks(rng, count: int) -> list[tuple[dict, str, str]]:
    """count scheduled tasks with distinct times and kinds: (setup call, time, how a pupil names it)."""
    tasks, times = [], set()
    for _, instruction, _, label in rng.sample(SCHEDULED_KINDS, k=count):
        time, _ = _time(rng)
        while time in times:
            time, _ = _time(rng)
        times.add(time)
        recurrence, _ = _recurrence(rng, int(time[:2]))
        tasks.append((_call("create_scheduled_task", time=time, **recurrence, instruction=instruction), time, label))
    return tasks


def cancel_task(seed, today, rng):
    tasks = _existing_tasks(rng, rng.randint(1, 4))
    target = rng.randrange(len(tasks))
    _, time, label = tasks[target]
    hour, minute = int(time[:2]), time[3:]
    at = f"{'об' if hour == 11 else 'о'} {rng.choice([time, f'{hour}:{minute}'])}"
    if rng.random() < 0.5:
        user = rng.choice([f"Видали нагадування, яке спрацьовує {at}.", f"Нагадування {at} мені вже не потрібне, скасуй його.", f"Прибери те, що приходить {at}.",
                           f"Вимкни нагадування {at}.", f"Не надсилай мені більше нічого {at}."])
    else:
        user = rng.choice([f"Скасуй {label}.", f"Мені більше не потрібне {label}, прибери його.", f"Вимкни, будь ласка, {label}.", f"Видали {label}."])
    return Episode("cancel_task", seed, today, user, [_call("list_scheduled_tasks"), _call("cancel_scheduled_task", task_id=f"task{target + 1}")],
                   _expect(), "", setup=[t[0] for t in tasks])


def list_tasks(seed, today, rng):
    tasks = _existing_tasks(rng, rng.randint(0, 3))
    user = rng.choice(["Які в мене зараз стоять нагадування?", "Покажи всі мої нагадування.", "Що в мене заплановано?", "Нагадування якісь увімкнені?",
                       "Які регулярні сповіщення ти мені надсилаєш?", "Перелічи заплановані завдання."])
    return Episode("list_tasks", seed, today, user, [_call("list_scheduled_tasks")], _expect(), "", setup=[t[0] for t in tasks])


def mark_done(seed, today, rng):
    world = SchoolWorld(seed, today)
    undo = rng.random() < 0.2
    by_subject: dict[str, list] = {}
    for h in world.call("list_homework", {"status": "done" if undo else "todo"})["homework"]:
        by_subject.setdefault(h["subject"], []).append(h)
    unique = [items[0] for items in by_subject.values() if len(items) == 1]
    if not unique:
        return None
    hw = rng.choice(unique)
    subject, s = hw["subject"], _subject(rng, hw["subject"])
    if undo:
        user = rng.choice([f"Я помилився, домашка {s} ще не зроблена. Зніми позначку.", f"Познач завдання {s} як невиконане, я його ще доробляю.",
                           f"Ой, {s} я ще не доробив, поверни завдання в невиконані."])
        gold = [_call("list_homework", subject=subject, status="done"), _call("mark_homework_done", homework_id=hw["id"], done=False)]
    else:
        user = rng.choice([f"{subject} — готово, відміть як зроблене.", f"Домашнє {s} я вже виконав, зазнач це.", f"Зробив домашку {s}, постав галочку.",
                           f"Я закінчив завдання {s}.", f"Відміть, що {s} я все зробив.", f"Можеш позначити домашку {s} виконаною?",
                           f"Усе, {s} зроблено. Познач, будь ласка."])
        gold = [_call("list_homework", subject=subject, status="todo"), _call("mark_homework_done", homework_id=hw["id"])]
    return Episode("mark_done", seed, today, user, gold, _expect(), "")


TEACHER_REASONS = [
    "завтра не прийду на урок, бо йду до лікаря", "не встигаю здати проєкт до п'ятниці й прошу ще два дні", "захворів і мене не буде до кінця тижня",
    "запізнюся на перший урок через транспорт", "хочу перескласти останню контрольну", "не зрозумів останню тему й прошу про консультацію",
    "здам домашнє завдання на день пізніше", "буду відсутній у четвер через олімпіаду", "загубив роздруківку й прошу надіслати її ще раз",
    "хочу взяти додаткове завдання, щоб виправити оцінку", "не зміг відкрити файл із завданням", "прошу перевірити мою роботу ще раз",
]
PARENT_REASONS = [
    "затримаюся після уроків на годину", "сьогодні маю сім уроків і буду вдома пізніше", "забув удома ключі", "отримав дванадцять за контрольну",
    "піду після школи до бібліотеки", "прошу забрати мене о 15:00", "нагадую про батьківські збори в п'ятницю о 18:00",
    "маю принести гроші на екскурсію до середи",
]
CLASSMATE_REASONS = [
    "не прийду завтра на тренування", "чекатиму біля входу о 8:15", "принесу зошит завтра", "захворів і прошу скинути конспекти",
    "зробив свою частину проєкту", "займу місце в їдальні", "не встигаю на першу репетицію", "знайшов загублений пенал",
]


def message_someone(seed, today, rng):
    world = SchoolWorld(seed, today)
    role = rng.choice(["teacher", "teacher", "parent", "classmate"])
    if role == "teacher":
        person = rng.choice([p for p in world.recipients if p["role"] == "teacher"])
        reason = rng.choice(TEACHER_REASONS)
        s = person["subject"]
        user = rng.choice([f"Передай учителю, який веде предмет «{s}», що я {reason}.", f"Надішли повідомлення вчителю ({s}): я {reason}.",
                           f"Напиши вчителю {SUBJECT_GEN[s]}, що я {reason}.", f"Повідом учителя {SUBJECT_GEN[s]}: я {reason}.",
                           f"Треба написати вчителю {SUBJECT_GEN[s]}. Скажи, що я {reason}."])
        lookup, text = _call("list_recipients", role="teacher", query=s), f"Добрий день! Повідомляю, що я {reason}."
    elif role == "parent":
        person = rng.choice([p for p in world.recipients if p["role"] == "parent"])
        reason = rng.choice(PARENT_REASONS)
        whom = "мамі" if person["name"].startswith("Мама") else "татові"
        user = rng.choice([f"Напиши {whom}, що я {reason}.", f"Скинь {whom} повідомлення: я {reason}.", f"Передай {whom}, що я {reason}.",
                           f"Повідом {whom}, будь ласка, що я {reason}."])
        lookup, text = _call("list_recipients", role="parent"), f"Я {reason}."
    else:
        person = rng.choice([p for p in world.recipients if p["role"] == "classmate"])
        reason = rng.choice(CLASSMATE_REASONS)
        whom = CLASSMATE_DATIVE[person["name"]]
        user = rng.choice([f"Напиши {whom}, що я {reason}.", f"Передай {whom}: я {reason}.", f"Скинь повідомлення {whom}, що я {reason}."])
        lookup, text = _call("list_recipients", role="classmate", query=person["name"].split()[0]), f"Привіт! Я {reason}."
    return Episode("message_teacher", seed, today, user, [lookup, _call("send_message", recipient_id=person["id"], text=text)], _expect(), "")


# ---------- no tool call ----------

SMALL_TALK = [
    ("Привіт!", "Привіт! Чим допомогти?"),
    ("Що ти вмієш?", "Можу пояснити тему чи задачу, показати домашні завдання, оцінки й розклад, написати вчителю або поставити нагадування."),
    ("Хто ти?", "Я Дикань, твій шкільний асистент. Допомагаю з навчанням і шкільними справами."),
    ("Дякую, ти дуже допоміг!", "Радий допомогти! Звертайся, якщо буде ще щось потрібно."),
    ("Дякую", "Будь ласка!"),
    ("Добре, зрозумів.", "Чудово. Якщо щось іще знадобиться — пиши."),
    ("Ти працюєш без інтернету?", "Так, я працюю просто на твоєму телефоні й можу відповідати без інтернету."),
    ("Ти можеш помилятися?", "Так, можу. Важливі речі — дати, формули, правила — варто перевіряти за підручником або спитати вчителя."),
    ("Добраніч!", "Добраніч! Гарно відпочити."),
    ("Мені нудно.", "Буває. Хочеш, запропоную цікаву задачу, розкажу щось незвичне з науки чи допоможу розібратися з домашкою?"),
    ("Я втомився від уроків.", "Розумію. Зроби коротку перерву: попий води, пройдися кілька хвилин. Потім можемо розбити решту завдань на маленькі кроки."),
    ("Як краще готуватися до контрольної?", "Почни заздалегідь: переглянь конспект, розв'яжи кілька типових задач без підказок, випиши формули чи правила на один аркуш і добре виспися напередодні."),
    ("Як мені все встигати?", "Запиши всі справи, вибери три найважливіші на день і починай із найскладнішої. Працюй відрізками по 25 хвилин із короткими перервами."),
    ("Якою мовою ти розмовляєш?", "Найкраще — українською. Можу також відповідати англійською, якщо потрібно."),
    ("Ок", "Добре. Якщо буде потрібна допомога — я поруч."),
    ("Ти справжня людина?", "Ні, я програма — шкільний асистент Дикань. Але намагаюся допомагати так, як допоміг би уважний репетитор."),
]
MISSING_DETAIL = [
    ("Напиши, що мене завтра не буде.", "Кому написати — класному керівнику, комусь із учителів чи батькам?"),
    ("Нагадай мені про репетитора.", "Коли нагадати — якого дня і о котрій годині?"),
    ("Познач домашку як зроблену.", "Яке саме завдання позначити — з якого предмета?"),
    ("Постав нагадування на завтра.", "О котрій годині й про що саме нагадати?"),
    ("Нагадай о 7:30.", "Про що нагадати і якого дня — завтра чи щодня?"),
    ("Надішли повідомлення вчителю.", "Якому вчителю — з якого предмета — і що саме написати?"),
    ("Напиши мамі.", "Що саме написати мамі?"),
    ("Покажи статистику.", "За який період — тиждень, місяць чи семестр?"),
    ("Скасуй нагадування.", "Яке саме нагадування скасувати? Можу показати список усіх, якщо треба."),
    ("Передай, що я запізнюся.", "Кому передати — вчителю (якого предмета?), батькам чи комусь з однокласників?"),
    ("Зроби мені щоденне нагадування.", "О котрій годині воно має приходити і про що нагадувати?"),
    ("Відміть, що я зробив.", "Що саме ти зробив — домашнє завдання з якого предмета?"),
    ("Постав будильник про контрольну.", "На який день і час поставити нагадування про контрольну?"),
    ("Попроси вчителя перенести здачу.", "Якого вчителя попросити — з якого предмета — і здачу якої роботи перенести?"),
    ("Надсилай мені розклад.", "Коли надсилати розклад — щодня чи лише в будні, і о котрій годині?"),
    ("Напиши однокласнику про проєкт.", "Кому саме з однокласників написати і що саме сказати про проєкт?"),
]
UNSUPPORTED = [
    ("Зміни мені оцінку з алгебри на 12.", "Я не можу змінювати оцінки — їх виставляє лише вчитель. Можу показати твої останні оцінки або написати вчителю."),
    ("Видали домашку з фізики, я не хочу її робити.", "Видаляти завдання я не можу. Можу показати, що саме задано, і допомогти розібратися."),
    ("Замов мені піцу.", "Замовляти їжу я не вмію. Можу допомогти з навчанням, розкладом, домашніми завданнями й нагадуваннями."),
    ("Подзвони мамі.", "Дзвонити я не можу. Можу надіслати мамі повідомлення — що написати?"),
    ("Який сьогодні курс долара?", "У мене немає доступу до інтернету, тож актуального курсу я не знаю."),
    ("Яка завтра погода?", "Прогнозу погоди в мене немає — я працюю без інтернету. Можу показати розклад чи домашні завдання на завтра."),
    ("Додай мені новий урок у розклад.", "Змінювати розклад я не можу — його складає школа. Можу показати розклад на будь-який день."),
    ("Постав мені 11 за твір.", "Оцінки ставить учитель, я цього зробити не можу. Якщо хочеш, допоможу покращити твір."),
    ("Увімкни музику.", "Керувати музикою я не можу. Можу допомогти з навчанням або поставити нагадування."),
    ("Відкрий мені відповіді на завтрашню контрольну.", "У мене немає відповідей на контрольні, і це було б нечесно. Зате можу допомогти підготуватися: поясню тему й дам задачі для тренування."),
    ("Перенеси контрольну на наступний тиждень.", "Переносити контрольні може лише вчитель. Можу надіслати вчителю повідомлення з проханням — написати?"),
    ("Зареєструй мене на НМТ.", "Зареєструвати на НМТ я не можу — це робиться в кабінеті вступника. Можу допомогти з підготовкою до тестів."),
]


def _scripted(family: str, pairs: list):
    def make(seed, today, rng):
        user, final = rng.choice(pairs)
        return Episode(family, seed, today, user, [], _expect(), final)
    return make


FAMILIES = [
    homework_for_day, homework_details, day_overview, schedule_for_day, last_grades, analytics, unknown_subject,
    digest_now, schedule_recurring, schedule_once, cancel_task, list_tasks, mark_done, message_someone,
    # Chained and scheduling tasks are the hard ones, so they appear twice per cycle.
    digest_now, schedule_recurring, schedule_once, cancel_task, mark_done, message_someone, homework_details,
    _scripted("missing_detail", MISSING_DETAIL), _scripted("unsupported", UNSUPPORTED), _scripted("no_tool_needed", SMALL_TALK),
]


def generate_train(n: int, seed: int = 0, start: date = date(2026, 9, 1)) -> list[Episode]:
    """n training episodes, cycling through the families, each in its own school world, day and tool list."""
    rng = random.Random(seed)
    episodes: list[Episode] = []
    world_seed = seed * 1_000_003
    while len(episodes) < n:
        family = FAMILIES[len(episodes) % len(FAMILIES)]
        world_seed += 1
        today = start + timedelta(days=rng.randint(0, 250))
        if today.weekday() >= 5:
            today += timedelta(days=7 - today.weekday())
        episode = family(world_seed, today, rng)
        if episode is not None:
            episode.tools = _tool_subset(episode, random.Random(world_seed))
            episodes.append(episode)
    return episodes
