"""Terminal chat with a Dykan model on a Mac, with the school tools simulated.

    uv run --extra mac python -m demo.chat --model ~/models/dykan-e4b-mlx-4bit

Runs fully offline through MLX. The model gets the Dykan system prompt (with the
date, time and a short calendar) and the example school tools; when it calls a
tool, the call is executed on the simulated school backend (data/tools/sim.py)
and shown the way a coding assistant shows its tool calls. Nothing leaves the Mac.

Pictures: with a model converted by mlx-vlm (it keeps the vision encoder), put the
path of an image file in the message, or drag the file into the terminal:

    › що тут написано? ~/Desktop/zadacha.jpg

Commands: /help  /think  /tools  /state  /system  /calendar  /new  /quit
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import sys
from datetime import date, datetime
from pathlib import Path

from data.tools.episodes import system_prompt
from data.tools.schemas import TOOLS
from data.tools.sim import SchoolWorld
from eval.gemma_calls import parse_tool_calls, strip_markup

MAX_TOOL_STEPS = 8
STOP_MARKERS = ("<turn|>", "<|tool_response>", "<end_of_turn>", "<eos>")
DIM, BOLD, CYAN, GREEN, YELLOW, RED, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".gif", ".bmp"}

HELP = """a path to an image file inside a message attaches that picture (vision models only)
/tools     the tools the model can call
/state     what the model has changed: homework marked done, messages, notifications, reminders
/system    the system prompt the model sees
/calendar  switch the short calendar in the system prompt on or off
/think     switch thinking mode on or off (the model reasons first, shown dimmed, then answers)
/raw       switch showing the model's raw output (special tokens included) on or off
/new       start a new conversation in a fresh school world
/quit      leave"""


def short(value, limit: int = 110) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def split_images(line: str) -> tuple[str, list[str]]:
    """(the message without image paths, the image files it named). Handles quoted and
    backslash-escaped paths, which is what dragging a file into a terminal produces."""
    try:
        parts = shlex.split(line)
    except ValueError:
        return line, []
    images = [str(Path(p).expanduser()) for p in parts if Path(p).suffix.lower() in IMAGE_TYPES and Path(p).expanduser().is_file()]
    if not images:
        return line, []
    text = " ".join(p for p in parts if str(Path(p).expanduser()) not in images)
    return text, images


def split_thought(text: str) -> tuple[str, str | None]:
    """(the thinking so far, the text after it; None while the model is still thinking)."""
    start = text.find("<|channel>")
    if start < 0:
        return "", text
    body = text[start + len("<|channel>"):]
    body = body[len("thought"):].lstrip("\n") if body.startswith("thought") else body
    end = body.find("<channel|>")
    if end < 0:
        return body.rstrip(), None
    return body[:end].strip(), text[:start] + body[end + len("<channel|>"):]


def describe_result(result: dict) -> str:
    """One line about a tool result, the way a person would glance at it."""
    if "error" in result:
        return f"{RED}помилка: {result['error']}{RESET}"
    for key, value in result.items():
        if isinstance(value, list):
            return f"{key}: {len(value)}" + (f" — {short(value[0], 80)}" if value else "")
    return short(result)


class Session:
    def __init__(self, model_path: str, today: date, clock: str, calendar: bool, seed: int, max_tokens: int, temperature: float):
        # A folder converted by mlx-vlm carries the image processor; one converted by mlx-lm is text only.
        self.vision = (Path(model_path) / "processor_config.json").exists()
        print(f"{DIM}loading {model_path} ({'text and images' if self.vision else 'text only'}) …{RESET}", flush=True)
        if self.vision:
            import warnings

            warnings.filterwarnings("ignore", message="At least one mel filter")  # noise from the unused audio front end
            from mlx_vlm import load

            self.model, self.processor = load(model_path)
            self.tokenizer = self.processor.tokenizer
        else:
            from mlx_lm import load
            from mlx_lm.sample_utils import make_sampler

            self.model, self.tokenizer = load(model_path)
            self.sampler = make_sampler(temp=temperature)
        self.temperature = temperature
        self.today, self.clock, self.calendar, self.seed = today, clock, calendar, seed
        self.max_tokens, self.raw, self.thinking = max_tokens, False, False
        self.new()

    def new(self) -> None:
        self.world = SchoolWorld(self.seed, self.today)
        self.messages: list[dict] = []
        self.images: list[str] = []  # every picture of the conversation, in the order the template meets them

    @property
    def system(self) -> str:
        return system_prompt(self.today, self.clock, calendar=self.calendar)

    def generate(self) -> tuple[str, float]:
        """Streams one model turn to the terminal; returns (raw text, tokens per second)."""
        prompt = self.tokenizer.apply_chat_template([{"role": "system", "content": self.system}] + self.messages, tools=TOOLS,
                                                    add_generation_prompt=True, tokenize=False, enable_thinking=self.thinking)
        if self.vision:
            from mlx_vlm import stream_generate

            stream = stream_generate(self.model, self.processor, prompt, image=self.images or None, max_tokens=self.max_tokens, temperature=self.temperature)
        else:
            from mlx_lm import stream_generate

            stream = stream_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=self.max_tokens, sampler=self.sampler)
        text, shown_thought, shown, speed = "", 0, 0, 0.0
        for chunk in stream:
            text += chunk.text
            speed = getattr(chunk, "generation_tps", 0.0) or speed
            cut = min((text.find(m) for m in STOP_MARKERS if m in text), default=-1)
            if cut >= 0 and "<|tool_call>" not in text[:cut]:
                text = text[:cut]
            thought, answer = split_thought(text)
            # Thinking is printed dimmed as it arrives; then the answer. A tool call is shown once it is complete.
            if len(thought) > shown_thought:
                if shown_thought == 0:
                    sys.stdout.write(f"{DIM}✻ ")
                sys.stdout.write(f"{DIM}{thought[shown_thought:]}{RESET}")
                shown_thought = len(thought)
            if answer is not None and "<|tool_call>" not in answer and not answer.lstrip().startswith("<"):
                visible = answer if self.raw else strip_markup(answer)
                if visible and shown == 0 and shown_thought:
                    sys.stdout.write("\n\n")
                sys.stdout.write(visible[shown:])
                shown = len(visible)
            sys.stdout.flush()
            if cut >= 0 or "<tool_call|>" in text:
                break
        return text, speed

    def turn(self, user: str) -> None:
        user, images = split_images(user)
        if images and not self.vision:
            print(f"{YELLOW}this model folder is text only; convert with mlx-vlm to send pictures{RESET}")
            return
        if images:
            self.images += images
            print(f"{DIM}  📎 {', '.join(Path(i).name for i in images)}{RESET}")
            self.messages.append({"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": user or "Що на зображенні?"}]})
        else:
            self.messages.append({"role": "user", "content": user})
        for _ in range(MAX_TOOL_STEPS):
            sys.stdout.write(f"\n{GREEN}⏺{RESET} ")
            text, speed = self.generate()
            calls = parse_tool_calls(text)
            if not calls:
                self.messages.append({"role": "assistant", "content": strip_markup(text)})
                print(f"\n{DIM}  {speed:.0f} tok/s{RESET}")
                return
            if self.raw:
                print(f"{DIM}{text}{RESET}")
            for call in calls:
                arguments = call["arguments"]
                shown = ", ".join(f"{k}={short(v, 60)!r}" if isinstance(v, str) else f"{k}={v}" for k, v in (arguments or {}).items())
                sys.stdout.write(f"\r{CYAN}⏺ {BOLD}{call['name']}{RESET}{CYAN}({shown}){RESET}\n")
                result = self.world.call(call["name"], arguments) if arguments is not None else {"error": "Не вдалося розібрати аргументи виклику"}
                print(f"  {DIM}⎿ {RESET}{describe_result(result)}")
                self.messages.append({"role": "assistant",
                                      "tool_calls": [{"type": "function", "function": {"name": call["name"], "arguments": arguments or {}}}],
                                      "tool_responses": [{"name": call["name"], "response": result}]})
        print(f"{YELLOW}stopped after {MAX_TOOL_STEPS} tool calls{RESET}")

    def state(self) -> str:
        world, fresh = self.world, SchoolWorld(self.seed, self.today)
        before = {h["id"] for h in fresh.homework if h["done"]}
        lines = [f"{BOLD}Домашні завдання, позначені в цій розмові:{RESET} " + (", ".join(
            f"{h['subject']} — {h['title']}" for h in world.homework if h["done"] and h["id"] not in before) or "немає")]
        names = {p["id"]: p["name"] for p in world.recipients}
        lines.append(f"{BOLD}Надіслані повідомлення:{RESET}" + ("".join(f"\n  → {names.get(m['recipient_id'], m['recipient_id'])}: {m['text']}" for m in world.messages) or " немає"))
        lines.append(f"{BOLD}Сповіщення на телефон:{RESET}" + ("".join(f"\n  🔔 {n['title']}: {short(n['text'], 300)}" for n in world.notifications) or " немає"))
        lines.append(f"{BOLD}Заплановані завдання:{RESET}" + ("".join(
            f"\n  ⏰ {t['id']} {t['time']} {t['recurrence']}{' ' + t['date'] if t.get('date') else ''}{' ' + t['weekday'] if t.get('weekday') else ''}: {t['instruction']}"
            for t in world.scheduled) or " немає"))
        return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="folder with an MLX model (see deploy notes in the README)")
    ap.add_argument("--date", default=None, help="today's date for the school world, YYYY-MM-DD (default: the real date)")
    ap.add_argument("--seed", type=int, default=None, help="school world to simulate (default: a random one)")
    ap.add_argument("--no-calendar", action="store_true", help="leave the short calendar out of the system prompt")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    today = date.fromisoformat(args.date) if args.date else date.today()
    seed = args.seed if args.seed is not None else random.randrange(10**6)
    session = Session(str(Path(args.model).expanduser()), today, datetime.now().strftime("%H:%M"), not args.no_calendar, seed, args.max_tokens, args.temperature)
    print(f"{BOLD}Дикань{RESET} {DIM}— офлайн, {Path(args.model).name}, школа №{seed}, сьогодні {today.isoformat()}. /help — команди.{RESET}")
    print(f"{DIM}Предмети: {', '.join(session.world.subjects)}{RESET}")

    while True:
        try:
            user = input(f"\n{BOLD}›{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user in ("/quit", "/exit", "/q"):
            return
        if user == "/help":
            print(HELP)
        elif user == "/tools":
            print("\n".join(f"{CYAN}{t['function']['name']}{RESET}({', '.join(t['function']['parameters']['properties'])}) — {t['function']['description']}" for t in TOOLS))
        elif user == "/state":
            print(session.state())
        elif user == "/system":
            print(session.system)
        elif user == "/calendar":
            session.calendar = not session.calendar
            print(f"calendar in the system prompt: {'on' if session.calendar else 'off'}")
        elif user == "/think":
            session.thinking = not session.thinking
            print(f"thinking mode: {'on' if session.thinking else 'off'}")
        elif user == "/raw":
            session.raw = not session.raw
            print(f"raw output: {'on' if session.raw else 'off'}")
        elif user == "/new":
            session.new()
            print("new conversation, fresh school world")
        elif user.startswith("/"):
            print(f"unknown command {user}; /help lists them")
        else:
            try:
                session.turn(user)
            except KeyboardInterrupt:
                print(f"\n{YELLOW}interrupted{RESET}")


if __name__ == "__main__":
    main()
