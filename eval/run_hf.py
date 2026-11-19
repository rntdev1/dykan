"""Run the eval suite on a Hugging Face Gemma 4 checkpoint (full precision, GPU).

    uv run --extra gpu python -m eval.run_hf --model google/gemma-4-E2B-it --zno all.jsonl.gz

Two evals, both greedy and with thinking off:
  zno    held-out ZNO/NMT tasks (2020 and later), answer-only protocol
  tools  multi-step assistant episodes executed on the simulated school backend

The same script scores stock and fine-tuned checkpoints, so numbers are comparable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from data.filters import analyze
from data.sources.zno_eval import load
from data.split import carve_dev, split_exam_tasks
from data.tools.episodes import generate, system_prompt
from data.tools.rename import HELDOUT, Renaming
from eval import toolcall, zno
from eval.gemma_calls import parse_tool_calls, strip_markup

MAX_TOOL_STEPS = 8


class Runner:
    def __init__(self, model_id: str, revision: str | None, adapter: Path | None = None):
        import torch
        import transformers
        from transformers import AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        # Gemma 4 loads as a multimodal LM; Gemma 3 models (MamayLM) as image-text-to-text.
        self.model = None
        for loader in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM"):
            try:
                self.model = getattr(transformers, loader).from_pretrained(model_id, revision=revision, dtype=torch.bfloat16, device_map="auto").eval()
                break
            except (ValueError, AttributeError, KeyError):
                continue
        if self.model is None:
            raise RuntimeError(f"no loader worked for {model_id}")
        if adapter is not None:  # a LoRA adapter from train/sft.py, folded into the base weights
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter)).merge_and_unload().eval()
        tok = self.processor.tokenizer
        # Some repos (MamayLM) ship the chat template on the tokenizer only, or not at all.
        if not getattr(self.processor, "chat_template", None):
            self.processor.chat_template = getattr(tok, "chat_template", None)
        self.gemma3_fallback = not self.processor.chat_template
        eos = self.model.generation_config.eos_token_id
        stop = set(eos if isinstance(eos, list) else [eos])
        # The model ends a tool call by predicting the start of the tool response.
        for token in ("<turn|>", "<|tool_response>", "<end_of_turn>"):
            token_id = tok.convert_tokens_to_ids(token)
            if token_id is not None and token_id != tok.unk_token_id:
                stop.add(token_id)
        self.stop_ids = sorted(i for i in stop if i is not None)

    def generate(self, messages: list[dict], tools: list[dict] | None, max_new_tokens: int) -> str:
        if self.gemma3_fallback:
            # Plain Gemma 3 turn format, text only, for repos without any template.
            assert tools is None, "the Gemma 3 fallback has no tool format"
            text = "<bos>" + "".join(
                f"<start_of_turn>{'model' if m['role'] == 'assistant' else 'user'}\n{m['content'].strip()}<end_of_turn>\n" for m in messages
            ) + "<start_of_turn>model\n"
            inputs = self.processor.tokenizer(text, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.processor.apply_chat_template(
                messages, tools=tools, tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=True, enable_thinking=False,
            ).to(self.model.device)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=self.stop_ids)
        return self.processor.decode(ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)


def run_zno(runner: Runner, path: Path, out: Path, limit: int | None, split: str = "test") -> dict:
    """split: "test" is the held-out years (the reported number); "dev" is the slice of
    earlier years kept out of training for choosing hyperparameters."""
    train, test, _ = split_exam_tasks(list(load(path)))
    tasks = [t for t in (test if split == "test" else carve_dev(train)[1]) if zno.eligible(t)]
    tasks = tasks[:limit] if limit else tasks
    rows = []
    with out.open("w", encoding="utf-8") as f:
        for n, task in enumerate(tasks, 1):
            prompt = zno.build_prompt(task) + " Пояснення не потрібні."
            reply = strip_markup(runner.generate([{"role": "user", "content": prompt}], None, 64))
            parsed = zno.parse_answer(task, reply)
            row = {"uid": task.uid, "subject": task.subject, "question_type": task.question_type,
                   "correct": zno.is_correct(task, reply), "parsed": parsed is not None, "gold": list(task.correct), "reply": reply}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if n % 50 == 0:
                print(f"zno {n}/{len(tasks)} acc={sum(r['correct'] for r in rows) / n:.3f}", flush=True)
    return zno.summarize(rows)


def run_tools(runner: Runner, n: int, seed: int, out: Path, calendar: bool = False, skin: Renaming | None = None) -> dict:
    """skin: declare the tools under other names (see data/tools/rename.py); calls are
    translated back for the simulator, so the scoring is the same."""
    skin = skin or Renaming()
    results = []
    with out.open("w", encoding="utf-8") as f:
        for i, episode in enumerate(generate(n, seed=seed), 1):
            world = episode.make_world()
            system = system_prompt(episode.today, calendar=True) if calendar else episode.system
            messages = [{"role": "system", "content": system}, {"role": "user", "content": episode.user}]
            final, raw = "", []
            for _ in range(MAX_TOOL_STEPS):
                text = runner.generate(messages, skin.declarations(episode.tools), 400)
                raw.append(text)
                calls = parse_tool_calls(text)
                if not calls:
                    final = strip_markup(text)
                    break
                for call in calls:
                    if call["arguments"] is None:
                        result = {"error": "Не вдалося розібрати аргументи виклику"}
                        world.calls.append({"name": call["name"], "arguments": None, "result": result})
                    else:
                        result = skin.response(world.call(*skin.call_back(call["name"], call["arguments"])))
                    messages.append({
                        "role": "assistant",
                        "tool_calls": [{"type": "function", "function": {"name": call["name"], "arguments": call["arguments"] or {}}}],
                        "tool_responses": [{"name": call["name"], "response": result}],
                    })
            score = toolcall.score_world(episode, world)
            score["answered"] = bool(final)
            score["degenerate"] = analyze(final).degenerate
            results.append(score)
            f.write(json.dumps({**score, "seed": episode.seed, "user": episode.user, "final": final, "raw": raw,
                                "calls": [{"name": c["name"], "arguments": c["arguments"]} for c in world.calls]}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 13 == 0:
                print(f"tools {i}/{n} completion={sum(r['success'] for r in results) / i:.3f}", flush=True)
    summary = toolcall.summarize(results)
    summary["answered_rate"] = round(sum(r["answered"] for r in results) / len(results), 4)
    summary["degenerate_answers"] = sum(r["degenerate"] for r in results)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA adapter directory to apply on top of --model")
    ap.add_argument("--zno", type=Path, default=None, help="path to ZNO-Eval all.jsonl.gz; omit to skip the exam eval")
    ap.add_argument("--zno-limit", type=int, default=None)
    ap.add_argument("--zno-split", choices=["test", "dev"], default="test", help="dev: pre-2020 tasks held out of training, for tuning")
    ap.add_argument("--episodes", type=int, default=130, help="0 skips the tool eval")
    ap.add_argument("--seed", type=int, default=20261114, help="episode seed; keep fixed across checkpoints")
    ap.add_argument("--calendar", action="store_true", help="tool eval with the short calendar in the system prompt")
    ap.add_argument("--skin", choices=["original", "heldout"], default="original",
                    help="heldout: same episodes, tools declared under names never used in training")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or Path("outputs/eval") / args.model.rstrip("/").split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.model, args.revision, args.adapter)
    summary = {"model": args.model, "revision": args.revision, "adapter": str(args.adapter) if args.adapter else None, "stop_ids": runner.stop_ids}
    started = time.time()
    if args.episodes:
        summary["tools"] = run_tools(runner, args.episodes, args.seed, out / "tools.jsonl", calendar=args.calendar,
                                     skin=HELDOUT if args.skin == "heldout" else None)
        summary["tools"]["calendar"], summary["tools"]["skin"] = args.calendar, args.skin
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.zno:
        summary["zno"] = run_zno(runner, args.zno, out / "zno.jsonl", args.zno_limit, args.zno_split)
        summary["zno"]["split"] = args.zno_split
    summary["minutes"] = round((time.time() - started) / 60, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
