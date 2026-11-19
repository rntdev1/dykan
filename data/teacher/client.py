"""Minimal concurrent client for an OpenAI-compatible server (vLLM) running the teacher.

Standard library only, so it runs on a bare pod. Results are appended to a JSONL
file as they finish and finished ids are skipped on restart, so a crashed or
interrupted generation run resumes where it stopped.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable


class Teacher:
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:8000/v1", temperature: float = 0.7, max_tokens: int = 700, extra: dict | None = None):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.temperature, self.max_tokens, self.extra = temperature, max_tokens, extra or {}

    def chat(self, messages: list[dict], retries: int = 4, temperature: float | None = None, max_tokens: int | None = None) -> tuple[str, int]:
        """Returns (text, completion_tokens)."""
        body = json.dumps({"model": self.model, "messages": messages,
                           "temperature": self.temperature if temperature is None else temperature,
                           "max_tokens": max_tokens or self.max_tokens, **self.extra}).encode()
        for attempt in range(retries):
            try:
                req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as resp:
                    data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"] or "", data.get("usage", {}).get("completion_tokens", 0)
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")


def run_jobs(jobs: Iterable[dict], work: Callable[[dict], dict], out: Path, workers: int = 32, log_every: int = 100) -> dict:
    """Run `work(job)` for every job concurrently, appending each result to `out`.

    Each job needs a unique "id". Jobs whose id is already in `out` are skipped.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open(encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    pending = [j for j in jobs if j["id"] not in done]
    lock, started = threading.Lock(), time.time()
    stats = {"skipped_existing": len(done), "finished": 0, "failed": 0, "completion_tokens": 0}

    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, job): job for job in pending}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:  # one bad job must not stop a long run
                stats["failed"] += 1
                result = {"id": futures[future]["id"], "error": repr(error)[:300]}
            with lock:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                stats["finished"] += 1
                stats["completion_tokens"] += result.get("completion_tokens", 0)
                if stats["finished"] % log_every == 0:
                    elapsed = time.time() - started
                    print(f"{stats['finished']}/{len(pending)} | failed {stats['failed']} | {stats['completion_tokens'] / elapsed:.0f} tok/s", flush=True)
    stats["seconds"] = round(time.time() - started, 1)
    stats["tok_per_s"] = round(stats["completion_tokens"] / max(stats["seconds"], 1e-9), 1)
    return stats
