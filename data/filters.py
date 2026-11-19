"""Degeneration detectors for Ukrainian model output.

Two failure modes, both seen in Gemma-4-E2B-it on-device:
  * loops: one word or phrase repeated until the token limit ("який який який ...")
  * script salad: bursts of unrelated scripts (Thai, Devanagari, CJK, Arabic ...)

Used both as an eval metric (eval/degeneration.py) and as a training-data filter,
so it has no dependencies beyond the standard library.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass

# Scripts a Ukrainian school answer may legitimately contain: Cyrillic text,
# Latin (English terms, formulas, units) and Greek (maths and physics symbols).
EXPECTED_SCRIPTS = frozenset({"CYRILLIC", "LATIN", "GREEK"})

_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class Degeneration:
    max_word_run: int  # longest run of one word repeated back to back
    repeated_4gram_frac: float  # share of word 4-grams that are repeats
    foreign_script_frac: float  # share of letters outside EXPECTED_SCRIPTS
    foreign_scripts: int  # number of distinct unexpected scripts
    looping: bool
    script_salad: bool

    @property
    def degenerate(self) -> bool:
        return self.looping or self.script_salad

    def to_dict(self) -> dict:
        return {**asdict(self), "degenerate": self.degenerate}


def _script(ch: str) -> str | None:
    """Script of a letter, taken from the first word of its Unicode name."""
    if not ch.isalpha():
        return None
    try:
        return unicodedata.name(ch).split(" ", 1)[0]
    except ValueError:
        return "UNKNOWN"


def _max_word_run(words: list[str]) -> int:
    best = run = 0
    prev = None
    for w in words:
        run = run + 1 if w == prev else 1
        prev = w
        best = max(best, run)
    return best


def _max_phrase_run(words: list[str], max_period: int = 12) -> int:
    """Longest back-to-back repetition count of any phrase up to max_period words."""
    best = 1
    n = len(words)
    for period in range(2, max_period + 1):
        run = 1
        i = period
        while i + period <= n:
            if words[i : i + period] == words[i - period : i]:
                run += 1
                best = max(best, run)
                i += period
            else:
                run = 1
                i += 1
    return best


def _repeated_ngram_frac(words: list[str], n: int = 4) -> float:
    if len(words) < n * 2:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeats = sum(c - 1 for c in counts.values())
    return repeats / len(grams)


def analyze(
    text: str,
    *,
    word_run_limit: int = 6,
    phrase_run_limit: int = 4,
    ngram_frac_limit: float = 0.35,
    foreign_frac_limit: float = 0.02,
    foreign_scripts_limit: int = 3,
) -> Degeneration:
    """Score one model output. Thresholds are deliberately loose: a normal
    explanation scores far below them, and the on-device failures far above."""
    words = [w.lower() for w in _WORD_RE.findall(text)]
    word_run = _max_word_run(words)
    phrase_run = _max_phrase_run(words)
    ngram_frac = _repeated_ngram_frac(words)

    scripts = Counter(s for s in map(_script, text) if s is not None)
    letters = sum(scripts.values())
    foreign = {s: c for s, c in scripts.items() if s not in EXPECTED_SCRIPTS}
    foreign_frac = sum(foreign.values()) / letters if letters else 0.0

    looping = (
        word_run >= word_run_limit
        or phrase_run >= phrase_run_limit
        or ngram_frac >= ngram_frac_limit
    )
    # One foreign word (a Japanese term in a history answer) is fine; a burst of
    # several unrelated scripts is not.
    script_salad = (
        foreign_frac >= foreign_frac_limit and len(foreign) >= foreign_scripts_limit
    ) or foreign_frac >= 0.15

    return Degeneration(
        max_word_run=word_run,
        repeated_4gram_frac=round(ngram_frac, 4),
        foreign_script_frac=round(foreign_frac, 4),
        foreign_scripts=len(foreign),
        looping=looping,
        script_salad=script_salad,
    )


# Letters used in Russian but not in Ukrainian. One of them in a Ukrainian answer
# means Russian text or a Russian-style spelling slipped in.
_RUSSIAN_ONLY = frozenset("ыэъёЫЭЪЁ")


def russian_letter_share(text: str) -> float:
    """Share of Cyrillic letters that exist only in Russian."""
    cyrillic = [ch for ch in text if "Ѐ" <= ch <= "ӿ"]
    return sum(ch in _RUSSIAN_ONLY for ch in cyrillic) / len(cyrillic) if cyrillic else 0.0


def clean_ukrainian(text: str, max_russian_share: float = 0.0) -> bool:
    """True if the text is usable as Ukrainian training data: not degenerate and
    free of Russian-only letters."""
    return not analyze(text).degenerate and russian_letter_share(text) <= max_russian_share
