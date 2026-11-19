"""Score how "school-like" a text is, to pick curriculum-relevant documents out of a large corpus.

A vocabulary is learned by contrast: words that are much more frequent in school
material (exam tasks, explanations, tutoring dialogues) than in the corpus at
large get a positive weight. A document's score is the average weight of the
words in its opening part. No model and no GPU are needed, so it runs over
millions of documents on a CPU.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

_WORD_RE = re.compile(r"[а-щьюяґєії']{3,}", re.IGNORECASE)


def words(text: str, limit: int | None = None) -> list[str]:
    found = [w.lower() for w in _WORD_RE.findall(text.replace("’", "'").replace("ʼ", "'"))]
    return found[:limit] if limit else found


def document_frequency(texts: Iterable[str], limit: int | None = None) -> tuple[Counter, int]:
    df: Counter[str] = Counter()
    n = 0
    for text in texts:
        df.update(set(words(text, limit)))
        n += 1
    return df, n


def school_vocabulary(school: Iterable[str], background: Iterable[str], min_school_docs: int = 8, min_ratio: float = 3.0,
                      min_background_docs: int = 3) -> dict[str, float]:
    """word -> weight (log of how many times likelier it is in school texts than in the background).

    A word must also occur in the background a few times. Words that never do
    ("розв'яжи", "запам'ятай", the assistant's name) mark the style of a task or
    a chat, not a subject, and cannot help to find documents."""
    s_df, s_n = document_frequency(school)
    b_df, b_n = document_frequency(background, limit=500)
    vocab = {}
    for word, count in s_df.items():
        if count < min_school_docs or b_df.get(word, 0) < min_background_docs:
            continue
        ratio = (count / s_n) / ((b_df.get(word, 0) + 1) / (b_n + 1))
        if ratio >= min_ratio:
            vocab[word] = min(math.log(ratio), 6.0)
    return vocab


def score(text: str, vocab: dict[str, float], head_words: int = 500) -> float:
    """Average vocabulary weight over the first `head_words` words; 0 for an empty text."""
    head = words(text, head_words)
    return sum(vocab.get(w, 0.0) for w in head) / len(head) if head else 0.0
