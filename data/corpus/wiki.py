"""School-relevant articles from Ukrainian Wikipedia (CC BY-SA), as plain text.

    python -m data.corpus.wiki --parquet-dir wiki_uk --school outputs/train/dykan-v1.jsonl \\
        --out outputs/corpus/wiki.jsonl --target-words 60000000

Input: the parquet files of the Hugging Face dataset wikimedia/wikipedia, config 20231101.uk.
Steps: drop stubs, lists, date pages and mass-produced articles (settlements, minor
planets, sports seasons); cut reference sections; score what is left against a
school vocabulary (see relevance.py) and keep the best-scoring articles up to a word budget.
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import re
from pathlib import Path

from data.corpus.relevance import school_vocabulary, score, words
from data.filters import russian_letter_share

MIN_WORDS = 300
_TAIL_RE = re.compile(r"\n(Примітки|Посилання|Література|Джерела|Див\. також|Виноски|Бібліографія|Зовнішні посилання)\s*\n", re.IGNORECASE)
_BAD_TITLE_RE = re.compile(r"^(Список|Перелік|Хронологія|\d{1,4}( до н\. е\.)?$|\d{1,2} (січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня|листопада|грудня)$)|"
                           r"\((значення|фільм|альбом|пісня|серіал|гурт|футболіст|станція|аеропорт)\)|\b(сезон|чемпіонат|кубок|ліга|турнір)\b|\d{4}[–-]\d{2,4}", re.IGNORECASE)
# Opening phrases of mass-produced articles that carry little school knowledge.
_BOILERPLATE_RE = re.compile(
    r"— (село|селище|хутір|присілок|муніципалітет|комуна|громада|містечко|населений пункт|залізнична станція|зупинний пункт|астероїд|"
    r"вид [а-яіїє]+ (родини|роду)|рід [а-яіїє]+ родини|футболіст|футбольний клуб|хокеїст|баскетболіст|тенісист|"
    r"американський фільм|художній фільм|студійний альбом|телесеріал|комп'ютерна гра|відеогра)", re.IGNORECASE)


def clean(text: str) -> str:
    cut = _TAIL_RE.search(text)
    text = text[: cut.start()] if cut else text
    # Brackets left half-empty or empty when templates were stripped: "(; застаріле )", "()".
    text = re.sub(r"\(\s*[;,:]\s*", "(", text)
    text = re.sub(r"\s*[;,:]?\s+\)", ")", text)
    text = re.sub(r" ?\((?:[\s;,:—–-]|або|англ\.|лат\.|нім\.|фр\.|рос\.|грец\.|пол\.)*\)", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def acceptable(title: str, text: str) -> bool:
    if _BAD_TITLE_RE.search(title) or _BOILERPLATE_RE.search(text[:400]):
        return False
    return len(words(text)) >= MIN_WORDS and russian_letter_share(text[:3000]) < 0.01


def school_texts(path: Path) -> list[str]:
    """Exam tasks and the teacher's subject material from the training set, as the "school" side of the contrast."""
    keep = {"exam_key", "exam_explained", "tasks", "explain"}
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["source"] in keep:
                # Exam questions as asked; otherwise the answer side, which carries the subject matter.
                roles = ("user",) if row["source"] == "exam_key" else ("user", "assistant")
                out.append("\n".join(m["content"] for m in row["messages"] if m["role"] in roles and isinstance(m.get("content"), str)))
    return out


def articles(folder: Path):
    import pyarrow.parquet as pq

    for path in sorted(folder.glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=["title", "url", "text"]):
            for title, url, text in zip(*(batch.column(c).to_pylist() for c in ("title", "url", "text"))):
                yield title, url, text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", type=Path, required=True)
    ap.add_argument("--school", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-words", type=int, default=60_000_000)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--max-words", type=int, default=6000, help="longer articles are cut at a paragraph boundary")
    args = ap.parse_args()

    rng = random.Random(0)
    kept, background = [], []
    seen = 0
    for title, url, text in articles(args.parquet_dir):
        seen += 1
        text = clean(text)
        if acceptable(title, text):
            kept.append((title, url, text))
            if rng.random() < 0.2:
                background.append(text)
        if seen % 200_000 == 0:
            print(f"read {seen:,} | passed basic filters {len(kept):,}", flush=True)
    vocab = school_vocabulary(school_texts(args.school), background)
    print(f"articles {seen:,} | passed {len(kept):,} | vocabulary {len(vocab):,} words", flush=True)

    scored = sorted(((score(text, vocab), title, url, text) for title, url, text in kept), reverse=True)
    with args.out.with_suffix(".ranking.tsv").open("w", encoding="utf-8") as f:  # every candidate, for choosing the cut-off
        for s, title, _, text in scored:
            f.write(f"{s:.4f}\t{len(text.split())}\t{title}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total, n = 0, 0
    with args.out.open("w", encoding="utf-8") as f:
        for s, title, url, text in scored:
            if total >= args.target_words or s < args.min_score:
                break
            parts, count = [], 0
            for paragraph in text.split("\n"):
                count += len(paragraph.split())
                parts.append(paragraph)
                if count >= args.max_words:
                    break
            total += count
            n += 1
            f.write(json.dumps({"source": "wiki", "title": title, "url": url, "score": round(s, 4), "text": "\n".join(parts).strip()}, ensure_ascii=False) + "\n")
    stats = {"articles_read": seen, "passed_basic_filters": len(kept), "kept": n, "words": total, "lowest_score_kept": round(scored[n - 1][0], 4) if n else None,
             "top_vocabulary": heapq.nlargest(40, vocab, key=vocab.get)}
    args.out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "top_vocabulary"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
