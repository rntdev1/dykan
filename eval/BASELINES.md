# Baselines: stock Gemma 4

Run 2026-11-14 with `eval/run_hf.py`: bf16, transformers 5.23.0, greedy, thinking off.
Models: `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it` (latest revisions on that date).
Raw outputs: `outputs/eval/` (not in git).

## Held-out ZNO/NMT, 2020–2024

1,111 text-only tasks, answer-only protocol (no explanation, 64 new tokens).

| Subject | Tasks | E2B | E4B |
|---|---|---|---|
| Ukrainian language and literature | 521 | 26.5% | 36.7% |
| History of Ukraine | 316 | 33.5% | 47.5% |
| Geography | 156 | 60.3% | 71.2% |
| Maths | 118 | 22.0% | 27.1% |
| **Overall** | **1,111** | **32.8%** | **43.6%** |

| Question type | Tasks | E2B | E4B |
|---|---|---|---|
| Single choice | 889 | 39.4% | 51.2% |
| Matching (all pairs must be right) | 176 | 6.8% | 14.2% |
| Sequence | 21 | 9.5% | 9.5% |
| Multi-select | 25 | 0.0% | 8.0% |

Replies with no readable answer: E2B 2.8%, E4B 3.1% (counted as wrong).
Open-text maths answers (43 tasks) are not scored yet.

## Multi-step tool use

130 episodes (10 per task type, seed 20261114) on the simulated school backend,
scored on the end state. No degenerate answers in either run.

| Task type | E2B | E4B |
|---|---|---|
| Homework for a given day | 0/10 | 10/10 |
| Homework digest sent as a notification | 0/10 | 9/10 |
| Daily or weekday scheduled digest | 9/10 | 9/10 |
| One-off reminder | 10/10 | 10/10 |
| Mark homework done | 0/10 | 3/10 |
| Message a teacher | 0/10 | 10/10 |
| Last grades | 10/10 | 10/10 |
| Analytics | 9/10 | 9/10 |
| Timetable for a weekday | 1/10 | 6/10 |
| Cancel a reminder | 0/10 | 3/10 |
| No tool needed | 10/10 | 10/10 |
| Missing detail (should ask) | 10/10 | 10/10 |
| Unknown subject (tool error) | 10/10 | 10/10 |
| **Task completion** | **53.1%** | **83.8%** |

## What the failures look like

- **E2B does not chain calls.** When an action needs an identifier it has not seen
  (a homework id, a teacher id, a task id), it asks the user for it instead of
  calling the list tool first. It also asks "which subject?" when asked for
  tomorrow's homework.
- **E4B chains, but stops before the final action** in "mark done" and "cancel":
  it lists the items and then asks for confirmation or an id.
- **Date arithmetic is weak in both.** Wrong dates for a named weekday, and E4B
  produced `2026-11-31` for "tomorrow" on 30 November.

## Consequences

- Training targets: chained calls, finishing the action, and dates.
- App-side: put a short calendar in the system prompt (today, tomorrow and the
  next seven days with weekdays), so the model looks dates up instead of computing them.

## Pilot and MamayLM comparison (2026-11-14)

Same 1,111 held-out ZNO tasks and protocol as above.

| Subject | Stock E2B | Pilot E2B | Stock E4B | MamayLM 4B v1.0 | MamayLM 12B v2.0 |
|---|---|---|---|---|---|
| Ukrainian (521) | 26.5% | 33.4% | 36.7% | 28.8% | 44.2% |
| History (316) | 33.5% | 40.2% | 47.5% | 43.3% | 67.4% |
| Geography (156) | 60.3% | 69.9% | 71.2% | 60.9% | 82.7% |
| Maths (118) | 22.0% | 28.8% | 27.1% | 22.9% | 46.6% |
| **Overall** | **32.8%** | **40.0%** | **43.6%** | **36.8%** | **56.4%** |
| Single choice (889) | 39.4% | 45.7% | 51.2% | 45.0% | 64.3% |
| Matching (176) | 6.8% | 17.6% | 14.2% | 2.8% | 25.0% |

MamayLM models: `INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0` and
`INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0` (Gemma 3 based; ZNO only, since
their tool format differs).

**Pilot E2B** = stock E2B plus one LoRA run (`train/sft.py`, rank 16 on all
language-model projections, 2 epochs, 23 minutes on an RTX 5090, about 6,500
tokens/sec) on 4,487 pre-2020 exam tasks with answer keys only and 2,600
generated tool episodes (`data/build_pilot.py`). Adapter: `checkpoints/pilot-e2b/adapter`.

Tool use, 130 episodes: stock E2B 53.1% → pilot E2B **92.3%** (stock E4B 83.8%).
Chained tasks went from 0/10 to 8–10/10 (homework for a day, digest, mark done,
message a teacher, cancel a reminder). Timetable for a named weekday is still
4/10: date arithmetic. Training and eval episodes share task types but not
wording or school worlds.

Caveats: no explanations were trained; the pilot proves the benchmarks move, it
is not the final recipe.

## Unseen tool names and the dev split (2026-11-15)

Two additions to the eval:

- `--skin heldout`: the same 130 episodes, with every tool and parameter declared
  under a name that is never used in training (`data/tools/rename.py`). A call
  under the old, undeclared name counts as an error.
- `--zno-split dev`: 310 pre-2020 exam tasks (Ukrainian 138, history 116,
  geography 56) that are kept out of training, for choosing settings without
  looking at the reported 1,111 tasks.

| Model | Tools, original names | Tools, held-out names | ZNO dev (310) | ZNO test (1,111) |
|---|---|---|---|---|
| Stock E2B | 53.1% | 53.8% | 39.0% | 32.8% |
| Stock E4B | 83.8% | 73.8% | 51.9% | 43.6% |
| Pilot E2B | 92.3% | 83.1% | not comparable (trained on these tasks) | 40.0% |
| Smoke E2B | 95.4% (100% with the calendar) | 87.7% | 42.6% | 36.5% |

**Smoke E2B** = LoRA rank 64, one epoch on a partial training set of 22k examples
(exam keys without shuffled copies, 2.4k verified exam explanations, 5.2k
generated tasks, 4.8k explanations, 5.8k school-tool episodes half of them
renamed; no tutoring, chat or invented-tool data yet). 65 minutes on an RTX 5090.

Reading:
- Renaming costs the pilot 9 points and the stock E4B 10: both lean on the
  familiar names. Renamed training data narrows the gap, and the remaining
  failures of the smoke model are "mark done" under held-out names, where it
  wrote a parameter `hw_id` when the declared one was `hw`.
- With the calendar in the system prompt the smoke model completes all 130
  episodes; without it, the weekday timetable is still 4/10.
- On exams the smoke model is above stock but below the pilot, which saw only
  exam keys and tool episodes for two epochs. Exam keys supervise a few tokens
  each and get little weight in a mixed set, so the full set adds two
  option-shuffled copies of every single-choice task.

A first attempt at the smoke run is not in the table: 9,696 of its 9,725 batches
were skipped for lack of GPU memory and the result was an untrained model. The
trainer now computes the output layer in chunks, uses the leaner attention
implementation for the language model, and stops if more than 2% of batches are skipped.

## Full training runs (2026-11-15/16)

All LoRA on the language model only, 2 epochs over the instruction set
(`data/build_dataset.py`, about 62–69k examples). "Wikipedia" runs add one pass
over 36,000 school-relevant Ukrainian Wikipedia articles (39.8M words,
`data/corpus/`) and recast 4,415 worked solutions into thinking mode.

| Model | Exam test (1,111) | Exam dev (310) | Tools | Tools, unseen names | Tools + calendar |
|---|---|---|---|---|---|
| Stock E2B | 32.8% | 39.0% | 53.1% | 53.8% | — |
| E2B v1 (rank 64, shuffled exam copies) | 30.8% | 41.3% | 94.6% | 95.4% | 100% |
| E2B v2 (rank 16) | 38.2% | 48.7% | 98.5% | 98.5% | 99.2% |
| E2B v2 + Wikipedia + thinking (rank 16) | **39.3%** | **50.3%** | 95.4% | 96.2% | 98.5% |
| Stock E4B | 43.6% | 51.9% | 83.8% | 73.8% | — |
| E4B v1 (rank 64, shuffled exam copies) | 49.2% | **60.3%** | **98.5%** | **97.7%** | 100% |
| E4B + Wikipedia + thinking (rank 64) | **50.3%** | 58.7% | 97.7% | **97.7%** | 100% |
| MamayLM 4B v1.0 | 36.8% | — | — | — | — |
| MamayLM 12B v2.0 | 56.4% | — | — | — | — |

No degenerate answers in any of these evals. In every run the half-way
snapshot (end of epoch 1) scored lower than the final one.

### Why E2B v1 collapsed on exams

E2B v1 answered "В" to most single-choice questions. Reading the probability
it gives each letter (`eval/letter_probs.py`) showed nearly uniform
distributions: the model had stopped telling options apart. Small ablations
on exam keys alone (`outputs/eval/abl-*`):

| Exam keys only, E2B | Test |
|---|---|
| rank 64, lr 2e-4 | 17.8% (all "В") |
| rank 64, lr 2e-4, with shuffled copies | 18.0% (all "В") |
| rank 64, lr 5e-5 | 40.7% |
| rank 16, lr 2e-4 | 38.0% |
| pilot data, rank 16 (reproduction of the pilot) | 40.2% |
| pilot data, rank 64 | 30.0% |

So the cause was the update size (rank 64 at lr 2e-4) for the 2B model, not
the data and not the trainer; E4B tolerates it. The option-shuffled exam
copies did not help either and are off by default now.
