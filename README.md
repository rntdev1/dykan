# Dykan LM

Ukrainian fine-tunes of Google's Gemma 4 E2B and E4B for **Dykan**, an offline school assistant for students in Ukraine. The model runs entirely on the phone: it explains and solves school and exam (ZNO/NMT) tasks, reads photos of tasks, and carries out multi-step requests with tools.

Models on Hugging Face: [rntdev/dykan](https://huggingface.co/rntdev/dykan)

## Results

Held-out ZNO/NMT exams 2020–2024 (1,111 text tasks, answer only) and 130 multi-step tool episodes on a simulated school backend, scored on the end state. "Unseen names" declares every tool under names never used in training. Full tables and the history of each run: [`eval/BASELINES.md`](eval/BASELINES.md).

| Model | Exams | Tools | Tools, unseen names |
|---|---|---|---|
| Gemma 4 E2B (stock) | 32.8% | 53.1% | 53.8% |
| **Dykan E2B** | **39.3%** | **95.4%** | **96.2%** |
| Gemma 4 E4B (stock) | 43.6% | 83.8% | 73.8% |
| **Dykan E4B** | **50.3%** | **97.7%** | **97.7%** |
| MamayLM 4B v1.0 (Gemma 3) | 36.8% | — | — |
| MamayLM 12B v2.0 (Gemma 3) | 56.4% | — | — |

Scores are for the full-precision (bf16) models; the 4-bit phone builds have not been evaluated yet.

## How it is built

- **Base:** `google/gemma-4-E2B-it` and `google/gemma-4-E4B-it`, LoRA on the language model only (vision and audio encoders frozen), loss only on what the model writes.
- **Data** (about 62k examples plus 36k Ukrainian Wikipedia articles), all generated or verified without commercial APIs:
  - real exam tasks from 2019 and earlier with official answer keys, plus worked solutions that reached the key in at least two of three attempts and passed a fact check;
  - generated curriculum tasks, explanations, tutoring dialogues and general chat, written by Gemma 4 31B on our own GPUs, with answer checks, blind re-solving, fact checks and a clean-Ukrainian filter;
  - tool episodes on a simulated school backend (homework, grades, timetable, messages, scheduled tasks, notifications), half of them under randomly renamed tools, plus episodes with tools invented for other domains, so the model reads whatever tools the app declares.
- **Held-out exams:** forms from 2020 onward are eval-only, and every training source is filtered against them for near-duplicates.
- **Wikipedia and thinking mode:** one pass over 36k school-relevant Ukrainian Wikipedia articles, and 4,415 worked solutions recast so the working goes in Gemma's thinking channel (used when the app turns thinking on).

## Repository

```
data/       exam split and leak filter, teacher generation recipes, tool schemas
            and simulated school backend, dataset assembly, Wikipedia selection
train/      LoRA fine-tuning (train/sft.py) and loss masking
eval/       exam and tool-use evals, tool-call parser, results (BASELINES.md)
demo/       terminal chat on a Mac with the school tools simulated (MLX, offline)
configs/    pinned Gemma 4 chat template
tests/
```

## Running

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11–3.13.

```bash
uv run pytest -q                                   # tests, no GPU or downloads

# chat on an Apple-silicon Mac, fully offline, with an MLX build of a model (from Hugging Face)
uv run --extra mac python -m demo.chat --model path/to/dykan-e4b-mlx

# GPU: build the training set, train, evaluate
uv run python -m data.build_dataset --zno all.jsonl.gz --teacher outputs/teacher --out outputs/train/dykan.jsonl
python -m train.sft --model google/gemma-4-E4B-it --data outputs/train/dykan.jsonl --out checkpoints/e4b \
    --rank 64 --epochs 2 --token-budget 7000 --max-len 3500 --attn sdpa      # E2B: --rank 16 --token-budget 2500 --max-len 2500
python -m eval.run_hf --model google/gemma-4-E4B-it --adapter checkpoints/e4b/adapter --zno all.jsonl.gz
python -m eval.run_hf --model google/gemma-4-E4B-it --adapter checkpoints/e4b/adapter --skin heldout
```

`all.jsonl.gz` is `data/tasks/all.jsonl.gz` from [NLPForUA/zno-eval](https://huggingface.co/datasets/NLPForUA/zno-eval). Teacher generation needs an OpenAI-compatible server (vLLM) running the teacher model; see `data/teacher/recipes.py`.

The released models were trained on the instruction set plus `outputs/corpus/wiki_school.jsonl` (see `data/corpus/`). E2B needs a smaller adapter than E4B: at rank 64 its exam answers collapsed onto a single letter (diagnosis in `eval/BASELINES.md`).

## Data and licences

- Code: Apache 2.0 (`LICENSE`). Base models: Gemma 4, Apache 2.0.
- ZNO/NMT exam text belongs to the Ukrainian Center for Educational Quality Assessment (UCEQA). It is used for training and evaluation with attribution and is not republished here; neither is the generated training data, which contains it.
- Teacher outputs come from openly licensed models run on our own hardware. Solution-book (GDZ) sites, the state online school and non-commercially licensed sets are excluded.
- Wikipedia text used in training is CC BY-SA.

## Known limitations

- Knowledge ends before early 2025 and the model can state wrong facts confidently, i.e. - hallucinate.
- Without the date and a short calendar in the system prompt it gets weekday arithmetic wrong; the app should always provide them, as `demo/chat.py` does.
- Not trained on images yet; picture understanding is stock Gemma 4.

## References

- Gemma 4 Technical Report. [arXiv:2607.02770](https://arxiv.org/abs/2607.02770). Models: [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it), [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it)
- MamayLM (INSAIT), Ukrainian Gemma 3 models used for comparison: <https://huggingface.co/INSAIT-Institute>
- BgGPT 1.0: Extending English-centric LLMs to other languages. [arXiv:2412.10893](https://arxiv.org/abs/2412.10893)
- Mitigating Catastrophic Forgetting in Language Transfer via Model Merging. [arXiv:2407.08699](https://arxiv.org/abs/2407.08699)
- ZNO-Eval: Benchmarking reasoning capabilities of large language models in Ukrainian. [arXiv:2501.06715](https://arxiv.org/abs/2501.06715). Data: [`NLPForUA/zno-eval`](https://huggingface.co/datasets/NLPForUA/zno-eval)
- Benchmarking Multimodal Models for Ukrainian Language Understanding Across Academic and Cultural Domains. [arXiv:2411.14647](https://arxiv.org/abs/2411.14647)
- Exam papers and answer keys: Ukrainian Center for Educational Quality Assessment, <https://testportal.gov.ua>
