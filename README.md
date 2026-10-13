# SupportGen — GenAI-Powered Customer Support Intelligence and Response Generation Using Fine-Tuned Transformers

> **Status:** the ML pipeline (Colab notebooks) is ready to run. The backend, frontend and final
> documentation are next. No results are reported here until the notebooks have actually been run.

## What this project is
SupportGen is a copilot for customer-support agents. It analyses a ticket and **generates a support
response** using an encoder-decoder Transformer (**FLAN-T5**) that **we fine-tuned** on customer-support
conversations. Around the generator sit supporting components:
- predicted queue, priority and ticket type
- inferred sentiment
- zero-shot summary and recommended action

The project is **not** an API wrapper. The central contribution is:

**customer-support dataset → Transformer fine-tuning → generative response model → evaluation
(ROUGE, BERTScore, human) → web application.**

Gemini is only an optional benchmark.

## Workflow
```
Google Colab (notebooks/01–06)              Mac
───────────────────────────────            ──────────────────────────────────────────
EDA → cleaning → leakage-safe split          unzip supportgen_models.zip into project/
baseline FLAN-T5 (Model A)            ──►    Django + DRF loads models/flan-t5-supportgen
fine-tune FLAN-T5-small (Model B)            React frontend calls the REST API
evaluation + classifiers + export ZIP
```
See [`notebooks/README_COLAB.md`](notebooks/README_COLAB.md) for step-by-step Colab instructions.

## Project structure (current)
```
notebooks/   Colab notebooks 01–06 + README_COLAB.md   (all ML work)
models/      trained artefacts from the Colab export   (gitignored)
results/     metrics / plots from the Colab export
backend/     Django + DRF (next phase)
frontend/    React (next phase)
docs/        theory, report, viva Q&A, demo script (final phase)
```

## Dataset
[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
on Hugging Face.
- **License:** CC BY-NC 4.0 (non-commercial use with attribution).
- **Synthetic data:** the dataset card describes it as **synthetically generated**. Results therefore
  reflect performance on synthetic support conversations, not on a real company's tickets.
- **Language:** the data is English and German. We use the English subset.
- **Statistics:** split sizes and cleaning statistics are computed in notebook 01 and saved to
  `results/data/eda_stats.json`. They will be added here after the run.

## Authors
_TBD_
