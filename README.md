# SupportGen — GenAI-Powered Customer Support Intelligence and Response Generation Using Fine-Tuned Transformers

> **Status:** ML pipeline run in Colab ✅ · Django REST backend ✅ · React frontend ✅ ·
> final documentation (report, viva Q&A, demo script) — last phase.

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
backend/     Django + DRF API; AI services in backend/ai/ (no Django imports)
frontend/    React (Vite) app: dashboard, analyzer, history, comparison, evaluation
docs/        theory, report, viva Q&A, demo script (final phase)
```

## Results (FLAN-T5-small, 500 held-out test tickets)
All numbers come from `results/gen_metrics.json` (notebook 04).

| Metric | A: base FLAN-T5-small | B: fine-tuned (ours) |
|---|---|---|
| ROUGE-1 / ROUGE-2 / ROUGE-L | 0.205 / 0.080 / 0.158 | **0.474 / 0.240 / 0.343** |
| BLEU | 1.17 | **19.57** |
| BERTScore F1 (raw / rescaled) | 0.867 / 0.210 | **0.899 / 0.403** |
| Embedding cosine similarity | 0.580 | **0.781** |

Every improvement is significant (paired Wilcoxon p < 1e-50; bootstrap 95 % CIs exclude 0).
The model was trained for 4 epochs in 30 minutes on a Colab T4. Best validation loss was 1.122 and test loss 1.111 (perplexity 3.04).

**Known limitations** (see notebook 04):
- Replies are formulaic: 85 % offer a call, and distinct-2 drops from 0.55 to 0.16.
- Factual slips occur, e.g. "the refund you received" for a refund that was *not* received.
- Placeholders such as `<tel_num>` lose their `<`, because T5's vocabulary has no `<`. The backend repairs them.

## Running the backend (Mac)
Requires Python 3.12, plus PostgreSQL (or set `USE_POSTGRES=False` to use SQLite).
```bash
cd project
unzip -o ~/Downloads/supportgen_models.zip -d .       # → models/ and results/
~/.pyenv/versions/3.12.10/bin/python -m venv .venv
.venv/bin/pip install -r requirements-backend.txt
cp .env.example .env                                   # set POSTGRES_USER etc.
createdb supportgen
cd backend
../.venv/bin/python manage.py migrate
../.venv/bin/python manage.py test tickets             # 32 tests
../.venv/bin/python manage.py runserver                # http://127.0.0.1:8000/api/health/
```
The first `/analyze/` request downloads the summary and sentiment models (~1.5 GB, once).
After that, a request takes about 2 s on an M4 CPU.

| Endpoint | Purpose |
|---|---|
| `POST /api/tickets/analyze/` | `{subject, body}` → response, summary, category, priority, type, sentiment, action (saved) |
| `GET /api/tickets/`, `GET /api/tickets/<id>/` | ticket history |
| `GET /api/analytics/` | dashboard statistics |
| `GET /api/evaluation/`, `GET /api/evaluation/plots/<key>/` | metrics and plots exported from Colab |
| `POST /api/compare/` | same ticket through base FLAN-T5, fine-tuned FLAN-T5 and Gemini (if `GEMINI_API_KEY` is set) |
| `GET /api/health/` | which models, database and providers are available |

**Design notes**
- Models load once per process (`ai/registry.py`), never per request.
- **Errors:** a missing model returns 503; empty, too-short (< 10 chars) or too-long (> 5000 chars) tickets return 400. A database failure still returns the analysis, with a warning.
- **Summary:** zero-shot FLAN-T5-base with an extractive fallback when the output is degenerate.
- **Recommended action:** rule-based from the predicted queue and priority, because zero-shot actions were unusable in evaluation.
- **Sentiment:** inferred by a pretrained model, not evaluated against ground truth.

## Running the frontend
Requires Node 18 or newer. Start the backend first; the Vite dev server forwards `/api` to `127.0.0.1:8000`.
```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```
| Page | What it shows |
|---|---|
| Dashboard | Totals and distributions of analysed tickets (category, priority, type, inferred sentiment), recent tickets, model headline metrics |
| Analyze ticket | Customer ticket → fine-tuned FLAN-T5 draft reply, summary, recommended action, category, priority, type, sentiment |
| Ticket history | Every analysed ticket, with a detail view |
| Model comparison | Live: the same ticket through base FLAN-T5, fine-tuned FLAN-T5 and Gemini (if configured), plus the offline benchmark |
| Model evaluation | Training set-up, loss curves, ROUGE/BLEU/BERTScore/cosine with significance tests, efficiency, classifier metrics and confusion matrices, dataset preparation, example replies, limitations |

Every number on the Evaluation and Comparison pages is read from `results/` (the Colab export). None is typed into the UI.

## Dataset
[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
on Hugging Face.
- **License:** CC BY-NC 4.0 (non-commercial use with attribution).
- **Synthetic data:** the dataset card describes it as **synthetically generated**. Results therefore
  reflect performance on synthetic support conversations, not on a real company's tickets.
- **Language:** the data is English and German. We use the English subset.
- **Statistics** (`results/data/eda_stats.json`):
  - 61,765 raw rows, of which 28,261 are English.
  - Cleaning leaves 23,557 unique English tickets. Exact-duplicate removal dropped 4,597 rows, mostly 4,592 tickets shared between two of the CSV files.
  - Near-duplicate grouping plus a leakage audit (8 val + 8 test tickets removed) gives the final split: **18,848 train / 2,354 val / 2,339 test**.

## Authors
_TBD_
