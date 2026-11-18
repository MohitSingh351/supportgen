# SupportGen

**GenAI-powered customer support intelligence and response generation using fine-tuned Transformers.**

SupportGen is a copilot for customer-support agents. Paste in a customer ticket and it returns:

- a **draft reply**, written by a FLAN-T5 model that **we fine-tuned** on customer-support conversations
- the ticket's **category** (queue), **priority** and **type**, each with a confidence score
- the customer's **sentiment** (positive / neutral / negative)
- a **one-line summary** and a **recommended next action** for the agent

A human agent always stays in charge: SupportGen writes a first draft for a person to check, and it never replies to a customer on its own.

> **This is not an API wrapper.** The core of the project is our own pipeline:
> **customer-support dataset → Transformer fine-tuning → generative response model → evaluation (ROUGE, BLEU, BERTScore, LLM-as-a-judge) → web application.**
> Gemini appears only as an optional benchmark.

Viva preparation: see [`viva.md`](viva.md).

---

## Table of contents
1. [How it works, in plain language](#1-how-it-works-in-plain-language)
2. [The dataset](#2-the-dataset)
3. [The ML pipeline (Colab notebooks)](#3-the-ml-pipeline-colab-notebooks)
4. [Results](#4-results)
5. [The backend (Django REST API)](#5-the-backend-django-rest-api)
6. [The frontend (React)](#6-the-frontend-react)
7. [Known limitations](#7-known-limitations)
8. [How to run the whole project](#8-how-to-run-the-whole-project)
9. [How to demo it to someone](#9-how-to-demo-it-to-someone)
10. [Project structure](#10-project-structure)
11. [Tech stack](#11-tech-stack)
12. [License and credits](#12-license-and-credits)

---

## 1. How it works, in plain language

The project has four parts:

```
① Data  ──►  ② Training (Google Colab)  ──►  ③ Backend (Django, on a laptop)  ──►  ④ Website (React)
```

1. **Data:** thousands of example support tickets, each paired with the reply a support agent wrote.
2. **Training:** we start from **FLAN-T5-small**, a small, free language model from Google (~77M parameters), and **fine-tune** it on those ticket → reply pairs so it learns how support agents write. We also train three small classifiers. All heavy ML work runs on Colab's free GPU.
3. **Backend:** a Django server loads the trained models once and exposes a REST API. When a ticket arrives, it runs every model and saves the result to a database.
4. **Website:** a React app where you paste a ticket, view the results, browse the history, compare models side by side and read the evaluation.

We kept two versions of the reply model so the effect of training can be measured:

| | Model | What it is |
|---|---|---|
| **A** | `google/flan-t5-small` | The original model, used without any training on our data (zero-shot baseline) |
| **B** | `flan-t5-supportgen` | The same model after our fine-tuning |

---

## 2. The dataset

[**Tobi-Bueck/customer-support-tickets**](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) on Hugging Face.

- **License:** CC BY-NC 4.0 (non-commercial use with attribution).
- **Synthetic:** the dataset card says the data is **synthetically generated**. Results therefore describe performance on synthetic support conversations, not on a real company's tickets.
- **Language:** English and German. We use only English.
- **Fields used:** `subject`, `body` (the customer's ticket), `answer` (the agent's reply), `queue`, `priority`, `type`.

**Cleaning and splitting** (notebook 01):

| Step | Rows |
|---|---|
| Raw rows (all languages) | 61,765 |
| English only | 28,261 |
| After cleaning and exact-duplicate removal (−4,597 rows, mostly 4,592 tickets shared between two of the CSV files) | 23,557 |
| Near-duplicate grouping plus a leakage audit (8 val and 8 test tickets removed) | **18,848 train / 2,354 val / 2,339 test** |

*Leakage* means the same or almost the same ticket appearing in both training and test data, which would make test scores look better than they really are. We grouped near-duplicates so each group lands in only one split, then audited the test and validation sets against training.

---

## 3. The ML pipeline (Colab notebooks)

All ML work runs in Google Colab (`notebooks/`). The laptop only runs the finished models. Step-by-step Colab instructions are in [`notebooks/README_COLAB.md`](notebooks/README_COLAB.md).

| Notebook | What it does |
|---|---|
| `01_dataset_preparation` | Exploratory analysis, cleaning, English filtering, de-duplication and the leakage-safe split |
| `02_baseline_flan_t5` | Runs the original FLAN-T5-small with no training: **Model A** |
| `03_train_flan_t5` | Fine-tunes FLAN-T5-small on ticket → reply pairs: **Model B** |
| `04_evaluate_model` | Scores A and B against the real replies, with significance tests and error analysis |
| `05_classification_models` | Trains the TF-IDF classifiers for queue, priority and type |
| `06_model_comparison` | Puts everything side by side and exports `supportgen_models.zip` |

### Fine-tuning setup (Model B)

| Setting | Value |
|---|---|
| Base model | `google/flan-t5-small` (encoder-decoder Transformer) |
| Method | Full fine-tuning (all weights updated) |
| Input prompt | `Generate a professional customer support response.\nSubject: {subject}\nCustomer ticket: {body}` |
| Max input / target length | 160 / 128 tokens |
| Optimizer / learning rate | AdamW, 3e-4 |
| Batch size | 16 |
| Epochs | 4 (1,178 steps per epoch, 4,712 steps in total) |
| Hardware / time | Colab Tesla T4, about 30 minutes |
| Losses | train 1.351 · best validation 1.122 · test 1.111 (perplexity 3.04) |
| Decoding | Beam search, 4 beams, `no_repeat_ngram_size=3` |

### Classifiers

Three separate **TF-IDF + Linear SVM** pipelines (scikit-learn) predict `queue_group` (10 classes), `priority` (low / medium / high) and `type` (Incident / Request / Problem / Change). The input is `subject + " . " + body`.

### Other components (no training of our own)
- **Sentiment:** the pretrained `cardiffnlp/twitter-roberta-base-sentiment-latest`. The dataset has no sentiment labels, so this output is labelled *inferred* and is not evaluated.
- **Summary:** zero-shot `google/flan-t5-base`, with an **extractive fallback** (key sentences from the ticket) when the generated text is degenerate, for example when it only repeats the subject.
- **Recommended action:** **rule-based**, chosen from the predicted queue and priority. Zero-shot generated actions were unusable in evaluation, so rules give a predictable suggestion that never invents facts.

---

## 4. Results

All numbers come from `results/` (the Colab export). None is typed in by hand.

### Reply generation (500 held-out test tickets, `results/gen_metrics.json`)

| Metric | A: base FLAN-T5-small | **B: fine-tuned (ours)** |
|---|---|---|
| ROUGE-1 / ROUGE-2 / ROUGE-L | 0.205 / 0.080 / 0.158 | **0.474 / 0.240 / 0.343** |
| BLEU | 1.17 | **19.57** |
| BERTScore F1 (raw / rescaled) | 0.867 / 0.210 | **0.899 / 0.403** |
| Embedding cosine similarity | 0.580 | **0.781** |
| Average reply length (words; references average 59.8) | 18.5 | 67.8 |

- **ROUGE / BLEU** measure how many words and phrases overlap with the real reply.
- **BERTScore / cosine similarity** measure whether the *meaning* matches, even when the wording differs.

Every improvement is statistically significant: paired Wilcoxon test p < 1e-50, and the bootstrap 95% confidence intervals exclude 0. Model B scores higher ROUGE-1 than Model A on 92% of tickets.

### LLM-as-a-judge (50 tickets, `results/llm_judge_summary.json`)

Claude (`claude-opus-5-5`) rated anonymised, shuffled replies from 1 to 5 using the rubric in `results/llm_judge_rubric.md`. **This is a model's rating, not a human evaluation.**

| Criterion (1–5) | A: base | **B: fine-tuned** |
|---|---|---|
| Relevance | 1.48 | **4.34** |
| Correctness | 1.10 | **3.58** |
| Fluency | 3.30 | **4.68** |
| Professional tone | 1.08 | **4.74** |
| Helpfulness | 1.00 | **2.82** |
| **Overall** | 1.59 | **4.03** |

- The fine-tuned reply was preferred on **50 of 50** tickets (Wilcoxon p = 7.2e-10).
- 46 of 50 base replies were not support replies at all. None of the fine-tuned replies had that problem.
- 5 of 50 fine-tuned replies still scored ≤ 2 on correctness.

### Classifiers (2,339 test tickets, `results/cls_metrics.json`)

| Target | Accuracy | Macro F1 | Majority-class baseline accuracy |
|---|---|---|---|
| Type (4 classes) | **0.881** | 0.871 | 0.391 |
| Priority (3 classes) | 0.643 | 0.580 | 0.413 |
| Queue (10 classes) | 0.620 | 0.593 | 0.279 |

All three beat the baselines clearly. Queue and priority are harder because several queues overlap in meaning and priority is partly subjective.

Plots (loss curves, confusion matrices, metric box plots, label distributions) are in `results/plots/`. The app's Evaluation page draws the loss curves and metric charts from the result files and explains the classifier scores in plain language. The significance tables and confusion matrices stay in `results/` for the report.

---

## 5. The backend (Django REST API)

Located in `backend/`. The AI code sits in `backend/ai/` and has no Django imports, so it can be tested on its own.

For every ticket, `TicketAnalyzer.analyze()` in `backend/ai/services.py` runs five steps:

1. **Draft reply:** fine-tuned FLAN-T5 (required; if the model is missing, the API returns 503)
2. **Queue, priority, type:** the three classifiers, each with a confidence score
3. **Sentiment:** the pretrained RoBERTa model
4. **Summary:** zero-shot FLAN-T5-base, with an extractive fallback
5. **Recommended action:** rules based on queue and priority

The result, including per-step timings and any warnings, is saved to the database (PostgreSQL, or SQLite).

**Design notes**
- **Models load once per process** (`ai/registry.py`), not on every request. After warm-up, a request takes about 2 s on an Apple M4 CPU.
- **Placeholder repair** (`ai/postprocess.py`): the training data contains placeholders such as `<tel_num>`. T5's vocabulary has no `<` character, so the model emits `tel_num>`. The backend rewrites every variant into a readable form such as `[Phone]`.
- **Swappable providers** (`ai/providers.py`): the base model, the fine-tuned model and Gemini share one interface, so the comparison endpoint treats them the same way.
- **Errors:** a missing model returns 503. Empty, too-short (< 10 characters) or too-long (> 5,000 characters) tickets return 400. If the database fails, the analysis is still returned, with a warning.
- **Tests:** 32 tests in `backend/tickets/tests/` (unit tests for the AI helpers, API tests with mocked models, and optional tests that use the real models).

| Endpoint | Purpose |
|---|---|
| `POST /api/tickets/analyze/` | `{subject, body}` → reply, summary, category, priority, type, sentiment and action (saved) |
| `GET /api/tickets/`, `GET /api/tickets/<id>/` | Ticket history |
| `GET /api/analytics/` | Dashboard statistics |
| `GET /api/evaluation/`, `GET /api/evaluation/plots/<key>/` | Metrics and plots exported from Colab |
| `POST /api/compare/` | The same ticket through base FLAN-T5, fine-tuned FLAN-T5 and Gemini (if `GEMINI_API_KEY` is set) |
| `GET /api/health/` | Which models, database and providers are available |

---

## 6. The frontend (React)

Located in `frontend/` (React + Vite). In development, the Vite dev server forwards `/api` to the Django backend.

| Page | What it shows |
|---|---|
| **Dashboard** | Totals and distributions of analysed tickets (category, priority, type, inferred sentiment), recent tickets and the model's headline metrics |
| **How it works** | The whole project on one page in plain language: the problem, a 4-box overview, then the full pipeline step by step in four phases (A: prepare the data, B: build the models, C: test the models, D: run the app, including the 9 steps behind "Analyze ticket"), headline results, the pages and the limits. Good for presenting. |
| **Analyze ticket** | Paste a ticket and get the fine-tuned reply, summary, recommended action, category, priority, type and sentiment |
| **Ticket history** | Every analysed ticket, with a detail view |
| **Model comparison** | Live: one ticket through base FLAN-T5, fine-tuned FLAN-T5 and Gemini (if configured), plus the offline benchmark |
| **Model evaluation** | Training setup, loss curves, before/after reply metrics, AI-judge scores, classifier metrics with an easy explanation of macro-F1, dataset preparation, example replies and limitations |

---

## 7. Known limitations

- **Formulaic replies:** 85% of fine-tuned replies offer a phone call, and vocabulary diversity (distinct-2) drops from 0.55 to 0.16. The model learned the most common pattern in the data.
- **Factual slips:** for example, "the refund you received" when the customer said the refund had *not* arrived. A small model does not reliably track negation.
- **Synthetic data:** results show performance on AI-generated tickets, not a real company's.
- **Moderate classifiers:** queue and priority are only around 60% accurate.
- **Small model:** FLAN-T5-small was chosen to fit the free Colab GPU and to run on a laptop CPU. A larger model would likely do better.
- **Unevaluated parts:** sentiment has no ground-truth labels. The summary and action have no reference data, so neither was trained.

---

## 8. How to run the whole project

### What you need

| Tool | Version |
|---|---|
| Python | **3.12** (some ML packages do not support newer versions yet) |
| Node.js | 18 or newer |
| PostgreSQL | Optional; set `USE_POSTGRES=False` to use SQLite |
| Disk / RAM | About 3 GB free disk space, 8 GB RAM |
| Internet | Only for the first run, to download the summary and sentiment models (about 1.5 GB, once) |

### Step 0: get the trained models

The trained models are **not in this repository** because they are too large for git. They come from the Colab export `supportgen_models.zip`, which contains `models/` and `results/`. Either:

- **Option A:** run notebooks 01 → 06 in Google Colab (see [`notebooks/README_COLAB.md`](notebooks/README_COLAB.md)) and download `supportgen_models.zip` at the end, or
- **Option B:** ask the authors for their copy of `supportgen_models.zip`.

When unzipped into the project root, it creates:

```
models/flan-t5-supportgen/   ← fine-tuned reply model (Model B)
models/classifiers/          ← queue / priority / type classifiers + meta.json
results/                     ← metrics and plots (already in this repo)
```

### Step 1: clone the repository and unzip the models

```bash
git clone https://github.com/MohitSingh351/supportgen.git
cd supportgen
unzip -o ~/Downloads/supportgen_models.zip -d .
```

### Step 2: start the backend (terminal 1)

```bash
python3.12 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements-backend.txt

cp .env.example .env
# Edit .env:
#   - no PostgreSQL? set USE_POSTGRES=False (SQLite is used instead)
#   - with PostgreSQL: set POSTGRES_USER / POSTGRES_PASSWORD, then run:  createdb supportgen
#   - optional: set GEMINI_API_KEY to add Gemini to the comparison page (also: pip install google-genai)

cd backend
python manage.py migrate
python manage.py test tickets          # optional: runs the 32 tests
python manage.py runserver             # API at http://127.0.0.1:8000
```

Check that everything loaded: open <http://127.0.0.1:8000/api/health/>. It lists which models, database and providers are available.

### Step 3: start the frontend (terminal 2)

```bash
cd frontend
npm install
npm run dev                            # app at http://localhost:5173
```

Open **<http://localhost:5173>** in a browser.

### Troubleshooting

| Problem | Fix |
|---|---|
| `503` on Analyze | The fine-tuned model was not found. Check that `models/flan-t5-supportgen/` exists, or the `SUPPORTGEN_MODEL_DIR` path in `.env`. |
| The first analysis is very slow | Normal: the summary and sentiment models (about 1.5 GB) download once. Later requests take about 2 s. |
| Database connection error | Set `USE_POSTGRES=False` in `.env`, or start PostgreSQL and run `createdb supportgen`. |
| The frontend shows network errors | The backend is not running on port 8000. Start it first. |
| `pip install` fails on torch or tokenizers | Use Python 3.12, not 3.13 or newer. |
| No Gemini column on Comparison | Expected without `GEMINI_API_KEY`; Gemini is optional. |

---

## 9. How to demo it to someone

A 5–7 minute walkthrough. **Before the audience arrives**, start both servers and analyse one ticket so the models are already loaded and downloaded.

**1. The problem and the pipeline (1 min).** Open **How it works**. "Support agents answer hundreds of similar tickets. SupportGen reads a ticket and drafts the reply, tags it and suggests what to do next, so the agent only reviews and edits." Walk through the four boxes (Data → Training → Backend → Website) and the before/after reply.

**2. Analyze a ticket (2 min).** Open **Analyze ticket** and paste:

> **Subject:** Charged twice for my subscription
> **Body:** Hello, I was charged twice for my monthly subscription this month. Please refund the duplicate payment as soon as possible. My account email is the one I am writing from.

Point out the draft reply, the predicted category (*Billing and Payments*), priority, type, sentiment, summary and recommended action, and the confidence score next to each prediction. Then try a technical one:

> **Subject:** Server keeps crashing after update
> **Body:** Since installing the latest update yesterday, our server crashes every few hours and the dashboard shows a memory error. This is affecting all our users. Please help urgently.

**3. Show the effect of training (2 min).** Open **Model comparison** and run the same ticket. The **base model** usually returns a short or off-topic sentence; the **fine-tuned model** writes a proper support reply. "Same model, same size; the only difference is our fine-tuning."

**4. Show the evidence (1.5 min).** Open **Model evaluation**: the loss curves (training worked and did not overfit), ROUGE-L more than doubling (0.158 → 0.343), BLEU 1.2 → 19.6, the LLM judge preferring the fine-tuned reply on 50 of 50 tickets, and the "Is a macro-F1 of 0.59 too low?" panel under the classifiers.

**5. Be honest about the limits (30 s).** Replies are formulaic (they often offer a call), there are occasional factual slips, and the data is synthetic. That is why a human reviews every draft.

**6. Wrap up (30 s).** Open the **Dashboard** and **Ticket history** to show that every analysed ticket is stored and summarised.

---

## 10. Project structure

```
.
├── README.md                    this file
├── viva.md                      viva questions and answers
├── .env.example                 configuration template (copy to .env)
├── requirements-backend.txt     Python dependencies for the backend
├── notebooks/                   Colab notebooks 01–06 + README_COLAB.md (all ML work)
├── models/                      trained models from the Colab export (gitignored)
├── results/                     metrics, plots and example outputs from the Colab export
│   ├── gen_metrics.json             reply-generation metrics + significance tests
│   ├── cls_metrics.json             classifier metrics
│   ├── llm_judge_*                  LLM-as-a-judge rubric, ratings and summary
│   └── plots/                       loss curves, confusion matrices, box plots
├── scripts/
│   └── llm_judge_summary.py     aggregates the LLM-judge ratings
├── backend/
│   ├── supportgen_api/          Django project settings and URLs
│   ├── ai/                      AI services (no Django imports)
│   │   ├── services.py              TicketAnalyzer: runs all five steps
│   │   ├── providers.py             FLAN-T5 (local) and Gemini providers
│   │   ├── classify.py              TF-IDF + SVM classifiers
│   │   ├── sentiment.py             pretrained sentiment model
│   │   ├── assist.py                summary fallback and rule-based actions
│   │   ├── postprocess.py           placeholder repair
│   │   └── registry.py              loads each model once per process
│   └── tickets/                 Django app: model, API views, serializers, tests
└── frontend/
    └── src/
        ├── pages/                   Dashboard, HowItWorks, Analyze, History, TicketDetail, Compare, Evaluation
        ├── components/              result card, charts, UI pieces
        └── api.js                   calls to the backend
```

---

## 11. Tech stack

| Layer | Tools |
|---|---|
| ML / training | Python, PyTorch, Hugging Face Transformers, Datasets, scikit-learn, Google Colab (T4 GPU) |
| Evaluation | ROUGE, BLEU (sacreBLEU), BERTScore (roberta-large), sentence-transformers (all-MiniLM-L6-v2), SciPy (Wilcoxon), bootstrap CIs, LLM-as-a-judge |
| Backend | Django, Django REST Framework, PostgreSQL / SQLite |
| Frontend | React, Vite |
| Optional benchmark | Google Gemini (`gemini-2.5-flash`) |

---

## 12. License and credits

- **Dataset:** [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets), CC BY-NC 4.0. Because of the dataset license, the trained models and this project are for **non-commercial, educational use only**.
- **Models:** [google/flan-t5-small](https://huggingface.co/google/flan-t5-small), [google/flan-t5-base](https://huggingface.co/google/flan-t5-base), [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
- Built as the course project for **Generative AI**, 7th semester.

**Author:** Mohit Singh
