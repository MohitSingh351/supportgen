# Running the SupportGen ML pipeline in Google Colab

All ML work (EDA, cleaning, splitting, baseline, fine-tuning, evaluation, classifiers, export) runs
in these notebooks on Google Colab. The Mac only runs the finished app.

## How to open a notebook
1. Go to <https://colab.research.google.com> → **File → Upload notebook** → pick the `.ipynb`.
   (Or put the `notebooks/` folder in Google Drive and open each file with *Google Colaboratory*.)
2. Set the runtime type: **Runtime → Change runtime type**.
3. Run all the cells from top to bottom (**Runtime → Run all**). The first cell asks for permission to mount Google Drive. Allow it.

## Run order

| # | Notebook | Runtime | What it produces (in `MyDrive/SupportGen/`) |
|---|---|---|---|
| 01 | `01_dataset_preparation.ipynb` | CPU | `supportgen_common.py`, `data/{train,val,test}.parquet`, `eval_subset_ids.json`, `config.json`, `eda_stats.json`, plots |
| 02 | `02_baseline_flan_t5.ipynb` | GPU (T4) | `results/outputs_A_*.csv`, `results/metrics_A_*.json` (Model A, no fine-tuning) |
| 03 | `03_train_flan_t5.ipynb` | GPU (T4) | `models/flan-t5-supportgen/` (Model B), loss curves, `experiments.csv` |
| 04 | `04_evaluate_model.ipynb` | GPU (T4) | `results/gen_metrics.json`, significance tests, efficiency, human-eval sheet, summary/action examples |
| 05 | `05_classification_models.ipynb` | CPU | `models/classifiers/*.joblib`, `results/cls_metrics.json`, confusion matrices |
| 06 | `06_model_comparison.ipynb` | GPU optional | `results/comparison_metrics.json`, **`exports/supportgen_models.zip`** |

Notebook 01 must run first because it writes the shared helper module and the data splits. You can run 05 any time after 01.

**Tip:** in notebook 03, set `QUICK_RUN = True` first. That does a short pipeline check on 1,000 examples.
Then set it back to `False` for the real training run.

## What's in Drive
```
MyDrive/SupportGen/
  supportgen_common.py          shared helpers (written by notebook 01)
  data/                         splits, config.json, eda_stats.json
  models/flan-t5-supportgen/    fine-tuned model (Model B)
  models/classifiers/           TF-IDF classifiers
  results/                      outputs, metrics, plots, human-eval sheets
  experiments.csv               one row per model run
  exports/supportgen_models.zip what you download to your Mac
```

## Optional extras
- **Model D (FLAN-T5-base):** in notebook 03, set `MODEL_NAME = "google/flan-t5-base"`. In notebook 04,
  switch `FT_RUN/FT_DIR` to `sg.RUN_FT_BASE / sg.FINETUNED_BASE_DIR`. An L4 or A100 runtime is faster for this.
- **Gemini benchmark (Model C):** add a Colab secret named `GEMINI_API_KEY` (🔑 icon in the sidebar). Without it,
  notebook 06 skips Gemini.

## Moving the model to the Mac
```bash
cd "<path-to>/project"
unzip -o ~/Downloads/supportgen_models.zip -d .
```
This creates `project/models/flan-t5-supportgen/`, `project/models/classifiers/` and `project/results/`.
The Django backend loads them from the paths set in `.env` (see `.env.example`).
