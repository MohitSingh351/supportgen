"""Reads the evaluation results exported from Colab (results/ folder) for the Evaluation page.

Only reads files — nothing is recomputed or altered here, so the page always shows the
actual numbers produced by the notebooks.
"""
import csv
import json
from pathlib import Path

PLOT_FILES = {
    "loss_curves": "loss_curves_B_ft_flan-t5-small.png",
    "metric_boxplots": "metric_boxplots_A_vs_B.png",
    "cm_queue_group": "cm_queue_group.png",
    "cm_priority": "cm_priority.png",
    "cm_type": "cm_type.png",
    "label_distributions": "label_distributions_raw.png",
    "token_lengths": "token_lengths.png",
}


def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _csv(path: Path, limit: int | None = None):
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def _drop_per_class(cls_metrics: dict | None):
    """Keep per-class F1 only (the full report is large)."""
    if not cls_metrics:
        return cls_metrics
    out = {}
    for target, m in cls_metrics.items():
        m = dict(m)
        if "per_class" in m:
            m["per_class_f1"] = {k: v["f1-score"] for k, v in m.pop("per_class").items()
                                 if isinstance(v, dict) and k not in ("macro avg", "weighted avg")}
        out[target] = m
    return out


def load_evaluation(results_dir: Path, model_dir: Path) -> dict:
    gen = _json(results_dir / "gen_metrics.json")
    if gen:
        gen.pop("fine_tuned_model_dir", None)  # Colab path, meaningless locally
    plots_dir = results_dir / "plots"
    return {
        "available": results_dir.exists(),
        "training_config": _json(model_dir / "training_config.json"),
        "dataset": {
            "config": _json(results_dir / "data" / "config.json"),
            "eda": _json(results_dir / "data" / "eda_stats.json"),
            "split_sizes": _json(results_dir / "data" / "split_sizes.json"),
        },
        "generation": gen,
        "comparison": _json(results_dir / "comparison_metrics.json"),
        "classification": _drop_per_class(_json(results_dir / "cls_metrics.json")),
        "human_eval": _json(results_dir / "human_eval_summary.json"),
        "llm_judge": _json(results_dir / "llm_judge_summary.json"),
        "experiments": _csv(results_dir / "experiments.csv"),
        "train_log": _csv(results_dir / "train_log_B_ft_flan-t5-small.csv"),
        "qualitative_examples": _csv(results_dir / "qualitative_examples_A_vs_B.csv"),
        "comparison_examples": _csv(results_dir / "comparison_examples.csv", limit=10),
        "summary_action_examples": _csv(results_dir / "summary_action_examples.csv"),
        "plots": [key for key, name in PLOT_FILES.items() if (plots_dir / name).exists()],
    }
