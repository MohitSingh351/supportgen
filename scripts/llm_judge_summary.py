"""Unblinds the LLM-as-a-judge ratings and summarises them.

Input : results/llm_judge_ratings_blind.csv  (ratings made without the key)
        results/human_eval_key.csv           (X/Y → model, written by notebook 04)
Output: results/llm_judge_ratings.csv, results/llm_judge_summary.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R = Path(__file__).resolve().parents[1] / "results"
CRITERIA = ["relevance", "correctness", "fluency", "professional_tone", "helpfulness"]
NAMES = {"A_base": "A: base FLAN-T5-small", "B_finetuned": "B: fine-tuned FLAN-T5-small"}

blind = pd.read_csv(R / "llm_judge_ratings_blind.csv")
key = pd.read_csv(R / "human_eval_key.csv")
assert len(blind) == len(key) == 100 and not blind.duplicated(["item", "response_label"]).any()
df = blind.merge(key, on=["item", "response_label"], validate="one_to_one")
df["overall"] = df[CRITERIA].mean(axis=1)
df.to_csv(R / "llm_judge_ratings.csv", index=False)

wide = df.pivot(index="item", columns="system", values="overall")
summary = {
    "judge": "Claude (claude-opus-5-5), LLM-as-a-judge — not a human evaluation",
    "items": int(df["item"].nunique()),
    "protocol": "see results/llm_judge_rubric.md",
    "mean_scores": {NAMES[s]: {c: round(float(g[c].mean()), 2) for c in CRITERIA + ["overall"]}
                    for s, g in df.groupby("system")},
    "std_scores": {NAMES[s]: {c: round(float(g[c].std()), 2) for c in CRITERIA + ["overall"]}
                   for s, g in df.groupby("system")},
    "pairwise": {
        "finetuned_better": int((wide.B_finetuned > wide.A_base).sum()),
        "base_better": int((wide.A_base > wide.B_finetuned).sum()),
        "ties": int((wide.A_base == wide.B_finetuned).sum()),
        "wilcoxon_p_overall": float(wilcoxon(wide.B_finetuned, wide.A_base).pvalue),
    },
    "failure_counts": {
        NAMES[s]: {
            "not_a_support_reply (tone = 1)": int((g.professional_tone == 1).sum()),
            "correctness <= 2 (wrong, invented or no reply)": int((g.correctness <= 2).sum()),
            "helpfulness >= 4": int((g.helpfulness >= 4).sum()),
        } for s, g in df.groupby("system")
    },
}
# fine-tuned replies flagged for invented facts / contradictions
ft = df[(df.system == "B_finetuned") & (df.correctness <= 2)]
summary["finetuned_low_correctness_examples"] = ft[["item", "correctness", "comment"]].to_dict(orient="records")
(R / "llm_judge_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
