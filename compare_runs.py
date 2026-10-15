import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--baseline", nargs="+")
    p.add_argument("--output", default="outputs/seed_summary.json")
    p.add_argument("--holm-p", type=float, nargs="+")
    args = p.parse_args()
    subsets = [set(json.loads((Path(r) / "selected.json").read_text())) for r in args.runs]
    metrics = [json.loads((Path(r) / "metrics.json").read_text()) for r in args.runs]
    result = dict(pairwise_jaccard=float(np.mean([len(a & b) / max(1, len(a | b)) for a, b in combinations(subsets, 2)])) if len(subsets) > 1 else None)
    for key in ("macro_f1", "worst_client_macro_f1", "attack_target_selection", "malicious_detection_auroc", "honest_extreme_fpr", "server_seconds", "upload_bytes"):
        values = [m[key] for m in metrics if m.get(key) is not None]
        result[key] = dict(mean=float(np.mean(values)), sample_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None) if values else None
    if args.baseline:
        if len(args.baseline) != len(args.runs):
            raise ValueError("Pass one matched baseline directory per run, in the same seed order")
        baseline = [json.loads((Path(r) / "metrics.json").read_text()) for r in args.baseline]
        difference = np.array([m["worst_client_macro_f1"] - b["worst_client_macro_f1"] for m, b in zip(metrics, baseline)])
        if np.any(difference != 0):
            result["paired_wilcoxon_p"] = float(wilcoxon(difference).pvalue)
            from scipy.stats import rankdata
            nonzero = difference[difference != 0]
            ranks = rankdata(np.abs(nonzero))
            result["matched_rank_biserial"] = float(np.sum(ranks * np.sign(nonzero)) / ranks.sum())
        else:
            result["paired_wilcoxon_p"] = 1.
            result["matched_rank_biserial"] = 0.
    if args.holm_p:
        values = np.asarray(args.holm_p)
        order = np.argsort(values)
        adjusted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order]).clip(0, 1)
        corrected = np.empty_like(adjusted)
        corrected[order] = adjusted
        result["holm_corrected_p_in_input_order"] = corrected.tolist()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
