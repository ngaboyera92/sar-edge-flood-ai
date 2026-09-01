import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def ratio(n, d):
    return n / d if d else float("nan")


def metrics(tp, fp, fn):
    return {
        "iou": ratio(tp, tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--repo-root", default="/content/sar-edge-flood-ai-fw02b")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    matrix = json.loads((repo_root / "experiments" / "fw02b_kuro_multiseed_matrix.json").read_text())
    parent = project_root / matrix["stage_b_parent_output"]
    expected_shards = int(matrix["dataset"]["expected_shards"])
    expected_samples = int(matrix["dataset"]["expected_samples"])
    expected_events = int(matrix["dataset"]["expected_events"])

    shard_dirs = [parent / f"shard-{i:05d}" for i in range(expected_shards)]
    missing = [str(p) for p in shard_dirs if not (p / "COMPLETE.json").exists()]
    if missing:
        raise RuntimeError("Stage B is incomplete; missing COMPLETE.json for:\n" + "\n".join(missing))

    completes = [json.loads((p / "COMPLETE.json").read_text()) for p in shard_dirs]
    if sum(int(x["samples"]) for x in completes) != expected_samples:
        raise RuntimeError("Total sample count does not match predeclared 67,490 tiles")

    overall_parts = [pd.read_csv(p / "overall_metrics_by_model.csv") for p in shard_dirs]
    event_parts = [pd.read_csv(p / "event_metrics_by_model.csv") for p in shard_dirs]
    overall = pd.concat(overall_parts, ignore_index=True)
    events = pd.concat(event_parts, ignore_index=True)

    keys = ["run_name", "seed", "beta", "target"]
    counts = overall.groupby(keys, as_index=False)[["tp", "fp", "fn", "tn"]].sum()
    rows = []
    for r in counts.itertuples(index=False):
        d = r._asdict()
        rows.append({**d, **metrics(d["tp"], d["fp"], d["fn"])})
    overall_final = pd.DataFrame(rows)

    ekeys = keys + ["event_id"]
    event_counts = events.groupby(ekeys, as_index=False)[["tp", "fp", "fn", "tn"]].sum()
    erows = []
    for r in event_counts.itertuples(index=False):
        d = r._asdict()
        erows.append({**d, **metrics(d["tp"], d["fp"], d["fn"])})
    event_final = pd.DataFrame(erows)
    if event_final["event_id"].nunique() != expected_events:
        raise RuntimeError(f"Expected {expected_events} unique events, found {event_final['event_id'].nunique()}")

    macro = (event_final.groupby(keys, as_index=False)
             .agg(events=("event_id", "nunique"), event_macro_iou=("iou", "mean"),
                  event_macro_precision=("precision", "mean"), event_macro_recall=("recall", "mean"),
                  event_macro_f1=("f1", "mean")))

    merged = overall_final.merge(macro, on=keys, validate="one_to_one")
    metric_cols = ["iou", "precision", "recall", "f1", "event_macro_iou", "event_macro_precision", "event_macro_recall", "event_macro_f1"]
    condition_rows = []
    for (beta, target), g in merged.groupby(["beta", "target"]):
        row = {"beta": beta, "target": target, "seeds": int(g["seed"].nunique())}
        for c in metric_cols:
            row[c + "_mean"] = float(g[c].mean())
            row[c + "_sd"] = float(g[c].std(ddof=1))
        condition_rows.append(row)
    condition = pd.DataFrame(condition_rows)

    paired_rows = []
    for target in sorted(merged["target"].unique()):
        a = merged[(merged.beta == 0.0) & (merged.target == target)].set_index("seed")
        b = merged[(merged.beta == 0.3) & (merged.target == target)].set_index("seed")
        if sorted(a.index.tolist()) != [0, 1, 2, 3, 4] or sorted(b.index.tolist()) != [0, 1, 2, 3, 4]:
            raise RuntimeError(f"Missing paired seeds for {target}")
        for seed in range(5):
            row = {"target": target, "seed": seed}
            for c in metric_cols:
                row["delta_" + c] = float(b.loc[seed, c] - a.loc[seed, c])
            paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)

    paired_summary_rows = []
    for target, g in paired.groupby("target"):
        row = {"target": target, "pairs": len(g)}
        for c in metric_cols:
            s = g["delta_" + c]
            row["delta_" + c + "_mean"] = float(s.mean())
            row["delta_" + c + "_sd"] = float(s.std(ddof=1))
            row["beta03_wins_" + c] = int((s > 0).sum())
        paired_summary_rows.append(row)
    paired_summary = pd.DataFrame(paired_summary_rows)

    parent.mkdir(parents=True, exist_ok=True)
    overall_final.to_csv(parent / "01_overall_metrics_all_10_models.csv", index=False)
    event_final.to_csv(parent / "02_event_metrics_all_10_models.csv", index=False)
    macro.to_csv(parent / "03_event_macro_metrics_all_10_models.csv", index=False)
    condition.to_csv(parent / "04_condition_mean_sd_across_seeds.csv", index=False)
    paired.to_csv(parent / "05_paired_seed_differences_beta03_minus_beta00.csv", index=False)
    paired_summary.to_csv(parent / "06_paired_difference_summary.csv", index=False)

    summary = {
        "completed": True,
        "samples": expected_samples,
        "events": expected_events,
        "models": int(merged["run_name"].nunique()),
        "targets": sorted(merged["target"].unique().tolist()),
        "condition_mean_sd": condition.to_dict(orient="records"),
        "paired_difference_summary": paired_summary.to_dict(orient="records"),
    }
    (parent / "FW-02B_final_summary.json").write_text(json.dumps(summary, indent=2))
    hashes = {}
    for name in [
        "01_overall_metrics_all_10_models.csv", "02_event_metrics_all_10_models.csv",
        "03_event_macro_metrics_all_10_models.csv", "04_condition_mean_sd_across_seeds.csv",
        "05_paired_seed_differences_beta03_minus_beta00.csv", "06_paired_difference_summary.csv",
        "FW-02B_final_summary.json",
    ]:
        hashes[name] = sha256_file(parent / name)
    (parent / "SHA256SUMS_final.json").write_text(json.dumps(hashes, indent=2))
    print("FW-02B AGGREGATION COMPLETE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
