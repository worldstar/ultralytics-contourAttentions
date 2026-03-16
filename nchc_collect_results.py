"""
Collect results from all NCHC experiment runs and generate summary tables.

Usage (on NCHC after all jobs complete):
  python nchc_collect_results.py
"""

import csv
import json
from pathlib import Path

import numpy as np

EXPERIMENTS = {
    "baseline": "ultralytics/cfg/models/v9/yolov9c.yaml",
    "contourcbam": "ultralytics/cfg/models/v9/yolov9c-contourcbam.yaml",
    "stdcbam": "ultralytics/cfg/models/v9/yolov9c-stdcbam.yaml",
    "se": "ultralytics/cfg/models/v9/yolov9c-se.yaml",
    "eca": "ultralytics/cfg/models/v9/yolov9c-eca.yaml",
    "simam": "ultralytics/cfg/models/v9/yolov9c-simam.yaml",
    "coordatt": "ultralytics/cfg/models/v9/yolov9c-coordatt.yaml",
    "gam": "ultralytics/cfg/models/v9/yolov9c-gam.yaml",
}
SEEDS = [0, 1, 2, 3, 4]


def parse_test_log(log_dir):
    """Parse test validation results from the test run directory."""
    # Try to read from the val results
    results_csv = log_dir / "results.csv"
    if not results_csv.exists():
        return {}

    rows = []
    with open(results_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})

    if not rows:
        return {}

    last = rows[-1]
    result = {}
    for k, v in last.items():
        try:
            result[k] = float(v)
        except (ValueError, TypeError):
            result[k] = v
    return result


def collect_all():
    """Collect metrics from all experiment runs."""
    all_results = {}

    for exp_name in EXPERIMENTS:
        for seed in SEEDS:
            run_name = f"{exp_name}_seed{seed}"
            run_dir = Path(f"runs/detect/{run_name}")
            test_dir = Path(f"runs/detect/{run_name}_test")

            if not run_dir.exists():
                print(f"MISSING: {run_name}")
                continue

            # Get training results
            train_csv = run_dir / "results.csv"
            train_metrics = {}
            if train_csv.exists():
                rows = []
                with open(train_csv) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rows.append({k.strip(): v.strip() for k, v in row.items()})
                if rows:
                    last = rows[-1]
                    for k, v in last.items():
                        try:
                            train_metrics[k] = float(v)
                        except (ValueError, TypeError):
                            train_metrics[k] = v

            # Get test metrics from Slurm log output (parse stdout)
            test_metrics = {}
            # Check if test validation directory exists with results
            if test_dir.exists():
                test_metrics = parse_test_log(test_dir)

            # Also try to parse from the Slurm log files
            # Look for test metrics printed by our script
            best_weights = run_dir / "weights" / "best.pt"

            all_results[run_name] = {
                "exp_name": exp_name,
                "seed": seed,
                "run_name": run_name,
                "val_metrics": train_metrics,
                "test_metrics": test_metrics,
                "best_weights": str(best_weights) if best_weights.exists() else None,
            }
            print(f"OK: {run_name}")

    return all_results


def generate_summary(all_results):
    """Generate mean +/- std summary tables."""
    summary = {}

    for exp_name in EXPERIMENTS:
        runs = [all_results[f"{exp_name}_seed{s}"] for s in SEEDS if f"{exp_name}_seed{s}" in all_results]
        if not runs:
            continue

        # Collect test metrics across seeds
        metrics_keys = set()
        for r in runs:
            metrics_keys.update(r.get("test_metrics", {}).keys())

        exp_summary = {}
        for key in sorted(metrics_keys):
            values = [
                r["test_metrics"][key]
                for r in runs
                if key in r.get("test_metrics", {}) and isinstance(r["test_metrics"][key], (int, float))
            ]
            if values:
                exp_summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "values": values,
                }
        summary[exp_name] = exp_summary

    return summary


def print_tables(summary):
    """Print formatted result tables."""
    key_metrics = [
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
    ]

    print("\n" + "=" * 100)
    print("  RESULTS SUMMARY (Validation Set, mean +/- std over 5 seeds)")
    print("=" * 100)

    print(f"\n{'Model':<15}", end="")
    for _, label in key_metrics:
        print(f" {label:>18}", end="")
    print()
    print("-" * (15 + 18 * len(key_metrics)))

    for exp_name in EXPERIMENTS:
        if exp_name not in summary:
            continue
        s = summary[exp_name]
        print(f"{exp_name:<15}", end="")
        for key, _ in key_metrics:
            if key in s:
                m = s[key]["mean"]
                sd = s[key]["std"]
                print(f" {m:.4f}+/-{sd:.4f} ", end="")
            else:
                print(f"{'N/A':>18}", end="")
        print()

    # Significance tests
    print("\n" + "=" * 100)
    print("  PAIRED t-TESTS (baseline vs. each)")
    print("=" * 100)

    from scipy import stats

    baseline_key = "baseline"
    if baseline_key not in summary:
        print("No baseline results for significance tests.")
        return

    for exp_name in EXPERIMENTS:
        if exp_name == baseline_key or exp_name not in summary:
            continue
        print(f"\n--- {exp_name} vs baseline ---")
        for key, label in key_metrics:
            if key in summary[baseline_key] and key in summary[exp_name]:
                base_vals = summary[baseline_key][key]["values"]
                exp_vals = summary[exp_name][key]["values"]
                if len(base_vals) == len(exp_vals) and len(base_vals) > 1:
                    t_stat, p_val = stats.ttest_rel(exp_vals, base_vals)
                    sig = "*" if p_val < 0.05 else ""
                    sig = "**" if p_val < 0.01 else sig
                    base_m = np.mean(base_vals)
                    exp_m = np.mean(exp_vals)
                    delta = exp_m - base_m
                    print(f"  {label:<18}: delta={delta:+.4f}, t={t_stat:.3f}, p={p_val:.4f} {sig}")


if __name__ == "__main__":
    out_dir = Path("experiment_results")
    out_dir.mkdir(exist_ok=True)

    print("Collecting results from runs/detect/...")
    all_results = collect_all()

    # Save raw results
    results_file = out_dir / "all_results.json"
    results_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nRaw results saved to {results_file}")

    # Generate and save summary
    summary = generate_summary(all_results)
    summary_file = out_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"Summary saved to {summary_file}")

    # Print tables
    print_tables(summary)
