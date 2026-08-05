import os
import numpy as np
from collect.dataset import TimingDataset
from analysis.mutual_info import compute_mi_total, compute_mi_per_op, compute_mi_bootstrap
from analysis.correlation import (timing_kruskal_wallis, timing_anova,
                                   timing_spearman, timing_cohens_d,
                                   timing_summary_stats)
from analysis.classifier import train_timing_classifier, evaluate_classifier
from analysis.leakage_report import generate_report, format_report_table
from analysis.plots import (plot_timing_distributions, plot_confusion_matrix,
                              plot_per_layer_breakdown, plot_mi_heatmap)

DATASET_PATHS = [
    "data/cartpole_fp32.npz",
    "data/cartpole_int8.npz",
    "data/grid_nav_fp32.npz",
    "data/grid_nav_int8.npz",
]

FIGURE_DIR = "figures"
N_BOOTSTRAP = 100


def analyze_single(dataset_path: str) -> dict:
    label = os.path.splitext(os.path.basename(dataset_path))[0]
    ds = TimingDataset.load(dataset_path)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  samples={len(ds)}, obs_dim={ds.obs_dim}, act_dim={ds.act_dim}")
    print(f"{'='*60}")

    stats = timing_summary_stats(ds)
    print("\n  Per-Action Timing Stats:")
    print(f"  {'Action':<12} {'Mean':>12} {'Std':>12} {'Median':>12} {'CV':>8} {'Count':>8}")
    print(f"  {'-'*64}")
    for key in sorted(stats.keys()):
        s = stats[key]
        print(f"  {key:<12} {s['mean']:>12.1f} {s['std']:>12.1f} "
              f"{s['median']:>12.1f} {s['cv']:>8.4f} {s['count']:>8d}")

    mi = compute_mi_total(ds)
    mi_boot = compute_mi_bootstrap(ds, n_bootstrap=N_BOOTSTRAP)
    print(f"\n  Mutual Information:")
    print(f"    MI = {mi['mi_bits']:.4f} bits  "
          f"(max possible = {mi['max_possible_bits']:.4f} bits)")
    print(f"    95% CI = [{mi_boot['ci_low']:.4f}, {mi_boot['ci_high']:.4f}]")
    print(f"    Leakage ratio = {mi['mi_bits'] / mi['max_possible_bits'] * 100:.1f}%")

    mi_ops = compute_mi_per_op(ds)
    active_ops = {k: v for k, v in mi_ops.items() if v > 0.001}
    if active_ops:
        print(f"\n  Per-Operator MI (bits):")
        for op_idx in sorted(active_ops.keys()):
            bar = "#" * int(active_ops[op_idx] * 20)
            print(f"    Op {op_idx}: {active_ops[op_idx]:.4f}  {bar}")

    kw = timing_kruskal_wallis(ds)
    anova = timing_anova(ds)
    print(f"\n  Hypothesis Tests:")
    print(f"    Kruskal-Wallis: H={kw['statistic']:.2f}, p={kw['p_value']:.2e}")
    print(f"    ANOVA:          F={anova['f_statistic']:.2f}, p={anova['p_value']:.2e}")

    cohens = timing_cohens_d(ds)
    if cohens:
        max_pair = max(cohens.items(), key=lambda x: abs(x[1]))
        print(f"    Max Cohen's d:  {max_pair[0]} = {max_pair[1]:.4f}")

    spearman = timing_spearman(ds)
    sig_dims = {k: v for k, v in spearman.items() if v["significant"]}
    if sig_dims:
        print(f"\n  Significant Obs-Timing Correlations (Spearman):")
        for dim, v in sorted(sig_dims.items()):
            print(f"    {dim}: r={v['correlation']:.4f}, p={v['p_value']:.2e}")

    rf_result = train_timing_classifier(ds, model_type="rf")
    rf_eval = evaluate_classifier(rf_result)
    mlp_result = train_timing_classifier(ds, model_type="mlp")
    mlp_eval = evaluate_classifier(mlp_result)

    print(f"\n  Classifiers (timing -> action):")
    print(f"    Random baseline: {rf_eval['random_baseline']:.3f}")
    print(f"    RF:  acc={rf_eval['accuracy']:.3f}, "
          f"F1={rf_eval['f1_macro']:.3f}, "
          f"lift={rf_eval['lift_over_random']:.2f}x")
    print(f"    MLP: acc={mlp_eval['accuracy']:.3f}, "
          f"F1={mlp_eval['f1_macro']:.3f}, "
          f"lift={mlp_eval['lift_over_random']:.2f}x")

    vuln = "VULNERABLE" if rf_eval["accuracy"] > rf_eval["random_baseline"] * 1.5 else "LOW RISK"
    print(f"\n  >>> VERDICT: {vuln} <<<")

    os.makedirs(FIGURE_DIR, exist_ok=True)
    plot_timing_distributions(
        ds, title=f"Timing Distribution: {label}",
        save_path=os.path.join(FIGURE_DIR, f"{label}_violin.png")).clear()
    plot_confusion_matrix(
        np.array(rf_result["confusion_matrix"]),
        title=f"RF Confusion: {label}",
        save_path=os.path.join(FIGURE_DIR, f"{label}_confusion.png")).clear()
    plot_per_layer_breakdown(
        ds, n_ops=5, title=f"Per-Op: {label}",
        save_path=os.path.join(FIGURE_DIR, f"{label}_layers.png")).clear()

    report = generate_report(ds, label=label)
    return report


def run_analysis():
    reports = []
    for path in DATASET_PATHS:
        if not os.path.exists(path):
            print(f"\n  SKIPPED: {path} (not found)")
            continue
        report = analyze_single(path)
        reports.append(report)

    if len(reports) > 1:
        print(f"\n{'='*60}")
        print("  COMPARATIVE SUMMARY")
        print(f"{'='*60}\n")
        print(format_report_table(reports))

        mi_data = {}
        for r in reports:
            mi_data[r["label"]] = {
                "MI (bits)": r["mutual_information"]["mi_bits"],
                "RF Acc": r["rf_classifier"]["accuracy"],
                "MLP Acc": r["mlp_classifier"]["accuracy"],
            }
        plot_mi_heatmap(
            mi_data, title="Leakage Comparison",
            save_path=os.path.join(FIGURE_DIR, "comparison_heatmap.png")).clear()

    print(f"\nFigures saved to: {os.path.abspath(FIGURE_DIR)}/")
    return reports


if __name__ == "__main__":
    run_analysis()
