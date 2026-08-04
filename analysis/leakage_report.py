import numpy as np
from typing import Dict, List
from collect.dataset import TimingDataset
from analysis.mutual_info import compute_mi_total, compute_mi_bootstrap
from analysis.correlation import (timing_kruskal_wallis, timing_anova,
                                   timing_cohens_d, timing_summary_stats)
from analysis.classifier import train_timing_classifier, evaluate_classifier


def generate_report(dataset: TimingDataset, label: str = "") -> Dict:
    mi_result = compute_mi_total(dataset)
    mi_boot = compute_mi_bootstrap(dataset, n_bootstrap=50)
    kw_result = timing_kruskal_wallis(dataset)
    anova_result = timing_anova(dataset)
    cohens = timing_cohens_d(dataset)
    stats = timing_summary_stats(dataset)

    rf_result = train_timing_classifier(dataset, model_type="rf")
    rf_eval = evaluate_classifier(rf_result)

    mlp_result = train_timing_classifier(dataset, model_type="mlp")
    mlp_eval = evaluate_classifier(mlp_result)

    return {
        "label": label,
        "n_samples": len(dataset),
        "n_actions": dataset.act_dim,
        "mutual_information": mi_result,
        "mi_bootstrap": mi_boot,
        "kruskal_wallis": kw_result,
        "anova": anova_result,
        "cohens_d": cohens,
        "timing_stats": stats,
        "rf_classifier": rf_eval,
        "mlp_classifier": mlp_eval,
        "rf_confusion_matrix": rf_result["confusion_matrix"].tolist(),
        "mlp_confusion_matrix": mlp_result["confusion_matrix"].tolist(),
    }


def format_report_table(reports: List[Dict]) -> str:
    header = (
        f"{'Label':<25} {'MI(bits)':<10} {'MI CI':<20} "
        f"{'KW p-val':<12} {'RF Acc':<10} {'MLP Acc':<10} "
        f"{'Random':<10} {'Max Cohen d':<12}"
    )
    lines = [header, "-" * len(header)]
    for r in reports:
        mi = r["mutual_information"]["mi_bits"]
        ci_lo = r["mi_bootstrap"]["ci_low"]
        ci_hi = r["mi_bootstrap"]["ci_high"]
        kw_p = r["kruskal_wallis"]["p_value"]
        rf_acc = r["rf_classifier"]["accuracy"]
        mlp_acc = r["mlp_classifier"]["accuracy"]
        rand = r["rf_classifier"]["random_baseline"]
        cohens = r.get("cohens_d", {})
        max_d = max(abs(v) for v in cohens.values()) if cohens else 0.0
        line = (
            f"{r['label']:<25} {mi:<10.4f} [{ci_lo:.3f}, {ci_hi:.3f}]"
            f"{'':>2} {kw_p:<12.2e} {rf_acc:<10.3f} {mlp_acc:<10.3f} "
            f"{rand:<10.3f} {max_d:<12.4f}"
        )
        lines.append(line)
    return "\n".join(lines)
