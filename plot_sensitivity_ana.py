"""
plot_sensitivity.py
--------------------
Produces one comparison figure per swept parameter: x-axis = parameter
value, showing best-fitness and held-out success rate (mean +/- std
across seeds), with the baseline setting marked distinctly.

Run this after sensitivity_analysis.py has produced
sensitivity_analysis_results.json.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("sensitivity_analysis_results.json") as f:
    all_results = json.load(f)

for param_name, results in all_results.items():
    results = sorted(results, key=lambda r: r["value"])
    values = [r["value"] for r in results]
    is_baseline = [r["is_baseline"] for r in results]

    fit_mean = np.array([r["best_fitness_mean"] for r in results])
    fit_std = np.array([r["best_fitness_std"] for r in results])
    hold_mean = np.array([r["holdout_success_mean"] * 100 for r in results])
    hold_std = np.array([r["holdout_success_std"] * 100 for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(values))
    colors = ["#c0392b" if b else "#3f51b5" for b in is_baseline]

    ax1.errorbar(x, fit_mean, yerr=fit_std, fmt="o-", color="#3f51b5",
                 ecolor="#3f51b5", capsize=4, linewidth=1.5)
    for xi, c, b in zip(x, colors, is_baseline):
        if b:
            ax1.scatter([xi], [fit_mean[xi]], color=c, s=120, zorder=5, label="baseline")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(v) for v in values])
    ax1.set_xlabel(param_name)
    ax1.set_ylabel("Best training fitness")
    ax1.set_title(f"Best fitness vs {param_name}")
    ax1.grid(alpha=0.3)
    if any(is_baseline):
        ax1.legend()

    ax2.errorbar(x, hold_mean, yerr=hold_std, fmt="o-", color="#2a9d3f",
                 ecolor="#2a9d3f", capsize=4, linewidth=1.5)
    for xi, c, b in zip(x, colors, is_baseline):
        if b:
            ax2.scatter([xi], [hold_mean[xi]], color=c, s=120, zorder=5, label="baseline")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(v) for v in values])
    ax2.set_xlabel(param_name)
    ax2.set_ylabel("Held-out success rate (%)")
    ax2.set_title(f"Generalization vs {param_name}")
    ax2.set_ylim(-5, 105)
    ax2.grid(alpha=0.3)
    if any(is_baseline):
        ax2.legend()

    fig.suptitle(f"Sensitivity analysis: {param_name}")
    fig.tight_layout()
    out_path = f"sensitivity_{param_name}.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")