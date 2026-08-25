"""
Sensitivity_analysis.py
-------------------------------------

Sweeps one hyperparameter at a time (holding all others at baseline),
running multiple seeds per setting via multi_seed_run.run_multi_seed,
and records the same metrics used for the baseline: best training
fitness, generations-to-threshold, and held-out success rate.

Runtime warning: this runs (n_settings x n_seeds) full evolutionary
runs per swept parameter. With the defaults below (4 settings x 5
seeds x 6 parameters), that is 120 full runs -- budget accordingly.
Reduce N_SEEDS_PER_SETTING or the number of settings per parameter
first if this is too slow on your machine.


"""

import numpy as np
import json
import time

from multi_seed_run import run_multi_seed, N_GENERATIONS
from run_evolution import POPULATION_SIZE, ELITE_FRAC, SIGMA_INIT, N_HIDDEN, N_FLOWERS


N_SEEDS_PER_SETTING = 5   # fewer than the N_SEEDS=10 baseline, to keep runtime manageable


# One entry per parameter to sweep. "kwarg" is the keyword argument name
# expected by run_single_evolution (via es_kwargs in run_multi_seed).
# "baseline" is included explicitly in each sweep so the baseline run
# doesn't need to be re-run separately to appear in the comparison plot.
SWEEPS = {
    "population_size": {
        "kwarg": "population_size",
        "values": [10, 20, 40, 80],
        "baseline": POPULATION_SIZE,
    },
    "elite_frac": {
        "kwarg": "elite_frac",
        "values": [0.1, 0.25, 0.5],
        "baseline": ELITE_FRAC,
    },
    "sigma_init": {
        "kwarg": "sigma_init",
        "values": [0.1, 0.5, 1.0, 2.0],
        "baseline": SIGMA_INIT,
    },
    # "n_flowers": {
    #     "kwarg": "n_flowers",
    #     "values": [1,5,10,20],
    #     "baseline": N_FLOWERS
    # },

    "n_hidden": {
        "kwarg": "n_hidden",
        "values": [2, 4, 8, 16],
        "baseline": N_HIDDEN,
    },
}


def run_sweep(param_name, spec, n_seeds=N_SEEDS_PER_SETTING, n_generations=N_GENERATIONS):
    """Runs one parameter's full sweep, returning a list of per-setting
    result dicts (one per value tested, each itself an aggregate over
    n_seeds runs)."""
    values = sorted(set(spec["values"] + [spec["baseline"]]))
    kwarg = spec["kwarg"]
 
    results = []
    for value in values:
        print(f"\n########## Sweeping {param_name} = {value} "
              f"({'baseline' if value == spec['baseline'] else 'variant'}) ##########")
        t0 = time.time()
 
        es_kwargs = {kwarg: value}
        all_histories, summary_rows = run_multi_seed(
            n_seeds=n_seeds, n_generations=n_generations, **es_kwargs
        )
 
        best_fitnesses = [r["best_fitness"] for r in summary_rows]
        holdout_rates = [r["holdout_success_rate"] for r in summary_rows]
        gen_threshs = [r["gen_to_threshold"] for r in summary_rows if r["gen_to_threshold"] is not None]
        n_converged = len(gen_threshs)
 
        elapsed = time.time() - t0
        result = {
            "param": param_name,
            "value": value,
            "is_baseline": (value == spec["baseline"]),
            "n_seeds": n_seeds,
            "best_fitness_mean": float(np.mean(best_fitnesses)),
            "best_fitness_std": float(np.std(best_fitnesses)),
            "holdout_success_mean": float(np.mean(holdout_rates)),
            "holdout_success_std": float(np.std(holdout_rates)),
            "gen_to_threshold_mean": float(np.mean(gen_threshs)) if gen_threshs else None,
            "gen_to_threshold_std": float(np.std(gen_threshs)) if gen_threshs else None,
            "n_converged": n_converged,
            "elapsed_seconds": elapsed,
            "per_seed_summary": summary_rows,
        }
        results.append(result)
        print(f"  -> best_fitness={result['best_fitness_mean']:.2f}+/-{result['best_fitness_std']:.2f}  "
              f"holdout={result['holdout_success_mean']:.1%}+/-{result['holdout_success_std']:.1%}  "
              f"converged={n_converged}/{n_seeds}  ({elapsed:.0f}s)")
 
    return results
 
 
def main():
    all_results = {}
    for param_name, spec in SWEEPS.items():
        all_results[param_name] = run_sweep(param_name, spec)
 
        # save incrementally after each parameter, in case of interruption
        with open("sensitivity_analysis_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
 
    print("\nDone. Saved: sensitivity_analysis_results_10flowers.json")
 
 
if __name__ == "__main__":
    main()