"""
multi_seed_run.py
--------------------

Runs run_evolution.py independently across multiple random seeds, and produces results with stadistics.

Reporting a single run's numbers as the main result risks presenting run-to-run
noise as if it were a real finding. Repeating the full run across many
independent seeds and reporting mean +/- std separates the two, a
spread across seeds indicates a robus result, a wide spread indicates the outcome is sensitive to randomness
and any single run's number should not be over-interpreted.

"""

import numpy as np
import json

from run_evolution import run_single_evolution
from evaluate import rollout


N_SEEDS = 10
N_GENERATIONS = 150
N_HOLDOUT_EVALS = 30 #environment seeds per run
SUCCESS_THRESHOLD = 20.0 #whole populations average fitness needs to be above 20 to be considered succesful, 20 decided from the single baseline run (20.0 is determined from the baseline run)



def evaluate_holdout_success_rate(genome, n_hidden, n_flowers, max_steps,
                                  n_evals=N_HOLDOUT_EVALS, holdout_seed_offset=9_000_000):
    """
    Tests a trained individual (genome) on a fixed batch of environment seeds (different flower positions and bee initial position) that were never
    part of  ANY training run (offset outside the range used for training env seeds), and returns the fraction of episiodes where the flower was reached.    
    
    """

    successes = 0

    for i in range(n_evals):
        seed = holdout_seed_offset + i
        result = rollout(genome, n_hidden=n_hidden, n_flowers=n_flowers,
                         env_seed=seed, max_steps=max_steps)

        successes += int(result["reached"])

    return successes / n_evals


def generation_to_threshold(history, threshold=SUCCESS_THRESHOLD):
    """
    First generation at which the average fitness of the population exceeds the marked threshold,
    or None if it never does within the run.

    """

    for entry in history:
        if entry["mean"] > threshold:
            return entry["generation"]

    return None


def run_multi_seed(n_seeds=N_SEEDS, n_generations=N_GENERATIONS, **es_kwargs):

    all_histories = []
    summary_rows = []

    for seed in range(n_seeds):
        print(f"\n=== Run seed {seed} ({seed+1}/{n_seeds}) ===")

        #get the optimized weights for that specific seed
        es = run_single_evolution(run_seed=seed, n_generations=n_generations, verbose=False, **es_kwargs)

        #Let the ANN run with those weights and get the fitness values from the ANN
        n_hidden = es_kwargs.get("n_hidden", 8)
        n_flowers = es_kwargs.get("n_flowers", 10) 
        max_steps = es_kwargs.get("max_steps", 300)
 
        holdout_rate = evaluate_holdout_success_rate(
            es.best_genome, n_hidden=n_hidden, n_flowers=n_flowers, max_steps=max_steps
        )
        gen_thresh = generation_to_threshold(es.history)
 
        print(f"  best_fitness={es.best_fitness:.2f}  "
              f"holdout_success_rate={holdout_rate:.2%}  "
              f"gen_to_threshold={gen_thresh}")
 
        all_histories.append(es.history)
        summary_rows.append({
            "seed": seed,
            "best_fitness": float(es.best_fitness),
            "holdout_success_rate": holdout_rate,
            "gen_to_threshold": gen_thresh,
        })
 
    return all_histories, summary_rows


def aggregate_histories(all_histories):
    """
    Stacks per-generation best/mean/worst fitness across all seeds into
    arrays of shape (n_seeds, n_generations), for computing mean/std
    across seeds at each generation.
    """
    n_seeds = len(all_histories)
    n_gens = min(len(h) for h in all_histories)  # in case any run is shorter
 
    best = np.array([[h[g]["best"] for g in range(n_gens)] for h in all_histories])
    mean_ = np.array([[h[g]["mean"] for g in range(n_gens)] for h in all_histories])
    worst = np.array([[h[g]["worst"] for g in range(n_gens)] for h in all_histories])
 
    return {
        "generations": list(range(n_gens)),
        "best_mean_over_seeds": best.mean(axis=0).tolist(),
        "best_std_over_seeds": best.std(axis=0).tolist(),
        "mean_mean_over_seeds": mean_.mean(axis=0).tolist(),
        "mean_std_over_seeds": mean_.std(axis=0).tolist(),
        "worst_mean_over_seeds": worst.mean(axis=0).tolist(),
        "worst_std_over_seeds": worst.std(axis=0).tolist(),
        "n_seeds": n_seeds,
    }


def main():
    all_histories, summary_rows = run_multi_seed(n_seeds=N_SEEDS, n_generations=N_GENERATIONS)

    aggregated = aggregate_histories(all_histories)

    with open("multi_seed_histories.json", "w") as f:
        json.dump(all_histories, f)
    with open("multi_seed_aggregated.json", "w") as f:
        json.dump(aggregated, f, indent=2)
    with open("multi_seed_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
 
    best_fitnesses = [r["best_fitness"] for r in summary_rows]
    holdout_rates = [r["holdout_success_rate"] for r in summary_rows]
    gen_threshs = [r["gen_to_threshold"] for r in summary_rows if r["gen_to_threshold"] is not None]
 
    print("\n" + "=" * 60)
    print(f"Across {N_SEEDS} seeds:")
    print(f"  best_fitness:          {np.mean(best_fitnesses):.2f} +/- {np.std(best_fitnesses):.2f}")
    print(f"  holdout_success_rate:  {np.mean(holdout_rates):.2%} +/- {np.std(holdout_rates):.2%}")
    if gen_threshs:
        print(f"  gen_to_threshold:      {np.mean(gen_threshs):.1f} +/- {np.std(gen_threshs):.1f} "
              f"({len(gen_threshs)}/{N_SEEDS} runs reached threshold)")
    else:
        print("  gen_to_threshold:      no runs reached threshold")
    print("Saved: multi_seed_histories.json, multi_seed_aggregated.json, multi_seed_summary.json")
 
 
if __name__ == "__main__":
    main()



