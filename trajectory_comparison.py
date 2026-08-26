"""
trajectory_comparison.py
-----------------------------------
Compares two evolved genomes, one trained on a sparse field
(n_flowers=5), one trained directly on a dense field (n_flowers=100)
 by running both on the SAME held-out 100-flower layout and plotting
their actual trajectories side by side.

Also reports fraction_consumed statistics across multiple held-out
layouts (not just the one visualized), so the figure can be read
alongside a quantitative summary rather than a single cherry-picked
example.



Run locally:
    python transfer_trajectory_comparison.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import rollout

GENOME_5_PATH = "best_genome_5flowers_correct.npy"
GENOME_100_PATH = "best_genome_100flowers.npy"

N_FLOWERS_TEST = 30        # dense field both genomes are evaluated on
N_HOLDOUT_EVALS = 20        # for the fraction_consumed statistics
MAX_STEPS = 300
HOLDOUT_SEED_OFFSET = 9_000_000  # must not overlap any training seed range


def evaluate_fraction_consumed(genome, n_evals=N_HOLDOUT_EVALS):
    """Runs n_evals held-out episodes, returns list of fraction_consumed
    values (one per episode) plus the first episode's full trajectory
    (for plotting)."""
    fractions = []
    first_trajectory, first_flowers = None, None
    for i in range(n_evals):
        seed = HOLDOUT_SEED_OFFSET + i
        result = rollout(genome, n_hidden=8, n_flowers=N_FLOWERS_TEST,
                         env_seed=seed, max_steps=MAX_STEPS,
                         record_trajectory=(i == 0))
        
        # FIX: Calculate fraction_consumed manually if the key is missing
        if "fraction_consumed" in result:
            frac = result["fraction_consumed"]
        else:
            # Assumes the raw count of eaten flowers is returned as 'fitness' or 'reward'
            consumed = result.get("fitness", result.get("reward", 0))
            frac = consumed / N_FLOWERS_TEST
            
        fractions.append(frac)
        
        if i == 0:
            first_trajectory = result["trajectory"]
            first_flowers = result["flowers"]
    return fractions, first_trajectory, first_flowers


def plot_trajectory(ax, trajectory, flowers, title):
    traj = np.array(trajectory)
    fx = [f[0] for f in flowers]
    fy = [f[1] for f in flowers]

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.scatter(fx, fy, color="#e91e8c", s=25, alpha=0.5, zorder=1, label="flowers")
    ax.plot(traj[:, 0], traj[:, 1], color="#3f51b5", linewidth=1, alpha=0.8, zorder=2)
    ax.scatter([traj[0, 0]], [traj[0, 1]], color="#2a9d3f", s=80, zorder=3, label="start")
    ax.scatter([traj[-1, 0]], [traj[-1, 1]], color="black", s=80, marker="X", zorder=3, label="end")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def main():
    genome_5 = np.load("best_genome_5flowers.npy")
    genome_100 = np.load("best_genome_100flowers.npy")

    print("Evaluating genome trained on 5 flowers, tested on 100...")
    frac_5, traj_5, flowers_5 = evaluate_fraction_consumed(genome_5)

    print("Evaluating genome trained on 100 flowers, tested on 100...")
    frac_100, traj_100, flowers_100 = evaluate_fraction_consumed(genome_100)

    frac_5, frac_100 = np.array(frac_5), np.array(frac_100)
    print(f"\ntrained on 5,   tested on {N_FLOWERS_TEST}: fraction_consumed = "
          f"{frac_5.mean():.1%} +/- {frac_5.std():.1%}")
    print(f"trained on 100, tested on {N_FLOWERS_TEST}: fraction_consumed = "
          f"{frac_100.mean():.1%} +/- {frac_100.std():.1%}")

    with open("transfer_comparison_results.json", "w") as f:
        json.dump({
            "trained_on_5": frac_5.tolist(),
            "trained_on_100": frac_100.tolist(),
        }, f, indent=2)

    # --- figure: trajectories on the SAME held-out layout (seed 0) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    plot_trajectory(axes[0], traj_5, flowers_5,
                     f"Trained on 5 flowers\n(tested on {N_FLOWERS_TEST})")
    plot_trajectory(axes[1], traj_100, flowers_100,
                     f"Trained on 100 flowers\n(tested on {N_FLOWERS_TEST})")
    fig.suptitle("Trajectory on the SAME held-out 100-flower layout, same starting position")
    fig.tight_layout()
    fig.savefig("transfer_trajectory_comparison.png", dpi=130)
    print("\nSaved: transfer_trajectory_comparison.png, transfer_comparison_results.json")


if __name__ == "__main__":
    main()