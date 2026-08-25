import json
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Load baseline history
# ============================================================

with open("evolution_history_baseline.json", "r") as f:
    history = json.load(f)


# Convert history to arrays
generation = np.array([h["generation"] for h in history])

best = np.array([h["best"] for h in history])
mean = np.array([h["mean"] for h in history])
worst = np.array([h["worst"] for h in history])

sigma_mean = np.array([h["sigma_mean"] for h in history])
sigma_min = np.array([h["sigma_min"] for h in history])
sigma_max = np.array([h["sigma_max"] for h in history])


# ============================================================
# Function evaluations
# ============================================================

POPULATION_SIZE = 40

# Every generation evaluates POPULATION_SIZE offspring.
function_evaluations = (generation + 1) * POPULATION_SIZE


# ============================================================
# Best-so-far fitness
# ============================================================

best_so_far = np.maximum.accumulate(best)


# ============================================================
# Plot 1: Main convergence plot
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    function_evaluations,
    best_so_far,
    linewidth=2,
    label="Best-so-far"
)

plt.xlabel("Function evaluations")
plt.ylabel("Fitness")
plt.title("Baseline ES convergence")
plt.grid(alpha=0.25)
plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# Plot 1.2: Main convergence plot
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    generation,
    best_so_far,
    linewidth=2,
    label="Best-so-far"
)

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Baseline ES convergence")
plt.grid(alpha=0.25)
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# Plot 2: Population fitness
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    function_evaluations,
    best,
    label="Best",
    linewidth=1.8
)

plt.plot(
    function_evaluations,
    mean,
    label="Mean",
    linewidth=1.8
)

plt.plot(
    function_evaluations,
    worst,
    label="Worst",
    linewidth=1.5
)

plt.fill_between(
    function_evaluations,
    worst,
    best,
    alpha=0.15,
    label="Population range"
)

plt.xlabel("Function evaluations")
plt.ylabel("Fitness")
plt.title("Baseline ES population fitness")
plt.grid(alpha=0.25)
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# Plot 3: Mutation step size
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    function_evaluations,
    sigma_mean,
    label="Mean σ",
    linewidth=2
)

plt.plot(
    function_evaluations,
    sigma_min,
    label="Min σ",
    linewidth=1.5
)

plt.plot(
    function_evaluations,
    sigma_max,
    label="Max σ",
    linewidth=1.5
)

plt.yscale("log")

plt.xlabel("Function evaluations")
plt.ylabel("Mutation step size σ")
plt.title("Evolution of ES mutation step size")
plt.grid(alpha=0.25, which="both")
plt.legend()

plt.tight_layout()
plt.show()

######################################################
plt.figure(figsize=(9, 5))

plt.plot(
    function_evaluations,
    reached_rate * 100,
    linewidth=2
)

plt.xlabel("Function evaluations")
plt.ylabel("Successful rollouts (%)")
plt.title("Baseline ES task success")
plt.ylim(0, 100)
plt.grid(alpha=0.25)

plt.tight_layout()
plt.show()
########################################################
# ============================================================
# Plot 4: Summary statistics
# ============================================================

print("\n" + "=" * 50)
print("BASELINE ES SUMMARY")
print("=" * 50)

best_idx = np.argmax(best)

print(
    f"Best fitness:       {best[best_idx]:.4f}"
)

print(
    f"Best fitness found: {best_so_far[-1]:.4f}"
)

print(
    f"Generation:          {generation[best_idx]}"
)

print(
    f"Function evaluations: "
    f"{function_evaluations[best_idx]}"
)

print(
    f"Final mean fitness:  {mean[-1]:.4f}"
)

print(
    f"Final worst fitness: {worst[-1]:.4f}"
)

print(
    f"Final mean sigma:    {sigma_mean[-1]:.4f}"
)

print(
    f"Final min sigma:     {sigma_min[-1]:.6f}"
)

print(
    f"Final max sigma:     {sigma_max[-1]:.4f}"
)

print(
    f"Total evaluations:   {function_evaluations[-1]}"
)

print("=" * 50)