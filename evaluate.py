"""
evaluate.py
-----------

Bridge between ANN controller and Environment: runs one full 
loop ("rollout") with a given genome (flat weight vector) and 
computes its fitness.

Fitness Design:

    fitness = -DIST_WEIGHT * final_distance_to_flower
              -TIME_WEIGHT * (steps_taken / max_steps)
              +REACHED_BONUS (only if the flower was reached)
    
Reasoning for each term:

    - final_distance_to_flower: did the bee end up
    near the flower? 
    - time taken: among two bees that both reach the flower, the faster
    one should score higher, without this term, a bee that
    wanders for 300 steps before stumbling onto the flower scores
    identically to one that beelines there in 20.
    - reached bonus: a fixed reward for actually reaching the flower.
    This creates a clear fitness "step" between "got close" and
    "actually succeeded," which helps the ES's selection step
    successful controllers from ones that just happened to end up nearby without actually solving the task.

All three weights are exposed as constants so they're easy to vary
during the sensitivity analysis.
"""

import numpy as np
from bee_env import Environment
from ann import FeedforwardNN

DIST_WEIGHT = 1.0
TIME_WEIGHT = 5.0
REACHED_BONUS = 30.0

def rollout(genome, n_hidden=4, n_flowers=1, env_seed=None,
            max_steps=300, width=100.0, height=100.0,
            flower_positions=None, bee_start=None,
            record_trajectory=False):
    """
    Runs one episode with the given genome and returns a dict with the
    fitness and useful diagnostics (used by ES and the analysis of the solution found)
    
    """
    env = Environment(width=width, height=height, n_flowers=n_flowers,
                      max_steps=max_steps, seed=env_seed,
                      flower_positions=flower_positions, bee_start=bee_start)

    controller = FeedforwardNN(n_inputs=2, n_hidden=n_hidden, n_outputs=2)
    controller.set_flat_weights(genome)

    trajectory = [] if record_trajectory else None

    while not env.done:
        obs = env.get_observation()
        action = controller.forward(obs)
        env.step(action[0], action[1])
        if record_trajectory:
            trajectory.append((env.bee.x, env.bee.y))


    final_dist = env.nearest_flower_distance()
    reached = any(not f.active for f in env.flowers)
    steps_taken = env.step_count

    fitness = (
        -DIST_WEIGHT * final_dist
        - TIME_WEIGHT * (steps_taken / max_steps)
        + (REACHED_BONUS if reached else 0.0)
    )

    return {
        "fitness": fitness,
        "final_dist": final_dist,
        "reached": reached,
        "steps_taken": steps_taken,
        "trajectory": trajectory,
        "flowers": [(f.x, f.y) for f in env.flowers],
        "bee_start": (env.bee.trail[0] if env.bee.trail else (env.bee.x, env.bee.y)),
    }

def evaluate_population(population, **rollout_kwargs):
    """Evaluate a whole ES population; returns an array of fitnesses."""
    fitnesses = np.zeros(len(population))
    for i, genome in enumerate(population):
        result = rollout(genome, **rollout_kwargs)
        fitnesses[i] = result["fitness"]
    return fitnesses