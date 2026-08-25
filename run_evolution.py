"""
run_evolution.py
-----------------

Main script: 
    1. Runs the (mu,lambda)-ES  to evolve the bee's ANN controller from scratch (random weights).

Key evaluation design decision: within a single generation, EVERY
candidate in the population is evaluated on the SAME environment seed
(same flower position, same bee start). This is important, without
it, a candidate could score well simply by getting lucky with an easy
flower placement, and the ES would have no reliable signal to select
on. The seed changes BETWEEN generations, so across the whole run the
evolved controller still has to generalize across many different
flower positions, rather than overfitting to one fixed layout.

"""
import numpy as np
import json

from ann import FeedforwardNN
from es import EvolutionStrategy
from evaluate import rollout, DIST_WEIGHT, TIME_WEIGHT, REACHED_BONUS

N_GENERATIONS = 150
POPULATION_SIZE = 40
ELITE_FRAC = 0.25
SIGMA_INIT = 0.5
N_HIDDEN = 8
N_FLOWERS = 1
MAX_STEPS = 300


def run_single_evolution(run_seed=0, n_generations = N_GENERATIONS,
                          population_size = POPULATION_SIZE, elite_frac = ELITE_FRAC,
                          sigma_init = SIGMA_INIT, n_hidden = N_HIDDEN,
                          n_flowers = N_FLOWERS, max_steps = MAX_STEPS,
                          verbose = True):
    """
    Runs one evolutionary run and returns the EvolutionStrategy object (.history, .best_genome, .best_fitness)

    run_seed controls both points of randomness for the run:
    1. ES's internal randomness (mutation sampling)
    2. Sequence of environment seeds used across generations
    
    """

    #Generate a controller
    template_nn = FeedforwardNN(n_inputs=2, n_hidden=n_hidden, n_outputs=2)
    num_params = template_nn.num_weights

    #Create and Configure the ES
    es = EvolutionStrategy(
        num_params=num_params,
        population_size=population_size,
        elite_frac=elite_frac,
        sigma_init=sigma_init,
        seed=run_seed #first source of randomness: mutation noise
    )

    #Second source of randomness: Whcih flower layout each generation gets evaluated on
    env_rng = np.random.default_rng(run_seed + 1_000_000) #we add 1_000_000 to get a different seed as for es, to make them independent

    for gen in range(n_generations):
        env_seed = int(env_rng.integers(0, 1_000_000)) #same seed for whole population in the current generation

        population = es._sample_offspring()
        fitnesses = np.zeros(population_size)
        reached_flags = np.zeros(population_size, dtype=bool)

        for i, genome in enumerate(population):
            result = rollout(
                genome, n_hidden=n_hidden, n_flowers=n_flowers,
                env_seed=env_seed, max_steps=max_steps,
            )

            fitnesses[i] = result["fitness"]
            reached_flags[i] = result["reached"]

        es.tell(population, fitnesses) #generate the next generation from the current population = offspring

        if verbose and (gen % 5 == 0 or gen == n_generations - 1):
                n_reached = reached_flags.sum()
                reached_rate = reached_flags.mean()
                print(f"gen {gen:4d} | best={fitnesses.max():7.2f} "
                    f"mean={fitnesses.mean():7.2f} | "
                    f"reached={n_reached}/{population_size} | "
                    f"sigma(mean/min/max)={es.sigma.mean():.3f}/{es.sigma.min():.3f}/{es.sigma.max():.3f}")

    return es


def main(): #By default we run always on seed 0, multiple seeds are tried on multi_seed_run.py
    es = run_single_evolution(run_seed=0)


 
    np.save("best_genome_baseline.npy", es.best_genome)
    with open("evolution_history_baseline.json", "w") as f:
        json.dump(es.history, f, indent=2)
 
    print(f"\nDone. Best fitness ever: {es.best_fitness:.2f}")
    print("Saved: best_genome_baseline.npy, evolution_history_baseline.json")

 
 
if __name__ == "__main__":
    main()
 



    

# def main():

#     #---- Set up NN --------------------
#     template_nn = FeedforwardNN(n_inputs=2, n_hidden=N_HIDDEN, n_outputs=2)
#     num_params = template_nn.num_weights
#     print(f"Genome length (number of evolvable weights): {num_params}")

#     #--- Set up the ES ------------------
#     es = EvolutionStrategy(
#         num_params=num_params,
#         population_size=POPULATION_SIZE,
#         elite_frac=ELITE_FRAC,
#         sigma_init=SIGMA_INIT,
#         seed=0,
#     )

#     rng = np.random.default_rng(123)

#     for generation in range(N_GENERATIONS):
#         env_seed = int(rng.integers(0, 1_000_000))

#         population = es._sample_offspring()
#         fitnesses = np.zeros(POPULATION_SIZE)
#         reached_flags = np.zeros(POPULATION_SIZE, dtype=bool)

#         for i, genome in enumerate(population):
            
#             result = rollout(
#                 genome, n_hidden=N_HIDDEN, n_flowers=N_FLOWERS,
#                 env_seed=env_seed, max_steps=MAX_STEPS
#             )

#             fitnesses[i] = result["fitness"]
#             reached_flags[i] = result["reached"]

#         es.tell(population, fitnesses)

#         if generation % 5 == 0 or generation == N_GENERATIONS - 1:
#             n_reached = reached_flags.sum()
#             print(f"gen {generation:4d} | best={fitnesses.max():7.2f} "
#                   f"mean={fitnesses.mean():7.2f} | "
#                   f"reached={n_reached}/{POPULATION_SIZE} | "
#                   f"sigma(mean/min/max)={es.sigma.mean():.3f}/{es.sigma.min():.3f}/{es.sigma.max():.3f}")
 
#     # --- save results ---
#     np.save("best_genome.npy", es.best_genome)
#     with open("evolution_history.json", "w") as f:
#         json.dump(es.history, f, indent=2)
 
#     print(f"\nDone. Best fitness ever: {es.best_fitness:.2f}")
#     print("Saved: best_genome.npy, evolution_history.json")
 
 
# if __name__ == "__main__":
#     main()
