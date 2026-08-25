"""
es.py
-----

(mu, lambda) Evolution Strategy (ES), from scratch.

Multiple variances, no covariances strategy: each genome weight has its own variance, which is evolved alongside the weight itself. 

Algorithm per genetarion:

0. Update the std deviation of each weight through log-normal multipilcatively mutation: std_i = max{std_i * exp(z + z_i), 1e-8}, where z ~ N(0, tau) and z_i ~ N(0, tau_i**2). tau = 1/sqrt(2*sqrt(n_weights)), tau_i = 1/sqrt(2*n_weights).
1. Sample lambda offspring around the current mean of the population with the updated std deviation:
    child_i = mean + sigma * N(0, 1)
2. Evaluate fitsness of each offspring
3. Select the mu best offspring base on fitness values
4. Update the mean of to match the new population

"""

import numpy as np

class EvolutionStrategy:
    def __init__(self, num_params, population_size = 40, elite_frac = 0.25,
                 sigma_init = 0.5, init_mean = None, seed = None):

        """
        num_params: length of the genome (NN's flat weight vector)
        population_size: lambda, number of offspring sampled per generation
        elite_frac: mu/lambda, fraction of population selected to be the next generation from lambda offspring
        sigma_init: initial mutation standard deviation (same for every weight at gen 0, they diverge from there via self-adaptation)
        init_mean: starting point for the mean genome (default:zeros)
        """
        self.num_params = num_params
        self.population_size = population_size
        self.mu = max(2, int(elite_frac * population_size))
        self.rng = np.random.default_rng(seed)

        self.mean = np.zeros(num_params) if init_mean is None else np.array(init_mean, dtype=float)

        self.sigma = np.full(num_params, sigma_init, dtype=float) #one step size per weight

        n = num_params
        self.tau = 1.0 / np.sqrt(2.0 * n)  #global learning rate
        self.tau_i = 1.0 / np.sqrt(2.0 * np.sqrt(n))    # per-coordinate learning rate

        self.generation = 0
        self.best_genome = self.mean.copy()
        self.best_fitness = -np.inf
        self.history = []  # list of dicts: {gen, best, mean, worst, sigma stats}

    def _update_sigma(self):
        """
        0. log-normal self-adaptation of strategy parameters
        """
        z = self.rng.normal(0, self.tau) #shared, one scalar
        z_i = self.rng.normal(0, self.tau_i, size = self.num_params) #independent per weight

        self.sigma = np.maximum(self.sigma * np.exp(z + z_i), 1e-8)

    def _sample_offspring(self):
        """
        1. Mutation: Adapt step sizes, then sample a new population of lambda offspring around the current mean.
        """

        self._update_sigma()
        noise = self.rng.normal(0, 1, size=(self.population_size, self.num_params))
        population = self.mean + noise * self.sigma
        return population

    def tell(self, population, fitnesses):
        """
        2. Selection: Choose the mu best offspring based on fitness
        3. Update the mean to match the new population.
        """ 

        fitnesses = np.asarray(fitnesses)
        order = np.argsort(-fitnesses) #descending: best first
        elite_idx = order[:self.mu]

        new_population = population[elite_idx]
        self.mean = new_population.mean(axis=0) #update plain average, no recombination

        gen_best_idx = order[0]
        gen_best_fitness = fitnesses[gen_best_idx]
        if gen_best_fitness > self.best_fitness:
            self.best_fitness = gen_best_fitness
            self.best_genome = population[gen_best_idx].copy()


        self.history.append({
            "generation": self.generation,
            "best": float(gen_best_fitness),
            "mean": float(fitnesses.mean()),
            "worst": float(fitnesses.min()),
            "sigma_mean": float(self.sigma.mean()),
            "sigma_min": float(self.sigma.min()),
            "sigma_max": float(self.sigma.max()),
            
        })
 
        self.generation += 1