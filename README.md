# Project Overview

This repository contains a custom Evolution Strategy (ES) pipeline used to train a neural network to navigate a simulated bee towards flowers. Below is a summary of the core files included in this project.

## Core Simulation and Training Files

* **`ann.py`**: Implements a small feedforward neural network that serves as a reactive controller for the bee. The network's flat weight vector acts as the genome that the Evolution Strategy evolves.
* **`bee_env.py`**: Contains the pure simulation logic for the bounded 2D environment. It manages the bee's unicycle kinematics, static flower locations, and a simplified gradient-vector sensory model.
* **`es.py`**: Implements a custom (mu, lambda) Evolution Strategy algorithm from scratch. It evolves the population using independent variances for mutation and selects the best offspring based on their fitness scores.
* **`evaluate.py`**: Serves as the bridge between the neural network controller and the environment by running a full episode rollout. It calculates the fitness of a genome based on the final distance to the flower, time taken, and a bonus reward for successfully reaching the target.
* **`multi_seed_run.py`**: Executes the evolution process across multiple independent random seeds to generate statistical results. This helps separate run-to-run noise from actual findings and evaluates the generalization success rate on held-out environment layouts.

## Visualization and Demos

* **`play_demo.py`**: Provides a local, live Pygame visualization of the bee environment. It uses a simple heuristic or random placeholder controller to verify the environment and rendering logic.
* **`play_evolved.py`**: Runs a Pygame visualizer to observe the final evolved neural network controller navigating the environment. It loads the optimized weights from a saved genome file to drive the simulated bee.

## Plotting and Analysis Scripts

* **`plot_baseline.py`**: Extracts data from a baseline history JSON file to plot the Evolution Strategy convergence, population fitness range, and mutation step sizes over evaluations.
* **`plot_multi_seed.py`**: Aggregates JSON data to plot the fitness over generations across multiple seeds, as well as a scatter plot comparing training fitness versus generalization success.
* **`plot_sensitivity_ana.py`**: Generates comparison figures for a sensitivity analysis, plotting the best training fitness and held-out success rate against swept parameter values.