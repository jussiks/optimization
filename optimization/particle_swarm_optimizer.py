#!/usr/bin/env python

"""Particle swarm optimization algorithm with optional plotting.

Dependencies: numpy (also matplotlib if user wants to draw plots)

Plotting only works properly for problems with two variables.

author: jussiks
"""

from optimization import Problem, Particle, generate_variables
import math
import copy
try:
    from particle_swarm_plotter import Plotter
except ImportError as e:
    print(e)
    print("Plotting not available")


def optimize_swarm(problem, swarm_size=20, max_iterations=100,
                   max_velocity=3, acc_const=2, neighbour_count=2,
                   desired_value=None, epsilon=None, plotter=None):
    """Uses particle swarm algorithm to optimize given problem."""

    # Generate initial particles
    swarm = [Particle(problem, generate_variables(problem))
             for i in range(swarm_size)]

    # Set current best particle as global_best
    global_best = copy.deepcopy(min(swarm, key=lambda x: x.best_value))

    if plotter:
        plotter.set_up_value_plot(
            "Local paradigm" if neighbour_count else "Global paradigm")
        plotter.update_plots(swarm, global_best, 0)

    # Iterate until maximum iterations are reached
    for iteration in range(max_iterations):
        for i in range(len(swarm)):
            if neighbour_count:
                # Pick best from neighbour particles
                idx = range(
                    i - int(neighbour_count / 2),
                    i + int(neighbour_count / 2) + 1)
                neighbours = [swarm[j % len(swarm)] for j in idx]
                social_best = min(neighbours, key=lambda x: x.best_value)
            else:
                # Otherwise use global best for updating the
                # particle
                social_best = global_best
            swarm[i].update(social_best)

            if swarm[i].best_value < global_best.best_value:
                global_best = copy.deepcopy(swarm[i])

        if plotter:
            plotter.update_plots(swarm, global_best, iteration)

        if desired_value and epsilon:
            if global_best.best_value < desired_value + epsilon:
                break

    return global_best, iteration


if __name__ == "__main__":
    # Testing the algorithm with some problem
    func = lambda x: math.cos(x[0]) * math.sin(x[1]) - float(
        x[0]) / (x[1]**2 + 1)
    const_func = lambda x: [-1 <= x[0] <= 2, -1 <= x[1] <= 1]
    bounds = [(-1, 2), (-1, 1)]

    problem = Problem(
        func, constraint_function=const_func, variable_bounds=bounds)
    try:
        plotter = Plotter(
            swarm_plot_domain=[-5, 5, -5, 5],
            value_plot_domain=None,
            iterations_to_plot=50,
            sleep=0.04)
    except NameError as e:
        print(e)
        plotter = None

    sol, iterations = optimize_swarm(problem, plotter=plotter)

    print("Best solution: " + str(sol.best_variables))
    print("Value: " + str(sol.best_value))
    print("Iterations: " + str(iterations))
