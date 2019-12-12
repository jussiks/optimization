#!/usr/bin/env python

"""Tchebycheff and weighted sum approaches for multiobjective optimization.

author: jussiks
"""

from optimization import Problem, MOPSolution, WeightedSolution
import genetic_methods as gm
import numpy as np
import math
from enum import Enum
import itertools
import random
from typing import List


class Method(Enum):
    tchebycheff = 1
    weighted_sum = 2


def g_tchebycheff(solution: WeightedSolution, ref_point):
    """Returns the value of objective function used in Tchebycheff approach.

    ref_point is the reference point, i.e. the ideal objective vector.
    """
    return max(solution.weights * abs(solution.evaluate() - ref_point))


def g_weighted_sum(solution: WeightedSolution):
    """Returns the value of objective function used in Weighted Sum approach."""
    return sum(solution.weights * solution.evaluate())


def fitness_eval(population: List[WeightedSolution], method: Method,
                 ref_point=None):
    """Calculates the fitness values for every member of the population.

    Arguments:
    population  list of WeightedSolutions
    method      either Method.tchebycheff or Method.weighted_sum
    ref_point   reference point used in Tchebycheff method (numpy array)
    """
    if method == Method.tchebycheff:
        if ref_point is None:
            raise ValueError("Reference point (ref_point) must be defined when using Tchebycheff approach.")
        for p in population:
            p.fitness = g_tchebycheff(p, ref_point)
    elif method == Method.weighted_sum:
        for p in population:
            p.fitness = g_weighted_sum(p)
    else:
        raise ValueError("Method {0} not implemented.".format(method))


def find_neighbours(population: List[WeightedSolution], count):
    """Finds a given number of neighbours for each solution in population.

    Calculates nearests neighbours for each member of population using
    euclidean distace between weight vectors. Returns a dict where each key
    is an individual an value is a list of nearest neighbours of that
    individual.

    Arguments:
    population  list of WeightedSolutions
    count       how many neigbours to pick
    """
    distances = {p: [] for p in population}
    for x, y in itertools.combinations(population, 2):
        dis = np.linalg.norm(x.variables - y.variables)
        distances[x].append({"ind": y, "distance": dis})
        distances[y].append({"ind": x, "distance": dis})
        # TODO add values in sorted order so we don't have to sort them
        #      afterwards
    for key in distances:
        distances[key] = sorted(
            distances[key], key=lambda x: x["distance"])[:count]
    return {
        key: [x["ind"] for x in distances[key]] for key in distances
    }


def generate_next_gen(population: List[WeightedSolution], neighbour_count,
                      method, ref_point=None, mutation_probability=0.01):
    """Returns next generation of solutions"""

    # Find neigbours
    neighbour_dict = find_neighbours(population, neighbour_count)
    next_gen = []

    for solution in neighbour_dict:
        # Pick two random parents from neigbours
        x, y = random.sample(neighbour_dict[solution], 2)

        # Perform crossover, mutation and repair
        child_vars = gm.single_point_crossover(x, y)
        child = WeightedSolution(x.problem, child_vars, solution.weights)
        gm.gaussian_mutation(
            child, mutation_probability, [1] * len(child_vars))
        gm.repair(child)
        next_gen.append(child)
    return next_gen


if __name__ == "__main__":
    # Objective function.
    func = lambda x: [
        sum(-10.0 * math.exp(-0.2 * math.sqrt(
            x[i]**2 + x[i + 1]**2)) for i in range(2)),
        sum(abs(x[i])**0.8 + 5 * math.sin(x[i]**3) for i in range(3))
    ]
    # Minimum and maximum values for input vectors
    constraints = [(-5, 5) for i in range(3)]
    problem = Problem(func, variable_bounds=constraints)
    var_generator = lambda: [
        random.uniform(low, high) for (low, high) in constraints]

    sol_count = 12

    # initial solutions
    solutions = [
        WeightedSolution(
            problem,
            var_generator(),
            [0 + i / sol_count, 1 - i / sol_count])
        for i in range(sol_count)
    ]

    # Reference point used in Tchebycheff method
    z_star = np.array([-20, -12])

    # Number of neigbours picked in neigbour selection.
    neighbour_count = 4

    print("\n{0} nearest neighbours for each individual using euclidean distance:".format(
        neighbour_count))
    neighbour_dict = find_neighbours(solutions, neighbour_count)
    for solution in neighbour_dict:
        print(solution)
        for neighbour in neighbour_dict[solution]:
            print("\t{0}".format(neighbour))

    fitness_eval(solutions, Method.tchebycheff, ref_point=z_star)
    print("\nTchebycheff function values for each individual:")
    for s in solutions:
        print("{0}\t{1}".format(s, s.fitness))

    fitness_eval(solutions, Method.weighted_sum, ref_point=z_star)
    print("\nWeighted sum function values for each individual:")
    for s in solutions:
        print("{0}\t{1}".format(s, s.fitness))

    next_gen = generate_next_gen(
        solutions, neighbour_count, Method.tchebycheff, ref_point=z_star,
        mutation_probability=0.1)
    print("\nNext generation using Tchebycheff:")
    for s in next_gen:
        print(s)

    next_gen = generate_next_gen(
        solutions, neighbour_count, Method.weighted_sum,
        mutation_probability=0.1)
    print("\nNext generation using weighted sums:")
    for s in next_gen:
        print(s)
