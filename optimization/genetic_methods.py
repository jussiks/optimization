#!/usr/bin/env python

"""Selection, crossover and mutation methods used in genetic algorithms.

author: jussiks
"""

from optimization import Solution
from typing import List
import random
import numpy as np


def tournament_selection(population: List[Solution], k: int, p: float = 1.0,
                         selection_size: int = 2) -> List[Solution]:
    """Returns a list of Solutions using tournament selection.

    k number of Solutions are sampled from the population and the one
    with lowest fitness is added to selection.

    Arguments
    population      list of Solutions
    k               number of Solutions selected to the tournament,
                    must be non-negative
    p               likelihood of picking the the best solution from
                    the tournament
    selection_size  how many Solutions are returned (default 2)
    """
    if k < 1:
        raise ValueError("k must be non-negative")

    # Clone the population
    clone_population = list(population)

    selection = []

    for i in range(min(selection_size, len(clone_population))):
        # Draw a sample of k Solutions randomly
        sample = random.sample(clone_population, min(k, len(clone_population)))

        # Take the one with objective function value, remove it from further
        # selections and add it to parents
        # TODO add random tiebreaks
        # TODO handle multiobjective
        sample.sort(key=lambda x: x.value)
        if p == 1:
            winner = sample[0]
        else:
            r = random.random()
            for j in range(k):
                winner = sample[j]
                if r < p * (1 - p)**j:
                    break

        clone_population.remove(winner)
        selection.append(winner)

    return selection


def uniform_crossover(parent1: Solution, parent2: Solution) -> List:
    """Combines parent variables using uniform crossover."""
    if (len(parent1.variables) != len(parent2.variables)):
        raise ValueError(
            "Parents should have same number of decision variables.")

    new_variables = []
    for var1, var2 in zip(parent1.variables, parent2.variables):
        new_variables.append(var1 if random.random() < 0.5 else var2)
    return np.array(new_variables)


def single_point_crossover(parent1: Solution, parent2: Solution,
                           quarantee_both: bool = False, 
                           randomize_parent_order: bool = True) -> List:
    """Combines parent variables using single point crossover.

    quarantee_both argument can be used to ensure that at least one variable is
    picked from both parents."""
    if (len(parent1.variables) != len(parent2.variables)):
        raise ValueError(
            "Parents should have same number of decision variables.")
    # Calculate crossover point
    if len(parent1.variables) == 0:
        return np.array([])

    if quarantee_both and len(parent1.variables) > 1:
        crossover_pnt = random.randint(1, len(parent1.variables) - 1)
    else:
        crossover_pnt = random.randint(0, len(parent1.variables))

    parents = [parent1, parent2]
    if randomize_parent_order:
        random.shuffle(parents)

    return np.concatenate((
        parents[0].variables[:crossover_pnt],
        parents[1].variables[crossover_pnt:]
        ))


def real_value_crossover(parent1: Solution, parent2: Solution,
                         minmax: float = 1) -> List:
    """Combines parent values using a random coeffiecient
    Based on https://help.scilab.org/doc/5.5.2/en_US/crossover_ga_default.html

    Arguments:
    parent1, parent2    two Solution objects used as parents
    minmax
    """
    mix = random.uniform(-minmax, minmax)
    new_variables = mix * parent1.value + (1 - mix) * parent2.value
    return new_variables


def uniform_mutation(solution: Solution, mutation_p: float, continous=True):
    """Mutates given solution by drawing new variable values from
    uniform distribution.

    Calls the evaluate function of the solution to recalculate the
    value of objective function in case mutation has occurred.

    Arguments:
    solution        Solution to be mutated
    mutation_p      probability of individual variable getting mutated
    continous       are variables continous or not
    """
    def var_gen(a, b):
        return random.uniform(a, b) if continous else random.randint(a, b)

    mutated = False
    for i in range(len(solution.variables)):
        if random.random() < mutation_p:
            # TODO handle None types
            low, high = solution.problem.variable_bounds[i]
            solution.variables[i] = var_gen(low, high)
            mutated = True
    if mutated:
        solution.evaluate()


def gaussian_mutation(solution: Solution, mutation_p: float,
                      standard_devs: List[float]):
    """Mutates given solution by drawing new variable values from standard
    distribution.

    Original value of the variable is set as the mean of standard distribution.
    Calls the evaluate function of the solution to recalculate the value of
    objective function in case mutation has occurred.

    TODO take variable bounds into account

    Arguments:
    solution        Solution to be mutated
    mutation_p      probability of individual variable getting mutated
    standard_devs   list of standard deviations for each standard distribution
    """
    mutated = False
    for i in range(len(solution.variables)):
        if random.random() < mutation_p:
            solution.variables[i] = random.normalvariate(
                solution.variables[i],
                standard_devs[i]
                )
            low, high = solution.problem.variable_bounds[i]
            solution.variables[i] = min(
                high, max(low, solution.variables[i]))
            mutated = True
    if mutated:
        solution.evaluate()


def repair(solution: Solution):
    for i in range(len(solution.value)):
        low, high = solution.problem.variable_bounds[i]
        solution.value[i] = solution.variables[i] if solution.variables[i] > low else low
        solution.value[i] = solution.variables[i] if solution.variables[i] < high else high
