#!/usr/bin/env python

"""Class definitions for optimization problems and solutions

author: jussiks
"""

import inspect
import math
import random
import numpy as np
from typing import List, Any


class Problem:
    def __init__(self, objective_function, constraint_function=None,
                 variable_bounds=None):
        """Initializes a new optimization problem."""
        self.objective_function = objective_function
        self.constraint_function = constraint_function
        self.variable_bounds = variable_bounds
        # TODO determine variable count

    def __call__(self, solution):
        if isinstance(solution, Solution):
            return self.objective_function(solution.variables)
        raise TypeError("Expected Solution as argument")

    def __str__(self):
        s = inspect.getsource(self.objective_function).strip() + "\n"
        if self.constraint_function:
            s += inspect.getsource(self.constraint_function)
        return s


class Solution:
    def __init__(self, problem: Problem, variables: List[Any]):
        """Initializes a new Solution to optimization problem."""
        self.problem = problem
        self.variables = np.array(variables)
        self.evaluate()

    def __str__(self):
        return str({
            "variables": self.variables,
            "value": self.value
        })

    def evaluate(self):
        self.value = self.problem(self)
        return self.value

    def within_constraints(self) -> bool:
        return all(self.problem.constraint_function(self.variables))


class MOPSolution(Solution):
    def __init__(self, problem, variables):
        """Initializes a new multiobjective optimization solution."""
        Solution.__init__(self, problem, variables)
        self.rank = None
        self.distance = None

    def __str__(self):
        return str({
            "variables": self.variables,
            "value": self.value,
            "rank": self.rank,
            "self": self.distance
        })

    def dominates(self, other_solution) -> bool:
        """Returns True if solution dominates other solution, False otherwise.

        Domination is defined as having at least one lower objective function
        value and none that are higher.
        """
        isBetter = False
        for val1, val2 in zip(self.value, other_solution.value):
            if val2 < val1:
                return False
            if val1 < val2:
                isBetter = True
        return isBetter

    def weakly_dominates(self, other_solution) -> bool:
        """Returns True if solution weakly dominates other solution, False
        otherwise.

        A solution is said to weakly dominate other_solution if all of it's
        objective function values are lower or equal compared to other
        solution.
        """
        for val1, val2 in zip(self.value, other_solution.value):
            if val2 < val1:
                return False
        return True


class WeightedSolution(MOPSolution):
    def __init__(self, problem, variables, weights):
        MOPSolution.__init__(self, problem, variables)
        self.weights = np.array(weights)

    def __str__(self):
        return str({
            "variables": self.variables,
            "value": self.value,
            "weights": self.weights
        })


class Particle(Solution):
    """Solution that is used in particle swarm optimization.

    Has an update method, and retains knowledge of best position and value.
    """
    def __init__(self, problem: Problem, variables: List, acc_const: float = 2,
                 max_velocity: float = 2):
        """Initializes a new Particle to be used in swarm optimization.

        Arguments
        func:           function that is being minimized. Function takes
                        a vector (list of floats) as its input.
        domain:         list of tuples that define the minimum and maximum
                        values for each value in function's input vector.
        acc_const:      acceleration constant that is applied to both
                        personal and social movement.
        max_velocity:   defines the maximum distance the particle can
                        move in one generation.
        """
        Solution.__init__(self, problem, variables)
        self.acc_const = acc_const
        self.max_velocity = max_velocity

        # Initialize a random velocity vector with length between 0 and
        # max_velocity
        self.velocity = np.array([
            random.uniform(-1, 1) for cons in self.problem.variable_bounds
            ])
        if self.max_velocity:
            velocity_length = random.uniform(0, self.max_velocity)
            self.velocity = velocity_length / np.linalg.norm(
                self.velocity) * self.velocity

        self.best_value = self.value
        self.best_variables = np.array(self.variables)

    def update(self, other_solution):
        """Updates the position of the particle.

        Particle is allowed to move outside the boundaries, but
        these values will not be considered as best values.
        """
        personal = self.acc_const * random.random() * (
            self.best_variables - self.variables)
        social = self.acc_const * random.random() * (
            other_solution.best_variables - self.variables)
        self.velocity = self.velocity + personal + social

        # Scale the vector to fit maximum velocity
        if self.max_velocity:
            vnorm = np.linalg.norm(self.velocity)
            if vnorm > self.max_velocity:
                self.velocity = self.max_velocity / vnorm * self.velocity

        self.variables += self.velocity

        # Update personal best if position is in range and current value
        # is less than previous best.
        if self.within_constraints():
            if self.evaluate() < self.best_value:
                self.best_value = self.value
                self.best_variables = np.array(self.variables)


def generate_variables(problem: Problem, continous: bool = True) -> List:
    var_func = lambda low, high: random.uniform(low, high) if continous else random.randint(low, high) 
    initial_variables = [var_func(low, high) 
        for low, high in problem.variable_bounds] # TODO handle none types
    #TODO evaluate constraints and repair variables
    return initial_variables


eamon = Problem(
    lambda x: - math.cos(x[0]) * math.cos(x[1]) * math.exp(
        -(x[0] - math.pi)**2 - (x[0] - math.pi)**2),
    [(0, 10), (0, 10), (0, 10)]
    )
