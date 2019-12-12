#!/usr/bin/env python

"""Unit tests for optimization methods.

author: jussiks
"""


import unittest
import genetic_methods as gm
from optimization import Solution, Problem, MOPSolution
import scipy.stats
import numpy as np


class TestGeneticMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._variable_count = 7
        cls._population_count = 6
        cls._offspring_count = 10000
        cls._accuracy = 0.05
        cls._problem = Problem(lambda x: x[0], lambda x: [1 < 2])
        cls._parent1 = Solution(cls._problem, [0] * cls._variable_count)
        cls._parent2 = Solution(cls._problem, [1] * cls._variable_count)
        cls._population = [cls._parent1 for i in range(cls._population_count)]

    def test_solution(self):
        self.assertRaises(IndexError, lambda: Solution(self._problem, []))
        s = Solution(self._problem, [3, 2])
        self.assertEqual(3, s.value)
        s.variables[0] = 5
        self.assertEqual(3, s.value)
        self.assertEqual(5, s.evaluate())
        self.assertEqual(5, s.value)

    def test_tournament_selection(self):
        # testing various values for parameter selection size
        self.assertEqual(
            [], gm.tournament_selection([], 1, selection_size=1))
        self.assertEqual(
            2, len(gm.tournament_selection(self._population, 1)))
        self.assertEqual(
            [], gm.tournament_selection(self._population, 1, selection_size=-1))
        self.assertEqual(
            [], gm.tournament_selection(self._population, 1, selection_size=0))
        self.assertEqual(
            self._population,
            gm.tournament_selection(
                self._population, 1, selection_size=self._population_count))
        self.assertEqual(
            self._population,
            gm.tournament_selection(
                self._population, 1, selection_size=self._population_count + 1))
        self.assertEqual(
            3, len(gm.tournament_selection(
                self._population, 1, selection_size=3)))

        # testing parameter k
        for i in range(20):
            self.assertEqual(
                [self._parent1], gm.tournament_selection(
                    [self._parent1, self._parent2], 2 + i, selection_size=1))
        for i in range(20):
            self.assertEqual(
                [self._parent1, self._parent2], gm.tournament_selection(
                    [self._parent1, self._parent2], 2 + i, selection_size=2))
        self.assertRaises(
            ValueError, lambda: len(gm.tournament_selection(
                self._population, -1)))
        self.assertRaises(
            ValueError, lambda: len(gm.tournament_selection(
                self._population, 0)))

        # testing parameter p
        sel = [gm.tournament_selection(
            [self._parent1, self._parent2], 2, p=0.5, selection_size=1)
               for i in range(100)]
        # both parents should be picked if 0 < p < 1
        self.assertIn(
            [self._parent2], sel)
        self.assertIn(
            [self._parent1], sel)
        for i in range(20):
            self.assertEqual(
                [self._parent1],
                 gm.tournament_selection(
                     [self._parent1, self._parent2], 2, p=1.5 + i, selection_size=1))
        for i in range(20):
            self.assertEqual(
                [self._parent2],
                 gm.tournament_selection(
                     [self._parent1, self._parent2], 2, p=0 - i, selection_size=1))

    def test_uniform_crossover(self):
        """Tests for uniform crossover.

        The function takes variables from both parents randomly."""

        # Crossover cannot be performed if length of variables differs
        # between parents
        parent3 = Solution(
            self._problem, [1] * (len(self._parent2.variables) + 1))
        self.assertRaises(
            ValueError, lambda: gm.uniform_crossover(self._parent1, parent3))

        pr = Problem(lambda x: 0)
        s1 = Solution(pr, [])
        s2 = Solution(pr, [])
        self.assertTrue(
            np.array_equal(np.array([]), gm.uniform_crossover(s1, s2)))

        # Generate some example offspring to see how the variables
        # get distributed.
        offspring = [gm.uniform_crossover(self._parent1, self._parent2)
                     for i in range(self._offspring_count)]

        counts, next_is_same = _calculate_counts_and_neighbours(offspring)
        expected = [self._offspring_count / 2] * self._variable_count

        self.assertTrue(_is_uniform(counts, expected))
        self.assertTrue(_is_uniform(next_is_same, expected))

    def test_single_point_crossover(self):
        """Tests for single point crossover.

        Function takes first n variables from one parent and m last
        variables from second parent, n being random number between
        0 and len(parent.variables)
        """
        # Method should fail if parents do not have same number of variables
        parent2 = Solution(self._problem, [1] * (self._variable_count + 1))
        self.assertRaises(
            ValueError, lambda: gm.uniform_crossover(self._parent1, parent2))

        pr = Problem(lambda x: 0)
        s = Solution(pr, [])
        self.assertTrue(
            np.array_equal(np.array([]), gm.single_point_crossover(s, s)))

        self.assertTrue(
            np.array_equal(np.array([]), gm.single_point_crossover(
                s, s, quarantee_both=True)))

        s = Solution(pr, [1])
        self.assertTrue(
            np.array_equal(np.array([1]), gm.single_point_crossover(
                s, s, quarantee_both=True)))

        # Generate offspring
        offspring = [gm.single_point_crossover(self._parent1, self._parent2,
                     quarantee_both=False)
                     for i in range(self._offspring_count)]

        counts, next_is_same = _calculate_counts_and_neighbours(offspring)
        expected = [self._offspring_count / 2] * self._variable_count

        self.assertTrue(_is_uniform(counts, expected))
        self.assertFalse(_is_uniform(next_is_same, expected))
        self.assertTrue(
            _arreq_in_list(self._parent1.variables, offspring))
        self.assertTrue(
            _arreq_in_list(self._parent2.variables, offspring))

        # include at least one variable from both parents
        offspring = [gm.single_point_crossover(
            self._parent1, self._parent2, quarantee_both=True)
                     for i in range(self._offspring_count)]

        counts, next_is_same = _calculate_counts_and_neighbours(offspring)

        self.assertTrue(_is_uniform(counts, expected))
        self.assertFalse(_is_uniform(next_is_same, expected))
        self.assertFalse(
            _arreq_in_list(self._parent1.variables, offspring))
        self.assertFalse(
            _arreq_in_list(self._parent2.variables, offspring))

        self.assertFalse(_arreq_in_list(self._parent1.variables, offspring))
        self.assertFalse(_arreq_in_list(self._parent2.variables, offspring))

    def test_dominates(self):
        problem = Problem(
                lambda x: [x[0], x[1], x[2]],
                lambda x: 1 < 2
            )
        sol1 = MOPSolution(problem, [0, 0, 0])
        sol2 = MOPSolution(problem, [0, 0, 0])
        self.assertFalse(sol1.dominates(sol2))
        self.assertFalse(sol2.dominates(sol1))

        sol1 = MOPSolution(problem, [0, 1, 0])
        sol2 = MOPSolution(problem, [0, 0, 0])
        self.assertFalse(sol1.dominates(sol2))
        self.assertTrue(sol2.dominates(sol1))

        sol1 = MOPSolution(problem, [0, 1, 0])
        sol2 = MOPSolution(problem, [1, 0, 0])
        self.assertFalse(sol1.dominates(sol2))
        self.assertFalse(sol2.dominates(sol1))

        sol1 = MOPSolution(problem, [0, 0, 0])
        sol2 = MOPSolution(problem, [1, 1, 1])
        self.assertTrue(sol1.dominates(sol2))
        self.assertFalse(sol2.dominates(sol1))

        sol1 = MOPSolution(problem, [1, 1, 0])
        sol2 = MOPSolution(problem, [1, 1, 1])
        self.assertTrue(sol1.dominates(sol2))
        self.assertFalse(sol2.dominates(sol1))

        sol1 = MOPSolution(problem, [1, 2, 0])
        sol2 = MOPSolution(problem, [1, 1, 1])
        self.assertFalse(sol1.dominates(sol2))
        self.assertFalse(sol2.dominates(sol1))

    def test_weakly_dominates(self):
        problem = Problem(
                lambda x: [x[0], x[1], x[2]],
                lambda x: 1 < 2
            )
        sol1 = MOPSolution(problem, [0, 0, 0])
        sol2 = MOPSolution(problem, [0, 0, 0])
        self.assertTrue(sol1.weakly_dominates(sol2))
        self.assertTrue(sol2.weakly_dominates(sol1))

        sol1 = MOPSolution(problem, [0, 1, 0])
        sol2 = MOPSolution(problem, [0, 0, 0])
        self.assertFalse(sol1.weakly_dominates(sol2))
        self.assertTrue(sol2.weakly_dominates(sol1))

        sol1 = MOPSolution(problem, [0, 1, 0])
        sol2 = MOPSolution(problem, [1, 0, 0])
        self.assertFalse(sol1.weakly_dominates(sol2))
        self.assertFalse(sol2.weakly_dominates(sol1))

        sol1 = MOPSolution(problem, [0, 0, 0])
        sol2 = MOPSolution(problem, [1, 1, 1])
        self.assertTrue(sol1.weakly_dominates(sol2))
        self.assertFalse(sol2.weakly_dominates(sol1))

        sol1 = MOPSolution(problem, [1, 1, 0])
        sol2 = MOPSolution(problem, [1, 1, 1])
        self.assertTrue(sol1.weakly_dominates(sol2))
        self.assertFalse(sol2.weakly_dominates(sol1))

        sol1 = MOPSolution(problem, [1, 2, 0])
        sol2 = MOPSolution(problem, [1, 1, 1])
        self.assertFalse(sol1.weakly_dominates(sol2))
        self.assertFalse(sol2.weakly_dominates(sol1))


def _is_uniform(observed, expected=None, significance=0.05):
    """Checks if the observed values are from the expected distribution
    with given statistical significance."""
    if expected:
        chisq, p = scipy.stats.chisquare(observed, expected)
    else:
        chisq, p = scipy.stats.chisquare(observed)
    return p > significance


def _calculate_counts_and_neighbours(data):
    counts = []
    next_is_same = []
    for x in data:
        while len(x) > len(counts):
            counts.append(0)
            next_is_same.append(0)
        for i in range(len(x)):
            counts[i] += x[i]
            next_is_same[i] += x[i] == x[(i + 1) % len(x)]

    return counts, next_is_same


# From
# https://stackoverflow.com/questions/23979146/check-if-numpy-array-is-in-list-of-numpy-arrays
def _arreq_in_list(myarr, list_arrays):
    return next(
        (True for elem in list_arrays if np.array_equal(elem, myarr)), False)


if __name__ == "__main__":
    unittest.main()
