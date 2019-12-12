#!/usr/bin/env python

"""Implementations of various methods used in NSGA-2

Fast-non-dominated-sort, crowding-distance-assignment and partial order
selection based on the paper "A Fast and Elitist Multiobjective Genetic
Algorithm: NSGA-II" (by Deb & al. (2002)).

author: jussiks
"""

from optimization import MOPSolution, Problem
from typing import List, Dict


def fast_nondominated_sort(solutions: List[MOPSolution]) -> Dict:
    """Sorts solutions into domination fronts.

    Returns the fronts as a dictionary where keys indicate the number of
    the front and values are lists of solutions belonging to each front.

    Arguments:
    solutions       list of MOPSolutions
    """
    fronts = {1: []}
    S = {}
    n = {}
    for p in solutions:
        # Iterate through each solution to find a set of non-dominated
        # solutions, and set them as the first front.
        S[p] = []
        n[p] = 0
        for q in solutions:
            if p.dominates(q):
                S[p].append(q)
            elif q.dominates(p):
                n[p] += 1
        if n[p] == 0:
            p.rank = 1
            fronts[1].append(p)
    i = 1

    while fronts[i]:
        # Create next front by iterating through solutions dominated
        # by solutions in current front.
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    q.rank = i + 1
                    Q.append(q)
        i += 1
        fronts[i] = Q
    return fronts


def crowding_distance(solutions: List[MOPSolution]):
    """Calculates the crowding distance for each solution.

    Arguments:
    solutions       list of MOPSolutions
    """

    sol_count = len(solutions)
    for sol in solutions:
        sol.distance = 0
    for i in range(len(solutions[0].value)):
        # Make a sorted copy of the list to avoid changing the order of the
        # original list. Note that objects in list are not copied, so we can
        # still assign distance value original objects while iterating over
        # the copied list.
        solutions_sorted = sorted(solutions, key=lambda sol: sol.value[i])
        solutions_sorted[0].distance = float("inf")
        solutions_sorted[-1].distance = float("inf")
        for j in range(1, sol_count - 2):
            neighbour_distance = float(
                solutions_sorted[j + 1].value[i] - solutions_sorted[j - 1].value[i])
            if neighbour_distance == 0:
                continue
            max_distance = float(
                solutions_sorted[-1].value[i] - solutions_sorted[0].value[i])
            solutions_sorted[j].distance += neighbour_distance / max_distance


def partial_order_selection(solutions: List[MOPSolution], n: int):
    """Returns n solutions from current solutions using partial order.

    Lower ranked solutions are picked first and for solutions that have
    the same rank, a higher crowding distance is the deciding factor.

    Arguments
    solutions   list of MOPSolutions
    n           number of solutions to select
    """

    # Sort solutions into fronts.
    fronts = fast_nondominated_sort(solutions)
    P = []
    i = 1
    # TODO only calculate the crowding distance for the last front
    crowding_distance(solutions)
    while len(P) + len(fronts[i]) <= n:
        # Add fronts to next generation until N is reached.
        P += fronts[i]
        i += 1

    # Sort solutions in next front by distance and pick ones that
    # are most isolated.
    fronts[i].sort(key=lambda sol: sol.distance, reverse=True)
    P += fronts[i][0:n - len(P)]
    return P


if __name__ == "__main__":
    problem = Problem(
        lambda x: [x[0], x[1]],
        constraint_function=lambda x: [0 <= x[0] <= 10, 0 <= x[1] <= 10],
        variable_bounds=[(0, 10) for i in range(2)]
        )
    # Initialize a set of solutions
    solutions = [
        MOPSolution(problem, [0, 1]),
        MOPSolution(problem, [1, 0]),
        MOPSolution(problem, [2, 1.5]),
        MOPSolution(problem, [1.5, 3]),
        MOPSolution(problem, [3, 1.6]),
        MOPSolution(problem, [4, 3.5]),
        MOPSolution(problem, [4.5, 3.1]),
        MOPSolution(problem, [5, 2.5]),
        MOPSolution(problem, [6, 5]),
        MOPSolution(problem, [5.5, 7]),
        MOPSolution(problem, [4.2, 6]),
        MOPSolution(problem, [3.3, 6.5])
    ]

    # Sort solutions into fronts.
    fronts = fast_nondominated_sort(solutions)
    print("\nFRONTS")
    for front in fronts:
        print("Front {0}:".format(front))
        for s in fronts[front]:
            print(s.value)

    # Calculate crowding distances
    print("\nCROWDING DISTANCES")
    crowding_distance(solutions)
    for sol in solutions:
        print("{0}\t{1}".format(sol.value, sol.distance))

    # Select 6 solutions using partial order selection.
    next_gen = partial_order_selection(solutions, 6)
    print("\nSELECTION")
    for n in next_gen:
        print(n)
