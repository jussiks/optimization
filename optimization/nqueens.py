#!/usr/bin/env python

"""Genetic algorithm for solving the N-Queens problem.

In N-Queens problem the idea is to place n queens on a chess board so that
no two queens occupy the same row, column or diagonal. solve_nqueens method
provides a genetic algorithm to solve this problem. The steps of the algorithm
are as follows:

1. Create initial generation of size m and set it as current generation
2. Create an empty next generation
3. While size of next generation is lower than current generation
    3.1 Perform tournament selection to find two parents from current
        generation.
    3.2 Use uniform crossover and mutation to generate a child
    3.3 Add child to next generation
4. Set next generation as current generation

author: jussiks
"""

from optimization import Solution, Problem, generate_variables
import genetic_methods as gm
import math


def solve_nqueens(n=8, pop_size=10, mutation_p=0.1,
                  max_generations=10000, k=3, elitist=False,
                  print_gen=False):
    """Tries to solve N-Queens problem using genetic algorithm.

    Returns the best solution found and number of generations
    it took to find it.

    Arguments
    n                   number of queens
    pop_size            size of the population used in genetic
                        algorithm
    mutation_p          probability of mutation
    max_generations     maximum number of generations
    k                   number of solutions picked for tournament
                        selection
    elitist             if True, child is only added to the next
                        generation if it is better than the parents
    print_gen           if True, information about the current generation
                        is printed
    """

    # Set up the problem and generate first generation of solutions.
    problem = Problem(
        fitness,
        variable_bounds=[(1, n) for i in range(n)])
    current_gen = [
        Solution(problem, generate_variables(problem, continous=False))
        for i in range(pop_size)
        ]

    best_solution = min(current_gen, key=lambda x: x.value)

    for generation_count in range(max_generations):
        # Generate new generations until a correct solution is found or
        # maximum number of generations is reached
        next_gen = []
        while len(next_gen) < len(current_gen):
            # Select parents, perform crossover and mutations
            parents = gm.tournament_selection(current_gen, k, selection_size=2)
            child_vars = gm.uniform_crossover(parents[0], parents[1])
            child = Solution(problem, child_vars)
            gm.uniform_mutation(child, mutation_p, continous=False)

            if child.value < best_solution.value:
                best_solution = child
                if best_solution.value == 0:
                    break
            if elitist:
                parents.append(child)
                next_gen.append(min(parents, key=lambda x: x.value))
            else:
                next_gen.append(child)
        else:
            current_gen = next_gen
            if print_gen:
                print("---------- GENERATION " + str(generation_count))
                for s in current_gen:
                    print(s.variables, s.value)
            continue
        break

    return best_solution, generation_count


def fitness(chromosome):
    """Returns the fitness value of a chromosome.

    Fitness is is calculated by summing up all the occasions where a queen is
    under attack by some other queen (if three or more queens exists on the
    same row or diagonal, all of them are considered threatening each other).
    """
    fitness = 0
    for i in range(len(chromosome)):
        for j in range(i + 1, len(chromosome)):
            # Calculate the distances of chromosome[i] and chromosome[j]
            x_distance = math.fabs(j - i)
            y_distance = math.fabs(chromosome[j] - chromosome[i])
            # Increment fitness value if chromosome[i] and chromosome[j] are
            # on the same row or diagonal.
            fitness += y_distance == 0
            fitness += x_distance == y_distance
    return fitness


if __name__ == "__main__":
    # Parse command line arguments given to the script
    import argparse
    from timeit import default_timer as timer

    parser = argparse.ArgumentParser(
        description="Tries to find solution to N-Queens problem")
    parser.add_argument("-n", type=int, default=8,
                        help="number of queens")
    parser.add_argument("--pop_size", type=int, default=10,
                        help="size of population in evolutionary algorithm")
    parser.add_argument("--mutation_p", type=float, default=0.1,
                        help="likelihood of mutation in single gene")
    parser.add_argument("--max_gen", type=int, default=1000,
                        help="maximum number of generations")
    parser.add_argument("-k", type=int, default=3,
                        help="number of solutions picked for tournament selection")
    parser.add_argument("--use_elitism", dest="use_elitism",
                        action="store_true",
                        help="allows preservation of parents")
    parser.add_argument("--print_gen", dest="print_gen", action="store_true",
                        help="script will print out each generation")
    parser.set_defaults(use_elitism=False)
    parser.set_defaults(print_gen=False)
    args = parser.parse_args()

    # Solve the problem
    start = timer()
    sol, gen = solve_nqueens(
        n=args.n,
        pop_size=args.pop_size,
        mutation_p=args.mutation_p,
        max_generations=args.max_gen,
        k=args.k,
        elitist=args.use_elitism,
        print_gen=args.print_gen
        )
    stop = timer()

    # Print out the results
    print("Found solution {0} with fitness value of {1} in {2} generations and {3} seconds.".format(
        sol.variables, sol.value, gen, stop - start))
