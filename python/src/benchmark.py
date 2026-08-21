#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark script to compare the performance of different solvers.

The three solvers here are not in the same league, so the benchmark has two
halves. The first runs every solver over scrambles of increasing depth, which
is where BFS and A* run out of road. The second runs the two-phase solver
alone over random scrambles and reports the distribution of its solve times
and solution lengths, which is the only one of the three where a distribution
is a meaningful thing to ask for.

Every solver runs in its own process under a time limit, because BFS on
anything past six moves does not come back.
"""

import random
import statistics
import time

import pandas as pd

from cube import RubiksCube
from depth_wall import SOLVED, TIMEOUT, random_scramble, time_solver
from solvers.bfs_solver import BFSSolver
from solvers.astar_solver import AStarSolver
from solvers.kociemba_solver import KociembaSolver

# Seconds allowed per solver run before it is written off as a failure.
RUN_TIMEOUT = 10.0


def run_benchmark(scrambles, solvers, num_trials=1, timeout=RUN_TIMEOUT):
    """
    Run benchmark tests for different solvers on different scrambles.

    Args:
        scrambles (list): List of (name, algorithm) tuples for scrambles to test.
        solvers (list): List of (name, solver_instance) tuples for solvers to test.
        num_trials (int): Number of trials to run for each combination.
        timeout (float): Seconds allowed per run.

    Returns:
        pd.DataFrame: DataFrame with benchmark results.
    """
    results = []

    for scramble_name, scramble_alg in scrambles:
        for solver_name, solver in solvers:
            for trial in range(num_trials):
                outcome, length, nodes, seconds = time_solver(
                    solver, scramble_alg, timeout)

                results.append({
                    'Scramble': scramble_name,
                    'Solver': solver_name,
                    'Trial': trial + 1,
                    'Success': outcome == SOLVED,
                    'Outcome': outcome,
                    'Solution Length': length,
                    'Nodes Explored': nodes,
                    'Time (seconds)': seconds,
                })

    return pd.DataFrame(results)


def run_distribution(solver, depth=25, samples=100, seed=0):
    """
    Solve many random scrambles with one solver and collect the results.

    Args:
        solver: Solver instance exposing solve(cube).
        depth (int): Length of each random scramble.
        samples (int): Number of scrambles to solve.
        seed (int): Seed for the scramble generator.

    Returns:
        pd.DataFrame: One row per scramble.
    """
    rng = random.Random(seed)
    rows = []

    for sample in range(samples):
        scramble = random_scramble(depth, rng)
        cube = RubiksCube()
        cube.apply_algorithm(scramble)

        started = time.time()
        success, solution, nodes, _ = solver.solve(cube)
        seconds = time.time() - started

        # Never report a solve time without checking the solve.
        cube.apply_algorithm(' '.join(solution))
        if not (success and cube.is_solved()):
            raise AssertionError(f"solver failed on {scramble}")

        rows.append({
            'Sample': sample + 1,
            'Moves': len(solution),
            'Nodes Explored': nodes,
            'Time (seconds)': seconds,
        })

    return pd.DataFrame(rows)


def describe_distribution(name, values, unit='', digits=3):
    """
    Print a one-line summary of a distribution.

    Args:
        name (str): What is being summarised.
        values (list): The numbers.
        unit (str): Suffix for the numbers.
        digits (int): Decimal places to show.
    """
    ordered = sorted(values)
    percentile95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    summary = {
        'min': ordered[0],
        'median': statistics.median(ordered),
        'mean': statistics.mean(ordered),
        'p95': percentile95,
        'max': ordered[-1],
    }
    parts = ' '.join(f"{label} {value:.{digits}f}{unit}"
                     for label, value in summary.items())
    print(f"  {name:<18} {parts}")


def main():
    """Run the benchmarking."""
    rng = random.Random(1)

    # Depths chosen to straddle the point where the uninformed solvers give
    # up: they manage four and six moves, and nothing beyond.
    scrambles = [
        ('Solved', ''),
        ('Single Move', 'F'),
        ('Two Moves', 'F U'),
        ('Four Moves', "R U R' U'"),
        ('Six Moves', random_scramble(6, rng)),
        ('Eight Moves', random_scramble(8, rng)),
        ('Random 25', random_scramble(25, rng)),
        ('Random 25 (2)', random_scramble(25, rng)),
    ]

    print("Generating two-phase tables...")
    two_phase = KociembaSolver(verbose=True)

    solvers = [
        ('BFS (depth=8)', BFSSolver(max_depth=8)),
        ('A* (depth=8)', AStarSolver(max_depth=8)),
        ('Two-phase', two_phase),
    ]

    print(f"\nRunning benchmarks ({RUN_TIMEOUT:.0f}s limit per run)...")
    results = run_benchmark(scrambles, solvers, num_trials=1)

    print("\nBenchmark Results:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 140)
    print(results.to_string(index=False))

    print("\nSolved within the limit:")
    solved = results.pivot_table(index='Scramble', columns='Solver',
                                 values='Success', aggfunc='mean',
                                 sort=False)
    print(solved.to_string())

    print("\nSolving 100 random 25-move scrambles with the two-phase solver...")
    distribution = run_distribution(two_phase, depth=25, samples=100)

    describe_distribution('solve time', distribution['Time (seconds)'], 's')
    describe_distribution('solution moves', distribution['Moves'], digits=1)
    describe_distribution('nodes explored', distribution['Nodes Explored'],
                          digits=0)

    print("\n  Solution length distribution:")
    for moves, count in sorted(distribution['Moves'].value_counts().items()):
        print(f"    {moves:2d} moves  {'#' * count} ({count})")

    print("\nFor comparison, on the same scrambles the other two solvers "
          "return nothing:")
    for solver_name, solver in solvers[:2]:
        outcome, _, _, seconds = time_solver(solver, scrambles[-1][1],
                                             RUN_TIMEOUT)
        verdict = 'timed out' if outcome == TIMEOUT else outcome
        print(f"  {solver_name:<15} {verdict} after {seconds:.1f}s")


if __name__ == '__main__':
    main()
