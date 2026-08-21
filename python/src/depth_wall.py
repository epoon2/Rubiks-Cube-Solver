#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Measure the scramble depth at which the uninformed solvers stop working.

A 3x3 cube has 4.3e19 reachable states and a branching factor of 12 for
quarter turns, so any search that enumerates states from the solved position
outwards runs out of time and memory after a handful of moves. This script
pins down where that happens for the BFS and A* solvers in this repository
instead of leaving it as an assertion in the README.

Each solver run happens in its own process so that a run which never returns,
or one that eats all of the memory, can be cut off and recorded as a failure
rather than taking the experiment down with it.
"""

import argparse
import multiprocessing
import random
import resource
import sys
import time

from cube import RubiksCube
from solvers.bfs_solver import BFSSolver
from solvers.astar_solver import AStarSolver

QUARTER_TURNS = ['F', "F'", 'B', "B'", 'U', "U'", 'D', "D'", 'L', "L'", 'R', "R'"]

# Outcomes recorded for a single solver run.
SOLVED = 'solved'
EXHAUSTED = 'exhausted'
TIMEOUT = 'timeout'
OUT_OF_MEMORY = 'out of memory'


def random_scramble(depth, rng):
    """
    Build a random scramble of the requested depth.

    Args:
        depth (int): Number of quarter turns to apply.
        rng (random.Random): Source of randomness.

    Returns:
        str: Space-separated moves.
    """
    moves = []
    for _ in range(depth):
        # Never turn the same face twice in a row: that either cancels out or
        # collapses into one turn, which would make the scramble shallower
        # than advertised and flatter the searchers.
        choices = [move for move in QUARTER_TURNS
                   if not moves or move[0] != moves[-1][0]]
        moves.append(rng.choice(choices))
    return ' '.join(moves)


def _run_solver(solver, scramble, memory_limit_gb, results):
    """
    Solve one scramble and post the outcome back to the parent process.

    Args:
        solver: Solver instance exposing solve(cube).
        scramble (str): Scramble to apply before solving.
        memory_limit_gb (float): Address space cap for this process.
        results (multiprocessing.Queue): Queue to post the outcome on.
    """
    limit = int(memory_limit_gb * 1024 ** 3)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    cube = RubiksCube()
    cube.apply_algorithm(scramble)

    started = time.time()
    try:
        success, solution, nodes, _ = solver.solve(cube)
    except MemoryError:
        results.put((OUT_OF_MEMORY, None, None, time.time() - started))
        return

    outcome = SOLVED if success else EXHAUSTED
    results.put((outcome, len(solution), nodes, time.time() - started))


def time_solver(solver, scramble, timeout, memory_limit_gb=2.0):
    """
    Run a solver on a scramble under a wall-clock and a memory limit.

    Args:
        solver: Solver instance exposing solve(cube).
        scramble (str): Scramble to apply before solving.
        timeout (float): Seconds to allow before giving up on the run.
        memory_limit_gb (float): Address space cap for the solver process.

    Returns:
        tuple: (outcome, solution_length, nodes_explored, seconds). The last
               three are None when the run did not finish.
    """
    results = multiprocessing.Queue()
    worker = multiprocessing.Process(
        target=_run_solver, args=(solver, scramble, memory_limit_gb, results))

    started = time.time()
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        worker.terminate()
        worker.join()
        return TIMEOUT, None, None, time.time() - started

    if results.empty():
        # The process died without reporting, which in practice means the
        # kernel killed it for its memory use.
        return OUT_OF_MEMORY, None, None, time.time() - started

    return results.get()


def run_experiment(depths, trials, timeout, seed=0, memory_limit_gb=2.0):
    """
    Time BFS and A* on random scrambles of increasing depth.

    Each solver gets its depth limit set to the scramble depth, which is the
    most generous setting possible: it is told exactly how deep to look.

    Args:
        depths (iterable): Scramble depths to test.
        trials (int): Scrambles per depth.
        timeout (float): Seconds allowed per solver run.
        seed (int): Seed for the scramble generator.
        memory_limit_gb (float): Address space cap per solver process.

    Returns:
        list: One dict per (depth, solver, trial) run.
    """
    rng = random.Random(seed)
    results = []

    for depth in depths:
        for trial in range(trials):
            scramble = random_scramble(depth, rng)
            solvers = [
                ('BFS', BFSSolver(max_depth=depth)),
                ('A*', AStarSolver(max_depth=depth)),
            ]

            for name, solver in solvers:
                outcome, length, nodes, seconds = time_solver(
                    solver, scramble, timeout, memory_limit_gb)
                results.append({
                    'depth': depth,
                    'solver': name,
                    'trial': trial + 1,
                    'scramble': scramble,
                    'outcome': outcome,
                    'length': length,
                    'nodes': nodes,
                    'seconds': seconds,
                })
                print(f"  depth {depth:2d} trial {trial + 1} {name:4s} "
                      f"{outcome:13s} {seconds:8.2f}s "
                      f"nodes={nodes if nodes is not None else '-'}")

    return results


def summarise(results, timeout):
    """
    Print a per-depth summary of the experiment.

    Args:
        results (list): Rows returned by run_experiment.
        timeout (float): Timeout the experiment ran with, for the header.
    """
    print(f"\nSummary (limit: {timeout:.0f}s wall clock per run)")
    print(f"{'depth':>5} {'solver':>6} {'solved':>7} {'median s':>9} "
          f"{'median moves':>13} {'median nodes':>13}")

    depths = sorted({row['depth'] for row in results})
    for depth in depths:
        for solver in ['BFS', 'A*']:
            rows = [row for row in results
                    if row['depth'] == depth and row['solver'] == solver]
            solved = [row for row in rows if row['outcome'] == SOLVED]
            share = f"{len(solved)}/{len(rows)}"

            if solved:
                seconds = f"{_median([row['seconds'] for row in solved]):9.2f}"
                moves = f"{_median([row['length'] for row in solved]):13.0f}"
                nodes = f"{_median([row['nodes'] for row in solved]):13.0f}"
            else:
                outcomes = {row['outcome'] for row in rows}
                seconds = f"{'-':>9}"
                moves = f"{'-':>13}"
                nodes = f"{', '.join(sorted(outcomes)):>13}"

            print(f"{depth:>5} {solver:>6} {share:>7} {seconds} {moves} {nodes}")


def _median(values):
    """Return the median of a non-empty list of numbers."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def main():
    """Run the depth-wall experiment from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-depth', type=int, default=1)
    parser.add_argument('--max-depth', type=int, default=8)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=60.0,
                        help='Seconds allowed per solver run.')
    parser.add_argument('--memory-limit-gb', type=float, default=2.0,
                        help='Address space allowed per solver process.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    depths = range(args.min_depth, args.max_depth + 1)
    print(f"Timing BFS and A* on random scrambles of depth "
          f"{args.min_depth}-{args.max_depth}, {args.trials} scrambles each.")

    results = run_experiment(depths, args.trials, args.timeout,
                             args.seed, args.memory_limit_gb)
    summarise(results, args.timeout)


if __name__ == '__main__':
    sys.exit(main())
