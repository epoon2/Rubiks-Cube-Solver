#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the Kociemba two-phase solver.

The point of this solver is that it handles scrambles no other solver here can
touch, so most of what follows is randomised round trips: scramble a cube,
solve it, apply the solution, check it is solved.
"""

import sys
import os
import random
import unittest

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from cube import RubiksCube
from solvers.cubie import MOVE_NAMES, CubieCube, from_facelets
from solvers.kociemba_solver import (MAX_TOTAL_DEPTH, KociembaSolver,
                                     simplify)
from solvers.kociemba_tables import TwoPhaseTables

# The tables cost a couple of seconds to generate the first time and are then
# cached on disk, so every test in this module shares one copy of them.
TABLES = TwoPhaseTables()

QUARTER_TURNS = [move for move in MOVE_NAMES if not move.endswith('2')]


def scramble_cube(algorithm):
    """
    Return a cube with the given algorithm applied.

    Args:
        algorithm (list): Moves in standard notation.

    Returns:
        RubiksCube: The scrambled cube.
    """
    cube = RubiksCube()
    cube.apply_algorithm(' '.join(algorithm))
    return cube


class TestKociembaSolver(unittest.TestCase):
    """Test cases for the two-phase solver."""

    def setUp(self):
        self.solver = KociembaSolver(tables=TABLES)

    def assert_solves(self, algorithm):
        """
        Solve a scramble and check the solution actually solves it.

        Args:
            algorithm (list): The scramble, in standard notation.

        Returns:
            list: The solution.
        """
        cube = scramble_cube(algorithm)
        success, solution, _, _ = self.solver.solve(cube)

        self.assertTrue(success, f"no solution for {' '.join(algorithm)}")
        self.assertLessEqual(len(solution), MAX_TOTAL_DEPTH)

        cube.apply_algorithm(' '.join(solution))
        self.assertTrue(cube.is_solved(),
                        f"{' '.join(solution)} does not solve "
                        f"{' '.join(algorithm)}")
        return solution

    def test_solved_cube_needs_no_moves(self):
        """Test that a solved cube comes back with an empty solution."""
        success, solution, _, _ = self.solver.solve(RubiksCube())

        self.assertTrue(success)
        self.assertEqual(solution, [])

    def test_solves_random_scrambles(self):
        """Test round trips over many random scrambles."""
        rng = random.Random(2024)

        for _ in range(40):
            algorithm = [rng.choice(MOVE_NAMES) for _ in range(25)]
            self.assert_solves(algorithm)

    def test_solves_scrambles_of_every_depth(self):
        """Test round trips over scrambles from one to sixteen moves."""
        rng = random.Random(7)

        for depth in range(1, 17):
            for _ in range(2):
                algorithm = [rng.choice(QUARTER_TURNS) for _ in range(depth)]
                self.assert_solves(algorithm)

    def test_solution_lengths_are_sane(self):
        """Test that solutions come in near the target length."""
        rng = random.Random(555)
        lengths = []

        for _ in range(20):
            algorithm = [rng.choice(MOVE_NAMES) for _ in range(30)]
            lengths.append(len(self.assert_solves(algorithm)))

        # Two-phase solutions are not optimal, but a random cube is about 18
        # moves from solved and the detour through G1 costs a handful more.
        # Anything past the low twenties means the search is not working.
        self.assertLessEqual(max(lengths), 25)
        self.assertGreaterEqual(min(lengths), 15)

    def test_solution_never_repeats_a_face(self):
        """Test that solutions come back tidy."""
        rng = random.Random(31)

        for _ in range(10):
            algorithm = [rng.choice(MOVE_NAMES) for _ in range(25)]
            solution = self.assert_solves(algorithm)

            for previous, move in zip(solution, solution[1:]):
                self.assertNotEqual(previous[0], move[0],
                                    f"{' '.join(solution)} turns a face twice")

    def test_accepts_a_cubie_cube(self):
        """Test that a cubie cube can be handed straight to the solver."""
        rng = random.Random(8)
        algorithm = [rng.choice(MOVE_NAMES) for _ in range(25)]

        cubie = CubieCube().apply_algorithm(algorithm)
        success, solution, _, _ = self.solver.solve(cubie)

        self.assertTrue(success)
        self.assertTrue(cubie.apply_algorithm(solution).is_solved())

    def test_rejects_impossible_cubes(self):
        """Test that a cube no scramble could produce is refused."""
        cube = RubiksCube()
        cubie = from_facelets(cube)
        # A single flipped edge is the classic impossible cube: legal to build
        # out of stickers, unreachable by any sequence of moves.
        cubie.eo[0] = 1

        with self.assertRaises(ValueError):
            self.solver.solve(cubie.to_facelets())

    def test_reports_search_effort(self):
        """Test that the solver reports nodes and time like the others."""
        rng = random.Random(4)
        cube = scramble_cube([rng.choice(MOVE_NAMES) for _ in range(25)])

        success, _, nodes, seconds = self.solver.solve(cube)

        self.assertTrue(success)
        self.assertGreater(nodes, 0)
        self.assertGreater(seconds, 0.0)


class TestSimplify(unittest.TestCase):
    """Test cases for the move-sequence tidier."""

    def test_merges_adjacent_turns(self):
        """Test that turns of the same face are combined."""
        self.assertEqual(simplify(['R', 'R']), ['R2'])
        self.assertEqual(simplify(['R', "R'"]), [])
        self.assertEqual(simplify(['R2', 'R2']), [])
        self.assertEqual(simplify(['R', 'R2']), ["R'"])

    def test_merges_through_the_opposite_face(self):
        """Test that a turn of the opposite face does not block a merge."""
        # U and D commute, so the two U turns can still be combined.
        self.assertEqual(simplify(['U', 'D2', "U'"]), ['D2'])
        self.assertEqual(simplify(['U', 'D', 'U']), ['U2', 'D'])

    def test_leaves_other_sequences_alone(self):
        """Test that unrelated turns are untouched."""
        self.assertEqual(simplify(['R', 'U', "R'"]), ['R', 'U', "R'"])

    def test_keeps_the_cube_state(self):
        """Test that tidying never changes what a sequence does."""
        rng = random.Random(17)

        for _ in range(200):
            algorithm = [rng.choice(MOVE_NAMES) for _ in range(rng.randint(1, 20))]

            before = CubieCube().apply_algorithm(algorithm)
            after = CubieCube().apply_algorithm(simplify(algorithm))

            self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
