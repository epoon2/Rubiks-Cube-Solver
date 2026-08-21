#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the depth-wall experiment.

These keep the README's claim about the uninformed solvers checkable: shallow
scrambles come back quickly, deep ones do not come back at all. The budgets
here are deliberately tiny so the suite stays fast; depth_wall.py itself is
the version to run for real numbers.
"""

import sys
import os
import unittest

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from depth_wall import (SOLVED, TIMEOUT, random_scramble, time_solver)
from solvers.bfs_solver import BFSSolver
import random


class TestDepthWall(unittest.TestCase):
    """Test cases for the depth-wall experiment."""

    def test_random_scramble_length(self):
        """Test that a scramble has the requested depth."""
        rng = random.Random(1)

        for depth in range(1, 12):
            scramble = random_scramble(depth, rng).split()
            self.assertEqual(len(scramble), depth)

    def test_random_scramble_never_repeats_a_face(self):
        """Test that consecutive moves never turn the same face."""
        rng = random.Random(2)
        scramble = random_scramble(60, rng).split()

        for previous, move in zip(scramble, scramble[1:]):
            self.assertNotEqual(previous[0], move[0])

    def test_shallow_scramble_is_within_reach(self):
        """Test that BFS still handles a three-move scramble."""
        rng = random.Random(3)
        scramble = random_scramble(3, rng)

        outcome, length, _, _ = time_solver(BFSSolver(max_depth=3), scramble,
                                            timeout=30.0)

        self.assertEqual(outcome, SOLVED)
        self.assertLessEqual(length, 3)

    def test_deep_scramble_is_out_of_reach(self):
        """Test that BFS cannot get through a seven-move scramble."""
        # Seven quarter turns is already 12^7 sequences to enumerate, so this
        # is not a matter of the budget being a little too small.
        rng = random.Random(4)
        scramble = random_scramble(7, rng)

        outcome, _, _, _ = time_solver(BFSSolver(max_depth=7), scramble,
                                       timeout=5.0)

        self.assertEqual(outcome, TIMEOUT)


if __name__ == '__main__':
    unittest.main()
