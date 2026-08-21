#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the two-phase coordinate and pruning tables.
"""

import sys
import os
import random
import unittest
from itertools import permutations

import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from solvers.cubie import MOVE_CUBES, MOVE_NAMES, CubieCube
from solvers import kociemba_tables as kt

# Generating the tables is a one-off cost that is then cached on disk.
TABLES = kt.TwoPhaseTables()


class TestCoordinates(unittest.TestCase):
    """Test cases for the coordinate encodings."""

    def test_solved_cube_is_the_origin(self):
        """Test that every coordinate is 0 for a solved cube."""
        solved = CubieCube()

        self.assertEqual(kt.twist_coordinate(solved), 0)
        self.assertEqual(kt.flip_coordinate(solved), 0)
        self.assertEqual(kt.slice_coordinate(solved), 0)
        self.assertEqual(kt.corner_perm_coordinate(solved), 0)
        self.assertEqual(kt.ud_edge_perm_coordinate(solved), 0)
        self.assertEqual(kt.slice_perm_coordinate(solved), 0)

    def test_twist_encoding_is_a_bijection(self):
        """Test that every corner orientation coordinate round-trips."""
        for coordinate in range(kt.TWIST_COUNT):
            cube = kt.cube_with_twist(coordinate)
            self.assertEqual(sum(cube.co) % 3, 0)
            self.assertEqual(kt.twist_coordinate(cube), coordinate)

    def test_flip_encoding_is_a_bijection(self):
        """Test that every edge orientation coordinate round-trips."""
        for coordinate in range(kt.FLIP_COUNT):
            cube = kt.cube_with_flip(coordinate)
            self.assertEqual(sum(cube.eo) % 2, 0)
            self.assertEqual(kt.flip_coordinate(cube), coordinate)

    def test_slice_encoding_is_a_bijection(self):
        """Test that every slice coordinate round-trips."""
        seen = set()

        for coordinate in range(kt.SLICE_COUNT):
            slots = kt._unrank_combination(coordinate)
            self.assertEqual(len(set(slots)), 4)
            self.assertEqual(kt._rank_combination(slots), coordinate)
            seen.add(tuple(slots))

        self.assertEqual(len(seen), kt.SLICE_COUNT)

    def test_permutation_rank_is_lexicographic(self):
        """Test that permutation ranks match lexicographic order."""
        for rank, values in enumerate(permutations(range(5))):
            self.assertEqual(kt._rank_permutation(list(values)), rank)


class TestMoveTables(unittest.TestCase):
    """Test cases for the coordinate move tables."""

    def _random_cube(self, rng, moves=None):
        """Return a cube scrambled with random moves."""
        pool = moves if moves is not None else MOVE_NAMES
        return CubieCube().apply_algorithm(
            [rng.choice(pool) for _ in range(rng.randint(1, 25))])

    def test_phase1_tables_match_the_cube(self):
        """Test phase-1 move tables against moving a real cube."""
        rng = random.Random(3)

        for _ in range(200):
            cube = self._random_cube(rng)
            twist = kt.twist_coordinate(cube)
            flip = kt.flip_coordinate(cube)
            middle = kt.slice_coordinate(cube)

            for column, move in enumerate(kt.PHASE1_MOVES):
                moved = cube.multiply(MOVE_CUBES[move])

                self.assertEqual(TABLES.twist_move[twist, column],
                                 kt.twist_coordinate(moved))
                self.assertEqual(TABLES.flip_move[flip, column],
                                 kt.flip_coordinate(moved))
                self.assertEqual(TABLES.slice_move[middle, column],
                                 kt.slice_coordinate(moved))

    def test_phase2_tables_match_the_cube(self):
        """Test phase-2 move tables against moving a real cube."""
        # The U/D edge and slice permutation coordinates only mean anything
        # inside G1, so scramble with G1's own moves.
        rng = random.Random(5)

        for _ in range(200):
            cube = self._random_cube(rng, kt.PHASE2_MOVES)
            corners = kt.corner_perm_coordinate(cube)
            ud_edges = kt.ud_edge_perm_coordinate(cube)
            middle = kt.slice_perm_coordinate(cube)

            for column, move in enumerate(kt.PHASE2_MOVES):
                moved = cube.multiply(MOVE_CUBES[move])

                self.assertEqual(TABLES.corner_move[corners, column],
                                 kt.corner_perm_coordinate(moved))
                self.assertEqual(TABLES.ud_edge_move[ud_edges, column],
                                 kt.ud_edge_perm_coordinate(moved))
                self.assertEqual(TABLES.slice_perm_move[middle, column],
                                 kt.slice_perm_coordinate(moved))


class TestPruningTables(unittest.TestCase):
    """Test cases for the pruning tables."""

    PRUNING_TABLES = [
        ('flip_slice_prune', 'flip_move', 'slice_move', kt.SLICE_COUNT),
        ('twist_slice_prune', 'twist_move', 'slice_move', kt.SLICE_COUNT),
        ('corner_slice_prune', 'corner_move', 'slice_perm_move',
         kt.SLICE_PERM_COUNT),
        ('ud_edge_slice_prune', 'ud_edge_move', 'slice_perm_move',
         kt.SLICE_PERM_COUNT),
    ]

    def test_every_state_has_a_distance(self):
        """Test that the breadth-first search reached the whole space."""
        for name, _, _, _ in self.PRUNING_TABLES:
            table = getattr(TABLES, name)
            self.assertEqual(int((table == 255).sum()), 0,
                             f"{name} has unreachable entries")
            self.assertEqual(table[0], 0, f"{name} does not start at 0")

    def test_distances_change_by_at_most_one_per_move(self):
        """Test that the tables really are breadth-first distances."""
        # A table that is off by more than one move somewhere would make the
        # heuristic inadmissible and the solver could return a wrong answer.
        rng = random.Random(13)

        for name, first_name, second_name, second_count in self.PRUNING_TABLES:
            table = getattr(TABLES, name)
            first_table = getattr(TABLES, first_name)
            second_table = getattr(TABLES, second_name)

            for _ in range(2000):
                state = rng.randrange(len(table))
                first, second = divmod(state, second_count)
                distance = int(table[state])

                for column in range(first_table.shape[1]):
                    neighbour = (int(first_table[first, column]) * second_count
                                 + int(second_table[second, column]))
                    self.assertLessEqual(abs(int(table[neighbour]) - distance), 1,
                                         f"{name} jumps by more than one move")

    def test_tables_are_cached(self):
        """Test that a second load reuses the files on disk."""
        reloaded = kt.TwoPhaseTables(directory=TABLES.directory)

        self.assertEqual(reloaded.build_seconds, 0.0)
        for name in kt.TwoPhaseTables.TABLE_NAMES:
            self.assertTrue(np.array_equal(getattr(reloaded, name),
                                           getattr(TABLES, name)))


if __name__ == '__main__':
    unittest.main()
