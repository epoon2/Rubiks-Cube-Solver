#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the cubie-level cube representation.
"""

import sys
import os
import random
import unittest
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from cube import RubiksCube
from solvers.cubie import (CORNER_COUNT, EDGE_COUNT, MOVE_CUBES, MOVE_NAMES,
                           CubieCube, from_facelets)


class TestCubieCube(unittest.TestCase):
    """Test cases for the cubie cube."""

    def test_solved_cube_converts_to_identity(self):
        """Test that a solved sticker cube has every piece home."""
        cubie = from_facelets(RubiksCube())

        self.assertTrue(cubie.is_solved())
        self.assertEqual(cubie.check(), "")

    def test_quarter_turns_have_order_four(self):
        """Test that four quarter turns of a face come back to solved."""
        for face in ['U', 'R', 'F', 'D', 'L', 'B']:
            cube = CubieCube()
            for _ in range(4):
                cube = cube.apply_move(face)
            self.assertTrue(cube.is_solved(), f"{face} does not have order 4")

    def test_moves_agree_with_sticker_model(self):
        """Test that both representations of the cube stay in step."""
        rng = random.Random(11)

        for _ in range(50):
            algorithm = [rng.choice(MOVE_NAMES) for _ in range(rng.randint(0, 25))]

            stickers = RubiksCube()
            stickers.apply_algorithm(' '.join(algorithm))

            cubie = CubieCube().apply_algorithm(algorithm)

            self.assertEqual(from_facelets(stickers), cubie,
                             f"models disagree on {' '.join(algorithm)}")
            self.assertEqual(cubie.check(), "")

    def test_round_trip_through_stickers(self):
        """Test that converting to stickers and back is lossless."""
        rng = random.Random(12)

        for _ in range(50):
            algorithm = [rng.choice(MOVE_NAMES) for _ in range(rng.randint(0, 25))]
            cubie = CubieCube().apply_algorithm(algorithm)

            self.assertEqual(from_facelets(cubie.to_facelets()), cubie)

    def test_orientation_conventions(self):
        """Test which moves are allowed to twist and flip pieces."""
        # Phase 1 of the two-phase algorithm only works because orientation is
        # measured so that these are the only moves that disturb it.
        for name in MOVE_NAMES:
            move = MOVE_CUBES[name]
            quarter_turn = not name.endswith('2')

            self.assertEqual(any(move.eo), quarter_turn and name[0] in 'FB',
                             f"{name} flips the wrong edges")
            self.assertEqual(any(move.co), quarter_turn and name[0] in 'RLFB',
                             f"{name} twists the wrong corners")

    def test_check_rejects_illegal_cubes(self):
        """Test that impossible cubes are recognised as impossible."""
        flipped_edge = CubieCube()
        flipped_edge.eo[0] = 1
        self.assertNotEqual(flipped_edge.check(), "")

        twisted_corner = CubieCube()
        twisted_corner.co[0] = 1
        self.assertNotEqual(twisted_corner.check(), "")

        swapped_pair = CubieCube()
        swapped_pair.ep[0], swapped_pair.ep[1] = swapped_pair.ep[1], swapped_pair.ep[0]
        self.assertNotEqual(swapped_pair.check(), "")

    def test_from_facelets_rejects_broken_stickers(self):
        """Test that a sticker array that is not a cube is rejected."""
        cube = RubiksCube()
        # Paint a corner sticker the colour of the face opposite it, which
        # makes that corner a piece no Rubik's Cube has.
        cube.faces[2, 2, 2] = cube.faces[3, 1, 1]

        with self.assertRaises(ValueError):
            from_facelets(cube)

    def test_relabelled_stickers_still_convert(self):
        """Test that conversion reads the colours off the centres."""
        cube = RubiksCube()
        cube.apply_algorithm("R U R' U'")
        # Shift every colour by a constant; the cube is unchanged, only the
        # names of the colours are different.
        recoloured = RubiksCube()
        recoloured.faces = ((cube.faces + 3) % 6).astype(np.int8)

        self.assertEqual(from_facelets(recoloured), from_facelets(cube))

    def test_piece_counts(self):
        """Test the sizes of the piece arrays."""
        cube = CubieCube()

        self.assertEqual(len(cube.cp), CORNER_COUNT)
        self.assertEqual(len(cube.co), CORNER_COUNT)
        self.assertEqual(len(cube.ep), EDGE_COUNT)
        self.assertEqual(len(cube.eo), EDGE_COUNT)


if __name__ == '__main__':
    unittest.main()
