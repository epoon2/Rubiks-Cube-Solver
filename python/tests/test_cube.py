#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the RubiksCube class.
"""

import sys
import os
import unittest
import numpy as np

# Add the src directory to the path so we can import the cube module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from cube import RubiksCube

FRONT, BACK, UP, DOWN, LEFT, RIGHT = range(6)

# Which face each border row/column of a face touches, read straight off the
# strips that cube.py cycles in its move methods.
_NEIGHBOURS = {
    FRONT: {'row0': UP, 'row2': DOWN, 'col0': LEFT, 'col2': RIGHT},
    BACK: {'row0': UP, 'row2': DOWN, 'col0': RIGHT, 'col2': LEFT},
    UP: {'row0': BACK, 'row2': FRONT, 'col0': LEFT, 'col2': RIGHT},
    DOWN: {'row0': FRONT, 'row2': BACK, 'col0': LEFT, 'col2': RIGHT},
    LEFT: {'row0': UP, 'row2': DOWN, 'col0': BACK, 'col2': FRONT},
    RIGHT: {'row0': UP, 'row2': DOWN, 'col0': FRONT, 'col2': BACK},
}

# Where each of the 54 stickers starts out, keyed by its label.
_STICKER_HOME = {face * 9 + row * 3 + col: (face, row, col)
                 for face in range(6) for row in range(3) for col in range(3)}


def _labelled_faces():
    """Return a face array whose 54 stickers all carry a distinct label."""
    return np.arange(54, dtype=np.int16).reshape((6, 3, 3))


def _cubie_faces(face, row, col):
    """Return the set of faces the sticker at this position belongs to."""
    faces = {face}
    if row == 0:
        faces.add(_NEIGHBOURS[face]['row0'])
    if row == 2:
        faces.add(_NEIGHBOURS[face]['row2'])
    if col == 0:
        faces.add(_NEIGHBOURS[face]['col0'])
    if col == 2:
        faces.add(_NEIGHBOURS[face]['col2'])
    return frozenset(faces)


def _cubies():
    """Group the 54 sticker positions into the cubies they sit on."""
    groups = {}
    for position in _STICKER_HOME.values():
        groups.setdefault(_cubie_faces(*position), []).append(position)
    return [positions for positions in groups.values() if len(positions) > 1]


class TestRubiksCube(unittest.TestCase):
    """Test cases for the RubiksCube class."""
    
    def test_initialization(self):
        """Test that a new cube is properly initialized in solved state."""
        cube = RubiksCube()
        
        # Check dimensions
        self.assertEqual(cube.size, 3)
        self.assertEqual(cube.faces.shape, (6, 3, 3))
        
        # Check that each face has a uniform color
        for i in range(6):
            self.assertTrue(np.all(cube.faces[i] == i))
        
        # Check that the cube is recognized as solved
        self.assertTrue(cube.is_solved())
    
    def test_single_moves(self):
        """Test that single moves properly change the cube state."""
        cube = RubiksCube()
        
        # Apply a single move
        cube.move_F()
        
        # Cube should no longer be solved
        self.assertFalse(cube.is_solved())
        
        # Apply three more F moves to get back to solved
        for _ in range(3):
            cube.move_F()
        
        # Now the cube should be solved again
        self.assertTrue(cube.is_solved())
    
    def test_apply_move(self):
        """Test the apply_move method with various notations."""
        cube = RubiksCube()
        
        # Test F move
        cube.apply_move("F")
        self.assertFalse(cube.is_solved())
        
        # Test F' move (should undo F)
        cube.apply_move("F'")
        self.assertTrue(cube.is_solved())
        
        # Test R and R' moves
        cube.apply_move("R")
        self.assertFalse(cube.is_solved())
        cube.apply_move("R'")
        self.assertTrue(cube.is_solved())
    
    def test_half_turns(self):
        """Test that X2 is the same as turning X twice."""
        for move in ['F', 'B', 'U', 'D', 'L', 'R']:
            half_turn = RubiksCube()
            half_turn.apply_move(f"{move}2")

            twice = RubiksCube()
            twice.apply_move(move)
            twice.apply_move(move)

            self.assertTrue(np.array_equal(half_turn.faces, twice.faces))

            # A half turn is its own inverse.
            half_turn.apply_move(f"{move}2")
            self.assertTrue(half_turn.is_solved())

    def test_invalid_move(self):
        """Test that unknown notation is rejected."""
        cube = RubiksCube()

        for move in ['X', 'F3', 'RR', '']:
            with self.assertRaises(ValueError):
                cube.apply_move(move)

    def test_apply_algorithm(self):
        """Test applying a sequence of moves."""
        cube = RubiksCube()
        
        # Apply a simple algorithm
        cube.apply_algorithm("R U R' U'")
        self.assertFalse(cube.is_solved())
        
        # Apply the inverse to get back to solved
        cube.apply_algorithm("U R U' R'")
        self.assertTrue(cube.is_solved())
        
        # Test a longer algorithm
        # Sexy move six times returns to solved state
        cube.apply_algorithm("R U R' U' R U R' U' R U R' U' R U R' U' R U R' U' R U R' U'")
        self.assertTrue(cube.is_solved())
    
    def test_reset(self):
        """Test the reset method."""
        cube = RubiksCube()
        
        # Scramble the cube
        cube.apply_algorithm("F R U B' L D'")
        self.assertFalse(cube.is_solved())
        
        # Reset the cube
        cube.reset()
        self.assertTrue(cube.is_solved())

    def test_turn_directions(self):
        """Test that each move turns the face clockwise seen from outside."""
        # For a clockwise turn of face X, the strip of the face listed second
        # is the one that ends up on the face listed first.
        cases = [
            ('U', (FRONT, 0, slice(None)), RIGHT),
            ('D', (RIGHT, 2, slice(None)), FRONT),
            ('F', (RIGHT, slice(None), 0), UP),
            ('B', (LEFT, slice(None), 0), UP),
            ('L', (DOWN, slice(None), 0), FRONT),
            ('R', (UP, slice(None), 2), FRONT),
        ]

        for move, destination, source_face in cases:
            cube = RubiksCube()
            cube.faces = _labelled_faces()
            cube.apply_move(move)
            origins = {_STICKER_HOME[int(label)][0]
                       for label in cube.faces[destination]}
            self.assertEqual(origins, {source_face},
                             f"{move} should pull that strip from face {source_face}")

    def test_moves_keep_cubies_intact(self):
        """Test that every move permutes whole cubies, not loose stickers."""
        # A face turn that rotates its own face the opposite way from the
        # strips around it looks plausible sticker by sticker but tears the
        # corner and edge cubies apart, so check the three stickers of every
        # corner (and two of every edge) still sit on one cubie afterwards.
        cubies = _cubies()

        for move in ['F', 'B', 'U', 'D', 'L', 'R']:
            cube = RubiksCube()
            cube.faces = _labelled_faces()
            cube.apply_move(move)

            for stickers in cubies:
                landed = {_cubie_faces(*_STICKER_HOME[int(cube.faces[sticker])])
                          for sticker in stickers}
                self.assertEqual(len(landed), 1,
                                 f"{move} split a cubie across positions {stickers}")

    def test_two_generator_order(self):
        """Test that repeating "R U" returns to solved after 105 rounds."""
        # The order of every two adjacent face turns on a real cube is 105,
        # which a broken move would almost certainly not reproduce.
        cube = RubiksCube()

        for repetitions in range(1, 106):
            cube.apply_algorithm("R U")
            if repetitions < 105:
                self.assertFalse(cube.is_solved(),
                                 f"cube solved early after {repetitions} rounds")

        self.assertTrue(cube.is_solved())


if __name__ == '__main__':
    unittest.main() 