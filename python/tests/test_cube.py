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


if __name__ == '__main__':
    unittest.main() 