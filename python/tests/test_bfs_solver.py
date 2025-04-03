#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the BFS Solver.
"""

import sys
import os
import unittest
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from cube import RubiksCube
from solvers.bfs_solver import BFSSolver


class TestBFSSolver(unittest.TestCase):
    """Test cases for the BFS Solver."""
    
    def test_solve_scrambled_cube(self):
        """Test solving a cube that is a few moves away from solved."""
        # Create a solved cube
        cube = RubiksCube()
        
        # Apply a short sequence of moves (4 moves away from solved)
        cube.apply_algorithm("R U R' U'")
        
        # Solve with BFS
        solver = BFSSolver(max_depth=5)
        success, solution, nodes, time_taken = solver.solve(cube)
        
        # Verify solution
        self.assertTrue(success, "Solver should find a solution")
        self.assertLessEqual(len(solution), 5, "Solution should be at most 5 moves")
        
        # Apply the solution and check if cube is solved
        new_cube = RubiksCube()
        new_cube.faces = np.copy(cube.faces)
        new_cube.apply_algorithm(' '.join(solution))
        self.assertTrue(new_cube.is_solved(), "Cube should be solved after applying the solution")
    
    def test_already_solved_cube(self):
        """Test solving an already solved cube."""
        # Create a solved cube
        cube = RubiksCube()
        
        # Solve with BFS
        solver = BFSSolver(max_depth=5)
        success, solution, nodes, time_taken = solver.solve(cube)
        
        # Verify solution
        self.assertTrue(success, "Solver should find a solution")
        self.assertEqual(len(solution), 0, "Solution should be empty for already solved cube")
    
    def test_max_depth_limit(self):
        """Test that solver respects the max depth limit."""
        # Create a solved cube
        cube = RubiksCube()
        
        # Apply a sequence of moves that's beyond our depth limit
        cube.apply_algorithm("R U R' U' L D L' D' B F'")
        
        # Solve with BFS with very small depth limit
        solver = BFSSolver(max_depth=3)
        success, solution, nodes, time_taken = solver.solve(cube)
        
        # Verify no solution found
        self.assertFalse(success, "Solver should not find a solution within shallow depth limit")


if __name__ == '__main__':
    unittest.main() 