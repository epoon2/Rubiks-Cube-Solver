#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the A* Solver.
"""

import sys
import os
import unittest
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from cube import RubiksCube
from solvers.astar_solver import AStarSolver


class TestAStarSolver(unittest.TestCase):
    """Test cases for the A* Solver."""
    
    def test_heuristic(self):
        """Test the heuristic function of the A* solver."""
        # Create a solved cube
        cube = RubiksCube()
        solver = AStarSolver()
        
        # Heuristic of a solved cube should be 0
        self.assertEqual(solver._heuristic(cube), 0, "Heuristic of a solved cube should be 0")
        
        # Perform a single move
        cube.move_F()
        
        # Heuristic of a cube with one move should be > 0
        self.assertGreater(solver._heuristic(cube), 0, "Heuristic after a move should be > 0")
    
    def test_solve_scrambled_cube(self):
        """Test solving a cube that is a few moves away from solved."""
        # Create a solved cube
        cube = RubiksCube()
        
        # Apply a sequence of moves
        cube.apply_algorithm("R U R' U'")
        
        # Solve with A*
        solver = AStarSolver(max_depth=8)
        success, solution, nodes, time_taken = solver.solve(cube)
        
        # Verify solution
        self.assertTrue(success, "Solver should find a solution")
        self.assertLessEqual(len(solution), 8, "Solution should be at most 8 moves")
        
        # Apply the solution and check if cube is solved
        new_cube = RubiksCube()
        new_cube.faces = np.copy(cube.faces)
        new_cube.apply_algorithm(' '.join(solution))
        self.assertTrue(new_cube.is_solved(), "Cube should be solved after applying the solution")
    
    def test_slightly_more_complex_scramble(self):
        """Test solving a slightly more complex scramble."""
        # Create a solved cube
        cube = RubiksCube()
        
        # Apply a sequence of moves
        cube.apply_algorithm("R U R' U' R' F R F'")
        
        # Solve with A*
        solver = AStarSolver(max_depth=10)
        success, solution, nodes, time_taken = solver.solve(cube)
        
        # Verify solution
        self.assertTrue(success, "Solver should find a solution")
        self.assertLessEqual(len(solution), 10, "Solution should be at most 10 moves")
        
        # Apply the solution and check if cube is solved
        new_cube = RubiksCube()
        new_cube.faces = np.copy(cube.faces)
        new_cube.apply_algorithm(' '.join(solution))
        self.assertTrue(new_cube.is_solved(), "Cube should be solved after applying the solution")


if __name__ == '__main__':
    unittest.main() 