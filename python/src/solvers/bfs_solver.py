#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Breadth-First Search solver for the Rubik's Cube.

This is a naive implementation that can only solve cubes that are
a few moves away from the solved state. For a scrambled cube, it
would require too much memory and time to find a solution.
"""

import time
from collections import deque
import numpy as np
from cube import RubiksCube


class BFSSolver:
    """A Breadth-First Search solver for the Rubik's Cube."""
    
    def __init__(self, max_depth=5):
        """
        Initialize the BFS solver.
        
        Args:
            max_depth (int): Maximum search depth (number of moves).
        """
        self.max_depth = max_depth
        self.basic_moves = ['F', 'F\'', 'B', 'B\'', 'U', 'U\'', 'D', 'D\'', 'L', 'L\'', 'R', 'R\'']
    
    def _cube_state_to_hashable(self, cube):
        """
        Convert the cube state to a hashable representation for visited set.
        
        Args:
            cube (RubiksCube): The cube to convert.
            
        Returns:
            tuple: A hashable representation of the cube.
        """
        return tuple(tuple(tuple(face) for face in cube.faces))
    
    def _reconstruct_path(self, current, came_from):
        """
        Reconstruct the solution path from the search tree.
        
        Args:
            current: The current state.
            came_from: Dictionary mapping states to (previous_state, move).
            
        Returns:
            list: Sequence of moves to reach the current state.
        """
        path = []
        while current in came_from:
            current, move = came_from[current]
            path.append(move)
        path.reverse()
        return path
    
    def solve(self, cube):
        """
        Solve the given cube using BFS.
        
        Args:
            cube (RubiksCube): The cube to solve.
            
        Returns:
            tuple: (solution_found, solution_moves, nodes_explored, time_taken)
        """
        start_time = time.time()
        
        # If cube is already solved, return empty solution
        if cube.is_solved():
            return True, [], 0, 0
        
        # Create a copy of the cube to avoid modifying the original
        cube_copy = RubiksCube()
        cube_copy.faces = np.copy(cube.faces)
        
        # Initialize BFS
        queue = deque([(self._cube_state_to_hashable(cube_copy), 0)])
        came_from = {}
        visited = {self._cube_state_to_hashable(cube_copy)}
        nodes_explored = 0
        
        # BFS loop
        while queue:
            current_state, depth = queue.popleft()
            nodes_explored += 1
            
            # Check depth limit
            if depth >= self.max_depth:
                continue
            
            # Try each possible move
            for move in self.basic_moves:
                # Create a new cube from the current state
                new_cube = RubiksCube()
                new_cube.faces = np.array(current_state).reshape((6, 3, 3))
                
                # Apply the move
                new_cube.apply_move(move)
                
                # Check if solved
                if new_cube.is_solved():
                    end_time = time.time()
                    came_from[self._cube_state_to_hashable(new_cube)] = (current_state, move)
                    solution = self._reconstruct_path(self._cube_state_to_hashable(new_cube), came_from)
                    return True, solution, nodes_explored, end_time - start_time
                
                # Check if already visited
                new_state = self._cube_state_to_hashable(new_cube)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, depth + 1))
                    came_from[new_state] = (current_state, move)
        
        end_time = time.time()
        return False, [], nodes_explored, end_time - start_time


def main():
    """Demonstrate the BFS solver."""
    # Create a solved cube
    cube = RubiksCube()
    print("Initial cube (solved):")
    print(cube)
    
    # Apply a short sequence of moves to scramble slightly
    scramble = "R U R' U'"
    print(f"Applying scramble: {scramble}")
    cube.apply_algorithm(scramble)
    print("Scrambled cube:")
    print(cube)
    
    # Solve with BFS
    print("Solving with BFS...")
    solver = BFSSolver(max_depth=5)
    success, solution, nodes, time_taken = solver.solve(cube)
    
    if success:
        print(f"Solution found! Moves: {' '.join(solution)}")
        print(f"Solution length: {len(solution)}")
        print(f"Nodes explored: {nodes}")
        print(f"Time taken: {time_taken:.3f} seconds")
        
        # Verify solution
        new_cube = RubiksCube()
        new_cube.faces = np.copy(cube.faces)
        new_cube.apply_algorithm(' '.join(solution))
        print("After applying solution:")
        print(new_cube)
        print("Is solved:", new_cube.is_solved())
    else:
        print("No solution found within the depth limit.")
        print(f"Nodes explored: {nodes}")
        print(f"Time taken: {time_taken:.3f} seconds")


if __name__ == '__main__':
    main() 