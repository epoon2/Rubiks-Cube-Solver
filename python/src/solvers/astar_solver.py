#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A* search algorithm solver for the Rubik's Cube.

This implementation uses a heuristic to guide the search, making it
more efficient than plain BFS for slightly more complex scrambles.
"""

import time
import heapq
import numpy as np
from cube import RubiksCube


class AStarSolver:
    """An A* search algorithm solver for the Rubik's Cube."""
    
    def __init__(self, max_depth=8):
        """
        Initialize the A* solver.
        
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
    
    def _heuristic(self, cube):
        """
        Calculate a heuristic value for the cube state.
        
        The heuristic counts the number of misplaced stickers.
        A perfect cube would have a heuristic value of 0.
        
        Args:
            cube (RubiksCube): The cube to evaluate.
            
        Returns:
            int: The heuristic value.
        """
        # Count misplaced stickers on each face
        misplaced = 0
        for face_idx in range(6):
            center_color = cube.faces[face_idx, 1, 1]  # Center piece
            misplaced += np.sum(cube.faces[face_idx] != center_color)
        
        return misplaced
    
    def solve(self, cube):
        """
        Solve the given cube using A*.
        
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
        
        # Initialize A*
        start_state = self._cube_state_to_hashable(cube_copy)
        start_h = self._heuristic(cube_copy)
        
        # Priority queue entries: (f_score, count, state, depth)
        # count is used to break ties when f_scores are equal
        count = 0
        open_set = [(start_h, count, start_state, 0)]
        
        # Dictionary to track where we came from
        came_from = {}
        
        # Dictionary to store g_scores (cost from start to current)
        g_score = {start_state: 0}
        
        # Dictionary to store f_scores (g_score + heuristic)
        f_score = {start_state: start_h}
        
        # Set to track visited states
        visited = {start_state}
        
        # Counter for nodes explored
        nodes_explored = 0
        
        # A* search loop
        while open_set:
            # Get state with lowest f_score
            _, _, current_state, depth = heapq.heappop(open_set)
            nodes_explored += 1
            
            # Create a cube from the current state
            current_cube = RubiksCube()
            current_cube.faces = np.array(current_state).reshape((6, 3, 3))
            
            # Check if solved
            if current_cube.is_solved():
                end_time = time.time()
                solution = self._reconstruct_path(current_state, came_from)
                return True, solution, nodes_explored, end_time - start_time
            
            # Check depth limit
            if depth >= self.max_depth:
                continue
            
            # Try each possible move
            for move in self.basic_moves:
                # Create a new cube from the current state
                new_cube = RubiksCube()
                new_cube.faces = np.copy(current_cube.faces)
                
                # Apply the move
                new_cube.apply_move(move)
                
                # Get the new state
                new_state = self._cube_state_to_hashable(new_cube)
                
                # Calculate tentative g_score
                tentative_g_score = g_score[current_state] + 1
                
                # If we've found a better path to this state, or it's a new state
                if new_state not in g_score or tentative_g_score < g_score[new_state]:
                    # Update tracking dictionaries
                    came_from[new_state] = (current_state, move)
                    g_score[new_state] = tentative_g_score
                    
                    # Calculate f_score: g_score + heuristic
                    new_h = self._heuristic(new_cube)
                    new_f = tentative_g_score + new_h
                    f_score[new_state] = new_f
                    
                    # Add to open set if not already there
                    if new_state not in visited:
                        visited.add(new_state)
                        count += 1
                        heapq.heappush(open_set, (new_f, count, new_state, depth + 1))
        
        # If we get here, no solution was found
        end_time = time.time()
        return False, [], nodes_explored, end_time - start_time


def main():
    """Demonstrate the A* solver."""
    # Create a solved cube
    cube = RubiksCube()
    print("Initial cube (solved):")
    print(cube)
    
    # Apply a sequence of moves to scramble
    scramble = "R U R' U' R' F R F'"  # A simple 7-move OLL algorithm
    print(f"Applying scramble: {scramble}")
    cube.apply_algorithm(scramble)
    print("Scrambled cube:")
    print(cube)
    
    # Solve with A*
    print("Solving with A*...")
    solver = AStarSolver(max_depth=10)
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
        print("Is solved:", new_cube.is_solved())
    else:
        print("No solution found within the depth limit.")
        print(f"Nodes explored: {nodes}")
        print(f"Time taken: {time_taken:.3f} seconds")


if __name__ == '__main__':
    main() 