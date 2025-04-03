#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demonstration of the Rubik's Cube functionality.
"""

from cube import RubiksCube


def main():
    # Create a new Rubik's Cube
    cube = RubiksCube()
    
    print("=== Initial Solved Cube ===")
    print(cube)
    
    # Demonstrate a single move
    print("=== After performing an F move ===")
    cube.move_F()
    print(cube)
    
    # Demonstrate applying a sequence of moves
    print("=== After applying the sequence R U R' U' ===")
    cube.apply_algorithm("R U R' U'")
    print(cube)
    
    # Demonstrate scrambling the cube
    print("=== After a sequence of random-looking moves ===")
    cube.apply_algorithm("F R U R' U' F' U R U' R'")
    print(cube)
    
    # Reset the cube back to solved
    print("=== After resetting the cube ===")
    cube.reset()
    print(cube)
    
    print("Is the cube solved?", cube.is_solved())


if __name__ == '__main__':
    main() 