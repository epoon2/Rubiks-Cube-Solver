#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization tools for the Rubik's Cube.

This module provides functions to visualize the Rubik's Cube
using matplotlib for 2D representation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class CubeVisualizer:
    """A class for visualizing a Rubik's Cube."""
    
    def __init__(self):
        """Initialize the visualizer with colors."""
        # Define colors for each face (0-5)
        self.colors = [
            '#FFFFFF',  # White (0)
            '#FFFF00',  # Yellow (1)
            '#00FF00',  # Green (2)
            '#0000FF',  # Blue (3)
            '#FF0000',  # Red (4)
            '#FFA500'   # Orange (5)
        ]
    
    def _draw_face(self, ax, face, start_x, start_y, size=1.0):
        """
        Draw a single face of the cube.
        
        Args:
            ax: Matplotlib axis.
            face: 3x3 array representing the face.
            start_x: X-coordinate of the top-left corner.
            start_y: Y-coordinate of the top-left corner.
            size: Size of each sticker.
        """
        for i in range(3):
            for j in range(3):
                color = self.colors[face[i, j]]
                rect = Rectangle((start_x + j * size, start_y - i * size), 
                                size, size, 
                                facecolor=color, 
                                edgecolor='black', 
                                linewidth=2)
                ax.add_patch(rect)
    
    def visualize_2d(self, cube, title=None, save_path=None):
        """
        Visualize the cube in a 2D net layout.
        
        Args:
            cube: RubiksCube object.
            title: Optional title for the plot.
            save_path: Optional path to save the visualization.
        """
        # Create a figure with appropriate size
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up the plot
        ax.set_xlim(0, 9)
        ax.set_ylim(-9, 3)
        ax.set_aspect('equal')
        ax.axis('off')
        
        if title:
            plt.title(title, fontsize=16, pad=20)
        
        # Draw each face in the net layout
        # Up face (at the top)
        self._draw_face(ax, cube.faces[2], 3, 3)
        
        # Left, Front, Right, Back faces (in the middle row)
        self._draw_face(ax, cube.faces[4], 0, 0)
        self._draw_face(ax, cube.faces[0], 3, 0)
        self._draw_face(ax, cube.faces[5], 6, 0)
        self._draw_face(ax, cube.faces[1], 9, 0)
        
        # Down face (at the bottom)
        self._draw_face(ax, cube.faces[3], 3, -3)
        
        # Add face labels
        labels = ['F', 'B', 'U', 'D', 'L', 'R']
        positions = [(4.5, -1.5), (10.5, -1.5), (4.5, 2.5), (4.5, -5.5), (1.5, -1.5), (7.5, -1.5)]
        
        for label, (x, y) in zip(labels, positions):
            ax.text(x, y, label, fontsize=12, 
                   ha='center', va='center', 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()
    
    def visualize_sequence(self, cube, algorithm, title="Move Sequence", save_path=None):
        """
        Visualize a sequence of moves on the cube.
        
        Args:
            cube: RubiksCube object.
            algorithm: String of space-separated moves.
            title: Title for the visualization.
            save_path: Optional path to save the visualization. If provided, will append 
                      step numbers to the filename.
        """
        # Create a copy of the cube
        from cube import RubiksCube
        cube_copy = RubiksCube()
        cube_copy.faces = np.copy(cube.faces)
        
        # Visualize initial state
        self.visualize_2d(cube_copy, f"{title} - Initial State", 
                         save_path.replace('.png', '_step0.png') if save_path else None)
        
        # Apply and visualize each move
        moves = algorithm.split()
        for i, move in enumerate(moves, 1):
            cube_copy.apply_move(move)
            self.visualize_2d(cube_copy, f"{title} - Step {i}: {move}", 
                             save_path.replace('.png', f'_step{i}.png') if save_path else None)


def main():
    """Demonstrate the cube visualizer."""
    from cube import RubiksCube
    
    # Create a solved cube
    cube = RubiksCube()
    
    # Create a visualizer
    visualizer = CubeVisualizer()
    
    # Visualize the solved cube
    visualizer.visualize_2d(cube, "Solved Cube")
    
    # Apply a sequence and visualize the steps
    algorithm = "R U R' U'"
    visualizer.visualize_sequence(cube, algorithm, f"Applying {algorithm}")


if __name__ == '__main__':
    main() 