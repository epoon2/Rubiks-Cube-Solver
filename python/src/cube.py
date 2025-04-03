#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rubik's Cube representation and basic operations.
"""

import numpy as np


class RubiksCube:
    """
    A class representing a 3x3 Rubik's Cube.
    
    The cube is represented as 6 faces, each with a 3x3 grid of colors.
    Face indexing:
    0: Front (F)
    1: Back (B)
    2: Up (U)
    3: Down (D)
    4: Left (L)
    5: Right (R)
    
    Color indexing:
    0: White
    1: Yellow
    2: Green
    3: Blue
    4: Red
    5: Orange
    """
    
    def __init__(self, size=3):
        """
        Initialize a solved Rubik's Cube.
        
        Args:
            size (int): The size of the cube (3 for a standard 3x3x3 cube).
                        Currently only 3x3 is supported.
        """
        if size != 3:
            raise ValueError("Currently only 3x3 cube is supported")
            
        self.size = size
        # Initialize the cube with each face having a uniform color
        self.faces = np.zeros((6, size, size), dtype=np.int8)
        for i in range(6):
            self.faces[i, :, :] = i
            
    def __str__(self):
        """
        Return a string representation of the cube for debugging.
        """
        result = []
        face_names = ['Front', 'Back', 'Up', 'Down', 'Left', 'Right']
        color_names = ['W', 'Y', 'G', 'B', 'R', 'O']  # Short color names
        
        for i, face in enumerate(self.faces):
            result.append(f"{face_names[i]}:")
            for row in face:
                result.append(' '.join(color_names[color] for color in row))
            result.append('')
        
        return '\n'.join(result)
    
    def is_solved(self):
        """
        Check if the cube is solved (each face has a uniform color).
        
        Returns:
            bool: True if the cube is solved, False otherwise.
        """
        for face in self.faces:
            if not np.all(face == face[0, 0]):
                return False
        return True
    
    def _rotate_face(self, face_idx, clockwise=True):
        """
        Rotate a face of the cube.
        
        Args:
            face_idx (int): Index of the face to rotate.
            clockwise (bool): If True, rotate clockwise, otherwise counterclockwise.
        """
        k = 1 if clockwise else 3  # 3 clockwise rotations = 1 counterclockwise
        self.faces[face_idx] = np.rot90(self.faces[face_idx], k=k)
    
    def _get_face_edge(self, face_idx, edge_idx):
        """
        Get the edge of a face.
        
        Args:
            face_idx (int): Index of the face.
            edge_idx (int): Index of the edge (0=top, 1=right, 2=bottom, 3=left).
            
        Returns:
            np.array: 1D array of the edge.
        """
        face = self.faces[face_idx]
        if edge_idx == 0:
            return face[0, :].copy()  # Top edge
        elif edge_idx == 1:
            return face[:, -1].copy()  # Right edge
        elif edge_idx == 2:
            return face[-1, ::-1].copy()  # Bottom edge (reversed)
        elif edge_idx == 3:
            return face[::-1, 0].copy()  # Left edge (reversed)
        
    def _set_face_edge(self, face_idx, edge_idx, new_edge):
        """
        Set the edge of a face.
        
        Args:
            face_idx (int): Index of the face.
            edge_idx (int): Index of the edge (0=top, 1=right, 2=bottom, 3=left).
            new_edge (np.array): 1D array of the new edge.
        """
        if edge_idx == 0:
            self.faces[face_idx][0, :] = new_edge  # Top edge
        elif edge_idx == 1:
            self.faces[face_idx][:, -1] = new_edge  # Right edge
        elif edge_idx == 2:
            self.faces[face_idx][-1, :] = new_edge[::-1]  # Bottom edge (reversed)
        elif edge_idx == 3:
            self.faces[face_idx][:, 0] = new_edge[::-1]  # Left edge (reversed)
    
    def move_F(self, prime=False):
        """
        Perform a Front face move.
        
        Args:
            prime (bool): If True, perform F' (counterclockwise), otherwise F (clockwise).
        """
        # Rotate the front face
        self._rotate_face(0, not prime)
        
        # Define the affected edges and their new positions
        # Edges are defined as (face_idx, edge_idx)
        edges = [(2, 2), (5, 3), (3, 0), (4, 1)]
        
        # Shift edges
        shift = 1 if prime else -1
        edge_values = [self._get_face_edge(face, edge) for face, edge in edges]
        edge_values = edge_values[shift:] + edge_values[:shift]
        
        # Update edges
        for (face, edge), new_edge in zip(edges, edge_values):
            self._set_face_edge(face, edge, new_edge)
    
    def move_B(self, prime=False):
        """
        Perform a Back face move.
        
        Args:
            prime (bool): If True, perform B' (counterclockwise), otherwise B (clockwise).
        """
        # Rotate the back face
        self._rotate_face(1, not prime)
        
        # Define the affected edges and their new positions
        edges = [(2, 0), (4, 3), (3, 2), (5, 1)]
        
        # Shift edges (opposite direction from F move)
        shift = -1 if prime else 1
        edge_values = [self._get_face_edge(face, edge) for face, edge in edges]
        edge_values = edge_values[shift:] + edge_values[:shift]
        
        # Update edges
        for (face, edge), new_edge in zip(edges, edge_values):
            self._set_face_edge(face, edge, new_edge)
    
    def move_U(self, prime=False):
        """
        Perform an Up face move.
        
        Args:
            prime (bool): If True, perform U' (counterclockwise), otherwise U (clockwise).
        """
        # Rotate the up face
        self._rotate_face(2, not prime)
        
        # Define the affected edges and their new positions
        edges = [(0, 0), (5, 0), (1, 0), (4, 0)]
        
        # Shift edges
        shift = 1 if prime else -1
        edge_values = [self._get_face_edge(face, edge) for face, edge in edges]
        edge_values = edge_values[shift:] + edge_values[:shift]
        
        # Update edges
        for (face, edge), new_edge in zip(edges, edge_values):
            self._set_face_edge(face, edge, new_edge)
    
    def move_D(self, prime=False):
        """
        Perform a Down face move.
        
        Args:
            prime (bool): If True, perform D' (counterclockwise), otherwise D (clockwise).
        """
        # Rotate the down face
        self._rotate_face(3, not prime)
        
        # Define the affected edges and their new positions
        edges = [(0, 2), (4, 2), (1, 2), (5, 2)]
        
        # Shift edges (opposite direction from U move)
        shift = 1 if prime else -1
        edge_values = [self._get_face_edge(face, edge) for face, edge in edges]
        edge_values = edge_values[shift:] + edge_values[:shift]
        
        # Update edges
        for (face, edge), new_edge in zip(edges, edge_values):
            self._set_face_edge(face, edge, new_edge)
    
    def move_L(self, prime=False):
        """
        Perform a Left face move.
        
        Args:
            prime (bool): If True, perform L' (counterclockwise), otherwise L (clockwise).
        """
        # Rotate the left face
        self._rotate_face(4, not prime)
        
        # Define the affected edges and their new positions
        edges = [(0, 3), (2, 3), (1, 1), (3, 3)]
        
        # Shift edges
        shift = 1 if prime else -1
        edge_values = [self._get_face_edge(face, edge) for face, edge in edges]
        edge_values = edge_values[shift:] + edge_values[:shift]
        
        # Update edges
        for (face, edge), new_edge in zip(edges, edge_values):
            self._set_face_edge(face, edge, new_edge)
    
    def move_R(self, prime=False):
        """
        Perform a Right face move.
        
        Args:
            prime (bool): If True, perform R' (counterclockwise), otherwise R (clockwise).
        """
        # Rotate the right face
        self._rotate_face(5, not prime)
        
        # Define the affected edges and their new positions
        edges = [(0, 1), (3, 1), (1, 3), (2, 1)]
        
        # Shift edges
        shift = 1 if prime else -1
        edge_values = [self._get_face_edge(face, edge) for face, edge in edges]
        edge_values = edge_values[shift:] + edge_values[:shift]
        
        # Update edges
        for (face, edge), new_edge in zip(edges, edge_values):
            self._set_face_edge(face, edge, new_edge)
    
    def apply_move(self, move):
        """
        Apply a move to the cube.
        
        Args:
            move (str): Move in standard notation (F, F', B, B', U, U', D, D', L, L', R, R')
        """
        move = move.strip()
        prime = False
        if len(move) > 1 and move[1] == "'":
            prime = True
            move = move[0]
            
        if move == 'F':
            self.move_F(prime)
        elif move == 'B':
            self.move_B(prime)
        elif move == 'U':
            self.move_U(prime)
        elif move == 'D':
            self.move_D(prime)
        elif move == 'L':
            self.move_L(prime)
        elif move == 'R':
            self.move_R(prime)
        else:
            raise ValueError(f"Invalid move: {move}")
    
    def apply_algorithm(self, algorithm):
        """
        Apply a sequence of moves to the cube.
        
        Args:
            algorithm (str): Space-separated moves in standard notation.
        """
        for move in algorithm.split():
            self.apply_move(move)
    
    def reset(self):
        """
        Reset the cube to its solved state.
        """
        for i in range(6):
            self.faces[i, :, :] = i 