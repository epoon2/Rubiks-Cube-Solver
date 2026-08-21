#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coordinate move tables and pruning tables for the two-phase algorithm.

Kociemba's algorithm solves a cube in two stages through the subgroup

    G1 = <U, D, R2, L2, F2, B2>

Phase 1 turns any cube into a cube in G1; phase 2 solves it without ever
leaving G1. What makes each stage searchable is that neither one needs the
whole cube state, only a few coordinates:

    phase 1  twist   corner orientation                    3^7  = 2187
             flip    edge orientation                      2^11 = 2048
             slice   which four slots hold the FR/FL/BL/BR
                     edges, ignoring their order           C(12,4) = 495

    phase 2  corners     corner permutation                8! = 40320
             ud_edges    permutation of the eight U/D edges 8! = 40320
             slice_perm  order of the four slice edges      4! = 24

Each coordinate is small enough to tabulate every move on it, and each pair
(coordinate, slice) is small enough to hold an exact distance-to-goal table.
Those distances are admissible heuristics for the full stage, which turns an
otherwise hopeless search into an IDA* run that finishes in milliseconds.

The tables total roughly 8 MB. They are generated here and cached on disk;
nothing binary is checked in.
"""

import os
import time
from itertools import permutations
from math import comb, factorial

import numpy as np

from solvers.cubie import (CORNER_COUNT, EDGE_COUNT, MOVE_CUBES, MOVE_NAMES,
                           SLICE_EDGES, CubieCube)

TWIST_COUNT = 3 ** 7
FLIP_COUNT = 2 ** 11
SLICE_COUNT = comb(12, 4)
CORNER_PERM_COUNT = factorial(8)
UD_EDGE_PERM_COUNT = factorial(8)
SLICE_PERM_COUNT = factorial(4)

# Phase 1 may use every move; phase 2 is restricted to the generators of G1.
PHASE1_MOVES = list(MOVE_NAMES)
PHASE2_MOVES = ['U', 'U2', "U'", 'D', 'D2', "D'", 'R2', 'L2', 'F2', 'B2']

# The eight edges that belong in the U and D layers, in coordinate order.
UD_EDGES = [edge for edge in range(EDGE_COUNT) if edge not in SLICE_EDGES]

# The slice edges are numbered 8..11, so subtracting this turns them into a
# permutation of 0..3 that can be ranked like any other.
SLICE_PERM_OFFSET = min(SLICE_EDGES)

# Bumping this invalidates any cache on disk. Change it whenever a coordinate
# encoding or the move set changes, or stale tables will silently be reused.
TABLE_VERSION = 1

DEFAULT_TABLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'tables')


def twist_coordinate(cube):
    """
    Return the corner orientation coordinate of a cube.

    Args:
        cube (CubieCube): The cube to read.

    Returns:
        int: A value in 0..2186.
    """
    # The eighth corner's twist is fixed by the other seven, so it is not
    # part of the coordinate.
    coordinate = 0
    for corner in range(CORNER_COUNT - 1):
        coordinate = 3 * coordinate + cube.co[corner]
    return coordinate


def cube_with_twist(coordinate):
    """
    Return a cube whose corners carry the given orientation coordinate.

    Args:
        coordinate (int): A value in 0..2186.

    Returns:
        CubieCube: A cube with every piece home and that twist.
    """
    co = [0] * CORNER_COUNT
    for corner in range(CORNER_COUNT - 2, -1, -1):
        co[corner] = coordinate % 3
        coordinate //= 3
    co[CORNER_COUNT - 1] = (-sum(co)) % 3
    return CubieCube(co=co)


def flip_coordinate(cube):
    """
    Return the edge orientation coordinate of a cube.

    Args:
        cube (CubieCube): The cube to read.

    Returns:
        int: A value in 0..2047.
    """
    # As with the corners, the twelfth edge's flip follows from the rest.
    coordinate = 0
    for edge in range(EDGE_COUNT - 1):
        coordinate = 2 * coordinate + cube.eo[edge]
    return coordinate


def cube_with_flip(coordinate):
    """
    Return a cube whose edges carry the given orientation coordinate.

    Args:
        coordinate (int): A value in 0..2047.

    Returns:
        CubieCube: A cube with every piece home and that flip.
    """
    eo = [0] * EDGE_COUNT
    for edge in range(EDGE_COUNT - 2, -1, -1):
        eo[edge] = coordinate % 2
        coordinate //= 2
    eo[EDGE_COUNT - 1] = sum(eo) % 2
    return CubieCube(eo=eo)


def slice_coordinate(cube):
    """
    Return which slots hold the four middle-slice edges, ignoring their order.

    Args:
        cube (CubieCube): The cube to read.

    Returns:
        int: A value in 0..494.
    """
    occupied = [position for position in range(EDGE_COUNT)
                if cube.ep[position] in SLICE_EDGES]
    return _rank_combination(occupied)


def _rank_combination(positions):
    """
    Rank a set of four edge slots in the combinatorial number system.

    Args:
        positions (list): Four distinct slots in 0..11.

    Returns:
        int: A value in 0..494.
    """
    # Slots are mirrored before ranking so that the solved cube, whose slice
    # edges sit in the last four slots, comes out as coordinate 0 like every
    # other coordinate here.
    mirrored = sorted(EDGE_COUNT - 1 - position for position in positions)
    return sum(comb(position, index + 1)
               for index, position in enumerate(mirrored))


def _unrank_combination(coordinate):
    """
    Recover the four occupied edge slots from a slice coordinate.

    Args:
        coordinate (int): A value in 0..494.

    Returns:
        list: The four slots, ascending.
    """
    mirrored = []
    remaining = 3
    for position in range(EDGE_COUNT - 1, -1, -1):
        if remaining < 0:
            break
        size = comb(position, remaining + 1)
        if coordinate >= size:
            coordinate -= size
            mirrored.append(position)
            remaining -= 1
    return sorted(EDGE_COUNT - 1 - position for position in mirrored)


def corner_perm_coordinate(cube):
    """
    Return the corner permutation coordinate of a cube.

    Args:
        cube (CubieCube): The cube to read.

    Returns:
        int: A value in 0..40319.
    """
    return _rank_permutation(cube.cp)


def ud_edge_perm_coordinate(cube):
    """
    Return the permutation coordinate of the eight U- and D-layer edges.

    Only meaningful once the cube is in G1, where those eight edges are
    guaranteed to occupy the eight U/D slots.

    Args:
        cube (CubieCube): The cube to read.

    Returns:
        int: A value in 0..40319.
    """
    return _rank_permutation([cube.ep[position] for position in UD_EDGES])


def slice_perm_coordinate(cube):
    """
    Return the permutation coordinate of the four middle-slice edges.

    Args:
        cube (CubieCube): The cube to read.

    Returns:
        int: A value in 0..23.
    """
    return _rank_permutation([cube.ep[position] - SLICE_PERM_OFFSET
                              for position in SLICE_EDGES])


def _rank_permutation(values):
    """
    Return the lexicographic rank of a permutation.

    Args:
        values (list): A permutation of 0..n-1.

    Returns:
        int: The rank, from 0 for the identity.
    """
    rank = 0
    size = len(values)
    for index, value in enumerate(values):
        smaller = sum(1 for other in values[index + 1:] if other < value)
        rank += smaller * factorial(size - 1 - index)
    return rank


def _rank_permutations(rows):
    """
    Rank many permutations of the same length at once.

    Args:
        rows (np.ndarray): Shape (n, k), each row a permutation of 0..k-1.

    Returns:
        np.ndarray: Shape (n,) of ranks.
    """
    size = rows.shape[1]
    ranks = np.zeros(len(rows), dtype=np.int32)
    for index in range(size):
        smaller = (rows[:, index + 1:] < rows[:, index:index + 1]).sum(axis=1)
        ranks += smaller * factorial(size - 1 - index)
    return ranks


def _build_orientation_move_table(count, make_cube, read_coordinate, moves):
    """
    Tabulate an orientation coordinate against every move.

    Args:
        count (int): Number of values the coordinate takes.
        make_cube (callable): Coordinate to a representative CubieCube.
        read_coordinate (callable): CubieCube back to a coordinate.
        moves (list): Move names, in table column order.

    Returns:
        np.ndarray: Shape (count, len(moves)) of int32.
    """
    table = np.zeros((count, len(moves)), dtype=np.int32)
    for coordinate in range(count):
        cube = make_cube(coordinate)
        for column, move in enumerate(moves):
            table[coordinate, column] = read_coordinate(
                cube.multiply(MOVE_CUBES[move]))
    return table


def _build_slice_move_table(moves):
    """
    Tabulate the slice coordinate against every move.

    Args:
        moves (list): Move names, in table column order.

    Returns:
        np.ndarray: Shape (495, len(moves)) of int32.
    """
    table = np.zeros((SLICE_COUNT, len(moves)), dtype=np.int32)
    for coordinate in range(SLICE_COUNT):
        occupied = set(_unrank_combination(coordinate))
        for column, move in enumerate(moves):
            edge_perm = MOVE_CUBES[move].ep
            moved = [position for position in range(EDGE_COUNT)
                     if edge_perm[position] in occupied]
            table[coordinate, column] = _rank_combination(moved)
    return table


def _build_permutation_move_table(size, slots, moves):
    """
    Tabulate a permutation coordinate against every move.

    Args:
        size (int): Number of pieces being permuted.
        slots (callable): Move name to the list of source slots, expressed as
                          indices into the permutation being tracked.
        moves (list): Move names, in table column order.

    Returns:
        np.ndarray: Shape (size!, len(moves)) of int32.
    """
    rows = np.array(list(permutations(range(size))), dtype=np.int8)
    table = np.zeros((len(rows), len(moves)), dtype=np.int32)
    for column, move in enumerate(moves):
        table[:, column] = _rank_permutations(rows[:, slots(move)])
    return table


def _build_pruning_table(first_table, second_table, first_count, second_count):
    """
    Breadth-first search the product of two coordinates from the solved state.

    Args:
        first_table (np.ndarray): Move table of the first coordinate.
        second_table (np.ndarray): Move table of the second coordinate.
        first_count (int): Number of values the first coordinate takes.
        second_count (int): Number of values the second coordinate takes.

    Returns:
        np.ndarray: uint8 distances indexed by first * second_count + second.
    """
    size = first_count * second_count
    distances = np.full(size, 255, dtype=np.uint8)
    distances[0] = 0
    frontier = np.zeros(1, dtype=np.int64)

    depth = 0
    while frontier.size:
        reached = np.zeros(size, dtype=bool)
        first = frontier // second_count
        second = frontier % second_count

        for column in range(first_table.shape[1]):
            successors = (first_table[first, column].astype(np.int64) * second_count
                          + second_table[second, column])
            reached[successors[distances[successors] == 255]] = True

        frontier = np.flatnonzero(reached).astype(np.int64)
        depth += 1
        distances[frontier] = depth

    return distances


class TwoPhaseTables:
    """The move and pruning tables the two-phase solver searches over."""

    #: Files written to the cache directory, in build order.
    TABLE_NAMES = [
        'twist_move', 'flip_move', 'slice_move',
        'corner_move', 'ud_edge_move', 'slice_perm_move',
        'flip_slice_prune', 'twist_slice_prune',
        'corner_slice_prune', 'ud_edge_slice_prune',
    ]

    def __init__(self, directory=None, verbose=False):
        """
        Load the tables, generating and caching them if necessary.

        Args:
            directory (str): Where to cache the tables. Defaults to a
                             ``tables`` directory beside this module, or to
                             ``$KOCIEMBA_TABLE_DIR`` if that is set.
            verbose (bool): Print progress while generating.
        """
        self.directory = (directory or os.environ.get('KOCIEMBA_TABLE_DIR')
                          or DEFAULT_TABLE_DIR)
        self.verbose = verbose
        self.build_seconds = 0.0

        arrays = self._load()
        if arrays is None:
            arrays = self._build()
            self._save(arrays)

        for name in self.TABLE_NAMES:
            setattr(self, name, arrays[name])

        # The search reads these tables millions of times, and plain Python
        # lists and bytes are meaningfully quicker to index than numpy arrays
        # for one element at a time.
        self.twist_move_flat = self.twist_move.ravel().tolist()
        self.flip_move_flat = self.flip_move.ravel().tolist()
        self.slice_move_flat = self.slice_move.ravel().tolist()
        self.corner_move_flat = self.corner_move.ravel().tolist()
        self.ud_edge_move_flat = self.ud_edge_move.ravel().tolist()
        self.slice_perm_move_flat = self.slice_perm_move.ravel().tolist()
        self.flip_slice_prune_bytes = self.flip_slice_prune.tobytes()
        self.twist_slice_prune_bytes = self.twist_slice_prune.tobytes()
        self.corner_slice_prune_bytes = self.corner_slice_prune.tobytes()
        self.ud_edge_slice_prune_bytes = self.ud_edge_slice_prune.tobytes()

    def _path(self, name):
        """Return the cache path of one table."""
        return os.path.join(self.directory, f"v{TABLE_VERSION}_{name}.npy")

    def _load(self):
        """Return the cached tables, or None if any of them is missing."""
        arrays = {}
        for name in self.TABLE_NAMES:
            path = self._path(name)
            if not os.path.exists(path):
                return None
            arrays[name] = np.load(path)
        return arrays

    def _save(self, arrays):
        """Write the tables to the cache directory."""
        os.makedirs(self.directory, exist_ok=True)
        for name, array in arrays.items():
            np.save(self._path(name), array)

    def _report(self, message, started):
        """Print one progress line if the caller asked for progress."""
        if self.verbose:
            print(f"  {message:<28} {time.time() - started:6.2f}s")

    def _build(self):
        """
        Generate every table from scratch.

        Returns:
            dict: Table name to array.
        """
        overall = time.time()
        if self.verbose:
            print("Generating two-phase tables (one-off, then cached)...")

        arrays = {}

        started = time.time()
        arrays['twist_move'] = _build_orientation_move_table(
            TWIST_COUNT, cube_with_twist, twist_coordinate, PHASE1_MOVES)
        self._report('twist move table', started)

        started = time.time()
        arrays['flip_move'] = _build_orientation_move_table(
            FLIP_COUNT, cube_with_flip, flip_coordinate, PHASE1_MOVES)
        self._report('flip move table', started)

        started = time.time()
        arrays['slice_move'] = _build_slice_move_table(PHASE1_MOVES)
        self._report('slice move table', started)

        started = time.time()
        arrays['corner_move'] = _build_permutation_move_table(
            CORNER_COUNT, lambda move: MOVE_CUBES[move].cp, PHASE2_MOVES)
        self._report('corner move table', started)

        started = time.time()
        arrays['ud_edge_move'] = _build_permutation_move_table(
            len(UD_EDGES),
            lambda move: [MOVE_CUBES[move].ep[position] for position in UD_EDGES],
            PHASE2_MOVES)
        self._report('u/d edge move table', started)

        started = time.time()
        arrays['slice_perm_move'] = _build_permutation_move_table(
            len(SLICE_EDGES),
            lambda move: [MOVE_CUBES[move].ep[position] - SLICE_PERM_OFFSET
                          for position in SLICE_EDGES],
            PHASE2_MOVES)
        self._report('slice permutation table', started)

        started = time.time()
        arrays['flip_slice_prune'] = _build_pruning_table(
            arrays['flip_move'], arrays['slice_move'],
            FLIP_COUNT, SLICE_COUNT)
        self._report('flip/slice pruning table', started)

        started = time.time()
        arrays['twist_slice_prune'] = _build_pruning_table(
            arrays['twist_move'], arrays['slice_move'],
            TWIST_COUNT, SLICE_COUNT)
        self._report('twist/slice pruning table', started)

        started = time.time()
        arrays['corner_slice_prune'] = _build_pruning_table(
            arrays['corner_move'], arrays['slice_perm_move'],
            CORNER_PERM_COUNT, SLICE_PERM_COUNT)
        self._report('corner/slice pruning table', started)

        started = time.time()
        arrays['ud_edge_slice_prune'] = _build_pruning_table(
            arrays['ud_edge_move'], arrays['slice_perm_move'],
            UD_EDGE_PERM_COUNT, SLICE_PERM_COUNT)
        self._report('u/d edge pruning table', started)

        self.build_seconds = time.time() - overall
        if self.verbose:
            print(f"  {'total':<28} {self.build_seconds:6.2f}s")

        return arrays


def main():
    """Generate the tables from the command line."""
    tables = TwoPhaseTables(verbose=True)
    print(f"Tables cached in {tables.directory}")


if __name__ == '__main__':
    main()
