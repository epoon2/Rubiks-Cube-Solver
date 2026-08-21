#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kociemba's two-phase solver for the Rubik's Cube.

Unlike the BFS and A* solvers in this package, this one does not search the
cube's 4.3e19 states. It searches two much smaller quotients of them:

    Phase 1 gets the cube into G1 = <U, D, R2, L2, F2, B2>, the states with
            every edge and corner correctly oriented and the four
            middle-slice edges back in the middle slice. Only three
            coordinates matter, and 12 moves always suffice.

    Phase 2 solves the cube from there using only G1's own moves, which can
            no longer disturb orientation. Three more coordinates cover it,
            and 18 moves always suffice.

Both phases run IDA* against exact distance tables for pairs of coordinates
(see kociemba_tables.py), so the heuristic is admissible and the searches are
short. The price is that a two-phase solution is not necessarily optimal: the
detour through G1 costs a few moves. Solutions here typically come out in the
low twenties against an optimal solve of around 18.

The first phase-1 solution found is rarely the best, so the search keeps
looking for shorter ones until it either reaches a target length or runs out
of its time budget.
"""

import time

from cube import RubiksCube
from solvers.cubie import CubieCube, from_facelets
from solvers.kociemba_tables import (PHASE1_MOVES, PHASE2_MOVES, SLICE_COUNT,
                                     SLICE_PERM_COUNT, TwoPhaseTables,
                                     corner_perm_coordinate, flip_coordinate,
                                     slice_coordinate, slice_perm_coordinate,
                                     twist_coordinate, ud_edge_perm_coordinate)

# Phase 1 needs at most 12 moves and phase 2 at most 18; both bounds are
# proven, so a search that reaches them without an answer has a bug.
MAX_PHASE1_DEPTH = 12
MAX_PHASE2_DEPTH = 18
MAX_TOTAL_DEPTH = MAX_PHASE1_DEPTH + MAX_PHASE2_DEPTH

FACE_ORDER = ['U', 'R', 'F', 'D', 'L', 'B']
_PHASE1_FACES = [index // 3 for index in range(len(PHASE1_MOVES))]
_PHASE2_FACES = [FACE_ORDER.index(move[0]) for move in PHASE2_MOVES]

# U and D (and R/L, F/B) commute, so a sequence that turns D before U has an
# equivalent twin that turns U first. Only the first ordering is searched.
_OPPOSITE = [(face + 3) % 6 for face in range(6)]


class KociembaSolver:
    """A two-phase solver that handles any legal cube."""

    def __init__(self, target_length=23, timeout=0.5, tables=None,
                 verbose=False):
        """
        Initialize the solver, generating the lookup tables if needed.

        Args:
            target_length (int): Stop looking once a solution this short or
                                 shorter has been found.
            timeout (float): Overall budget in seconds. The hunt for a first
                             solution is never cut short, so a solve can run
                             past this, but no time is spent looking for a
                             shorter one once the budget is gone.
            tables (TwoPhaseTables): Prebuilt tables, shared between solvers.
            verbose (bool): Print progress while generating tables.
        """
        self.target_length = target_length
        self.timeout = timeout
        self.tables = tables if tables is not None else TwoPhaseTables(
            verbose=verbose)

        self._reset()

    def _reset(self):
        """Clear the per-solve search state."""
        self._start = CubieCube()
        self._best = None
        self._nodes = 0
        self._deadline = None
        self._allowance = MAX_TOTAL_DEPTH

    def solve(self, cube):
        """
        Solve a cube.

        Args:
            cube (RubiksCube or CubieCube): The cube to solve.

        Returns:
            tuple: (solution_found, solution_moves, nodes_explored, time_taken)

        Raises:
            ValueError: If the cube could not be reached from a solved cube.
        """
        started = time.time()
        self._reset()
        self._deadline = started + self.timeout

        cubie = cube if isinstance(cube, CubieCube) else from_facelets(cube)
        problem = cubie.check()
        if problem:
            raise ValueError(f"cube cannot be solved: {problem}")

        if cubie.is_solved():
            return True, [], 0, time.time() - started

        self._start = cubie

        flip = flip_coordinate(cubie)
        twist = twist_coordinate(cubie)
        middle = slice_coordinate(cubie)

        # A phase-1 solution always leads to a phase-2 solution, but not
        # always a short one, and a long phase 2 is expensive to find. Capping
        # the total length keeps those searches cheap and simply moves on to
        # the next phase-1 solution instead, of which there are plenty. The
        # cap is lifted for a second pass on the rare cube that defeats it.
        for allowance in (self.target_length, MAX_TOTAL_DEPTH):
            self._allowance = allowance

            for depth in range(1, MAX_PHASE1_DEPTH + 1):
                if self._best is not None and depth >= len(self._best):
                    break
                if self._search_phase1(flip, twist, middle, depth, [], -1):
                    break
                if self._best is not None and self._out_of_time():
                    break

            if self._best is not None:
                break

        if self._best is None:
            return False, [], self._nodes, time.time() - started

        return True, list(self._best), self._nodes, time.time() - started

    def _out_of_time(self):
        """Return True once the time budget has been spent."""
        return time.time() > self._deadline

    def _search_phase1(self, flip, twist, middle, remaining, path, last_face):
        """
        Look for a way into G1 in exactly ``remaining`` moves.

        Args:
            flip (int): Edge orientation coordinate.
            twist (int): Corner orientation coordinate.
            middle (int): Middle-slice edge location coordinate.
            remaining (int): Moves left to spend.
            path (list): Moves played so far.
            last_face (int): Face of the previous move, or -1 at the root.

        Returns:
            bool: True if the whole search should stop.
        """
        self._nodes += 1
        # Checking the clock at every node would cost more than it saves, and
        # the search only has to stop promptly once an answer already exists.
        if not self._nodes & 0xFFF and self._best is not None and self._out_of_time():
            return True

        if remaining == 0:
            # Coordinate 0 of each pruning table is reached only by the
            # coordinate pair (0, 0), so this is exactly the G1 test.
            if flip == 0 and twist == 0 and middle == 0:
                return self._search_phase2(path)
            return False

        flip_move = self.tables.flip_move_flat
        twist_move = self.tables.twist_move_flat
        slice_move = self.tables.slice_move_flat
        flip_prune = self.tables.flip_slice_prune_bytes
        twist_prune = self.tables.twist_slice_prune_bytes

        moves = len(PHASE1_MOVES)
        flip_row = flip * moves
        twist_row = twist * moves
        slice_row = middle * moves
        budget = remaining - 1

        for index in range(moves):
            face = _PHASE1_FACES[index]
            if face == last_face or (_OPPOSITE[face] == last_face
                                     and face < last_face):
                continue

            next_slice = slice_move[slice_row + index]
            next_flip = flip_move[flip_row + index]
            if flip_prune[next_flip * SLICE_COUNT + next_slice] > budget:
                continue
            next_twist = twist_move[twist_row + index]
            if twist_prune[next_twist * SLICE_COUNT + next_slice] > budget:
                continue

            path.append(PHASE1_MOVES[index])
            stop = self._search_phase1(next_flip, next_twist, next_slice,
                                       budget, path, face)
            path.pop()
            if stop:
                return True

        return False

    def _search_phase2(self, phase1_path):
        """
        Finish a cube that phase 1 has driven into G1.

        Args:
            phase1_path (list): The moves phase 1 played.

        Returns:
            bool: True if the whole search should stop.
        """
        cubie = self._start.apply_algorithm(phase1_path)
        corners = corner_perm_coordinate(cubie)
        ud_edges = ud_edge_perm_coordinate(cubie)
        middle = slice_perm_coordinate(cubie)

        # Only bother with phase 2 if it could come in under the allowance,
        # and beat what we already have.
        limit = min(MAX_PHASE2_DEPTH, self._allowance - len(phase1_path))
        if self._best is not None:
            limit = min(limit, len(self._best) - len(phase1_path) - 1)
        if limit < 0:
            return False

        for depth in range(0, limit + 1):
            path = []
            if self._search_phase2_depth(corners, ud_edges, middle, depth,
                                         path, -1):
                self._record(phase1_path + path)
                return (self._best is not None
                        and len(self._best) <= self.target_length)

        return self._best is not None and self._out_of_time()

    def _search_phase2_depth(self, corners, ud_edges, middle, remaining, path,
                             last_face):
        """
        Look for a G1 solution in exactly ``remaining`` moves.

        Args:
            corners (int): Corner permutation coordinate.
            ud_edges (int): U/D edge permutation coordinate.
            middle (int): Middle-slice edge permutation coordinate.
            remaining (int): Moves left to spend.
            path (list): Moves played so far, appended to in place.
            last_face (int): Face of the previous move, or -1 at the root.

        Returns:
            bool: True if a solution was written into ``path``.
        """
        self._nodes += 1

        if remaining == 0:
            return corners == 0 and ud_edges == 0 and middle == 0

        corner_move = self.tables.corner_move_flat
        ud_edge_move = self.tables.ud_edge_move_flat
        slice_perm_move = self.tables.slice_perm_move_flat
        corner_prune = self.tables.corner_slice_prune_bytes
        ud_edge_prune = self.tables.ud_edge_slice_prune_bytes

        moves = len(PHASE2_MOVES)
        corner_row = corners * moves
        ud_edge_row = ud_edges * moves
        slice_row = middle * moves
        budget = remaining - 1

        for index in range(moves):
            face = _PHASE2_FACES[index]
            if face == last_face or (_OPPOSITE[face] == last_face
                                     and face < last_face):
                continue

            next_middle = slice_perm_move[slice_row + index]
            next_corners = corner_move[corner_row + index]
            if corner_prune[next_corners * SLICE_PERM_COUNT + next_middle] > budget:
                continue
            next_ud_edges = ud_edge_move[ud_edge_row + index]
            if ud_edge_prune[next_ud_edges * SLICE_PERM_COUNT + next_middle] > budget:
                continue

            path.append(PHASE2_MOVES[index])
            if self._search_phase2_depth(next_corners, next_ud_edges,
                                         next_middle, budget, path, face):
                return True
            path.pop()

        return False

    def _record(self, solution):
        """
        Keep a solution if it is the shortest seen so far.

        Args:
            solution (list): The full move sequence.
        """
        solution = simplify(solution)
        if self._best is None or len(solution) < len(self._best):
            self._best = solution


def simplify(moves):
    """
    Collapse redundant turns in a move sequence.

    Phase 1 and phase 2 are searched independently, so the join between them
    can leave two turns of the same face next to each other (possibly with a
    turn of the opposite face, which commutes, in between).

    Args:
        moves (list): Moves in standard notation.

    Returns:
        list: An equivalent, no longer sequence.
    """
    powers = {'': 1, '2': 2, "'": 3}
    suffixes = {1: '', 2: '2', 3: "'"}

    result = [(move[0], powers[move[1:]]) for move in moves]

    changed = True
    while changed:
        changed = False
        for index in range(len(result)):
            for other in (index + 1, index + 2):
                if other >= len(result):
                    break
                if result[other][0] == result[index][0]:
                    total = (result[index][1] + result[other][1]) % 4
                    result[index] = (result[index][0], total)
                    del result[other]
                    if total == 0:
                        del result[index]
                    changed = True
                    break
                # Turns of different faces only commute if the faces are
                # opposite, so stop looking past anything else.
                if FACE_ORDER.index(result[other][0]) != _OPPOSITE[
                        FACE_ORDER.index(result[index][0])]:
                    break
            if changed:
                break

    return [face + suffixes[power] for face, power in result]


def main():
    """Demonstrate the two-phase solver."""
    scramble = "R U2 D' B D' F2 L2 R2 D' F' D2 B2 R2 U' L2 F2 U R2 D2"

    print("Generating tables...")
    solver = KociembaSolver(verbose=True)

    cube = RubiksCube()
    print(f"Applying scramble: {scramble}")
    cube.apply_algorithm(scramble)

    print("Solving with the two-phase algorithm...")
    success, solution, nodes, seconds = solver.solve(cube)

    if not success:
        print("No solution found.")
        return

    print(f"Solution found! Moves: {' '.join(solution)}")
    print(f"Solution length: {len(solution)}")
    print(f"Nodes explored: {nodes}")
    print(f"Time taken: {seconds:.3f} seconds")

    cube.apply_algorithm(' '.join(solution))
    print("Is solved:", cube.is_solved())


if __name__ == '__main__':
    main()
