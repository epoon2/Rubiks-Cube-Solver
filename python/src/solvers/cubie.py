#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cubie-level representation of a Rubik's Cube.

The sticker array in cube.py is convenient to look at but a poor thing to
search over: it has 54 entries, most combinations of which are not cubes at
all. A cubie cube instead records where the 8 corner pieces and 12 edge
pieces are and how they are twisted, which is exactly the information a
coordinate solver needs and nothing more.

Pieces are numbered the way Kociemba's two-phase literature numbers them, so
that the coordinate definitions in kociemba_tables.py can be read against the
published description:

    corners  0..7:  URF UFL ULB UBR DFR DLF DBL DRB
    edges    0..11: UR UF UL UB DR DF DL DB FR FL BL BR

Corner orientation counts clockwise twists needed to bring the piece's U or D
sticker back onto the U or D face. Edge orientation is 0 when the sticker that
belongs on U/D (or on F/B, for the four middle-slice edges) sits in the first
of the edge's two facelets. Those two conventions are what make phase 1 of the
two-phase algorithm meaningful: quarter turns of F and B are the only moves
that change edge orientation, and quarter turns of R, L, F and B are the only
ones that change corner orientation.
"""

from cube import RubiksCube

# Face indices, matching the order used by RubiksCube.
FRONT, BACK, UP, DOWN, LEFT, RIGHT = range(6)

CORNER_COUNT = 8
EDGE_COUNT = 12

CORNER_NAMES = ['URF', 'UFL', 'ULB', 'UBR', 'DFR', 'DLF', 'DBL', 'DRB']
EDGE_NAMES = ['UR', 'UF', 'UL', 'UB', 'DR', 'DF', 'DL', 'DB',
              'FR', 'FL', 'BL', 'BR']

# The (face, row, column) of every sticker of every corner, listed clockwise
# from the U or D sticker as seen from outside that corner.
CORNER_FACELETS = [
    ((UP, 2, 2), (RIGHT, 0, 0), (FRONT, 0, 2)),      # URF
    ((UP, 2, 0), (FRONT, 0, 0), (LEFT, 0, 2)),       # UFL
    ((UP, 0, 0), (LEFT, 0, 0), (BACK, 0, 2)),        # ULB
    ((UP, 0, 2), (BACK, 0, 0), (RIGHT, 0, 2)),       # UBR
    ((DOWN, 0, 2), (FRONT, 2, 2), (RIGHT, 2, 0)),    # DFR
    ((DOWN, 0, 0), (LEFT, 2, 2), (FRONT, 2, 0)),     # DLF
    ((DOWN, 2, 0), (BACK, 2, 2), (LEFT, 2, 0)),      # DBL
    ((DOWN, 2, 2), (RIGHT, 2, 2), (BACK, 2, 0)),     # DRB
]

# The (face, row, column) of both stickers of every edge, U/D sticker first
# for the eight U- and D-layer edges and F/B sticker first for the four
# middle-slice edges.
EDGE_FACELETS = [
    ((UP, 1, 2), (RIGHT, 0, 1)),      # UR
    ((UP, 2, 1), (FRONT, 0, 1)),      # UF
    ((UP, 1, 0), (LEFT, 0, 1)),       # UL
    ((UP, 0, 1), (BACK, 0, 1)),       # UB
    ((DOWN, 1, 2), (RIGHT, 2, 1)),    # DR
    ((DOWN, 0, 1), (FRONT, 2, 1)),    # DF
    ((DOWN, 1, 0), (LEFT, 2, 1)),     # DL
    ((DOWN, 2, 1), (BACK, 2, 1)),     # DB
    ((FRONT, 1, 2), (RIGHT, 1, 0)),   # FR
    ((FRONT, 1, 0), (LEFT, 1, 2)),    # FL
    ((BACK, 1, 2), (LEFT, 1, 0)),     # BL
    ((BACK, 1, 0), (RIGHT, 1, 2)),    # BR
]

# The four middle-slice edges, which phase 1 of the two-phase algorithm has to
# drive back into the middle layer.
SLICE_EDGES = [EDGE_NAMES.index(name) for name in ['FR', 'FL', 'BL', 'BR']]

# Face turns in the order the coordinate tables index them, three powers each.
FACE_ORDER = ['U', 'R', 'F', 'D', 'L', 'B']
MOVE_NAMES = [face + power for face in FACE_ORDER for power in ('', '2', "'")]

# Which face each sticker of each cubie belongs to when the cube is solved.
CORNER_COLOURS = [tuple(face for face, _, _ in facelets)
                  for facelets in CORNER_FACELETS]
EDGE_COLOURS = [tuple(face for face, _, _ in facelets)
                for facelets in EDGE_FACELETS]

_CORNER_LOOKUP = {colours: index
                  for index, colours in enumerate(CORNER_COLOURS)}
_EDGE_LOOKUP = {colours: index for index, colours in enumerate(EDGE_COLOURS)}


class CubieCube:
    """A cube described by piece positions and orientations."""

    def __init__(self, cp=None, co=None, ep=None, eo=None):
        """
        Initialize a cubie cube, solved unless told otherwise.

        Args:
            cp (list): Corner permutation; cp[i] is the corner sitting in
                       position i.
            co (list): Corner orientation; co[i] is 0, 1 or 2.
            ep (list): Edge permutation; ep[i] is the edge sitting in
                       position i.
            eo (list): Edge orientation; eo[i] is 0 or 1.
        """
        self.cp = list(cp) if cp is not None else list(range(CORNER_COUNT))
        self.co = list(co) if co is not None else [0] * CORNER_COUNT
        self.ep = list(ep) if ep is not None else list(range(EDGE_COUNT))
        self.eo = list(eo) if eo is not None else [0] * EDGE_COUNT

    def __eq__(self, other):
        return (self.cp == other.cp and self.co == other.co
                and self.ep == other.ep and self.eo == other.eo)

    def __repr__(self):
        return (f"CubieCube(cp={self.cp}, co={self.co}, "
                f"ep={self.ep}, eo={self.eo})")

    def copy(self):
        """Return an independent copy of this cube."""
        return CubieCube(self.cp, self.co, self.ep, self.eo)

    def is_solved(self):
        """
        Check whether every piece is home and untwisted.

        Returns:
            bool: True if the cube is solved.
        """
        return (self.cp == list(range(CORNER_COUNT))
                and self.co == [0] * CORNER_COUNT
                and self.ep == list(range(EDGE_COUNT))
                and self.eo == [0] * EDGE_COUNT)

    def multiply(self, other):
        """
        Return this cube with another cube's transformation applied after it.

        Args:
            other (CubieCube): Transformation to apply, usually a face turn.

        Returns:
            CubieCube: The composed cube.
        """
        cp = [self.cp[other.cp[i]] for i in range(CORNER_COUNT)]
        co = [(self.co[other.cp[i]] + other.co[i]) % 3
              for i in range(CORNER_COUNT)]
        ep = [self.ep[other.ep[i]] for i in range(EDGE_COUNT)]
        eo = [(self.eo[other.ep[i]] + other.eo[i]) % 2
              for i in range(EDGE_COUNT)]
        return CubieCube(cp, co, ep, eo)

    def apply_move(self, name):
        """
        Return this cube with a single named move applied.

        Args:
            name (str): Move in standard notation, e.g. "R", "R'" or "R2".

        Returns:
            CubieCube: The cube after the move.
        """
        return self.multiply(MOVE_CUBES[name])

    def apply_algorithm(self, algorithm):
        """
        Return this cube with a sequence of moves applied.

        Args:
            algorithm (str or iterable): Moves in standard notation.

        Returns:
            CubieCube: The cube after the moves.
        """
        moves = algorithm.split() if isinstance(algorithm, str) else algorithm
        cube = self
        for move in moves:
            cube = cube.apply_move(move)
        return cube

    def check(self):
        """
        Check that this cube could be reached from a solved cube.

        Returns:
            str: An empty string if the cube is legal, otherwise a description
                 of the first problem found.
        """
        if sorted(self.cp) != list(range(CORNER_COUNT)):
            return "corner permutation is not a permutation"
        if sorted(self.ep) != list(range(EDGE_COUNT)):
            return "edge permutation is not a permutation"
        if sum(self.co) % 3 != 0:
            return "corner twists do not cancel out"
        if sum(self.eo) % 2 != 0:
            return "edge flips do not cancel out"
        if _parity(self.cp) != _parity(self.ep):
            return "corner and edge permutations have different parity"
        return ""

    def to_facelets(self):
        """
        Convert this cube back to a sticker representation.

        Returns:
            RubiksCube: The same cube as a sticker array.
        """
        cube = RubiksCube()

        for position in range(CORNER_COUNT):
            piece = self.cp[position]
            twist = self.co[position]
            for slot in range(3):
                # A twist of t moves the piece's sticker s to slot s + t.
                face, row, col = CORNER_FACELETS[position][(slot + twist) % 3]
                cube.faces[face, row, col] = CORNER_COLOURS[piece][slot]

        for position in range(EDGE_COUNT):
            piece = self.ep[position]
            flip = self.eo[position]
            for slot in range(2):
                face, row, col = EDGE_FACELETS[position][(slot + flip) % 2]
                cube.faces[face, row, col] = EDGE_COLOURS[piece][slot]

        return cube


def _parity(permutation):
    """
    Return the parity of a permutation, 0 for even and 1 for odd.

    Args:
        permutation (list): A permutation of 0..n-1.

    Returns:
        int: 0 or 1.
    """
    swaps = 0
    for i in range(len(permutation)):
        for j in range(i + 1, len(permutation)):
            if permutation[i] > permutation[j]:
                swaps += 1
    return swaps % 2


def from_facelets(cube):
    """
    Convert a sticker representation into a cubie cube.

    Args:
        cube (RubiksCube): The cube to convert.

    Returns:
        CubieCube: The same cube described by piece positions.

    Raises:
        ValueError: If the stickers do not describe a set of real cubies.
    """
    # Read the colours off the centres rather than assuming the default
    # numbering, so a cube whose stickers were relabelled still converts.
    face_of_colour = {int(cube.faces[face, 1, 1]): face for face in range(6)}
    if len(face_of_colour) != 6:
        raise ValueError("cube does not have six distinct centre colours")

    def face_at(position):
        face, row, col = position
        colour = int(cube.faces[face, row, col])
        if colour not in face_of_colour:
            raise ValueError(f"sticker colour {colour} is not a centre colour")
        return face_of_colour[colour]

    cp, co, ep, eo = [], [], [], []

    for position in range(CORNER_COUNT):
        faces = tuple(face_at(facelet)
                      for facelet in CORNER_FACELETS[position])
        # The U or D sticker tells us how far the piece is twisted.
        twists = [slot for slot in range(3)
                  if faces[slot] in (UP, DOWN)]
        if len(twists) != 1:
            raise ValueError(
                f"corner at {CORNER_NAMES[position]} has {len(twists)} "
                "up/down stickers")
        twist = twists[0]

        untwisted = faces[twist:] + faces[:twist]
        if untwisted not in _CORNER_LOOKUP:
            raise ValueError(f"{untwisted} is not a corner of a Rubik's Cube")

        cp.append(_CORNER_LOOKUP[untwisted])
        co.append(twist)

    for position in range(EDGE_COUNT):
        faces = tuple(face_at(facelet) for facelet in EDGE_FACELETS[position])
        # U/D stickers identify the eight outer-layer edges; the four
        # middle-slice edges carry no U/D sticker, so F/B plays that role.
        flips = [slot for slot in range(2) if faces[slot] in (UP, DOWN)]
        if not flips:
            flips = [slot for slot in range(2)
                     if faces[slot] in (FRONT, BACK)]
        if len(flips) != 1:
            raise ValueError(
                f"edge at {EDGE_NAMES[position]} is not a cube edge")
        flip = flips[0]

        unflipped = (faces[flip], faces[1 - flip])
        if unflipped not in _EDGE_LOOKUP:
            raise ValueError(f"{unflipped} is not an edge of a Rubik's Cube")

        ep.append(_EDGE_LOOKUP[unflipped])
        eo.append(flip)

    return CubieCube(cp, co, ep, eo)


def _build_move_cubes():
    """
    Build the cubie cube of every move by reading it off the sticker model.

    Deriving the tables instead of hard-coding them keeps the two
    representations from drifting apart: if cube.py's moves ever change, the
    cubie moves change with them.

    Returns:
        dict: Move name to CubieCube.
    """
    cubes = {}
    for name in MOVE_NAMES:
        cube = RubiksCube()
        cube.apply_move(name)
        cubes[name] = from_facelets(cube)
    return cubes


MOVE_CUBES = _build_move_cubes()
