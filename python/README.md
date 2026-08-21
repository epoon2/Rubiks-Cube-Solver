# Python Implementation of Rubik's Cube Solver

This directory contains the Python implementation of the Rubik's Cube Solver.

## Structure

- `src/` - Source code
  - `cube.py` - Sticker representation of a 3x3 cube and its six face turns
  - `demo.py` - Demonstration script for cube functionality
  - `depth_wall.py` - Experiment measuring where BFS and A* stop working
  - `benchmark.py` - Performance comparison of all three solvers
  - `solvers/`
    - `bfs_solver.py` - Breadth-first search, good for a few moves
    - `astar_solver.py` - A* with a misplaced-sticker heuristic
    - `cubie.py` - Piece-level cube representation and sticker conversion
    - `kociemba_tables.py` - Coordinate move tables and pruning tables
    - `kociemba_solver.py` - Kociemba's two-phase algorithm
  - `visualization/`
    - `cube_visualizer.py` - 2D matplotlib view of the cube as an unfolded net
- `tests/` - Unit tests, one module per source module

## Usage

### Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Demo

```bash
cd src
python demo.py
```

### Solving a scramble

```bash
cd src
python -m solvers.kociemba_solver
```

The solver modules import `cube` from the `src` directory, so they have to be
run as modules from `src` rather than by path.

The first run generates about 8 MB of move and pruning tables, which takes
around two seconds and is then cached in `src/solvers/tables/`. That directory
is not checked in; delete it to force a rebuild. Set `KOCIEMBA_TABLE_DIR` to
cache them somewhere else.

### Running the Tests

```bash
python -m pytest tests
```

The two-phase tests generate the tables on first run, so the suite takes a few
seconds longer the first time.

### Benchmarking

```bash
cd src
python depth_wall.py     # where BFS and A* run out of road
python benchmark.py      # all three solvers, plus the two-phase distribution
```

Both scripts run each solver in its own process under a time and memory limit,
because BFS on anything past six moves does not come back.

## Features

- Representation of a 3x3x3 Rubik's Cube, as stickers and as pieces
- Face turns in standard notation, including half turns (`R`, `R'`, `R2`)
- Application of algorithms using standard notation
- Verification of solved state, and rejection of cubes no scramble could
  produce
- Three solvers: BFS, A*, and Kociemba's two-phase algorithm
- Performance benchmarking and a reproducible measurement of the search wall

## Next Steps

- A standalone IDA* solver with a pattern database, for provably optimal
  solutions
- An interactive GUI, and animation of a solution
- Support for larger cubes (4x4, 5x5)
