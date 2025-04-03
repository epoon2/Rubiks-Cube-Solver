# Python Implementation of Rubik's Cube Solver

This directory contains the initial Python implementation of the Rubik's Cube Solver.

## Structure

- `src/` - Source code
  - `cube.py` - Basic Rubik's Cube representation and operations
  - `demo.py` - Demonstration script for cube functionality
- `tests/` - Unit tests
  - `test_cube.py` - Tests for the cube implementation

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

### Running the Tests

```bash
cd tests
python test_cube.py
```

## Features

- Representation of a 3x3x3 Rubik's Cube
- Basic cube operations (F, B, U, D, L, R moves)
- Application of algorithms using standard notation
- Verification of solved state

## Next Steps

- Implement solving algorithms (BFS, A*)
- Add visualization
- Support for larger cubes (4x4, 5x5) 