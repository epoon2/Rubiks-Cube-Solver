#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark script to compare the performance of different solvers.
"""

import time
import numpy as np
import pandas as pd
from cube import RubiksCube
from solvers.bfs_solver import BFSSolver
from solvers.astar_solver import AStarSolver


def run_benchmark(scrambles, solvers, num_trials=1):
    """
    Run benchmark tests for different solvers on different scrambles.
    
    Args:
        scrambles (list): List of (name, algorithm) tuples for scrambles to test.
        solvers (list): List of (name, solver_instance) tuples for solvers to test.
        num_trials (int): Number of trials to run for each combination.
        
    Returns:
        pd.DataFrame: DataFrame with benchmark results.
    """
    results = []
    
    for scramble_name, scramble_alg in scrambles:
        for solver_name, solver in solvers:
            for trial in range(num_trials):
                # Create a fresh cube
                cube = RubiksCube()
                
                # Apply the scramble
                cube.apply_algorithm(scramble_alg)
                
                # Time the solver
                start_time = time.time()
                success, solution, nodes, _ = solver.solve(cube)
                end_time = time.time()
                solve_time = end_time - start_time
                
                # Record the results
                results.append({
                    'Scramble': scramble_name,
                    'Solver': solver_name,
                    'Trial': trial + 1,
                    'Success': success,
                    'Solution Length': len(solution) if success else None,
                    'Nodes Explored': nodes,
                    'Time (seconds)': solve_time
                })
    
    # Convert to DataFrame and return
    df = pd.DataFrame(results)
    return df


def main():
    """Run the benchmarking."""
    # Define scrambles to test
    scrambles = [
        ('Solved', ''),  # Already solved cube
        ('Single Move', 'F'),  # One move away
        ('Two Moves', 'F U'),  # Two moves away
        ('Simple Sequence', 'R U R\' U\''),  # 4 moves away
        ('OLL Case', 'R U R\' U\' R\' F R F\''),  # 7 moves (common OLL algorithm)
    ]
    
    # Define solvers to test (with different depth limits)
    solvers = [
        ('BFS (depth=5)', BFSSolver(max_depth=5)),
        ('BFS (depth=8)', BFSSolver(max_depth=8)),
        ('A* (depth=5)', AStarSolver(max_depth=5)),
        ('A* (depth=8)', AStarSolver(max_depth=8)),
    ]
    
    # Number of trials
    num_trials = 3
    
    # Run the benchmark
    print("Running benchmarks...")
    results = run_benchmark(scrambles, solvers, num_trials)
    
    # Print the results
    print("\nBenchmark Results:")
    print(results.to_string(index=False))
    
    # Generate summary statistics
    summary = results.groupby(['Scramble', 'Solver']).agg({
        'Success': 'mean',
        'Solution Length': ['mean', 'min'],
        'Nodes Explored': ['mean', 'max'],
        'Time (seconds)': ['mean', 'min', 'max']
    }).reset_index()
    
    print("\nSummary Statistics:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(summary.to_string(index=False))
    
    # Compare solvers
    print("\nComparison between BFS and A*:")
    for scramble_name, _ in scrambles:
        bfs_data = results[(results['Scramble'] == scramble_name) & 
                          (results['Solver'] == 'BFS (depth=8)')]
        astar_data = results[(results['Scramble'] == scramble_name) & 
                            (results['Solver'] == 'A* (depth=8)')]
        
        if not bfs_data.empty and not astar_data.empty:
            bfs_time = bfs_data['Time (seconds)'].mean()
            astar_time = astar_data['Time (seconds)'].mean()
            bfs_nodes = bfs_data['Nodes Explored'].mean()
            astar_nodes = astar_data['Nodes Explored'].mean()
            
            print(f"Scramble: {scramble_name}")
            print(f"  - Time: BFS = {bfs_time:.6f}s, A* = {astar_time:.6f}s, Ratio = {bfs_time/astar_time if astar_time > 0 else 'N/A':.2f}x")
            print(f"  - Nodes: BFS = {bfs_nodes:.1f}, A* = {astar_nodes:.1f}, Ratio = {bfs_nodes/astar_nodes if astar_nodes > 0 else 'N/A':.2f}x")
            
            if all(bfs_data['Success']) and all(astar_data['Success']):
                bfs_sol_len = bfs_data['Solution Length'].mean()
                astar_sol_len = astar_data['Solution Length'].mean()
                print(f"  - Solution Length: BFS = {bfs_sol_len:.1f}, A* = {astar_sol_len:.1f}")
            print()


if __name__ == '__main__':
    main() 