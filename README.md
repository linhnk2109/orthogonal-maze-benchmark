# orthogonal-maze-benchmark
Python code for generating grid-based maze environments from orthogonal polygons, with automatic start–goal sampling for path planning benchmarking.

This repository contains the generation pipeline and benchmarking is introduced in “A Dataset of Orthogonal Polygon-Derived Maze Environments for Path Planning Benchmarking” by Nguyễn Kiều Linh.

---

## Description

This repository provides a simple pipeline for:

- Generating orthogonal polygons  

- Converting polygons into grid-based maze environments  

- Automatically sampling start–goal pairs  

- Benchmarking classical path planning algorithms  

---

## Files

- `generate_and_benchmark.py`  

  Main script: generates polygons, converts them to grid mazes, samples start–goal pairs, and runs path planning algorithms.

- `benchmark_pathPlanning.py`  

  Runs benchmarking experiments on existing maze datasets. The suite evaluates 6 algorithms across two fundamental paradigms:

  + Discrete Grid-Based: BFS, Dijkstra, A*.

  + Continuous Sampling-Based: PRM, RRT-Connect, RRT*.

- `benchmark_stochastic_algos.py`
This script executes the **extended stochastic evaluation** for sampling-based path planning algorithms (PRM, RRT-Connect, and RRT*). 

* Multi-trial Aggregation: Runs 5 independent trials per maze instance for each stochastic algorithm to ensure statistical robustness, calculating the average success rate, path length, and execution time.
* RRT Asymptotic Optimality:** Implements a convergence-based termination criterion for RRT*. Instead of halting upon finding the first path, it utilizes an optimization budget to continuously rewire the tree until the path cost stagnates, demonstrating its asymptotic optimality.
* Data Output: Aggregates and exports the averaged benchmarking metrics to `stochastic_extended_benchmark.csv`.
* Visualization: Automatically generates and saves high-resolution visualizations of the optimized RRT* trajectories (omitting intermediate exploration trees for clarity) into the `images/rrt_star_optimized/` directory.
---

## Usage

Run the full pipeline:

```bash

python generate_and_benchmark.py


```
## Repository Structure

After executing the pipeline, the generated dataset is organized as follows:
```text
datasets/
├── polygons/                  # Raw mathematical coordinates of orthogonal polygons (.csv)
├── mazes/                     # Rasterized binary grids and configuration parameters (.json)
└── metadata/                  # Analytical reports
    ├── dataset_summary.csv    # Structural metadata mapping polygons to mazes
    ├── pathfinding_summary.csv # Raw pathfinding metrics (Success, Length, Time)


```
## How to Cite

If you use this code or dataset in your research, please cite:

Nguyen, K. L. (2026).  
"A Dataset of Orthogonal Polygon-Derived Maze Environments for Path Planning Benchmarking".  
Zenodo. https://doi.org/10.5281/zenodo.19640621

and the associated article:

Nguyen, K. L. (2026).  
"A Dataset of Orthogonal Polygon-Derived Maze Environments for Path Planning Benchmarking".  
Data in Brief (submitted).


