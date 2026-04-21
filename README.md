# orthogonal-maze-benchmark
Python code for generating grid-based maze environments from orthogonal polygons, with automatic start–goal sampling for path planning benchmarking.

This repository contains the generation pipeline and benchmarking is introduced in “A Dataset of Orthogonal Polygon-Derived Maze Environments for Path Planning Benchmarking” by Nguyễn Kiều Linh.

# Orthogonal Polygon Maze Generator

Python code for generating grid-based maze environments from orthogonal polygons, with benchmarking of path planning algorithms.

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

---

## Usage

Run the full pipeline:

```bash

python generate_and_benchmark.py


## Repository Structure

After executing the pipeline, the generated dataset is organized as follows:

```text
datasets/
├── polygons/                  # Raw mathematical coordinates of orthogonal polygons (.csv)
├── mazes/                     # Rasterized binary grids and configuration parameters (.json)
└── metadata/                  # Analytical reports
    ├── dataset_summary.csv    # Structural metadata mapping polygons to mazes
    ├── pathfinding_summary.csv # Raw pathfinding metrics (Success, Length, Time)




