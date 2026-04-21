# orthogonal-maze-benchmark
Python code for generating grid-based maze environments from orthogonal polygons, with automatic start–goal sampling for path planning benchmarking.

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

- `benchmark_only.py`  

  Runs benchmarking experiments on existing maze datasets.

---

## Usage

Run the full pipeline:

```bash

python generate_and_benchmark.py
