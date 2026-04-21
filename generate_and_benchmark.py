import os
import json
import time
import csv
import random
import tracemalloc
import numpy as np
from collections import deque
from shapely.geometry import Polygon
from PIL import Image, ImageDraw

# Import pathfinding algorithms from the benchmark module
from benchmark_pathplanning import (
    run_bfs, 
    run_astar, 
    run_dijkstra,
    run_prm, 
    run_rrt_connect, 
    run_rrt_star,
    path_length
)

# =================================================================
# 1. ORTHOGONAL POLYGON GENERATION
# =================================================================
class OrthogonalPolygonGenerator:
    """Generates random orthogonal polygons with a specified number of vertices."""
    
    def __init__(self):
        self.d = {}  # Dictionary to track used coordinates
        self.data = [[0, 0], [1, 0], [1, 1], [0, 1]]  # Initial seed shape
        self.a = [[1, 0], [-1, 0], [0, 1], [0, -1]]   # Movement directions
        
        # Tracking metrics
        self.start_time = 0
        self.end_time = 0
        self.iteration_count = 0
        self.successful_additions = 0
        self.peak_memory_kb = 0
        
        # Initialize base coordinates
        for pt in self.data:
            self.d[tuple(pt)] = 1

    def solve1(self, x, y, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i):
        f = [[x0, y0], [x2, y2], [x6, y6], [x5, y5], [x3, y3]]
        self.d[(x1, y1)] = 0; self.d[(x2, y2)] = 1; self.d[(x3, y3)] = 1
        self.d[(x4, y4)] = 0; self.d[(x5, y5)] = 1; self.d[(x6, y6)] = 1
        self.d[(x2, y2)] = 1; self.d[(x, y)] = 0
        self.data = self.data[:i] + f + self.data[i + 1:]
        self.successful_additions += 1

    def solve2(self, x, y, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i):
        f = [[x3, y3], [x5, y5], [x6, y6], [x2, y2], [x0, y0]]
        self.d[(x1, y1)] = 0; self.d[(x2, y2)] = 1; self.d[(x3, y3)] = 1
        self.d[(x4, y4)] = 0; self.d[(x5, y5)] = 1; self.d[(x6, y6)] = 1
        self.d[(x2, y2)] = 1; self.d[(x, y)] = 0
        self.data = self.data[:i] + f + self.data[i + 1:]
        self.successful_additions += 1
        
    def solve3(self, x, y, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i):
        f = [[x6, y6], [x5, y5], [x2, y2], [x4, y4], [x3, y3]]
        self.d[(x1, y1)] = 0; self.d[(x2, y2)] = 1; self.d[(x3, y3)] = 1
        self.d[(x4, y4)] = 1; self.d[(x5, y5)] = 1; self.d[(x6, y6)] = 1
        self.d[(x, y)] = 0
        self.data = self.data[:i] + f + self.data[i + 1:]
        self.successful_additions += 1

    def solve4(self, x, y, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i):
        f = [[x3, y3], [x4, y4], [x2, y2], [x5, y5], [x6, y6]]
        self.d[(x1, y1)] = 0; self.d[(x2, y2)] = 1; self.d[(x3, y3)] = 1
        self.d[(x4, y4)] = 1; self.d[(x5, y5)] = 1; self.d[(x6, y6)] = 1
        self.d[(x, y)] = 0
        self.data = self.data[:i] + f + self.data[i + 1:]
        self.successful_additions += 1

    def generate(self, n):
        """Generate orthogonal polygon with n vertices."""
        tracemalloc.start()
        self.start_time = time.time()
        self.iteration_count = 0
        self.successful_additions = 0
        
        if n > 4:
            if n % 4 != 0:
                self.data = [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2], [0, 2]]
                self.d[(2, 1)] = 1; self.d[(2, 2)] = 1; self.d[(1, 2)] = 0
                self.d[(0, 2)] = 1; self.d[(0, 1)] = 0
                
            while True:
                if len(self.data) == n: break
                    
                self.iteration_count += 1
                m = random.randint(0, len(self.data) - 1)
                
                for i in range(m, len(self.data)):
                    if len(self.data) == n: break
                        
                    x, y = self.data[i]
                    
                    if ((x, y + 1) in self.d) and ((x, y - 1) in self.d) and \
                       ((x + 1, y) in self.d) and ((x - 1, y) in self.d):
                        continue
                        
                    j = random.randint(0, 3)
                    x0, y0 = x + self.a[j][0], y + self.a[j][1]
                    
                    if ((x0, y0 + 1) in self.d) and ((x0, y0 - 1) in self.d) and \
                       ((x0 + 1, y0) in self.d) and ((x0 - 1, y0) in self.d):
                        continue
                        
                    if (x0, y0) in self.d:
                        if len(self.data) == n: break
                        self._process_vertex(x, y, x0, y0, j, i)
                        
        self.end_time = time.time()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        self.peak_memory_kb = peak_mem / 1024 
        return self.data

    def _process_vertex(self, x, y, x0, y0, j, i):
        """Core logic for vertex extrusion and manipulation to maintain orthogonality."""
        if j == 0:
            x1 = x; y1 = y + 1; x2 = x0; y2 = y0 + 1
            x22 = x0 + 1; y22 = y0
            if ((x22, y22) in self.d) and (self.d[(x0, y0)] == 0):
                if ((x1, y1) not in self.d) and ((x2, y2) not in self.d):
                    x3 = x1; y3 = y1 + 1; x4 = x2; y4 = y2 + 1
                    x5 = x3 + 2; y5 = y3; x6 = x2 + 1; y6 = y2
                    if ((x3, y3) not in self.d) and ((x4, y4) not in self.d) and ((x5, y5) not in self.d) and ((x6, y6) not in self.d):
                        self.solve1(x, y, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i)
                else:
                    x1 = x; y1 = y - 1; x2 = x0; y2 = y0 - 1
                    if ((x1, y1) not in self.d) and ((x2, y2) not in self.d):
                        x3 = x1; y3 = y1 - 1; x4 = x2; y4 = y2 - 1
                        x5 = x3 + 2; y5 = y3; x6 = x2 + 1; y6 = y2
                        if ((x3, y3) not in self.d) and ((x4, y4) not in self.d) and ((x5, y5) not in self.d) and ((x6, y6) not in self.d):
                            self.solve2(x, y, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i)
            else:
                if ((x1, y1) not in self.d) and ((x2, y2) not in self.d):
                    x3 = x1; y3 = y1 + 1; x4 = x2; y4 = y2 + 1
                    x5 = x1 + 2; y5 = y1; x6 = x0 + 1; y6 = y0
                    if ((x3, y3) not in self.d) and ((x4, y4) not in self.d) and ((x5, y5) not in self.d) and ((x6, y6) not in self.d):
                        self.solve3(x, y, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i)
                else:
                    x1 = x; y1 = y - 1; x2 = x0; y2 = y0 - 1
                    if ((x1, y1) not in self.d) and ((x2, y2) not in self.d):
                        x3 = x1; y3 = y1 - 1; x4 = x2; y4 = y2 - 1
                        x5 = x1 + 2; y5 = y1; x6 = x0 + 1; y6 = y0
                        if ((x3, y3) not in self.d) and ((x4, y4) not in self.d) and ((x5, y5) not in self.d) and ((x6, y6) not in self.d):
                            self.solve4(x, y, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i)
        elif j == 1:
            x1 = x; y1 = y + 1; x2 = x0; y2 = y0 + 1
            x22 = x0 - 1; y22 = y0
            if ((x22, y22) in self.d) and (self.d[(x0, y0)] == 0):
                if ((x1, y1) not in self.d) and ((x2, y2) not in self.d):
                    x3 = x1; y3 = y1 + 1; x4 = x2; y4 = y2 + 1
                    x5 = x3 - 2; y5 = y3; x6 = x2 - 1; y6 = y2
                    if ((x3, y3) not in self.d) and ((x4, y4) not in self.d) and ((x5, y5) not in self.d) and ((x6, y6) not in self.d):
                        self.solve2(x, y, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i)
                else:
                    x1 = x; y1 = y - 1; x2 = x0; y2 = y0 - 1
                    if ((x1, y1) not in self.d) and ((x2, y2) not in self.d):
                        x3 = x1; y3 = y1 - 1; x4 = x2; y4 = y2 - 1
                        x5 = x3 - 2; y5 = y3; x6 = x2 - 1; y6 = y2
                        if ((x3, y3) not in self.d) and ((x4, y4) not in self.d) and ((x5, y5) not in self.d) and ((x6, y6) not in self.d):
                            self.solve1(x, y, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i)
            else:
                if ((x1, y1) not in self.d) and ((x2, y2) not in self.d):
                    x3 = x1; y3 = y1 + 1; x4 = x2; y4 = y2 + 1
                    x5 = x1 - 2; y5 = y1; x6 = x0 - 1; y6 = y0
                    if ((x3, y3) not in self.d) and ((x4, y4) not in self.d) and ((x5, y5) not in self.d) and ((x6, y6) not in self.d):
                        self.solve4(x, y, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i)
                else:
                    x1 = x; y1 = y - 1; x2 = x0; y2 = y0 - 1
                    if ((x1, y1) not in self.d) and ((x2, y2) not in self.d):
                        x3 = x1; y3 = y1 - 1; x4 = x2; y4 = y2 - 1
                        x5 = x1 - 2; y5 = y1; x6 = x0 - 1; y6 = y0
                        if ((x3, y3) not in self.d) and ((x4, y4) not in self.d) and ((x5, y5) not in self.d) and ((x6, y6) not in self.d):
                            self.solve3(x, y, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i)
        elif j == 2:
            x1 = x + 1; y1 = y; x2 = x0 + 1; y2 = y0
            x22 = x0; y22 = y0 + 1
            if ((x22, y22) in self.d) and (self.d[(x0, y0)] == 0):
                if ((x1, y1) not in self.d) and ((x2, y2) not in self.d):
                    x3 = x1 + 1; y3 = y1; x4 = x2 + 1; y4 = y2
                    x5 = x3; y5 = y3 + 2; x6 = x2; y6 = y2 + 1
                    if ((x3, y3) not in self.d) and ((x4, y4) not in self.d) and ((x5, y5) not in self.d) and ((x6, y6) not in self.d):
                        self.solve2(x, y, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i)
                else:
                    x1 = x - 1; y1 = y; x2 = x0 - 1; y2 = y0
                    if ((x1, y1) not in self.d) and ((x2, y2) not in self.d):
                        x3 = x1 - 1; y3 = y1; x4 = x2 - 1; y4 = y2
                        x5 = x3; y5 = y3 + 2; x6 = x2; y6 = y2 + 1
                        if ((x3, y3) not in self.d) and ((x4, y4) not in self.d) and ((x5, y5) not in self.d) and ((x6, y6) not in self.d):
                            self.solve1(x, y, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i)
            else:
                if ((x1, y1) not in self.d) and ((x2, y2) not in self.d):
                    x3 = x1 + 1; y3 = y1; x4 = x2 + 1; y4 = y2
                    x5 = x1; y5 = y1 + 2; x6 = x0 ; y6 = y0 + 1
                    if ((x3, y3) not in self.d) and ((x4, y4) not in self.d) and ((x5, y5) not in self.d) and ((x6, y6) not in self.d):
                        self.solve4(x, y, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i)
                else:
                    x1 = x - 1; y1 = y; x2 = x0 - 1; y2 = y0
                    if ((x1, y1) not in self.d) and ((x2, y2) not in self.d):
                        x3 = x1 - 1; y3 = y1; x4 = x2 - 1; y4 = y2
                        x5 = x1; y5 = y1 + 2; x6 = x0; y6 = y0 + 1
                        if ((x3, y3) not in self.d) and ((x4, y4) not in self.d) and ((x5, y5) not in self.d) and ((x6, y6) not in self.d):
                            self.solve3(x, y, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i)
        elif j == 3:
            x1 = x + 1; y1 = y; x2 = x0 + 1; y2 = y0
            x22 = x0; y22 = y0 - 1
            if ((x22, y22) in self.d) and (self.d[(x0, y0)] == 0):
                if ((x1, y1) not in self.d) and ((x2, y2) not in self.d):
                    x3 = x1 + 1; y3 = y1; x4 = x2 + 1; y4 = y2
                    x5 = x3; y5 = y3 - 2; x6 = x2; y6 = y2 - 1
                    if ((x3, y3) not in self.d) and ((x4, y4) not in self.d) and ((x5, y5) not in self.d) and ((x6, y6) not in self.d):
                        self.solve1(x, y, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i)
                else:
                    x1 = x - 1; y1 = y; x2 = x0 - 1; y2 = y0
                    if ((x1, y1) not in self.d) and ((x2, y2) not in self.d):
                        x3 = x1 - 1; y3 = y1; x4 = x2 - 1; y4 = y2
                        x5 = x3; y5 = y3 - 2; x6 = x2; y6 = y2 - 1
                        if ((x3, y3) not in self.d) and ((x4, y4) not in self.d) and ((x5, y5) not in self.d) and ((x6, y6) not in self.d):
                            self.solve2(x, y, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i)
            else:
                if ((x1, y1) not in self.d) and ((x2, y2) not in self.d):
                    x3 = x1 + 1; y3 = y1; x4 = x2 + 1; y4 = y2
                    x5 = x1; y5 = y1 - 2; x6 = x0 ; y6 = y0 - 1
                    if ((x3, y3) not in self.d) and ((x4, y4) not in self.d) and ((x5, y5) not in self.d) and ((x6, y6) not in self.d):
                        self.solve3(x, y, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i)
                else:
                    x1 = x - 1; y1 = y; x2 = x0 - 1; y2 = y0
                    if ((x1, y1) not in self.d) and ((x2, y2) not in self.d):
                        x3 = x1 - 1; y3 = y1; x4 = x2 - 1; y4 = y2
                        x5 = x1; y5 = y1 - 2; x6 = x0; y6 = y0 - 1
                        if ((x3, y3) not in self.d) and ((x4, y4) not in self.d) and ((x5, y5) not in self.d) and ((x6, y6) not in self.d):
                            self.solve4(x, y, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, i)

    def calculate_metrics(self):
        """Calculate geometric and performance metrics of the generated polygon."""
        if len(self.data) < 3: return None
            
        poly = Polygon(self.data)
        execution_time = (self.end_time - self.start_time) * 1000  # ms
        num_vertices = len(self.data)
        area = poly.area
        perimeter = poly.length
        
        xs = [p[0] for p in self.data]
        ys = [p[1] for p in self.data]
        bbox_width = max(xs) - min(xs)
        bbox_height = max(ys) - min(ys)
        
        aspect_ratio = poly.length / poly.area if poly.area > 0 else 0
        
        convex_hull = poly.convex_hull
        convexity = area / convex_hull.area if convex_hull.area > 0 else 0
        
        reflex_count = self._count_reflex_vertices()
        horizontal_edges, vertical_edges = self._count_edge_orientations()
        
        complexity_ratio = self.iteration_count / num_vertices if num_vertices > 0 else 0
        per_vertex_memory_bytes = (self.peak_memory_kb * 1024) / num_vertices if num_vertices > 0 else 0
        
        return {
            "execution_time_ms": round(execution_time, 3),
            "num_vertices": num_vertices,
            "area": round(area, 2),
            "perimeter": round(perimeter, 2),
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "aspect_ratio": round(aspect_ratio, 2),
            "convexity_measure": round(convexity, 3),
            "reflex_vertices": reflex_count,
            "horizontal_edges": horizontal_edges,
            "vertical_edges": vertical_edges,
            "iteration_count": self.iteration_count,
            "successful_additions": self.successful_additions,
            "complexity_ratio": round(complexity_ratio, 2),
            "peak_memory_kb": round(self.peak_memory_kb, 2),
            "per_vertex_memory_bytes": round(per_vertex_memory_bytes, 2)
        }
        
    def _count_reflex_vertices(self):
        """Count the number of reflex vertices (270-degree internal angles)."""
        n = len(self.data)
        reflex_count = 0
        for i in range(n):
            prev = self.data[(i - 1) % n]
            curr = self.data[i]
            next_v = self.data[(i + 1) % n]
            
            v1 = (curr[0] - prev[0], curr[1] - prev[1])
            v2 = (next_v[0] - curr[0], next_v[1] - curr[1])
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            
            if cross < 0:
                reflex_count += 1
        return reflex_count
    
    def _count_edge_orientations(self):
        """Count the distribution of horizontal and vertical edges."""
        horizontal = vertical = 0
        n = len(self.data)
        for i in range(n):
            curr = self.data[i]
            next_v = self.data[(i + 1) % n]
            if curr[1] == next_v[1]: horizontal += 1
            elif curr[0] == next_v[0]: vertical += 1
        return horizontal, vertical
    
    def save_to_csv(self, filename="polygon_points.csv"):
        """Export polygon coordinates to CSV format."""
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            for (x, y) in self.data:
                writer.writerow([x, y])

# =================================================================
# 2. GRID RASTERIZATION & START/GOAL SAMPLING
# =================================================================
def polygon_to_grid(vertices, grid_size=None, padding=2):
    """
    Rasterize mathematical polygon into binary grid using scanline fill.
    Automatically scales grid resolution to prevent topological collapse (aliasing).
    """
    poly = Polygon(vertices)
    minx, miny, maxx, maxy = poly.bounds
    
    n_vertices = len(vertices)
    if grid_size is None:
        if n_vertices <= 100: grid_size = 128
        elif n_vertices <= 500: grid_size = 256
        else: grid_size = int(256 + (n_vertices / 100) * 50)  # Dynamic expansion for dense environments
        
    scale = min((grid_size - 2*padding)/(maxx - minx + 1e-8), 
                (grid_size - 2*padding)/(maxy - miny + 1e-8))
    
    # Render using PIL for fast and perfectly connected internal space (1=Wall, 0=Path)
    img = Image.new('L', (grid_size, grid_size), color=1)
    draw = ImageDraw.Draw(img)
    
    scaled_vertices = []
    for p in vertices:
        nx = (p[0] - minx) * scale + padding
        ny = (p[1] - miny) * scale + padding
        scaled_vertices.append((nx, ny))
        
    draw.polygon(scaled_vertices, fill=0)
    return np.array(img, dtype=np.uint8)

def sample_start_goal(grid):
    """
    Identify optimal Start/Goal nodes to maximize pathfinding difficulty 
    using a 2-sweep Breadth-First Search.
    """
    free_cells = np.argwhere(grid == 0)
    if len(free_cells) < 2:
        return [0, 0], [0, 0]

    def bfs_farthest(start_node):
        queue = deque([tuple(start_node)])
        visited = set([tuple(start_node)])
        farthest_node = tuple(start_node)
        max_dist = 0
        distances = {tuple(start_node): 0}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            curr = queue.popleft()
            curr_dist = distances[curr]

            if curr_dist > max_dist:
                max_dist = curr_dist
                farthest_node = curr

            for d in directions:
                nr, nc = curr[0] + d[0], curr[1] + d[1]
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if grid[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        distances[(nr, nc)] = curr_dist + 1
                        queue.append((nr, nc))

        return farthest_node, max_dist

    # Random starting seed
    A = free_cells[np.random.randint(len(free_cells))]
    # First sweep
    B, _ = bfs_farthest(A)
    # Second sweep
    C, _ = bfs_farthest(B)

    # Return coordinates in [x, y] format
    return [int(B[1]), int(B[0])], [int(C[1]), int(C[0])]

# =================================================================
# 3. MAIN PIPELINE (GENERATION + BENCHMARKING)
# =================================================================
def main_pipeline():
    """End-to-End data generation and benchmarking pipeline (Headless)."""
    BASE = "datasets"
    PATHS = {
        "poly": f"{BASE}/polygons",
        "maze": f"{BASE}/mazes",
        "meta": f"{BASE}/metadata"
    }
    for p in PATHS.values(): os.makedirs(p, exist_ok=True)

    vertex_counts = [50, 60]
    samples_per_n = 10

    f_ds = open(f"{PATHS['meta']}/dataset_summary.csv", 'w', newline='')
    f_pf = open(f"{PATHS['meta']}/pathfinding_summary.csv", 'w', newline='')
    writer_ds = csv.writer(f_ds)
    writer_pf = csv.writer(f_pf)
    
    writer_ds.writerow(["instance_id", "num_vertices", "poly_file", "maze_file", "start", "goal"])
    writer_pf.writerow(["maze_id", "algorithm", "success", "path_length", "time_sec"])

    print(">>> STARTING HEADLESS GENERATION & BENCHMARK PIPELINE...")

    algos = {
        "BFS": run_bfs, 
        "A_Star": run_astar,
        "Dijkstra": run_dijkstra,
        "PRM": run_prm,
        "RRT_Connect": run_rrt_connect,
        "RRT_Star": run_rrt_star
    }

    for n in vertex_counts:
        for i in range(1, samples_per_n + 1):
            inst_id = f"n{n}_{i:03}"
            print(f"\n--- Processing Instance: {inst_id} ---")

            # 1. Generate & Save Polygon
            gen = OrthogonalPolygonGenerator()
            vertices = gen.generate(n)
            poly_file = f"poly_{n}_{i:03}.csv"
            gen.save_to_csv(os.path.join(PATHS['poly'], poly_file))

            # 2. Rasterize & Save Maze Grid
            grid = polygon_to_grid(vertices)
            start, goal = sample_start_goal(grid)
            maze_file = f"maze_{n}_{i:03}.json"
            
            with open(os.path.join(PATHS['maze'], maze_file), 'w') as f:
                json.dump({
                    "grid": grid.tolist(), 
                    "start": start, 
                    "goal": goal, 
                    "size": grid.shape[0]
                }, f)

            # 3. Execute Benchmarks
            for name, func in algos.items():
                print(f"  > Benchmarking {name}...")
                
                # Fetch results (Path and Execution Time)
                result = func(grid, start, goal) 
                path = result[0]
                dt = result[-1]
                
                success = len(path) > 0
                plen = path_length(path)
                
                # Record metrics
                writer_pf.writerow([inst_id, name, success, round(plen, 2), round(dt, 5)])

            # Record dataset metadata
            writer_ds.writerow([inst_id, n, poly_file, maze_file, f"{start[0]},{start[1]}", f"{goal[0]},{goal[1]}"])

    f_ds.close()
    f_pf.close()
    print("\n>>> PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main_pipeline()
