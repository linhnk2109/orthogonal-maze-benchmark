import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import deque
from benchmark_pathplanning import (
    run_prm, 
    run_rrt_connect, 
    run_rrt_star,
    path_length 
)

def run_extended_stochastic_benchmark():
    base_dir = "dataset" 
    mazes_dir = os.path.join(base_dir, "mazes")
    
    rrt_img_dir = os.path.join(base_dir, "images", "rrt_star_optimized")
    os.makedirs(rrt_img_dir, exist_ok=True)
    
    stochastic_algos = {
        "PRM": run_prm,
        "RRT_Connect": run_rrt_connect,
        "RRT_Star": run_rrt_star  
    }
    
    NUM_TRIALS = 1
    results = []

    print("="*90)
    print(f"{'Maze File':<15} | {'Algorithm':<15} | {'Success':<8} | {'Avg Len':<10} | {'Avg Time(s)':<10}")
    print("="*90)

    def sort_by_n_and_i(filename):
        if not filename.endswith('.json'): return (float('inf'), float('inf'))
        try:
            parts = filename.replace(".json", "").split("_")
            return int(parts[1]), int(parts[2])
        except Exception:
            return (float('inf'), float('inf'))

    for root, _, files in os.walk(mazes_dir):
        for file in sorted(files, key=sort_by_n_and_i):
            if not file.endswith(".json"): continue
            
            filepath = os.path.join(root, file)
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            grid = np.array(data["grid"])
            start = data["start"]
            goal = data["goal"]
            file_basename = file.replace(".json", "")

            for algo_name, algo_func in stochastic_algos.items():
                success_count = 0
                total_len = 0.0
                total_time = 0.0
                best_path_to_draw = []
                
                for trial in range(NUM_TRIALS):
                    res = algo_func(grid, start, goal)
                    path = res[0]
                    elapsed_time = res[-1]
                    
                    if len(path) > 0:
                        success_count += 1
                        total_len += path_length(path)
                        best_path_to_draw = path
                    total_time += elapsed_time
                
                avg_success = success_count > 0
                avg_len = total_len / success_count if success_count > 0 else 0.0
                avg_time = total_time / NUM_TRIALS
                
                results.append({
                    "Maze": file_basename,
                    "Algorithm": algo_name,
                    "Success": avg_success,
                    "Path_Length": round(avg_len, 2),
                    "Time_sec": round(avg_time, 4)
                })
                
                print(f"{file_basename:<15} | {algo_name:<15} | {str(avg_success):<8} | {avg_len:<10.2f} | {avg_time:<10.4f}")

                if algo_name == "RRT_Star" and len(best_path_to_draw) > 0:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    cmap = ListedColormap(['white', '#34495e'])
                    ax.imshow(grid, cmap=cmap, origin='upper')
                    
                    px, py = zip(*best_path_to_draw)
                    ax.plot(px, py, color='#e74c3c', linewidth=2.5, label='Optimized Path')
                    
                    ax.scatter(start[0], start[1], c='#2ecc71', s=100, zorder=5)
                    ax.scatter(goal[0], goal[1], c='#f1c40f', s=150, marker='*', zorder=5)
                    
                    ax.set_title(f"Optimized RRT* | {file_basename}\nAvg Len: {avg_len:.1f} | Avg Time: {avg_time:.3f}s", fontweight='bold')
                    ax.axis('off')
                    
                    img_path = os.path.join(rrt_img_dir, f"{file_basename}_RRT_Star_Opt.png")
                    plt.tight_layout()
                    plt.savefig(img_path, dpi=200, bbox_inches='tight')
                    plt.close(fig)

    csv_path = os.path.join(base_dir, "metadata", "stochastic_extended_benchmark.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Maze", "Algorithm", "Success", "Path_Length", "Time_sec"])
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    run_extended_stochastic_benchmark()