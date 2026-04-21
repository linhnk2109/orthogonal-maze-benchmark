import os
import json
import time
import math
import csv
import numpy as np
import heapq
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree

# ==========================================
# 1. CÁC HÀM HỖ TRỢ DÙNG CHUNG (UTILS)
# ==========================================
def heuristic(p1, p2):
    """Khoảng cách Euclidean"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def path_length(path):
    """Tính tổng chiều dài của đường đi"""
    if not path or len(path) < 2: return 0.0
    return sum(heuristic(path[i], path[i+1]) for i in range(len(path)-1))

def is_collision_free(grid, p1, p2):
    """Kiểm tra xem đoạn thẳng nối p1, p2 có cắt chướng ngại vật không (Bresenham-style)"""
    x1, y1 = p1; x2, y2 = p2
    dist = heuristic(p1, p2)
    steps = max(int(dist * 2), 1) # Lấy mẫu mỗi nửa pixel
    for i in range(steps + 1):
        t = i / steps
        x, y = int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
        if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]): return False
        if grid[y, x] == 1: return False
    return True

# ==========================================
# 2. CÁC THUẬT TOÁN TÌM ĐƯỜNG
# ==========================================

def run_bfs(grid, start, goal):
    """1. Breadth-First Search (4-way grid)"""
    start_time = time.perf_counter()
    queue = deque([tuple(start)])
    visited = {tuple(start): None} # Node: Parent
    directions = [(0,1), (1,0), (0,-1), (-1,0)]
    
    path = []
    explored_edges = []
    
    while queue:
        curr = queue.popleft()
        if curr == tuple(goal):
            # Truy vết đường đi
            while curr is not None:
                path.append(curr)
                curr = visited[curr]
            path.reverse()
            break
            
        for dx, dy in directions:
            nx, ny = curr[0] + dx, curr[1] + dy
            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                if grid[ny, nx] == 0 and (nx, ny) not in visited:
                    visited[(nx, ny)] = curr
                    queue.append((nx, ny))
                    explored_edges.append((curr, (nx, ny)))

    return path, explored_edges, time.perf_counter() - start_time

def run_astar(grid, start, goal):
    """2. A* Search (8-way grid)"""
    start_time = time.perf_counter()
    open_set = []
    heapq.heappush(open_set, (0, tuple(start)))
    came_from = {}
    g_score = {tuple(start): 0}
    
    # 8 hướng di chuyển
    directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    path = []
    explored_edges = []
    
    while open_set:
        _, curr = heapq.heappop(open_set)
        
        if curr == tuple(goal):
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.append(tuple(start))
            path.reverse()
            break
            
        for dx, dy in directions:
            nx, ny = curr[0] + dx, curr[1] + dy
            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                # Chống đi chéo xuyên tường
                if grid[ny, nx] == 1: continue
                if dx != 0 and dy != 0: 
                    if grid[curr[1], nx] == 1 or grid[ny, curr[0]] == 1: continue
                
                tentative_g = g_score[curr] + math.hypot(dx, dy)
                neighbor = (nx, ny)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = curr
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    explored_edges.append((curr, neighbor))

    return path, explored_edges, time.perf_counter() - start_time


def run_dijkstra(grid, start, goal):
    """6. Dijkstra's Algorithm (8-way grid)"""
    t0 = time.perf_counter()
    open_set = [(0, tuple(start))]
    came_from = {}
    g_score = {tuple(start): 0}
    path = []
    explored_edges = []
    
    # 8 hướng di chuyển giống hệt A* để so sánh công bằng
    directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    while open_set:
        curr_cost, curr = heapq.heappop(open_set)
        
        # Nếu đã tìm thấy đường đi ngắn hơn trước khi pop ra thì bỏ qua
        if curr_cost > g_score.get(curr, float('inf')):
            continue
            
        if curr == tuple(goal):
            while curr in came_from: 
                path.append(curr)
                curr = came_from[curr]
            path.append(tuple(start))
            path.reverse()
            break
            
        for dx, dy in directions:
            nxt = (curr[0]+dx, curr[1]+dy)
            
            # Kiểm tra ranh giới và tường
            if 0 <= nxt[1] < grid.shape[0] and 0 <= nxt[0] < grid.shape[1] and grid[nxt[1], nxt[0]] == 0:
                # Chống đi chéo xuyên góc tường
                if dx != 0 and dy != 0: 
                    if grid[curr[1], nxt[0]] == 1 or grid[nxt[1], curr[0]] == 1: 
                        continue
                        
                cost = math.hypot(dx, dy)
                new_g = g_score[curr] + cost
                
                if nxt not in g_score or new_g < g_score[nxt]:
                    g_score[nxt] = new_g
                    came_from[nxt] = curr
                    
                    # Điểm khác biệt duy nhất với A*: Priority chỉ là g_score, không cộng thêm khoảng cách tới đích
                    heapq.heappush(open_set, (new_g, nxt))
                    
                    # Chỉ dùng để minh họa các cạnh khám phá (nếu cần)
                    explored_edges.append((curr, nxt))

    return path, explored_edges, time.perf_counter() - t0

def run_prm(grid, start, goal, num_samples=500, k_neighbors=10):
    """3. Probabilistic Roadmap (PRM)"""
    start_time = time.perf_counter()
    free_cells = np.argwhere(grid == 0)
    if len(free_cells) < num_samples: num_samples = len(free_cells)
    
    # Lấy mẫu
    samples = [tuple(start), tuple(goal)]
    idx = np.random.choice(len(free_cells), num_samples, replace=False)
    for i in idx:
        samples.append((free_cells[i][1], free_cells[i][0])) # (x, y)
        
    tree = cKDTree(samples)
    graph = {s: [] for s in samples}
    explored_edges = []
    
    # Xây dựng Roadmap
    for s in samples:
        distances, indices = tree.query(s, k=k_neighbors+1)
        for i in indices[1:]:
            neighbor = samples[i]
            if is_collision_free(grid, s, neighbor):
                graph[s].append(neighbor)
                graph[neighbor].append(s)
                explored_edges.append((s, neighbor))
                
    # Chạy Dijkstra trên Roadmap
    open_set = [(0, tuple(start))]
    came_from = {}
    g_score = {tuple(start): 0}
    path = []
    
    while open_set:
        _, curr = heapq.heappop(open_set)
        if curr == tuple(goal):
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.append(tuple(start))
            path.reverse()
            break
            
        for neighbor in graph[curr]:
            tentative_g = g_score[curr] + heuristic(curr, neighbor)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = curr
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g, neighbor))

    return path, explored_edges, time.perf_counter() - start_time





class RRTNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

# =================================================================
# 3. PROBABILISTIC ROADMAP (PRM) - OPTIMIZED
# =================================================================
def run_prm(grid, start, goal, num_samples=None, k_neighbors=15):
    """3. Probabilistic Roadmap (PRM) - Tự động mở rộng theo diện tích"""
    start_time = time.perf_counter()
    free_cells = np.argwhere(grid == 0)
    
    # TỐI ƯU: Lấy mẫu bằng 15% tổng số không gian trống (Tối đa 15000 mẫu để tránh tràn RAM)
    if num_samples is None:
        num_samples = min(int(len(free_cells) * 0.15), 15000)
        num_samples = max(num_samples, 1000) # Ít nhất 1000 mẫu cho bản đồ nhỏ
        
    # Lấy mẫu
    samples = [tuple(start), tuple(goal)]
    idx = np.random.choice(len(free_cells), num_samples, replace=False)
    for i in idx:
        samples.append((free_cells[i][1], free_cells[i][0])) # (x, y)
        
    tree = cKDTree(samples)
    graph = {s: [] for s in samples}
    explored_edges = []
    
    # Xây dựng Roadmap
    for s in samples:
        # TỐI ƯU: k_neighbors tăng lên 15 để đảm bảo kết nối qua các góc cua
        distances, indices = tree.query(s, k=k_neighbors+1)
        for i in indices[1:]:
            neighbor = samples[i]
            if is_collision_free(grid, s, neighbor):
                graph[s].append(neighbor)
                graph[neighbor].append(s)
                explored_edges.append((s, neighbor))
                
    # Chạy Dijkstra trên Roadmap
    open_set = [(0, tuple(start))]
    came_from = {}
    g_score = {tuple(start): 0}
    path = []
    
    while open_set:
        _, curr = heapq.heappop(open_set)
        if curr == tuple(goal):
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.append(tuple(start))
            path.reverse()
            break
            
        for neighbor in graph[curr]:
            tentative_g = g_score[curr] + heuristic(curr, neighbor)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = curr
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g, neighbor))

    return path, explored_edges, time.perf_counter() - start_time


# =================================================================
# 4. RRT-CONNECT (SMART SAMPLING) - CHỈ LẤY MẪU TRONG Ô TRỐNG
# =================================================================
def run_rrt_connect(grid, start, goal, step_size=4.0, max_iter=None):
    start_time = time.perf_counter()
    
    # 1. Gom toàn bộ không gian trống
    free_cells = np.argwhere(grid == 0)
    
    if max_iter is None:
        area = grid.shape[0] * grid.shape[1]
        max_iter = min(int(area * 0.5), 50000)
        max_iter = max(max_iter, 10000)
        
    tree_start = [RRTNode(start[0], start[1])]
    tree_goal = [RRTNode(goal[0], goal[1])]
    explored_edges = []
    path = []
    
    def get_random_point():
        # 2. Bốc ngẫu nhiên 1 ô trống thay vì random toàn lưới
        idx = np.random.randint(len(free_cells))
        return (free_cells[idx][1] + np.random.rand(), free_cells[idx][0] + np.random.rand())
        
    def get_nearest(tree, point):
        return min(tree, key=lambda n: heuristic((n.x, n.y), point))
        
    def steer(from_node, to_point):
        dist = heuristic((from_node.x, from_node.y), to_point)
        if dist <= step_size: return RRTNode(to_point[0], to_point[1])
        theta = math.atan2(to_point[1] - from_node.y, to_point[0] - from_node.x)
        return RRTNode(from_node.x + step_size * math.cos(theta), from_node.y + step_size * math.sin(theta))

    def extend(tree, point):
        nearest = get_nearest(tree, point)
        new_node = steer(nearest, point)
        if is_collision_free(grid, (nearest.x, nearest.y), (new_node.x, new_node.y)):
            new_node.parent = nearest
            tree.append(new_node)
            explored_edges.append(((nearest.x, nearest.y), (new_node.x, new_node.y)))
            return new_node
        return None

    for _ in range(max_iter):
        rand_p = get_random_point()
        new_s = extend(tree_start, rand_p)
        
        if new_s:
            new_g = extend(tree_goal, (new_s.x, new_s.y))
            if new_g and heuristic((new_s.x, new_s.y), (new_g.x, new_g.y)) <= step_size:
                # Connected!
                p1, p2 = [], []
                curr = new_s
                while curr: p1.append((curr.x, curr.y)); curr = curr.parent
                curr = new_g
                while curr: p2.append((curr.x, curr.y)); curr = curr.parent
                p1.reverse()
                path = p1 + p2
                break
                
        # Swap trees
        tree_start, tree_goal = tree_goal, tree_start

    # Đảm bảo đường đi chạy từ Start -> Goal
    if path and heuristic(path[0], start) > heuristic(path[-1], start):
        path.reverse()

    return path, explored_edges, time.perf_counter() - start_time

# =================================================================
# 5. RRT* (MICRO-STEPPING CHO MÊ CUNG HẸP)
# =================================================================
def run_rrt_star(grid, start, goal, step_size=2.0, search_radius=5.0, max_iter=None):
    """
    RRT* tinh chỉnh cho môi trường hẹp:
    - step_size giảm xuống 2.0 để luồn lách qua góc cua.
    - search_radius giảm xuống 5.0 để việc tối ưu (rewire) không cắt xéo qua tường.
    - goal_bias giảm xuống chỉ còn 5% để tránh đâm tường.
    """
    start_time = time.perf_counter()
    free_cells = np.argwhere(grid == 0)
    
    if max_iter is None:
        area = grid.shape[0] * grid.shape[1]
        # Bơm thêm "xăng": Tối đa 50,000 vòng lặp vì bước chân giờ đã ngắn đi một nửa
        max_iter = min(int(area * 0.8), 50000) 
        max_iter = max(max_iter, 10000)
        
    nodes = [RRTNode(start[0], start[1])]
    explored_edges = []
    path = []
    
    nodes_x = [start[0]]
    nodes_y = [start[1]]
    
    path_found = False
    smoothing_iters = 500  
    
    for _ in range(max_iter):
        # GIẢM GOAL BIAS: Chỉ 5% nhìn về đích
        if np.random.rand() < 0.05 and not path_found: 
            rand_p = tuple(goal)
        else:
            idx = np.random.randint(len(free_cells))
            rand_p = (free_cells[idx][1] + np.random.rand(), free_cells[idx][0] + np.random.rand())
            
        arr_x = np.array(nodes_x)
        arr_y = np.array(nodes_y)
        dists_sq = (arr_x - rand_p[0])**2 + (arr_y - rand_p[1])**2
        nearest_idx = np.argmin(dists_sq)
        nearest = nodes[nearest_idx]
        
        # Kiểm tra khoảng cách
        dist = math.sqrt(dists_sq[nearest_idx])
        if dist <= step_size:
            new_node = RRTNode(rand_p[0], rand_p[1])
        else:
            theta = math.atan2(rand_p[1] - nearest.y, rand_p[0] - nearest.x)
            new_node = RRTNode(nearest.x + step_size * math.cos(theta), nearest.y + step_size * math.sin(theta))
            
        if not is_collision_free(grid, (nearest.x, nearest.y), (new_node.x, new_node.y)):
            continue
            
        dists_to_new_sq = (arr_x - new_node.x)**2 + (arr_y - new_node.y)**2
        near_indices = np.where(dists_to_new_sq <= search_radius**2)[0]
        near_nodes = [nodes[i] for i in near_indices]
        
        new_node.parent = nearest
        new_node.cost = nearest.cost + math.sqrt(dists_to_new_sq[nearest_idx])
        
        for near, i in zip(near_nodes, near_indices):
            c = near.cost + math.sqrt(dists_to_new_sq[i])
            if c < new_node.cost and is_collision_free(grid, (near.x, near.y), (new_node.x, new_node.y)):
                new_node.parent = near
                new_node.cost = c
                
        nodes.append(new_node)
        nodes_x.append(new_node.x)
        nodes_y.append(new_node.y)
        explored_edges.append(((new_node.parent.x, new_node.parent.y), (new_node.x, new_node.y)))
        
        for near, i in zip(near_nodes, near_indices):
            c = new_node.cost + math.sqrt(dists_to_new_sq[i])
            if c < near.cost and is_collision_free(grid, (new_node.x, new_node.y), (near.x, near.y)):
                near.parent = new_node
                near.cost = c

        if not path_found:
            # Nới lỏng điều kiện chạm đích (cách đích 2 * step_size là coi như tới)
            if heuristic((new_node.x, new_node.y), goal) <= step_size * 2.0:
                if is_collision_free(grid, (new_node.x, new_node.y), goal):
                    path_found = True
        else:
            smoothing_iters -= 1
            if smoothing_iters <= 0:
                break

    goal_nodes = [n for n in nodes if heuristic((n.x, n.y), goal) <= step_size * 2.0]
    if goal_nodes:
        best_goal = min(goal_nodes, key=lambda n: n.cost)
        curr = best_goal
        while curr:
            path.append((curr.x, curr.y))
            curr = curr.parent
        path.reverse()
        path.append(tuple(goal))

    return path, explored_edges, time.perf_counter() - start_time
# ==========================================
# 3. HÀM VẼ VÀ LƯU ẢNH MINH HỌA
# ==========================================
def plot_algorithm_result(grid, start, goal, path, edges, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = ListedColormap(['white', '#34495e'])
    ax.imshow(grid, cmap=cmap, origin='upper')

    # Vẽ các cạnh đã khám phá (Exploration Graph/Tree)
    if edges:
        from matplotlib.collections import LineCollection
        # Giới hạn số cạnh vẽ để không bị giật lag nếu graph quá lớn
        if len(edges) > 5000: edges = edges[::2] 
        lc = LineCollection(edges, colors='#3498db', linewidths=0.5, alpha=0.4)
        ax.add_collection(lc)

    # Vẽ đường đi chính thức
    if path:
        px, py = zip(*path)
        ax.plot(px, py, color='#e74c3c', linewidth=3, label='Path')

    ax.scatter(start[0], start[1], c='#2ecc71', s=100, edgecolors='black', zorder=5, label='Start')
    ax.scatter(goal[0], goal[1], c='#f1c40f', s=150, marker='*', edgecolors='black', zorder=5, label='Goal')

    ax.set_title(title, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# 4. MODULE ĐÁNH GIÁ (BENCHMARK)
# ==========================================
def run_benchmark():
    base_dir = "dataset"
    mazes_dir = os.path.join(base_dir, "mazes")
    vis_dir = os.path.join(base_dir, "path_visuals")
    os.makedirs(vis_dir, exist_ok=True)
    
    if not os.path.exists(mazes_dir):
        print("[!] Không tìm thấy thư mục mazes. Vui lòng sinh dữ liệu trước.")
        return

    algorithms = {
        "BFS": run_bfs,
        "A_Star": run_astar,
        "PRM": run_prm,
        "RRT_Connect": run_rrt_connect,
        "RRT_Star": run_rrt_star
    }
    
    results = []

    print("="*80)
    print(f"{'Maze File':<15} | {'Algorithm':<15} | {'Success':<8} | {'Length':<10} | {'Time (s)':<10}")
    print("="*80)

    # Quét tất cả các file json
    for root, _, files in os.walk(mazes_dir):
        for file in sorted(files):
            if not file.endswith(".json"): continue
            
            filepath = os.path.join(root, file)
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            grid = np.array(data["grid"])
            start = data["start"]
            goal = data["goal"]
            
            # Thư mục lưu ảnh cho file json này
            file_basename = file.replace(".json", "")
            img_out_dir = os.path.join(vis_dir, file_basename)
            os.makedirs(img_out_dir, exist_ok=True)

            for algo_name, algo_func in algorithms.items():
                path, edges, elapsed_time = algo_func(grid, start, goal)
                
                success = len(path) > 0
                plength = path_length(path)
                
                results.append({
                    "Maze": file_basename,
                    "Algorithm": algo_name,
                    "Success": success,
                    "Path_Length": round(plength, 2),
                    "Time_sec": round(elapsed_time, 4)
                })
                
                print(f"{file_basename:<15} | {algo_name:<15} | {str(success):<8} | {plength:<10.2f} | {elapsed_time:<10.4f}")
                
                # Vẽ và lưu ảnh
                img_path = os.path.join(img_out_dir, f"{algo_name}.png")
                plot_title = f"{algo_name} | L: {plength:.1f} | T: {elapsed_time:.3f}s"
                plot_algorithm_result(grid, start, goal, path, edges, plot_title, img_path)

    # Lưu kết quả ra CSV
    csv_path = os.path.join(base_dir, "pathfinding_benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Maze", "Algorithm", "Success", "Path_Length", "Time_sec"])
        writer.writeheader()
        writer.writerows(results)
        
    print("="*80)
    print(f"[✓] Đã lưu kết quả Benchmark vào: {csv_path}")
    print(f"[✓] Đã xuất toàn bộ ảnh minh họa vào: {vis_dir}")

if __name__ == "__main__":
    run_benchmark()