import networkx as nx
import time
import json
import sys
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set
import heapq
from collections import deque

class GraphAlgorithms:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.results = {}
        
    def dijkstra(self, source: int) -> Dict[int, float]:
        """Dijkstra's algorithm for single source shortest path."""
        start_time = time.time()
        distances = {node: float('infinity') for node in self.graph.nodes()}
        distances[source] = 0
        pq = [(0, source)]
        visited = set()
        with open('dijkstra_trace.txt', 'w') as trace:
            trace.write(f"Insert: (0, {source})\n")
            while pq:
                current_distance, current_node = heapq.heappop(pq)
                trace.write(f"Pop: ({current_distance}, {current_node})\n")
                if current_node in visited:
                    continue
                visited.add(current_node)
                for neighbor in self.graph.neighbors(current_node):
                    weight = self.graph[current_node][neighbor].get('weight', 1)
                    distance = current_distance + weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(pq, (distance, neighbor))
                        trace.write(f"Insert: ({distance}, {neighbor})\n")
        
        self.results['dijkstra'] = {
            'distances': distances,
            'time': time.time() - start_time
        }
        with open('dijkstra_result.txt', 'w') as f:
            for node, dist in distances.items():
                f.write(f"{source} -> {node}: {dist}\n")
        return distances

    def bellman_ford(self, source: int) -> Dict[int, float]:
        """Bellman-Ford algorithm for single source shortest path."""
        start_time = time.time()
        distances = {node: float('infinity') for node in self.graph.nodes()}
        distances[source] = 0
        
        for _ in range(len(self.graph.nodes()) - 1):
            for u, v, data in self.graph.edges(data=True):
                weight = data.get('weight', 1)
                if distances[u] != float('infinity') and distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
        
        self.results['bellman_ford'] = {
            'distances': distances,
            'time': time.time() - start_time
        }
        with open('bellman_ford_result.txt', 'w') as f:
            for node, dist in distances.items():
                f.write(f"{source} -> {node}: {dist}\n")
        return distances

    def prim_mst(self) -> List[Tuple[int, int]]:
        start_time = time.time()
        mst = []
        visited = set()
        start_node = list(self.graph.nodes())[0]
        visited.add(start_node)
        while len(visited) < len(self.graph.nodes()):
            min_edge = None
            min_weight = float('infinity')
            for u in visited:
                for v in self.graph.neighbors(u):
                    if v not in visited:
                        weight = self.graph[u][v].get('weight', 1)
                        if weight < min_weight:
                            min_weight = weight
                            min_edge = (u, v)
            if min_edge:
                mst.append(min_edge)
                visited.add(min_edge[1])
            else:
                print("Warning: Graph is disconnected. Prim's MST covers only the connected component.")
                break
        self.results['prim_mst'] = {
            'mst': mst,
            'time': time.time() - start_time
        }
        with open('prim_mst_result.txt', 'w') as f:
            for u, v in mst:
                f.write(f"{u} - {v}\n")
        self.visualize_mst(mst, 'Prim MST')
        return mst

    def kruskal_mst(self) -> List[Tuple[int, int]]:
        """Kruskal's algorithm for Minimum Spanning Tree."""
        start_time = time.time()
        mst = []
        parent = {node: node for node in self.graph.nodes()}
        
        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]
        
        def union(u, v):
            parent[find(u)] = find(v)
        
        edges = sorted(self.graph.edges(data=True), 
                      key=lambda x: x[2].get('weight', 1))
        
        for u, v, data in edges:
            if find(u) != find(v):
                mst.append((u, v))
                union(u, v)
        
        self.results['kruskal_mst'] = {
            'mst': mst,
            'time': time.time() - start_time
        }
        with open('kruskal_mst_result.txt', 'w') as f:
            for u, v in mst:
                f.write(f"{u} - {v}\n")
        self.visualize_mst(mst, 'Kruskal MST')
        return mst

    def bfs(self, start: int) -> List[int]:
        """Breadth First Search traversal."""
        start_time = time.time()
        visited = set([start])
        queue = deque([start])
        traversal = []
        with open('bfs_trace.txt', 'w') as trace:
            trace.write(f"Insert: {start}\n")
            while queue:
                node = queue.popleft()
                trace.write(f"Pop: {node}\n")
                traversal.append(node)
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        trace.write(f"Insert: {neighbor}\n")
        
        self.results['bfs'] = {
            'traversal': traversal,
            'time': time.time() - start_time
        }
        with open('bfs_result.txt', 'w') as f:
            f.write(' -> '.join(map(str, traversal)))
        self.visualize_traversal(traversal, 'BFS Traversal')
        return traversal

    def dfs(self, start: int) -> List[int]:
        """Depth First Search traversal."""
        start_time = time.time()
        visited = set()
        traversal = []
        stack_trace = []
        
        def dfs_recursive(node):
            visited.add(node)
            traversal.append(node)
            stack_trace.append(f"Push: {node}")
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    dfs_recursive(neighbor)
            stack_trace.append(f"Pop: {node}")
        
        dfs_recursive(start)
        with open('dfs_trace.txt', 'w') as trace:
            for line in stack_trace:
                trace.write(line + '\n')
        
        self.results['dfs'] = {
            'traversal': traversal,
            'time': time.time() - start_time
        }
        with open('dfs_result.txt', 'w') as f:
            f.write(' -> '.join(map(str, traversal)))
        self.visualize_traversal(traversal, 'DFS Traversal')
        return traversal

    def graph_diameter(self) -> int:
        """Find the diameter of the graph."""
        start_time = time.time()
        diameter = 0
        
        for source in self.graph.nodes():
            distances = nx.single_source_shortest_path_length(self.graph, source)
            max_distance = max(distances.values())
            diameter = max(diameter, max_distance)
        
        self.results['diameter'] = {
            'diameter': diameter,
            'time': time.time() - start_time
        }
        with open('diameter_result.txt', 'w') as f:
            f.write(f"Diameter: {diameter}\n")
        return diameter

    def detect_cycle(self) -> bool:
        """Detect if the graph contains a cycle."""
        start_time = time.time()
        visited = set()
        rec_stack = set()
        
        def dfs_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    if dfs_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        has_cycle = any(dfs_cycle(node) for node in self.graph.nodes() 
                       if node not in visited)
        
        self.results['cycle_detection'] = {
            'has_cycle': has_cycle,
            'time': time.time() - start_time
        }
        with open('cycle_result.txt', 'w') as f:
            f.write(f"Cycle Detected: {has_cycle}\n")
        return has_cycle

    def average_degree(self):
        avg_deg = sum(dict(self.graph.degree()).values()) / len(self.graph)
        with open('average_degree.txt', 'w') as f:
            f.write(f"Average Degree: {avg_deg}\n")
        print(f"Average Degree: {avg_deg}")
        return avg_deg

    def save_results(self, filename: str = 'results.json'):
        """Save all results to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)

    def visualize_mst(self, mst_edges, title):
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, node_size=20, alpha=0.3, edge_color='gray')
        nx.draw_networkx_edges(self.graph, pos, edgelist=mst_edges, edge_color='r', width=2)
        plt.title(title)
        plt.savefig(f"{title.replace(' ', '_').lower()}.png")
        plt.close()

    def visualize_traversal(self, traversal, title):
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, node_size=20, alpha=0.3, edge_color='gray')
        path_edges = list(zip(traversal, traversal[1:]))
        nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color='b', width=2)
        plt.title(title)
        plt.savefig(f"{title.replace(' ', '_').lower()}.png")
        plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Graph Algorithms Project')
    parser.add_argument('--source', type=int, default=None, help='Source node for SSSP and traversals')
    parser.add_argument('--file', type=str, default='sample_graph.txt', help='Edge list file')
    args = parser.parse_args()
    try:
        graph = nx.read_edgelist(args.file, nodetype=int)
    except Exception as e:
        print(f"Error loading graph: {e}")
        sys.exit(1)
    if len(graph.nodes) < 1000:
        print(f"Warning: Graph has only {len(graph.nodes)} nodes. Requirement is >= 1000 nodes.")
    source_node = args.source if args.source is not None else list(graph.nodes())[0]
    if source_node not in graph:
        print(f"Error: Source node {source_node} not in graph.")
        sys.exit(1)
    algo = GraphAlgorithms(graph)
    print("Running Dijkstra...")
    algo.dijkstra(source_node)
    print("Running Bellman-Ford...")
    algo.bellman_ford(source_node)
    print("Running Prim's MST...")
    algo.prim_mst()
    print("Running Kruskal's MST...")
    algo.kruskal_mst()
    print("Running BFS...")
    algo.bfs(source_node)
    print("Running DFS...")
    algo.dfs(source_node)
    print("Calculating Diameter...")
    algo.graph_diameter()
    print("Detecting Cycle...")
    algo.detect_cycle()
    print("Calculating Average Degree...")
    algo.average_degree()
    algo.save_results()
    print("All results and traces saved. Visualizations generated for MSTs and traversals.")

if __name__ == "__main__":
    main()