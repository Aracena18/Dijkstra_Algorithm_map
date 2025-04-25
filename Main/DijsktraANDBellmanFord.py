import heapq
import time
import sys
import random
import matplotlib.pyplot as plt


def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    space_complexity = sys.getsizeof(distances) + sys.getsizeof(priority_queue)

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                space_complexity += sys.getsizeof(priority_queue)

    return distances, space_complexity


def bellman_ford(edges, vertices, start):
    distances = {vertex: float('inf') for vertex in vertices}
    distances[start] = 0
    space_complexity = sys.getsizeof(distances) + sys.getsizeof(edges)

    for _ in range(len(vertices) - 1):
        for u, v, weight in edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                space_complexity += sys.getsizeof(distances)

    for u, v, weight in edges:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            raise ValueError("Graph contains a negative-weight cycle")

    return distances, space_complexity


# Generate one shared graph as a list of edges
def generate_shared_graph(n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            weight = random.randint(1, 10)
            edges.append((str(i), str(j), weight))
            edges.append((str(j), str(i), weight))  # For undirected
    return edges


# Convert shared edges to adjacency list (for Dijkstra)
def convert_to_adjacency_list(edges, n):
    graph = {str(i): {} for i in range(n)}
    for u, v, w in edges:
        graph[u][v] = w
    return graph


def plot_performance(nodes_list, times_dijkstra, times_bellman):
    plt.plot(nodes_list, times_dijkstra, label='Dijkstra', marker='o')
    plt.plot(nodes_list, times_bellman, label='Bellman-Ford', marker='s')
    plt.title('Performance Comparison: Dijkstra vs Bellman-Ford')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    nodes_list = [5, 10, 15, 20, 25]
    times_dijkstra = []
    times_bellman = []

    for num_nodes in nodes_list:
        start_node = '0'
        vertices = [str(i) for i in range(num_nodes)]
        shared_edges = generate_shared_graph(num_nodes)

        graph_dijkstra = convert_to_adjacency_list(shared_edges, num_nodes)
        graph_bellman = shared_edges

        # Dijkstra Timing
        start_time = time.time()
        dijkstra(graph_dijkstra, start_node)
        times_dijkstra.append(time.time() - start_time)

        # Bellman-Ford Timing
        start_time = time.time()
        bellman_ford(graph_bellman, vertices, start_node)
        times_bellman.append(time.time() - start_time)

    # Print performance results
    print("--- Performance Data ---")
    for i in range(len(nodes_list)):
        print(f"{nodes_list[i]} nodes -> Dijkstra: {times_dijkstra[i]:.6f}s, Bellman-Ford: {times_bellman[i]:.6f}s")

    # Plot results
    plot_performance(nodes_list, times_dijkstra, times_bellman)


if __name__ == "__main__":
    main()
