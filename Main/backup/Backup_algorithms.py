import time
import heapq
import networkx as nx
from typing import Dict, List, Tuple, Set, Any


class PathAlgorithms:
    """
    Implements path-finding algorithms with detailed step tracking for educational purposes.
    """

    @staticmethod
    def dijkstra(G, source, target, weight='length') -> Tuple[Dict, Dict, List[Tuple]]:
        """
        Implementation of Dijkstra's algorithm with step-by-step tracking.

        Args:
            G: NetworkX graph
            source: Source node
            target: Target node
            weight: Edge weight attribute

        Returns:
            Tuple containing:
                - dist: Dictionary of shortest distances
                - prev: Dictionary of predecessors
                - steps: List of algorithm steps for visualization
        """
        # Initialize
        dist = {node: float('infinity') for node in G.nodes()}
        prev = {node: None for node in G.nodes()}
        dist[source] = 0

        # Priority queue
        pq = [(0, source)]
        visited = set()

        # Track steps for visualization
        steps = []

        while pq:
            # Get node with minimum distance
            current_dist, current = heapq.heappop(pq)

            # Record the step
            steps.append(("examine", current, current_dist))

            # Skip if we've processed this node already
            if current in visited:
                steps.append(("skip", current))
                continue

            # Visit the current node
            visited.add(current)
            steps.append(("visit", current))

            # If we reached the target
            if current == target:
                steps.append(("target_reached", current))
                break

            # Explore neighbors
            for neighbor in G.neighbors(current):
                if neighbor in visited:
                    continue

                # Calculate new distance
                edge_data = G.get_edge_data(current, neighbor)
                edge_weight = edge_data.get(weight, 1.0)
                new_dist = dist[current] + edge_weight

                # Record neighbor examination
                steps.append(("check_neighbor", current, neighbor, dist[neighbor], new_dist, edge_weight, dist[current]))

                # Update if shorter path found
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
                    steps.append(("update", neighbor, new_dist, current))

        return dist, prev, steps

    @staticmethod
    def bellman_ford(G, source, target, weight='length') -> Tuple[Dict, Dict, List[Tuple]]:
        """
        Implementation of Bellman-Ford algorithm with step-by-step tracking.

        Args:
            G: NetworkX graph
            source: Source node
            target: Target node
            weight: Edge weight attribute

        Returns:
            Tuple containing:
                - dist: Dictionary of shortest distances
                - prev: Dictionary of predecessors
                - steps: List of algorithm steps for visualization
        """
        # Initialize
        dist = {node: float('infinity') for node in G.nodes()}
        prev = {node: None for node in G.nodes()}
        dist[source] = 0

        # Track steps for visualization
        steps = []

        # Main algorithm
        num_nodes = len(G.nodes())
        for i in range(num_nodes - 1):
            steps.append(("iteration", i + 1))
            updated = False

            for u, v in G.edges():
                edge_data = G.get_edge_data(u, v)
                edge_weight = edge_data.get(weight, 1.0)

                steps.append(("check_edge", u, v, dist[u], dist[v], edge_weight))

                if dist[u] != float('infinity') and dist[u] + edge_weight < dist[v]:
                    dist[v] = dist[u] + edge_weight
                    prev[v] = u
                    updated = True
                    steps.append(("update", v, dist[v], u))

            if not updated:
                steps.append(("early_termination", i + 1))
                break

        # Check for negative cycles
        for u, v in G.edges():
            edge_data = G.get_edge_data(u, v)
            edge_weight = edge_data.get(weight, 1.0)

            if dist[u] != float('infinity') and dist[u] + edge_weight < dist[v]:
                steps.append(("negative_cycle", u, v))
                return None, None, steps

        return dist, prev, steps

    @staticmethod
    def reconstruct_path(prev: Dict, source: Any, target: Any) -> List:
        """
        Reconstruct the path from source to target using the predecessor dictionary.

        Args:
            prev: Dictionary of predecessors
            source: Source node
            target: Target node

        Returns:
            List of nodes representing the path from source to target
        """
        if prev.get(target) is None and source != target:
            return []  # No path exists

        path = []
        current = target

        while current is not None:
            path.append(current)
            current = prev.get(current)

        return list(reversed(path))

    @staticmethod
    def format_steps_explanation(G, steps: List[Tuple], prev: Dict, source: Any, target: Any) -> str:
        """
        Format algorithm steps into human-readable explanation.

        Args:
            G: NetworkX graph
            steps: List of algorithm steps
            prev: Dictionary of predecessors
            source: Source node
            target: Target node

        Returns:
            Formatted explanation string
        """
        explanation = f"Finding shortest path from Node {source} to Node {target}\n"
        explanation += "=" * 60 + "\n\n"

        # Keep track of current algorithm state for better explanations
        current_distances = {}
        current_predecessors = {}
        visited_nodes = set()

        for step in steps:
            step_type = step[0]

            if step_type == "examine":
                node, current_dist = step[1], step[2]
                explanation += f"📌 Examining Node {node} (current shortest distance: {current_dist:.2f})\n"

            elif step_type == "visit":
                node = step[1]
                explanation += f"✅ Marking Node {node} as visited\n"
                visited_nodes.add(node)
                explanation += f"   Visited nodes so far: {', '.join([f'Node {n}' for n in sorted(visited_nodes)])}\n"

            elif step_type == "skip":
                node = step[1]
                explanation += f"⏭️ Skipping already visited Node {node}\n"

            elif step_type == "check_neighbor":
                current, neighbor, old_dist, new_dist, edge_weight, current_node_dist = step[1], step[2], step[3], step[4], step[5], step[6]
                if old_dist == float('infinity'):
                    old_dist_str = "∞"
                else:
                    old_dist_str = f"{old_dist:.2f}"

                explanation += f"  👉 Analyzing edge: Node {current} → Node {neighbor} (weight: {edge_weight:.2f})\n"
                explanation += f"     • Current distance to Node {neighbor}: {old_dist_str}\n"
                explanation += f"     • Potential new distance via Node {current}: {new_dist:.2f}\n"
                explanation += f"       (= distance to Node {current}: {current_node_dist:.2f} + edge weight: {edge_weight:.2f})\n"

            elif step_type == "update":
                node, dist, prev_node = step[1], step[2], step[3]
                current_distances[node] = dist
                current_predecessors[node] = prev_node

                explanation += f"  ✨ IMPROVED PATH FOUND to Node {node}:\n"
                explanation += f"     • Updated distance: {dist:.2f}\n"
                explanation += f"     • New best predecessor: Node {prev_node}\n"

                # Show the current best path to this node
                path_to_node = PathAlgorithms.reconstruct_path(current_predecessors, source, node)
                path_str = " → ".join([f"Node {n}" for n in path_to_node])
                explanation += f"     • Current best path: {path_str}\n"

            elif step_type == "target_reached":
                node = step[1]
                explanation += f"\n🎯 TARGET REACHED! Node {node} has been processed.\n"
                explanation += f"   We can terminate early as we now have the optimal path from {source} to {target}.\n"

            elif step_type == "iteration":
                iteration = step[1]
                explanation += f"\n==== ITERATION {iteration} ====\n"

            elif step_type == "check_edge":
                u, v, dist_u, dist_v, weight = step[1], step[2], step[3], step[4], step[5]
                dist_u_str = f"{dist_u:.2f}" if dist_u != float('infinity') else "∞"
                dist_v_str = f"{dist_v:.2f}" if dist_v != float('infinity') else "∞"

                explanation += f"  👉 Checking edge: Node {u} → Node {v} (weight: {weight:.2f})\n"
                explanation += f"     • Current distance to Node {u}: {dist_u_str}\n"
                explanation += f"     • Current distance to Node {v}: {dist_v_str}\n"
                if dist_u != float('infinity'):
                    explanation += f"     • Potential new distance to Node {v}: {dist_u + weight:.2f}\n"

            elif step_type == "early_termination":
                iteration = step[1]
                explanation += f"\n🛑 No updates occurred in iteration {iteration}\n"
                explanation += "    Algorithm terminating early (converged to optimal solution)\n"

            elif step_type == "negative_cycle":
                u, v = step[1], step[2]
                explanation += f"\n⚠️ NEGATIVE CYCLE DETECTED involving edge Node {u} → Node {v}\n"
                explanation += "    The shortest path is undefined as it can be made arbitrarily small.\n"

        # Construct final path
        path = PathAlgorithms.reconstruct_path(prev, source, target)

        explanation += "\n" + "=" * 60 + "\n"
        if path:
            explanation += "🏁 FINAL RESULT:\n"
            path_str = " → ".join([f"Node {node}" for node in path])
            explanation += f"Shortest path: {path_str}\n"

            # Calculate total distance and show individual edge contributions
            total_dist = 0
            explanation += "\nPath breakdown:\n"
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_data = G.get_edge_data(u, v)
                weight = edge_data.get('length', 1.0)
                total_dist += weight
                explanation += f"  • Node {u} → Node {v}: {weight:.2f}\n"

            explanation += f"\nTotal distance: {total_dist:.2f}\n"
        else:
            explanation += "❌ NO PATH FOUND! The target is not reachable from the source.\n"

        return explanation