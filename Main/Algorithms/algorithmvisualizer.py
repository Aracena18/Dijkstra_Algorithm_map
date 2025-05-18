import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import numpy as np
from tkinter import Toplevel
import time
import threading
import gc
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import colorsys


class AlgorithmVisualizer:
    """Enhanced visualizer for comparing Dijkstra's and Bellman-Ford algorithms."""

    def __init__(self, parent, theme_var, theme_colors):
        """Initialize the visualizer with improved memory management."""
        self.parent = parent
        self.theme_var = theme_var
        self.theme_colors = theme_colors
        self.comparison_window = None
        self._is_animating = False

        # Constants for performance tuning - removed MAX_STEPS limitation
        self.MAX_NODES = 100
        self.MAX_EDGES = 300
        # self.MAX_STEPS = 500  # This line is removed to lift the limitation
        self.ANIMATION_FRAME_LIMIT = 0.05  # 50ms minimum between frame updates

        # Enhanced color schemes
        self.color_schemes = {
            "dark": {
                "bg": "#1E1E1E",
                "node": "#4D4D4D",
                "visited": "#3584e4",
                "examining": "#f9c440",
                "path": "#33d17a",
                "source": "#2ec27e",
                "target": "#e01b24",
                "text": "#ffffff"
            },
            "light": {
                "bg": "#F5F5F5",
                "node": "#D1D1D1",
                "visited": "#3584e4",
                "examining": "#e5a50a",
                "path": "#26a269",
                "source": "#2ec27e",
                "target": "#e01b24",
                "text": "#000000"
            }
        }

        self._cleanup_resources()

    def _cleanup_resources(self):
        """Improved resource cleanup to prevent memory leaks."""
        self.G = None
        self.source = None
        self.target = None
        self.pos = None

        # Algorithm data
        self.dijkstra_dist = {}
        self.dijkstra_path = []
        self.dijkstra_steps = []
        self.dijkstra_time = 0
        self.bellman_dist = {}
        self.bellman_path = []
        self.bellman_steps = []
        self.bellman_time = 0

        # Animation state
        self.dijkstra_step = 0
        self.bellman_step = 0
        self.dijkstra_visited = set()
        self.bellman_visited = set()
        self.dijkstra_distances = {}
        self.bellman_distances = {}
        self.current_dijkstra_node = None
        self.current_bellman_edge = None
        self.current_dijkstra_edge = None

        # Clear matplotlib figures
        for attr in ['ani_fig', 'ani_canvas']:
            if hasattr(self, attr):
                obj = getattr(self, attr)
                if attr == 'ani_fig' and obj:
                    plt.close(obj)
                delattr(self, attr)

        # Force garbage collection
        gc.collect()

    def show_comparison(self, G, dijkstra_results, bellman_ford_results, source_pos, target_pos):
        """Show algorithm comparison with enhanced visualization."""
        self._cleanup_resources()

        # Optimize graph for visualization
        if len(G.nodes()) > self.MAX_NODES:
            self.G = self._create_visualization_subgraph(G, dijkstra_results, bellman_ford_results)
        else:
            self.G = G

        # Extract data from results
        self._extract_algorithm_data(dijkstra_results, bellman_ford_results)

        # Create visualization window
        self._create_visualization_window()

    def _extract_algorithm_data(self, dijkstra_results, bellman_ford_results):
        """Extract and prepare algorithm data for visualization without limiting steps."""
        # Dijkstra data
        self.source = dijkstra_results.get('source')
        self.target = dijkstra_results.get('target')
        self.dijkstra_dist = dijkstra_results.get('dist', {})
        self.dijkstra_path = dijkstra_results.get('path', [])

        # Store full step data without limiting
        self.dijkstra_steps = dijkstra_results.get('steps', [])
        self.dijkstra_total_steps = len(self.dijkstra_steps)
        self.dijkstra_time = dijkstra_results.get('execution_time', 0)

        # Bellman-Ford data
        self.bellman_dist = bellman_ford_results.get('dist', {})
        self.bellman_path = bellman_ford_results.get('path', [])
        self.bellman_steps = bellman_ford_results.get('steps', [])
        self.bellman_total_steps = len(self.bellman_steps)
        self.bellman_time = bellman_ford_results.get('execution_time', 0)

        # Initialize animation state
        self.dijkstra_distances = {node: float('infinity') for node in self.G.nodes()}
        self.bellman_distances = {node: float('infinity') for node in self.G.nodes()}
        if self.source:
            self.dijkstra_distances[self.source] = 0
            self.bellman_distances[self.source] = 0

    def _create_visualization_subgraph(self, G, dijkstra_results, bellman_ford_results):
        """Create optimized subgraph for visualization with better algorithms."""
        important_nodes = set()
        source = dijkstra_results.get('source')
        target = dijkstra_results.get('target')

        # Add source, target, and paths to important nodes
        if source: important_nodes.add(source)
        if target: important_nodes.add(target)
        important_nodes.update(dijkstra_results.get('path', []))
        important_nodes.update(bellman_ford_results.get('path', []))

        # Add intermediate nodes based on algorithm steps
        for algo_steps in [dijkstra_results.get('steps', []), bellman_ford_results.get('steps', [])]:
            for step in algo_steps[:min(100, len(algo_steps))]:  # Limit to first 100 steps
                if len(step) > 1 and isinstance(step[1], (int, str)):
                    important_nodes.add(step[1])
                if len(step) > 2 and isinstance(step[2], (int, str)):
                    important_nodes.add(step[2])

        # Calculate max neighbors to include while staying under node limit
        remaining = self.MAX_NODES - len(important_nodes)
        if remaining > 0 and important_nodes:
            neighbors_per_node = max(1, remaining // len(important_nodes))
            neighbors = set()
            for node in important_nodes:
                if node in G:
                    neighbors.update(list(G.neighbors(node))[:neighbors_per_node])

            # Add neighbors up to the limit
            viz_nodes = list(important_nodes)
            for n in neighbors:
                if len(viz_nodes) < self.MAX_NODES:
                    viz_nodes.append(n)
                else:
                    break
        else:
            viz_nodes = list(important_nodes)

        # Create and return subgraph
        return G.subgraph(viz_nodes).copy()

    def _create_visualization_window(self):
        """Create the enhanced comparison visualization window."""
        if self.comparison_window:
            self.comparison_window.destroy()

        # Force garbage collection
        gc.collect()

        # Get current theme
        current_theme = self.theme_var.get()
        theme_config = self.theme_colors[current_theme]
        color_scheme = self.color_schemes["dark" if "dark" in current_theme.lower() else "light"]

        # Create window with improved styling
        self.comparison_window = Toplevel(self.parent)
        self.comparison_window.title("Algorithm Comparison: Dijkstra vs Bellman-Ford")
        self.comparison_window.geometry("1100x800")
        self.comparison_window.configure(bg=theme_config["fg"])
        self.comparison_window.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # Create main frame with improved styling
        main_frame = ctk.CTkFrame(self.comparison_window, fg_color=theme_config["fg"])
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)

        # Enhanced header with algorithm information
        header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 15))

        header = ctk.CTkLabel(
            header_frame,
            text="Algorithm Comparison: Dijkstra vs Bellman-Ford",
            font=("Helvetica", 20, "bold")
        )
        header.pack(pady=(0, 5))

        # Additional information about the algorithms
        algo_info = ctk.CTkLabel(
            header_frame,
            text="Comparing single-source shortest path algorithms: "
                 f"Dijkstra's (O((V+E)logV) time) and Bellman-Ford (O(VE) time)",
            font=("Helvetica", 12)
        )
        algo_info.pack(pady=(0, 5))

        # Tab view for different visualizations with improved styling
        tab_view = ctk.CTkTabview(main_frame)
        tab_view.pack(fill="both", expand=True)

        # Create tabs
        graph_tab = tab_view.add("Graph Comparison")
        stats_tab = tab_view.add("Performance Metrics")
        step_tab = tab_view.add("Animation")

        # Add warning for large graphs if needed
        if len(self.G.nodes()) > 50:
            warning_text = f"⚠️ Large graph detected ({len(self.G.nodes())} nodes, {len(self.G.edges())} edges). Visualization optimized for performance."
            warning_label = ctk.CTkLabel(
                main_frame,
                text=warning_text,
                font=("Helvetica", 12),
                text_color="#ff9900"
            )
            warning_label.pack(pady=(0, 10))

        # Populate tabs in background thread for responsiveness
        threading.Thread(target=self._populate_tabs,
                         args=(graph_tab, stats_tab, step_tab, color_scheme),
                         daemon=True).start()

    def _on_window_close(self):
        """Handle window close with thorough cleanup."""
        self._is_animating = False
        self._cleanup_resources()
        if self.comparison_window:
            self.comparison_window.destroy()
            self.comparison_window = None

    def _populate_tabs(self, graph_tab, stats_tab, step_tab, color_scheme):
        """Populate tabs with staggered timing to improve responsiveness."""
        self.comparison_window.after(0, lambda: self._create_graph_visualization(graph_tab, color_scheme))
        self.comparison_window.after(100, lambda: self._create_performance_stats(stats_tab, color_scheme))
        self.comparison_window.after(200, lambda: self._create_step_animation(step_tab, color_scheme))

    def _create_graph_visualization(self, parent, color_scheme):
        """Create enhanced graph visualization with better aesthetics."""
        try:
            # Create layout for node positions
            if not self.pos:
                self.pos = nx.spring_layout(self.G, seed=42)

            # Create figure with improved style
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor=color_scheme["bg"])
            fig.suptitle("Path Comparison", fontsize=16, color=color_scheme["text"])

            # Set titles with execution times
            ax1.set_title(f"Dijkstra's Algorithm\nTime: {self.dijkstra_time * 1000:.2f}ms",
                          color=color_scheme["text"])
            ax2.set_title(f"Bellman-Ford Algorithm\nTime: {self.bellman_time * 1000:.2f}ms",
                          color=color_scheme["text"])

            # Set axes background color
            ax1.set_facecolor(color_scheme["bg"])
            ax2.set_facecolor(color_scheme["bg"])

            # Draw optimized graph visualization
            self._draw_algorithm_graphs(ax1, ax2, color_scheme)

            # Improve layout
            plt.tight_layout()

            # Create canvas with draggable support
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill="both", expand=True, padx=15, pady=15)

            # Add legend if not too many nodes
            if len(self.G.nodes()) <= 50:
                self._add_graph_legend(fig, color_scheme)

        except Exception as e:
            self._show_error(parent, f"Error in graph visualization: {str(e)}")

    def _draw_algorithm_graphs(self, ax1, ax2, color_scheme):
        """Draw algorithm graphs with optimized rendering."""
        # Define edge subset if needed
        if len(self.G.edges()) > self.MAX_EDGES:
            # Prioritize path edges
            path_edges_dijkstra = [(self.dijkstra_path[i], self.dijkstra_path[i + 1])
                                   for i in range(len(self.dijkstra_path) - 1) if i + 1 < len(self.dijkstra_path)]
            path_edges_bellman = [(self.bellman_path[i], self.bellman_path[i + 1])
                                  for i in range(len(self.bellman_path) - 1) if i + 1 < len(self.bellman_path)]

            # Get remaining edges up to limit
            other_edges = [e for e in self.G.edges()
                           if e not in path_edges_dijkstra and e not in path_edges_bellman]
            import random
            random.shuffle(other_edges)
            remaining = self.MAX_EDGES - len(path_edges_dijkstra) - len(path_edges_bellman)
            display_edges = list(
                set(path_edges_dijkstra).union(set(path_edges_bellman)).union(set(other_edges[:remaining])))
        else:
            display_edges = list(self.G.edges())

        # Draw edges with weight-based styling
        edge_alphas = self._calculate_edge_alphas(display_edges)

        # Draw edges (optimized)
        nx.draw_networkx_edges(self.G, self.pos, ax=ax1, edgelist=display_edges,
                               alpha=0.3, edge_color="gray", width=edge_alphas)
        nx.draw_networkx_edges(self.G, self.pos, ax=ax2, edgelist=display_edges,
                               alpha=0.3, edge_color="gray", width=edge_alphas)

        # Draw nodes with enhanced styling
        node_colors = [color_scheme["source"] if n == self.source else
                       color_scheme["target"] if n == self.target else
                       color_scheme["node"] for n in self.G.nodes()]

        nx.draw_networkx_nodes(self.G, self.pos, ax=ax1, node_size=200,
                               node_color=node_colors, edgecolors='black', linewidths=0.5)
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax2, node_size=200,
                               node_color=node_colors, edgecolors='black', linewidths=0.5)

        # Draw labels only if reasonable node count
        if len(self.G.nodes()) <= 50:
            nx.draw_networkx_labels(self.G, self.pos, ax=ax1, font_size=8,
                                    font_color=color_scheme["text"])
            nx.draw_networkx_labels(self.G, self.pos, ax=ax2, font_size=8,
                                    font_color=color_scheme["text"])

            # Draw edge weights for small graphs
            if len(self.G.edges()) <= 50:
                edge_labels = {(u, v): f"{d.get('length', 1.0):.1f}"
                               for u, v, d in self.G.edges(data=True)}
                nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels,
                                             ax=ax1, font_size=7, font_color=color_scheme["text"])
                nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels,
                                             ax=ax2, font_size=7, font_color=color_scheme["text"])

        # Draw algorithm paths with enhanced styling
        if self.dijkstra_path and len(self.dijkstra_path) > 1:
            path_edges = [(self.dijkstra_path[i], self.dijkstra_path[i + 1])
                          for i in range(len(self.dijkstra_path) - 1)]
            nx.draw_networkx_edges(self.G, self.pos, ax=ax1, edgelist=path_edges,
                                   width=3, edge_color=color_scheme["path"], style='solid',
                                   arrows=True, arrowsize=15, arrowstyle='->')

        if self.bellman_path and len(self.bellman_path) > 1:
            path_edges = [(self.bellman_path[i], self.bellman_path[i + 1])
                          for i in range(len(self.bellman_path) - 1)]
            nx.draw_networkx_edges(self.G, self.pos, ax=ax2, edgelist=path_edges,
                                   width=3, edge_color=color_scheme["path"], style='solid',
                                   arrows=True, arrowsize=15, arrowstyle='->')

    def _calculate_edge_alphas(self, edges):
        """Calculate edge widths based on weights for better visualization."""
        # Get edge weights
        weights = []
        for u, v in edges:
            weight = self.G[u][v].get('length', 1.0)
            weights.append(weight)

        if not weights:
            return [1.0] * len(edges)

        # Normalize to range 0.5-2.5
        min_w, max_w = min(weights), max(weights)
        if min_w == max_w:
            return [1.0] * len(edges)

        widths = [0.5 + 2.0 * (1 - (w - min_w) / (max_w - min_w)) for w in weights]
        return widths

    def _add_graph_legend(self, fig, color_scheme):
        """Add a legend to the graph visualization."""
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_scheme["source"], edgecolor='black', label='Source'),
            Patch(facecolor=color_scheme["target"], edgecolor='black', label='Target'),
            Patch(facecolor=color_scheme["node"], edgecolor='black', label='Node'),
            Patch(facecolor=color_scheme["path"], edgecolor='black', label='Shortest Path')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, frameon=True,
                   facecolor=color_scheme["bg"], edgecolor='black')

    def _create_performance_stats(self, parent, color_scheme):
        """Create enhanced performance statistics visualization."""
        try:
            # Create main frame
            stats_frame = ctk.CTkFrame(parent, fg_color="transparent")
            stats_frame.pack(fill="both", expand=True, padx=15, pady=15)

            # Create two columns layout
            left_frame = ctk.CTkFrame(stats_frame)
            left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

            right_frame = ctk.CTkFrame(stats_frame)
            right_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))

            # Add metrics table to left frame with improved styling
            metrics_frame = ctk.CTkFrame(left_frame)
            metrics_frame.pack(fill="both", expand=True, padx=10, pady=10)

            # Create a header
            header = ctk.CTkLabel(
                metrics_frame,
                text="Algorithm Performance Metrics",
                font=("Helvetica", 16, "bold")
            )
            header.pack(pady=(10, 15))

            # Create improved metrics table
            self._create_metrics_table(metrics_frame)

            # Add visualization of performance to right frame
            chart_frame = ctk.CTkFrame(right_frame)
            chart_frame.pack(fill="both", expand=True, padx=10, pady=10)

            # Create performance charts with improved styling
            self._create_performance_charts(chart_frame, color_scheme)

        except Exception as e:
            self._show_error(parent, f"Error in performance stats: {str(e)}")

    def _create_metrics_table(self, parent):
        """Create an enhanced metrics comparison table with proper text coloring."""
        # Get current theme for proper text coloring
        current_theme = self.theme_var.get()
        color_scheme = self.color_schemes["dark" if "dark" in current_theme.lower() else "light"]
        text_color = color_scheme["text"]  # Define text_color from color scheme

        headers = ["Metric", "Dijkstra's Algorithm", "Bellman-Ford Algorithm", "Difference"]

        # Create table frame
        table_frame = ctk.CTkFrame(parent)
        table_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Add header row with improved styling and explicit text color
        for i, header_text in enumerate(headers):
            header_label = ctk.CTkLabel(
                table_frame,
                text=header_text,
                font=("Helvetica", 14, "bold"),
                text_color=text_color  # Explicitly set text color
            )
            header_label.grid(row=0, column=i, padx=10, pady=(0, 10), sticky="w")

        # Calculate differences with proper handling of the actual total steps
        time_diff = self.dijkstra_time - self.bellman_time
        time_diff_pct = (time_diff / self.bellman_time * 100) if self.bellman_time else 0
        time_faster = "faster" if time_diff < 0 else "slower"

        path_diff = (len(self.dijkstra_path) - len(
            self.bellman_path)) if self.dijkstra_path and self.bellman_path else "N/A"
        steps_diff = self.dijkstra_total_steps - self.bellman_total_steps

        # Enhanced metrics with comparison and explicit text coloring
        metrics = [
            ["Execution Time",
             f"{self.dijkstra_time * 1000:.2f} ms",
             f"{self.bellman_time * 1000:.2f} ms",
             f"{abs(time_diff * 1000):.2f} ms {time_faster} ({abs(time_diff_pct):.1f}%)"],

            ["Path Length",
             f"{len(self.dijkstra_path) - 1 if self.dijkstra_path else 'N/A'}",
             f"{len(self.bellman_path) - 1 if self.bellman_path else 'N/A'}",
             f"{path_diff if isinstance(path_diff, str) else path_diff - 1}"],

            ["Steps Analyzed",
             f"{self.dijkstra_total_steps}",
             f"{self.bellman_total_steps}",
             f"{steps_diff} steps difference"],

            ["Time Complexity",
             "O((V + E) log V)",
             "O(V·E)",
             "Dijkstra better for sparse graphs"],

            ["Space Complexity",
             "O(V)",
             "O(V)",
             "Equivalent"],

            ["Handles Negative Weights",
             "No",
             "Yes",
             "Bellman-Ford more flexible"],
        ]

        # Add metrics rows with proper text coloring
        for i, (metric, dijkstra_val, bellman_val, diff_val) in enumerate(metrics):
            row_bg = "#303030" if i % 2 == 0 else "#252525"  # Dark theme alternating rows

            row_frame = ctk.CTkFrame(table_frame, fg_color=row_bg)
            row_frame.grid(row=i + 1, column=0, columnspan=4, sticky="ew", pady=2)

            # Distribute columns within the row with explicit text color
            for j, val in enumerate([metric, dijkstra_val, bellman_val, diff_val]):
                col_width = 20 if j == 0 else 25 if j == 3 else 27
                cell = ctk.CTkLabel(
                    row_frame,
                    text=val,
                    font=("Helvetica", 12),
                    width=col_width,
                    text_color=text_color  # Explicitly set text color
                )
                cell.grid(row=0, column=j, padx=10, pady=5, sticky="w")

            # Configure row to expand properly
            for j in range(4):
                row_frame.grid_columnconfigure(j, weight=1)

    def _create_performance_charts(self, parent, color_scheme):
        """Create enhanced performance visualization charts."""
        # Create figure for performance charts
        fig = plt.figure(figsize=(8, 6), facecolor=color_scheme["bg"])

        # Time comparison bar chart (enhanced)
        ax1 = fig.add_subplot(211)
        ax1.set_facecolor(color_scheme["bg"])
        ax1.set_title("Execution Time Comparison", color=color_scheme["text"])
        ax1.tick_params(colors=color_scheme["text"])

        algorithms = ["Dijkstra's", "Bellman-Ford"]
        times = [self.dijkstra_time * 1000, self.bellman_time * 1000]

        # Create bars with gradients for better visualization
        bars = ax1.bar(algorithms, times, color=[color_scheme["source"], color_scheme["target"]])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{height:.2f} ms', ha='center', va='bottom',
                     color=color_scheme["text"], fontsize=10)

        ax1.set_ylabel('Time (ms)', color=color_scheme["text"])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(color_scheme["text"])
        ax1.spines['left'].set_color(color_scheme["text"])

        # Steps comparison chart (new)
        ax2 = fig.add_subplot(212)
        ax2.set_facecolor(color_scheme["bg"])
        ax2.set_title("Algorithm Steps Analysis", color=color_scheme["text"])
        ax2.tick_params(colors=color_scheme["text"])

        # Create more informative visualization comparing steps
        categories = ['Total Steps', 'Path Length']
        dijkstra_values = [len(self.dijkstra_steps), len(self.dijkstra_path) if self.dijkstra_path else 0]
        bellman_values = [len(self.bellman_steps), len(self.bellman_path) if self.bellman_path else 0]

        x = np.arange(len(categories))
        width = 0.35

        ax2.bar(x - width / 2, dijkstra_values, width, label="Dijkstra's",
                color=color_scheme["source"], edgecolor='black', linewidth=0.5)
        ax2.bar(x + width / 2, bellman_values, width, label='Bellman-Ford',
                color=color_scheme["target"], edgecolor='black', linewidth=0.5)

        # Add labels and formatting
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, color=color_scheme["text"])
        ax2.set_ylabel('Count', color=color_scheme["text"])
        ax2.legend(facecolor=color_scheme["bg"], edgecolor='black', labelcolor=color_scheme["text"])

        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_color(color_scheme["text"])
        ax2.spines['left'].set_color(color_scheme["text"])

        plt.tight_layout()

        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _create_step_animation(self, parent, color_scheme):
        """Create enhanced step-by-step animation with improved controls."""
        try:
            # Create main animation frame
            animation_container = ctk.CTkFrame(parent)
            animation_container.pack(fill="both", expand=True, padx=15, pady=15)

            # Enhanced controls frame with better layout
            control_frame = ctk.CTkFrame(animation_container)
            control_frame.pack(fill="x", pady=(0, 10))

            # Add better playback controls
            self._create_animation_controls(control_frame)

            # Add algorithm description
            description_frame = ctk.CTkFrame(animation_container)
            description_frame.pack(fill="x", pady=(0, 10))

            dijkstra_desc = ctk.CTkLabel(
                description_frame,
                text="Dijkstra's algorithm uses a priority queue to greedily select the next node with smallest distance.",
                font=("Helvetica", 11)
            )
            dijkstra_desc.pack(pady=(5, 0))

            bellman_desc = ctk.CTkLabel(
                description_frame,
                text="Bellman-Ford algorithm relaxes all edges V-1 times, allowing it to handle negative weights.",
                font=("Helvetica", 11)
            )
            bellman_desc.pack(pady=(0, 5))

            # Animation frame for matplotlib visualization
            self.animation_frame = ctk.CTkFrame(animation_container)
            self.animation_frame.pack(fill="both", expand=True)

            # Add step information display
            self.step_info_frame = ctk.CTkFrame(animation_container)
            self.step_info_frame.pack(fill="x", pady=(10, 0))

            # Set up the initial animation figure
            self._setup_animation_figure(color_scheme)

        except Exception as e:
            self._show_error(parent, f"Error in animation setup: {str(e)}")

    def _create_animation_controls(self, parent):
        """Create enhanced animation controls with improved speed slider."""
        # Get current theme for proper text coloring
        current_theme = self.theme_var.get()
        color_scheme = self.color_schemes["dark" if "dark" in current_theme.lower() else "light"]
        text_color = color_scheme["text"]

        # Create left section for playback controls
        playback_frame = ctk.CTkFrame(parent, fg_color="transparent")
        playback_frame.pack(side="left", fill="y", padx=(10, 5), pady=10)

        # Enhanced playback buttons
        self.play_button = ctk.CTkButton(
            playback_frame,
            text="▶ Play",
            command=self._play_animation,
            width=80
        )
        self.play_button.pack(side="left", padx=5)

        self.pause_button = ctk.CTkButton(
            playback_frame,
            text="⏸ Pause",
            command=self._pause_animation,
            width=80,
            state="disabled"
        )
        self.pause_button.pack(side="left", padx=5)

        self.reset_button = ctk.CTkButton(
            playback_frame,
            text="↺ Reset",
            command=self._reset_animation,
            width=80
        )
        self.reset_button.pack(side="left", padx=5)

        # Create right section for speed controls
        speed_frame = ctk.CTkFrame(parent, fg_color="transparent")
        speed_frame.pack(side="right", fill="y", padx=(5, 10), pady=10)

        # Improved speed control with higher maximum speed
        speed_label = ctk.CTkLabel(speed_frame, text="Animation Speed:", text_color=text_color)
        speed_label.pack(side="left", padx=(0, 5))

        self.speed_var = ctk.DoubleVar(value=1.0)

        # Improved speed slider with wider range (0.1-10x)
        speed_slider = ctk.CTkSlider(
            speed_frame,
            from_=0.1,
            to=10.0,  # Increased maximum speed to 10x
            variable=self.speed_var,
            width=150
        )
        speed_slider.pack(side="left", padx=5)

        self.speed_value = ctk.CTkLabel(speed_frame, text="1.0×", width=30, text_color=text_color)
        self.speed_value.pack(side="left", padx=5)

        # Update speed value display when slider changes
        def update_speed_display(*args):
            self.speed_value.configure(text=f"{self.speed_var.get():.1f}×")

        self.speed_var.trace_add("write", update_speed_display)

        # Step progress display
        self.progress_frame = ctk.CTkFrame(parent)
        self.progress_frame.pack(fill="x", padx=10, pady=(5, 0))

        # Create progress labels with proper text color
        self.dijkstra_progress = ctk.CTkLabel(
            self.progress_frame,
            text=f"Dijkstra: 0/{self.dijkstra_total_steps} steps",
            text_color=text_color
        )
        self.dijkstra_progress.pack(side="left", padx=(10, 0))

        self.bellman_progress = ctk.CTkLabel(
            self.progress_frame,
            text=f"Bellman-Ford: 0/{self.bellman_total_steps} steps",
            text_color=text_color
        )
        self.bellman_progress.pack(side="right", padx=(0, 10))

    def _setup_animation_figure(self, color_scheme):
        """Set up the enhanced animation figure with better visuals."""
        try:
            # Create a matplotlib figure with improved styling
            self.ani_fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5),
                                                              facecolor=color_scheme["bg"])
            self.ani_fig.suptitle("Algorithm Step-by-Step Comparison", fontsize=14,
                                  color=color_scheme["text"])

            # Set titles and styling
            self.ax1.set_title("Dijkstra's Algorithm", color=color_scheme["text"])
            self.ax2.set_title("Bellman-Ford Algorithm", color=color_scheme["text"])

            self.ax1.set_facecolor(color_scheme["bg"])
            self.ax2.set_facecolor(color_scheme["bg"])

            # Draw initial graph state
            self._draw_initial_animation_state(color_scheme)

            # Create canvas with improved performance
            self.ani_canvas = FigureCanvasTkAgg(self.ani_fig, master=self.animation_frame)
            self.ani_canvas.draw()
            self.ani_canvas.get_tk_widget().pack(fill="both", expand=True)

            # Initialize step counters and animation state
            self.dijkstra_step = 0
            self.bellman_step = 0
            self._is_animating = False

            # Create step information displays
            self.dijkstra_info = ctk.CTkLabel(
                self.step_info_frame,
                text="Dijkstra: Waiting to start...",
                font=("Helvetica", 11),
                anchor="w",
                justify="left"
            )
            self.dijkstra_info.pack(side="left", padx=10, fill="x", expand=True)

            self.bellman_info = ctk.CTkLabel(
                self.step_info_frame,
                text="Bellman-Ford: Waiting to start...",
                font=("Helvetica", 11),
                anchor="w",
                justify="left"
            )
            self.bellman_info.pack(side="right", padx=10, fill="x", expand=True)

        except Exception as e:
            print(f"Animation figure setup error: {e}")
            self._show_error(self.animation_frame, f"Error setting up animation: {str(e)}")

    def _draw_initial_animation_state(self, color_scheme):
        """Draw initial animation state with enhanced visuals."""
        # Create node color mapping
        node_colors = [color_scheme["source"] if node == self.source else
                       color_scheme["target"] if node == self.target else
                       color_scheme["node"] for node in self.G.nodes()]

        # Draw edges
        nx.draw_networkx_edges(self.G, self.pos, ax=self.ax1, alpha=0.3, edge_color="gray")
        nx.draw_networkx_edges(self.G, self.pos, ax=self.ax2, alpha=0.3, edge_color="gray")

        # Draw nodes with improved styling
        self.dijkstra_nodes = nx.draw_networkx_nodes(
            self.G, self.pos, ax=self.ax1, node_size=200,
            node_color=node_colors, edgecolors='black', linewidths=0.5
        )

        self.bellman_nodes = nx.draw_networkx_nodes(
            self.G, self.pos, ax=self.ax2, node_size=200,
            node_color=node_colors, edgecolors='black', linewidths=0.5
        )

        # Draw labels if reasonable node count
        if len(self.G.nodes()) <= 50:
            nx.draw_networkx_labels(self.G, self.pos, ax=self.ax1, font_size=8,
                                    font_color=color_scheme["text"])
            nx.draw_networkx_labels(self.G, self.pos, ax=self.ax2, font_size=8,
                                    font_color=color_scheme["text"])

        # Initialize state variables
        self.dijkstra_visited = set()
        self.bellman_visited = set()
        self.dijkstra_distances = {node: float('infinity') for node in self.G.nodes()}
        self.bellman_distances = {node: float('infinity') for node in self.G.nodes()}
        self.dijkstra_distances[self.source] = 0
        self.bellman_distances[self.source] = 0

        # Add initial step information
        self.ax1.text(0.5, -0.1, "Waiting to start algorithm", transform=self.ax1.transAxes,
                      ha="center", fontsize=10, color=color_scheme["text"])
        self.ax2.text(0.5, -0.1, "Waiting to start algorithm", transform=self.ax2.transAxes,
                      ha="center", fontsize=10, color=color_scheme["text"])

    def _play_animation(self):
        """Play the animation with improved controls."""
        if self._is_animating:
            return

        self._is_animating = True
        self.play_button.configure(state="disabled")
        self.pause_button.configure(state="normal")
        self._animate_steps()

    def _pause_animation(self):
        """Pause the animation."""
        self._is_animating = False
        self.play_button.configure(state="normal")
        self.pause_button.configure(state="disabled")

    def _reset_animation(self):
        """Reset the animation with improved state management."""
        # Stop any ongoing animation
        self._is_animating = False
        self.play_button.configure(state="normal")
        self.pause_button.configure(state="disabled")

        # Reset step counters
        self.dijkstra_step = 0
        self.bellman_step = 0

        # Reset state trackers
        self.dijkstra_visited = set()
        self.bellman_visited = set()
        self.dijkstra_distances = {node: float('infinity') for node in self.G.nodes()}
        self.bellman_distances = {node: float('infinity') for node in self.G.nodes()}
        self.dijkstra_distances[self.source] = 0
        self.bellman_distances[self.source] = 0

        # Update progress display
        self.dijkstra_progress.configure(text=f"Dijkstra: 0/{len(self.dijkstra_steps)} steps")
        self.bellman_progress.configure(text=f"Bellman-Ford: 0/{len(self.bellman_steps)} steps")

        # Reset step information
        self.dijkstra_info.configure(text="Dijkstra: Waiting to start...")
        self.bellman_info.configure(text="Bellman-Ford: Waiting to start...")

        # Get current theme
        current_theme = self.theme_var.get()
        color_scheme = self.color_schemes["dark" if "dark" in current_theme.lower() else "light"]

        # Redraw initial state
        self._draw_initial_animation_state(color_scheme)

        # Update the canvas
        if hasattr(self, 'ani_canvas'):
            self.ani_canvas.draw()

    def _animate_steps(self):
        """Animate algorithm steps with improved performance and speed handling."""
        if not self._is_animating or not self.comparison_window:
            return

        try:
            # Get current theme
            current_theme = self.theme_var.get()
            color_scheme = self.color_schemes["dark" if "dark" in current_theme.lower() else "light"]

            # Calculate dynamic delay based on speed (faster with higher speed values)
            # Lower minimum delay for higher speeds (25ms at 10x speed)
            min_delay = 25
            max_delay = 500
            speed = max(0.1, self.speed_var.get())
            delay = int(max(min_delay, max_delay / speed))

            # Process steps
            dijkstra_done = self.dijkstra_step >= len(self.dijkstra_steps)
            bellman_done = self.bellman_step >= len(self.bellman_steps)

            # Process one step from each algorithm
            dijkstra_step_info = ""
            if not dijkstra_done:
                dijkstra_step_info = self._process_dijkstra_step(self.dijkstra_steps[self.dijkstra_step], color_scheme)
                self.dijkstra_step += 1

                # Update progress display
                self.dijkstra_progress.configure(
                    text=f"Dijkstra: {self.dijkstra_step}/{self.dijkstra_total_steps} steps")
                self.dijkstra_info.configure(text=f"Dijkstra: {dijkstra_step_info}")

            bellman_step_info = ""
            if not bellman_done:
                bellman_step_info = self._process_bellman_step(self.bellman_steps[self.bellman_step], color_scheme)
                self.bellman_step += 1

                # Update progress display
                self.bellman_progress.configure(
                    text=f"Bellman-Ford: {self.bellman_step}/{self.bellman_total_steps} steps")
                self.bellman_info.configure(text=f"Bellman-Ford: {bellman_step_info}")

            # Update visualization with throttling
            current_time = time.time()
            if not hasattr(self,
                           '_last_frame_update') or current_time - self._last_frame_update >= self.ANIMATION_FRAME_LIMIT:
                self._update_animation_frame(color_scheme)
                self._last_frame_update = current_time

            # Continue animation if not done
            if (not dijkstra_done or not bellman_done) and self._is_animating:
                self.comparison_window.after(delay, self._animate_steps)
            else:
                # Animation complete, show final paths
                self._show_final_paths(color_scheme)
                self._is_animating = False
                self.play_button.configure(state="normal")
                self.pause_button.configure(state="disabled")

                # Update step information
                if dijkstra_done:
                    self.dijkstra_info.configure(
                        text=f"Dijkstra: Algorithm complete. Found path with {len(self.dijkstra_path) - 1} edges.")
                if bellman_done:
                    self.bellman_info.configure(
                        text=f"Bellman-Ford: Algorithm complete. Found path with {len(self.bellman_path) - 1} edges.")

        except Exception as e:
            print(f"Animation error: {e}")
            self._is_animating = False
            self.play_button.configure(state="normal")
            self.pause_button.configure(state="disabled")

    def _process_dijkstra_step(self, step, color_scheme):
        """Process Dijkstra step with enhanced information extraction and visualization."""
        step_info = "Processing step..."
        try:
            step_type = step[0]

            if step_type == "examine":
                if len(step) >= 3:
                    node, dist = step[1], step[2]
                    self.current_dijkstra_node = node
                    step_info = f"Examining node {node} with distance {dist:.2f}"

            elif step_type == "visit":
                if len(step) >= 2:
                    node = step[1]
                    self.dijkstra_visited.add(node)
                    step_info = f"Visiting node {node}"

            elif step_type == "update":
                if len(step) >= 4:
                    node, dist, prev = step[1], step[2], step[3]
                    self.dijkstra_distances[node] = dist
                    step_info = f"Updating distance to node {node}: {dist:.2f} via {prev}"

            elif step_type == "check_neighbor":
                if len(step) >= 7:
                    current, neighbor, curr_dist, edge_weight, new_dist, old_dist, improved = \
                        step[1], step[2], step[3], step[4], step[5], step[6], False
                    if len(step) >= 8:
                        improved = step[7]
                    self.current_dijkstra_edge = (current, neighbor)

                    status = "improved" if improved else "not improved"
                    step_info = f"Check edge ({current} -> {neighbor}): {old_dist:.2f} vs {new_dist:.2f} [{status}]"

            return step_info

        except Exception as e:
            print(f"Error processing Dijkstra step: {e}")
            return f"Error: {str(e)}"

    def _process_bellman_step(self, step, color_scheme):
        """Process Bellman-Ford step with enhanced information extraction."""
        step_info = "Processing step..."
        try:
            step_type = step[0]

            if step_type == "check_edge":
                if len(step) >= 6:
                    u, v, w, dist_u, dist_v = step[1], step[2], step[3], step[4], step[5]
                    self.current_bellman_edge = (u, v)
                    step_info = f"Check edge ({u} -> {v}), weight={w:.2f}"

            elif step_type == "update":
                if len(step) >= 4:
                    node, dist, prev = step[1], step[2], step[3]
                    self.bellman_distances[node] = dist
                    step_info = f"Updated distance to {node}: {dist:.2f} via {prev}"

            elif step_type == "iteration":
                if len(step) >= 2:
                    iteration = step[1]
                    self.current_bellman_iteration = iteration
                    step_info = f"Starting iteration {iteration + 1}"

            return step_info

        except Exception as e:
            print(f"Error processing Bellman-Ford step: {e}")
            return f"Error: {str(e)}"

    def _update_animation_frame(self, color_scheme):
        """Update animation frame with optimized rendering."""
        try:
            # Clear previous highlighted elements
            if hasattr(self, 'ax1') and self.ax1:
                # Clear any existing edge highlights
                for ax in [self.ax1, self.ax2]:
                    for artist in ax.get_children():
                        if hasattr(artist, 'get_color') and hasattr(artist, 'get_linewidth'):
                            if artist.get_linewidth() > 1:  # Assume it's a highlighted edge
                                artist.remove()

                # Clear previous text elements
                for ax in [self.ax1, self.ax2]:
                    for txt in ax.texts:
                        if hasattr(txt, 'get_position') and txt.get_position()[1] < 0:
                            txt.remove()

            # Update node colors based on algorithm state
            self._update_node_colors(color_scheme)

            # Draw current edges being examined
            self._draw_current_edges(color_scheme)

            # Add step information
            if hasattr(self, 'ax1') and self.ax1:
                if self.dijkstra_step < len(self.dijkstra_steps):
                    self.ax1.text(0.5, -0.1, f"Step {self.dijkstra_step}/{len(self.dijkstra_steps)}",
                                  transform=self.ax1.transAxes, ha="center", fontsize=10,
                                  color=color_scheme["text"])
                else:
                    self.ax1.text(0.5, -0.1, "Algorithm complete", transform=self.ax1.transAxes,
                                  ha="center", fontsize=10, color=color_scheme["text"])

            if hasattr(self, 'ax2') and self.ax2:
                if self.bellman_step < len(self.bellman_steps):
                    self.ax2.text(0.5, -0.1, f"Step {self.bellman_step}/{len(self.bellman_steps)}",
                                  transform=self.ax2.transAxes, ha="center", fontsize=10,
                                  color=color_scheme["text"])
                else:
                    self.ax2.text(0.5, -0.1, "Algorithm complete", transform=self.ax2.transAxes,
                                  ha="center", fontsize=10, color=color_scheme["text"])

            # Update canvas with optimized rendering
            if hasattr(self, 'ani_canvas'):
                self.ani_canvas.draw_idle()  # Use draw_idle for better performance

        except Exception as e:
            print(f"Error updating animation frame: {e}")

    def _update_node_colors(self, color_scheme):
        """Update node colors based on current algorithm state."""
        if not hasattr(self, 'dijkstra_nodes') or not hasattr(self, 'bellman_nodes'):
            return

        dijkstra_colors = []
        bellman_colors = []

        for node in self.G.nodes():
            # Skip source and target (fixed colors)
            if node == self.source:
                dijkstra_colors.append(color_scheme["source"])
                bellman_colors.append(color_scheme["source"])
                continue
            elif node == self.target:
                dijkstra_colors.append(color_scheme["target"])
                bellman_colors.append(color_scheme["target"])
                continue

            # Dijkstra node coloring
            if node == getattr(self, 'current_dijkstra_node', None):
                dijkstra_colors.append(color_scheme["examining"])
            elif node in self.dijkstra_visited:
                # Create gradient based on distance
                dist = self.dijkstra_distances.get(node, float('infinity'))
                if dist < float('infinity'):
                    # Get color gradient based on distance
                    color_val = self._get_color_gradient(dist, self.dijkstra_distances, color_scheme["visited"],
                                                         color_scheme["node"])
                    dijkstra_colors.append(color_val)
                else:
                    dijkstra_colors.append(color_scheme["node"])
            else:
                dijkstra_colors.append(color_scheme["node"])

            # Bellman-Ford node coloring
            dist = self.bellman_distances.get(node, float('infinity'))
            if dist < float('infinity'):
                # Get color gradient based on distance
                color_val = self._get_color_gradient(dist, self.bellman_distances, color_scheme["visited"],
                                                     color_scheme["node"])
                bellman_colors.append(color_val)
            else:
                bellman_colors.append(color_scheme["node"])

        # Update node colors
        self.dijkstra_nodes.set_color(dijkstra_colors)
        self.bellman_nodes.set_color(bellman_colors)

    def _get_color_gradient(self, dist, distances, base_color, default_color):
        """Generate a color gradient based on distance values."""
        if dist == float('infinity'):
            return default_color

        # Get all finite distances
        finite_distances = [d for d in distances.values() if d < float('infinity')]
        if not finite_distances:
            return base_color

        min_dist, max_dist = min(finite_distances), max(finite_distances)
        if min_dist == max_dist:
            return base_color

        # Normalize distance between 0 and 1
        normalized = (dist - min_dist) / (max_dist - min_dist)

        # Convert base color to RGB
        if isinstance(base_color, str) and base_color.startswith('#'):
            r = int(base_color[1:3], 16) / 255.0
            g = int(base_color[3:5], 16) / 255.0
            b = int(base_color[5:7], 16) / 255.0

            # Convert to HSV
            h, s, v = colorsys.rgb_to_hsv(r, g, b)

            # Adjust saturation and value based on normalized distance
            s = max(0.3, s * (1 - 0.7 * normalized))
            v = max(0.3, v * (1 - 0.5 * normalized))

            # Convert back to RGB
            r, g, b = colorsys.hsv_to_rgb(h, s, v)

            # Convert back to hex
            return f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'

        return base_color

    def _draw_current_edges(self, color_scheme):
        """Draw current edges being examined in the animation."""
        # Highlight current Dijkstra edge if available
        if hasattr(self, 'current_dijkstra_edge') and hasattr(self, 'ax1') and self.ax1:
            edge = self.current_dijkstra_edge
            if edge[0] in self.G and edge[1] in self.G and self.G.has_edge(*edge):
                nx.draw_networkx_edges(
                    self.G, self.pos, ax=self.ax1,
                    edgelist=[edge],
                    width=2.5, edge_color=color_scheme["examining"],
                    arrows=True, arrowstyle='->',
                    arrowsize=15
                )

        # Highlight current Bellman-Ford edge if available
        if hasattr(self, 'current_bellman_edge') and hasattr(self, 'ax2') and self.ax2:
            edge = self.current_bellman_edge
            if edge[0] in self.G and edge[1] in self.G and self.G.has_edge(*edge):
                nx.draw_networkx_edges(
                    self.G, self.pos, ax=self.ax2,
                    edgelist=[edge],
                    width=2.5, edge_color=color_scheme["examining"],
                    arrows=True, arrowstyle='->',
                    arrowsize=15
                )

    def _show_final_paths(self, color_scheme):
        """Show final paths after animation completes with enhanced styling."""
        try:
            # Draw Dijkstra's final path
            if hasattr(self, 'ax1') and self.ax1 and self.dijkstra_path and len(self.dijkstra_path) > 1:
                path_edges = [(self.dijkstra_path[i], self.dijkstra_path[i + 1])
                              for i in range(len(self.dijkstra_path) - 1)
                              if self.dijkstra_path[i] in self.G and self.dijkstra_path[i + 1] in self.G]

                nx.draw_networkx_edges(
                    self.G, self.pos, ax=self.ax1, edgelist=path_edges,
                    width=3.5, edge_color=color_scheme["path"],
                    arrows=True, arrowstyle='->', arrowsize=15,
                    label="Shortest Path"
                )

                # Add path cost information
                if self.target in self.dijkstra_dist:
                    self.ax1.text(0.5, -0.15, f"Total path cost: {self.dijkstra_dist[self.target]:.2f}",
                                  transform=self.ax1.transAxes, ha="center", fontsize=10,
                                  color=color_scheme["text"])

            # Draw Bellman-Ford final path
            if hasattr(self, 'ax2') and self.ax2 and self.bellman_path and len(self.bellman_path) > 1:
                path_edges = [(self.bellman_path[i], self.bellman_path[i + 1])
                              for i in range(len(self.bellman_path) - 1)
                              if self.bellman_path[i] in self.G and self.bellman_path[i + 1] in self.G]

                nx.draw_networkx_edges(
                    self.G, self.pos, ax=self.ax2, edgelist=path_edges,
                    width=3.5, edge_color=color_scheme["path"],
                    arrows=True, arrowstyle='->', arrowsize=15,
                    label="Shortest Path"
                )

                # Add path cost information
                if self.target in self.bellman_dist:
                    self.ax2.text(0.5, -0.15, f"Total path cost: {self.bellman_dist[self.target]:.2f}",
                                  transform=self.ax2.transAxes, ha="center", fontsize=10,
                                  color=color_scheme["text"])

            # Update canvas
            if hasattr(self, 'ani_canvas'):
                self.ani_canvas.draw()

        except Exception as e:
            print(f"Error showing final paths: {e}")

    def _show_error(self, parent, error_message):
        """Display error message in UI."""
        error_label = ctk.CTkLabel(
            parent,
            text=error_message,
            text_color="red",
            font=("Helvetica", 12)
        )
        error_label.pack(padx=15, pady=15)