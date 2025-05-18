import os
import warnings
from functools import lru_cache
import colorsys
import threading
import time
from queue import Queue

import osmnx as ox
import networkx as nx
from shapely.geometry import Point, LineString
from PIL import Image, ImageDraw, ImageTk

from Main.Algorithms.pathAlgorithms import PathAlgorithms

ox.settings.timeout = 300
MAX_PATHS = 100
MAX_NODES = 10000
PATH_SIMPLIFICATION_TOLERANCE = 0.0001  # Degrees
# Viewport-based rendering settings
MAX_EDGES_PER_BATCH = 500
EDGE_BATCH_DELAY = 0.05  # seconds between batches
EDGE_VISIBILITY_THRESHOLD = 200  # maximum number of edges visible at once

# Enhanced visualization settings
DEFAULT_EDGE_COLOR = '#3388FF'  # Brighter blue for default edges
HIGHLIGHT_EDGE_COLOR = '#39FF14'  # Bright orange for highlighted paths
NODE_SIZE = 8  # Size of node markers
NODE_COLOR = '#FF3333'  # Bright red for nodes
NODE_OUTLINE_COLOR = '#FFFFFF'  # White outline for nodes
SELECTED_NODE_COLOR = '#00FF00'  # Green for origin/destination nodes
EDGE_WIDTH = 2  # Default edge width
HIGHLIGHT_EDGE_WIDTH = 4  # Width for highlighted paths


@lru_cache(maxsize=None)
def _load_graphml_cached(path):
    return ox.load_graphml(path)


def load_or_build_graph(center_point, radius, graphml_path, network_type="drive"):  # returns an OSMnx graph
    if os.path.exists(graphml_path):
        return _load_graphml_cached(graphml_path)
    G = ox.graph_from_point(center_point, dist=radius,
                            network_type=network_type, simplify=True)
    if len(G.nodes) > MAX_NODES:
        raise MemoryError(f"Graph exceeds safety limit of {MAX_NODES} nodes")
    ox.save_graphml(G, graphml_path)
    _load_graphml_cached.cache_clear()
    return _load_graphml_cached(graphml_path)


def _enumerate_k_paths(G_simple, source, target, k, weight):
    if k is None:
        k = MAX_PATHS
    try:
        gen = nx.shortest_simple_paths(G_simple, source, target, weight=weight)
    except nx.NetworkXNoPath:
        return []
    return [next(gen) for _ in range(k)]


class OptimizedGraphBuilder:
    def __init__(self, map_widget, cache_dir='graphs', default_weight='length'):
        self.map_widget = map_widget
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.G_proj = None
        self.G_simple = None
        self.edge_coords = []
        self.orig_node = None
        self.dest_node = None
        self.node_coords = {}
        self.highlight_objects = []
        self.edge_objects = []
        self.node_objects = []
        self.node_labels = []
        self.distance_markers = []
        # hold onto PhotoImage refs so they don't get GC'd
        self._marker_icons = []

        self.weight = default_weight
        self.proj_s = None
        self.proj_e = None
        self.path_cache = {}
        self.algo_steps = []
        self.algorithm_results = {}

        # Progressive loading state
        self.loading_edges = False
        self.edge_queue = Queue()
        self.visible_edges = set()
        self.edge_batch_lock = threading.Lock()

        # Edge coloring based on road type
        self.edge_types = {}
        self.edge_colors = {
            'residential': '#8888FF',  # Light blue
            'primary': '#FF8800',  # Orange
            'secondary': '#FFCC00',  # Yellow
            'tertiary': '#BBBB00',  # Olive
            'motorway': '#FF0000',  # Red
            'trunk': '#FF3333',  # Light red
            'cycleway': '#00FF00',  # Green
            'path': '#00CC88',  # Teal
            'footway': '#00BBAA',  # Turquoise
            'service': '#AAAAAA',  # Gray
            'unclassified': '#666666',  # Dark gray
            'default': DEFAULT_EDGE_COLOR  # Default blue
        }

        # Add node renaming dictionary
        self.node_id_map = {}  # Maps original OSM IDs to sequential IDs
        self.reverse_node_id_map = {}  # Maps sequential IDs back to OSM IDs

        # Bind map events for viewport-based rendering
        self._setup_map_bindings()

        # Store spatial index for edges (for viewport optimization)
        self.edge_spatial_index = {}

    def _setup_map_bindings(self):
        """Set up event bindings for viewport-based rendering"""
        canvas = self.map_widget.canvas
        canvas.bind("<Configure>", self._on_viewport_change)
        canvas.bind("<ButtonRelease-1>", self._on_viewport_change)
        canvas.bind("<MouseWheel>", self._on_viewport_change)
        canvas.bind("<Button-4>", self._on_viewport_change)
        canvas.bind("<Button-5>", self._on_viewport_change)
        self._last_viewport_update = 0

    def _on_viewport_change(self, event):
        """Handle map viewport changes - update visible edges"""
        current_time = time.time()
        if current_time - self._last_viewport_update < 0.2:
            return
        self._last_viewport_update = current_time
        self.map_widget.after(200, self._update_visible_edges)

    def _update_visible_edges(self):
        """Update which edges are visible based on the current map viewport."""
        if not self.edge_coords:
            return

        try:
            # 1) If a get_bounds() method exists, use it:
            if hasattr(self.map_widget, "get_bounds"):
                northeast, southwest = self.map_widget.get_bounds()

            # 2) Otherwise, fall back to converting canvas corners:
            elif hasattr(self.map_widget, "convert_canvas_coords_to_decimal_coords"):
                canvas = self.map_widget.canvas
                w, h = canvas.winfo_width(), canvas.winfo_height()
                ne_lat, ne_lon = self.map_widget.convert_canvas_coords_to_decimal_coords(w, 0)
                sw_lat, sw_lon = self.map_widget.convert_canvas_coords_to_decimal_coords(0, h)
                northeast, southwest = (ne_lat, ne_lon), (sw_lat, sw_lon)

            # 3) If neither is available, we can't proceed:
            else:
                return

            lat_n, lon_e = northeast
            lat_s, lon_w = southwest

            visible = set()
            for edge_id, (start, end) in enumerate(self.edge_coords):
                lat1, lon1 = start
                lat2, lon2 = end
                if (min(lat1, lat2) <= lat_n and max(lat1, lat2) >= lat_s and
                        min(lon1, lon2) <= lon_e and max(lon1, lon2) >= lon_w):
                    visible.add(edge_id)

            to_add = visible - self.visible_edges
            to_remove = self.visible_edges - visible

            if len(to_add) > 0 or len(to_remove) > 10:
                self._update_edge_visibility(to_add, to_remove)
                self.visible_edges = visible

                # After updating edges, also update nodes to maintain consistency
                # Use a small delay to ensure edge rendering completes first
                self.map_widget.after(100, self.display_node_labels)

        except Exception as e:
            warnings.warn(f"Error updating visible edges: {e}")

    def _update_edge_visibility(self, to_add, to_remove):
        """Update which edges are displayed on the map"""
        with self.edge_batch_lock:
            for edge_id in to_remove:
                if edge_id < len(self.edge_objects) and self.edge_objects[edge_id]:
                    try:
                        self.edge_objects[edge_id].delete()
                        self.edge_objects[edge_id] = None
                    except Exception:
                        pass

            edges_to_add = list(to_add)[:EDGE_VISIBILITY_THRESHOLD]
            for edge_id in edges_to_add:
                if edge_id < len(self.edge_coords):
                    u, v = self.edge_coords[edge_id]
                    try:
                        if edge_id >= len(self.edge_objects):
                            self.edge_objects.extend([None] * (edge_id - len(self.edge_objects) + 1))

                        # Use edge-specific color if available
                        edge_color = self.edge_types.get(edge_id, DEFAULT_EDGE_COLOR)
                        path_obj = self.map_widget.set_path([u, v], color=edge_color, width=EDGE_WIDTH)
                        self.edge_objects[edge_id] = path_obj
                    except Exception as e:
                        warnings.warn(f"Error adding edge {edge_id}: {e}")

    def _make_circle_icon(self, color, size=NODE_SIZE, outline_color=NODE_OUTLINE_COLOR, outline_width=1):
        """Create a circular node marker with outline"""
        img_size = size + 2  # Add space for outline
        img = Image.new("RGBA", (img_size, img_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Draw outline first
        if outline_width > 0:
            draw.ellipse((0, 0, img_size - 1, img_size - 1), fill=outline_color)
        # Draw inner circle
        inner_offset = outline_width
        draw.ellipse((inner_offset, inner_offset,
                      img_size - 1 - inner_offset, img_size - 1 - inner_offset),
                     fill=color)
        return ImageTk.PhotoImage(img)

    def clear_highlights(self):
        """Remove all paths, edges and node markers."""
        for obj in self.highlight_objects + self.distance_markers + self.node_objects + self.node_labels:
            try:
                obj.delete()
            except Exception:
                pass

        with self.edge_batch_lock:
            for i, obj in enumerate(self.edge_objects):
                if obj:
                    try:
                        obj.delete()
                    except Exception:
                        pass
            self.edge_objects = []

        self._marker_icons.clear()
        self.highlight_objects.clear()
        self.distance_markers.clear()
        self.node_objects.clear()
        self.node_labels.clear()
        self.visible_edges.clear()
        self.path_cache.clear()
        self.algorithm_results.clear()

        self.loading_edges = False
        while not self.edge_queue.empty():
            try:
                self.edge_queue.get_nowait()
                self.edge_queue.task_done()
            except Exception:
                pass

    def highlight_paths(self, paths, width=HIGHLIGHT_EDGE_WIDTH):
        """Highlight selected paths and add nodes with clear markers"""
        for obj in self.highlight_objects + self.distance_markers:
            try:
                obj.delete()
            except Exception:
                pass
        self.highlight_objects.clear()
        self.distance_markers.clear()
        self._marker_icons.clear()

        simplified_paths = []
        for path in paths:
            if len(path) < 2:
                continue
            line = LineString([self.node_coords[n] for n in path])
            simp = line.simplify(PATH_SIMPLIFICATION_TOLERANCE)
            if simp.geom_type == 'LineString':
                simplified_paths.append(list(simp.coords))
            else:
                for part in getattr(simp, 'geoms', [line]):
                    simplified_paths.append(list(part.coords))

        for coords in simplified_paths:
            p = self.map_widget.set_path(coords, color=HIGHLIGHT_EDGE_COLOR, width=width)
            self.highlight_objects.append(p)

        # Determine text color based on background
        bg = self.map_widget.canvas.cget('bg')
        if isinstance(bg, str) and bg.startswith('#') and len(bg) == 7:
            r_bg = int(bg[1:3], 16)
            g_bg = int(bg[3:5], 16)
            b_bg = int(bg[5:7], 16)
            lum = (0.299 * r_bg + 0.587 * g_bg + 0.114 * b_bg) / 255
            text_color = '#FFFFFF' if lum < 0.5 else '#000000'
        else:
            text_color = '#FFFFFF'

        # Add distance markers with custom styling
        for idx_path, coords in enumerate(simplified_paths):
            hue = (idx_path * 0.618033988749895) % 1.0
            r, g, b = colorsys.hls_to_rgb(hue, 0.5, 1.0)
            node_color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            icon = self._make_circle_icon(node_color, size=NODE_SIZE)
            self._marker_icons.append(icon)

            step = max(1, len(coords) // 8)  # More frequent distance markers
            for i in range(step, len(coords), step):
                prev_lat, prev_lon = coords[i - step]
                curr_lat, curr_lon = coords[i]
                dist_m = ox.distance.great_circle(prev_lat, prev_lon,
                                                  curr_lat, curr_lon)
                label = (f"{dist_m:.0f} m"
                         if dist_m < 1000
                         else f"{dist_m / 1000:.2f} km")

                m = self.map_widget.set_marker(
                    curr_lat,
                    curr_lon,
                    text=label,
                    font=("Helvetica", 9, "bold"),
                    icon=icon,
                    icon_anchor="center",
                    text_color=text_color
                )
                self.distance_markers.append(m)

        # Add node markers for the path nodes with clearer styling
        if paths and paths[0]:
            path = paths[0]  # Use the first path for node labels

            # Create special icons for origin and destination
            origin_icon = self._make_circle_icon(SELECTED_NODE_COLOR, size=NODE_SIZE + 2)
            dest_icon = self._make_circle_icon(SELECTED_NODE_COLOR, size=NODE_SIZE + 2)
            self._marker_icons.extend([origin_icon, dest_icon])

            for i, node in enumerate(path):
                lat, lon = self.node_coords[node]
                # Choose icon based on position in path
                if i == 0:  # Origin
                    icon = origin_icon
                    # Use sequential ID for the label
                    seq_id = self.node_id_map.get(node, node)
                    node_label = f"START (Node {seq_id})"
                elif i == len(path) - 1:  # Destination
                    icon = dest_icon
                    seq_id = self.node_id_map.get(node, node)
                    node_label = f"END (Node {seq_id})"
                else:
                    # For intermediate nodes, use a standard icon
                    icon = self._make_circle_icon(NODE_COLOR)
                    self._marker_icons.append(icon)
                    seq_id = self.node_id_map.get(node, node)
                    node_label = f"Node {seq_id}"

                m = self.map_widget.set_marker(
                    lat, lon,
                    text=node_label,
                    font=("Helvetica", 10, "bold"),
                    text_color=text_color,
                    icon=icon,
                    icon_anchor="center"
                )
                self.node_labels.append(m)

        return self.highlight_objects + self.distance_markers + self.node_labels

    def display_node_labels(self):
        """Display labels for all nodes in the graph, ensuring full map coverage"""
        for obj in self.node_objects + self.node_labels:
            try:
                obj.delete()
            except Exception:
                pass
        self.node_objects.clear()
        self.node_labels.clear()

        # Get text color based on background
        bg = self.map_widget.canvas.cget('bg')
        if isinstance(bg, str) and bg.startswith('#') and len(bg) == 7:
            r_bg = int(bg[1:3], 16)
            g_bg = int(bg[3:5], 16)
            b_bg = int(bg[5:7], 16)
            lum = (0.299 * r_bg + 0.587 * g_bg + 0.114 * b_bg) / 255
            text_color = '#FFFFFF' if lum < 0.5 else '#000000'
        else:
            text_color = '#FFFFFF'

        # Create a node icon with better visibility
        node_icon = self._make_circle_icon(NODE_COLOR, size=NODE_SIZE)
        self._marker_icons.append(node_icon)

        # Special icons for origin and destination
        if self.orig_node and self.dest_node:
            origin_icon = self._make_circle_icon(SELECTED_NODE_COLOR, size=NODE_SIZE + 2)
            dest_icon = self._make_circle_icon(SELECTED_NODE_COLOR, size=NODE_SIZE + 2)
            self._marker_icons.extend([origin_icon, dest_icon])

        # Get all visible edges
        visible_edge_nodes = set()
        for edge_id in self.visible_edges:
            if edge_id < len(self.edge_coords):
                (lat1, lon1), (lat2, lon2) = self.edge_coords[edge_id]
                # Find which nodes match these coordinates
                for node, (lat, lon) in self.node_coords.items():
                    # Check if node is at either end of the edge (with small tolerance)
                    if ((abs(lat - lat1) < 1e-8 and abs(lon - lon1) < 1e-8) or
                            (abs(lat - lat2) < 1e-8 and abs(lon - lon2) < 1e-8)):
                        visible_edge_nodes.add(node)

        # Always include origin and destination nodes
        if self.orig_node:
            visible_edge_nodes.add(self.orig_node)
        if self.dest_node:
            visible_edge_nodes.add(self.dest_node)

        # For any visible edges with no nodes, add all nodes within the viewport
        if len(visible_edge_nodes) < len(self.visible_edges) * 0.5:  # If we're missing many nodes
            # Get current viewport
            try:
                if hasattr(self.map_widget, "get_bounds"):
                    northeast, southwest = self.map_widget.get_bounds()
                    lat_n, lon_e = northeast
                    lat_s, lon_w = southwest

                    # Filter nodes to those in the viewport
                    for node, (lat, lon) in self.node_coords.items():
                        if (lat <= lat_n and lat >= lat_s and lon <= lon_e and lon >= lon_w):
                            visible_edge_nodes.add(node)
            except Exception:
                pass

        # If we still don't have enough nodes, add nodes that are the endpoints of any edges
        if len(visible_edge_nodes) < 10:
            # Sample nodes from all edges (not just visible ones)
            edge_sample = list(range(min(len(self.edge_coords), 200)))
            for edge_id in edge_sample:
                (lat1, lon1), (lat2, lon2) = self.edge_coords[edge_id]
                # Find which nodes match these coordinates
                for node, (lat, lon) in self.node_coords.items():
                    if ((abs(lat - lat1) < 1e-8 and abs(lon - lon1) < 1e-8) or
                            (abs(lat - lat2) < 1e-8 and abs(lon - lon2) < 1e-8)):
                        visible_edge_nodes.add(node)
                        if len(visible_edge_nodes) >= 200:  # Safety limit
                            break
                if len(visible_edge_nodes) >= 200:
                    break

        # Add node markers with labels
        for node in visible_edge_nodes:
            lat, lon = self.node_coords[node]

            # Select appropriate icon
            if node == self.orig_node:
                icon = origin_icon
                seq_id = self.node_id_map.get(node, node)
                label = f"START (Node {seq_id})"
            elif node == self.dest_node:
                icon = dest_icon
                seq_id = self.node_id_map.get(node, node)
                label = f"END (Node {seq_id})"
            else:
                icon = node_icon
                seq_id = self.node_id_map.get(node, node)
                label = f"Node {seq_id}"

            m = self.map_widget.set_marker(
                lat, lon,
                text=label,
                font=("Helvetica", 8, "bold"),
                text_color=text_color,
                icon=icon,
                icon_anchor="center"
            )
            self.node_labels.append(m)

        return self.node_labels

    def _create_sequential_node_ids(self, path=None):
        """
        Create a mapping of original node IDs to randomized numbers.
        The origin node is always assigned ID 1, but other nodes get randomized IDs.
        If a path is provided, prioritize nodes along that path to be numbered.
        """
        import random

        # Reset the mapping dictionaries
        self.node_id_map = {}
        self.reverse_node_id_map = {}

        # Track nodes that have been assigned IDs
        path_nodes = set()

        # Collect all nodes that need IDs
        all_nodes = list(self.node_coords.keys())

        # Generate a list of random unique IDs (excluding 1 which is reserved for origin)
        max_id = len(all_nodes) + 1  # +1 to ensure we have enough IDs
        random_ids = list(range(2, max_id + 1))
        random.shuffle(random_ids)

        # First, assign ID 1 to origin
        if self.orig_node:
            self.node_id_map[self.orig_node] = 1
            self.reverse_node_id_map[1] = self.orig_node
            path_nodes.add(self.orig_node)

        # If path is provided, prioritize path nodes (but with randomized IDs)
        path_nodes_list = []
        if path:
            for node in path:
                if node != self.orig_node and node not in path_nodes:
                    path_nodes_list.append(node)
                    path_nodes.add(node)

        # Handle destination node specially (include in path if it exists)
        if self.dest_node and self.dest_node != self.orig_node and self.dest_node not in path_nodes:
            path_nodes_list.append(self.dest_node)
            path_nodes.add(self.dest_node)

        # Assign random IDs to path nodes first (including destination)
        for node in path_nodes_list:
            rand_id = random_ids.pop()
            self.node_id_map[node] = rand_id
            self.reverse_node_id_map[rand_id] = node

        # Then assign random IDs to all remaining nodes
        for node in all_nodes:
            if node not in self.node_id_map:
                rand_id = random_ids.pop()
                self.node_id_map[node] = rand_id
                self.reverse_node_id_map[rand_id] = node

        return self.node_id_map

    def build_graph(self, start, end, use_bbox=False, buffer=1000):
        self.clear_highlights()

        if len(start) != 2 or len(end) != 2:
            raise ValueError("Start and end must be (lat,lon) tuples")
        lat1, lon1 = start
        lat2, lon2 = end
        straight = ox.distance.great_circle(lat1, lon1, lat2, lon2)
        radius = straight / 2 + buffer
        center = ((lat1 + lat2) / 2, (lon1 + lon2) / 2)
        fname = os.path.join(self.cache_dir,
                             f"pt_{center[0]:.5f}_{center[1]:.5f}_{radius:.0f}.graphml")
        G0 = load_or_build_graph(center, radius, fname)
        G0 = ox.distance.add_edge_lengths(G0)
        Gp = ox.project_graph(G0)

        self.node_coords = {n: (d['y'], d['x'])
                            for n, d in G0.nodes(data=True)}

        # Build edge coordinates and assign edge types/colors
        self.edge_coords = []
        self.edge_types = {}

        for i, (u, v, data) in enumerate(G0.edges(data=True)):
            self.edge_coords.append(
                ((G0.nodes[u]['y'], G0.nodes[u]['x']),
                 (G0.nodes[v]['y'], G0.nodes[v]['x']))
            )

            # Assign edge color based on highway type
            highway = data.get('highway', 'default')
            if isinstance(highway, list):
                highway = highway[0] if highway else 'default'

            # Assign proper color from our color mapping
            color = self.edge_colors.get(highway, self.edge_colors['default'])
            self.edge_types[i] = color

        self.edge_objects = [None] * len(self.edge_coords)

        crs = G0.graph.get('crs')
        self.proj_s = ox.projection.project_geometry(Point(lon1, lat1), crs=crs)[0]
        self.proj_e = ox.projection.project_geometry(Point(lon2, lat2), crs=crs)[0]
        self.orig_node = ox.distance.nearest_nodes(Gp,
                                                   X=self.proj_s.x,
                                                   Y=self.proj_s.y)
        self.dest_node = ox.distance.nearest_nodes(Gp,
                                                   X=self.proj_e.x,
                                                   Y=self.proj_e.y)
        self.G_proj = Gp
        self.G_simple = ox.convert.to_digraph(Gp, weight=self.weight)

        # Create initial sequential node IDs
        self._create_sequential_node_ids()

        self._update_visible_edges()

        return self

    def _process_edge_queue(self):
        if not self.loading_edges:
            return

        with self.edge_batch_lock:
            edges_to_draw = []
            for _ in range(min(MAX_EDGES_PER_BATCH, self.edge_queue.qsize())):
                if not self.loading_edges:
                    break
                try:
                    edge_id = self.edge_queue.get_nowait()
                    edges_to_draw.append(edge_id)
                    self.edge_queue.task_done()
                except Exception:
                    break

            for edge_id in edges_to_draw:
                if edge_id in self.visible_edges and edge_id < len(self.edge_coords):
                    u, v = self.edge_coords[edge_id]
                    try:
                        if edge_id >= len(self.edge_objects):
                            self.edge_objects.extend([None] * (edge_id - len(self.edge_objects) + 1))

                        # Use edge-specific color
                        edge_color = self.edge_types.get(edge_id, DEFAULT_EDGE_COLOR)
                        path_obj = self.map_widget.set_path([u, v], color=edge_color, width=EDGE_WIDTH)
                        self.edge_objects[edge_id] = path_obj
                    except Exception:
                        pass

        if self.loading_edges and not self.edge_queue.empty():
            self.map_widget.after(int(EDGE_BATCH_DELAY * 1000), self._process_edge_queue)

    def display_graph(self, color=None, width=EDGE_WIDTH):
        """Display the complete graph with proper edge colors and node markers"""
        with self.edge_batch_lock:
            for i, obj in enumerate(self.edge_objects):
                if obj:
                    try:
                        obj.delete()
                    except Exception:
                        pass
            self.edge_objects = [None] * len(self.edge_coords)
            self.visible_edges.clear()

        self.loading_edges = True
        self._update_visible_edges()

        for edge_id in range(len(self.edge_coords)):
            self.edge_queue.put(edge_id)

        self._process_edge_queue()

        # Display node labels with clear markers - ensure every edge has nodes displayed
        self.map_widget.after(500, self._ensure_all_nodes_displayed)

        return self.edge_objects

    def _ensure_all_nodes_displayed(self):
        """Make sure all nodes are displayed for visible edges"""
        self.display_node_labels()

        # Schedule another node refresh after edge loading completes
        if self.loading_edges and not self.edge_queue.empty():
            self.map_widget.after(1000, self._ensure_all_nodes_displayed)

    def find_k_paths(self, k=5, weight=None):
        if not all([self.G_simple, self.orig_node, self.dest_node]):
            raise RuntimeError("Graph not initialized")
        key = (self.orig_node, self.dest_node, k, weight or self.weight)
        if key not in self.path_cache:
            self.path_cache[key] = _enumerate_k_paths(
                self.G_simple, self.orig_node, self.dest_node,
                k, weight or self.weight)
        return self.path_cache[key]

    def run_dijkstra(self):
        """
        Run Dijkstra's algorithm and format the explanation.

        Returns:
            Tuple containing:
                - path: List of nodes in the shortest path
                - explanation: Text explanation of the algorithm steps
                - details: Dictionary with detailed algorithm results for visualization
        """
        if not all([self.G_simple, self.orig_node, self.dest_node]):
            raise RuntimeError("Graph not initialized")

        try:
            # Run Dijkstra with step tracking
            result = PathAlgorithms.dijkstra(
                self.G_simple, self.orig_node, self.dest_node, self.weight)

            # Unpack the returned values - handle both old and new return format
            if len(result) == 4:
                dist, prev, steps, execution_time = result
            else:
                # Handle the case where the old function is still being used
                dist, prev, steps = result
                execution_time = None

            # Reconstruct path
            path = PathAlgorithms.reconstruct_path(prev, self.orig_node, self.dest_node)

            # Create sequential node IDs for the path
            if path:
                self._create_sequential_node_ids(path)

            # Generate explanation with node ID map and execution time
            explanation = PathAlgorithms.format_steps_explanation(
                self.G_simple, steps, prev, self.orig_node, self.dest_node,
                node_id_map=self.node_id_map, execution_time=execution_time)

            # Store results
            self.algorithm_results["dijkstra"] = {
                "path": path,
                "dist": dist,
                "prev": prev,
                "steps": steps,
                "explanation": explanation,
                "execution_time": execution_time,
                "source": self.orig_node,
                "target": self.dest_node
            }

            # Highlight the path with enhanced styling
            if path:
                self.highlight_paths([path], width=HIGHLIGHT_EDGE_WIDTH)

            # Return the result tuple with path, explanation and the complete results for visualization
            return path, explanation, self.algorithm_results["dijkstra"]

        except Exception as e:
            import traceback
            print(f"Error in run_dijkstra: {e}")
            print(traceback.format_exc())
            raise

    def run_bellman_ford(self):
        """
        Run Bellman-Ford algorithm and format the explanation.

        Returns:
            Tuple containing:
                - path: List of nodes in the shortest path
                - explanation: Text explanation of the algorithm steps
                - details: Dictionary with detailed algorithm results for visualization
        """
        if not all([self.G_simple, self.orig_node, self.dest_node]):
            raise RuntimeError("Graph not initialized")

        try:
            # Run Bellman-Ford with step tracking (now includes execution time)
            result = PathAlgorithms.bellman_ford(
                self.G_simple, self.orig_node, self.dest_node, self.weight)

            # Unpack the returned values - handle both old and new return format
            if result and len(result) == 4:
                dist, prev, steps, execution_time = result
            elif result and len(result) == 3:
                # Handle the case where the old function is still being used
                dist, prev, steps = result
                execution_time = None
            else:
                # Handle case where result is invalid
                dist, prev, steps, execution_time = None, None, [], None

            # Check if a path was found (could be None if negative cycle was detected)
            if dist is None or prev is None:
                explanation = "Negative cycle detected in the graph. Cannot find shortest path."
                path = []

                # Store results with negative cycle indication
                self.algorithm_results["bellman_ford"] = {
                    "path": [],
                    "dist": {},
                    "prev": {},
                    "steps": steps or [],
                    "explanation": explanation,
                    "execution_time": execution_time,
                    "source": self.orig_node,
                    "target": self.dest_node,
                    "negative_cycle": True
                }
            else:
                # Reconstruct path - handle potential errors
                try:
                    path = PathAlgorithms.reconstruct_path(prev, self.orig_node, self.dest_node)
                except Exception as e:
                    import traceback
                    print(f"Error reconstructing path: {e}")
                    print(traceback.format_exc())
                    path = []

                # Create sequential node IDs for the path
                if path:
                    try:
                        self._create_sequential_node_ids(path)
                    except Exception as e:
                        print(f"Error creating sequential node IDs: {e}")
                        # Continue without sequential IDs

                # Generate explanation with node ID map and execution time
                try:
                    explanation = PathAlgorithms.format_steps_explanation(
                        self.G_simple, steps, prev, self.orig_node, self.dest_node,
                        node_id_map=self.node_id_map, execution_time=execution_time)
                except Exception as e:
                    import traceback
                    print(f"Error formatting explanation: {e}")
                    print(traceback.format_exc())
                    explanation = f"Error generating explanation: {str(e)}"

                # Store results
                self.algorithm_results["bellman_ford"] = {
                    "path": path,
                    "dist": dist,
                    "prev": prev,
                    "steps": steps,
                    "explanation": explanation,
                    "execution_time": execution_time,
                    "source": self.orig_node,
                    "target": self.dest_node,
                    "negative_cycle": False
                }

                # Highlight the path with enhanced styling - handle potential errors
                if path:
                    try:
                        self.highlight_paths([path], width=HIGHLIGHT_EDGE_WIDTH)
                    except Exception as e:
                        print(f"Error highlighting paths: {e}")
                        # Continue without highlighting

            # Return the result tuple with path, explanation and the complete results for visualization
            return path, explanation, self.algorithm_results["bellman_ford"]

        except Exception as e:
            import traceback
            print(f"Error in run_bellman_ford: {e}")
            print(traceback.format_exc())

            # Create a graceful error response instead of crashing
            path = []
            explanation = f"Error running Bellman-Ford algorithm: {str(e)}"
            error_results = {
                "path": [],
                "dist": {},
                "prev": {},
                "steps": [],
                "explanation": explanation,
                "execution_time": None,
                "source": self.orig_node,
                "target": self.dest_node,
                "negative_cycle": False,
                "error": str(e)
            }
            self.algorithm_results["bellman_ford"] = error_results

            return path, explanation, error_results

    def _get_manageable_edge_subset(self, G, source, target=None, max_edges=5000):
        """Get a manageable subset of edges for large graphs to improve algorithm performance"""
        if len(G.edges()) <= max_edges:
            return list(G.edges())

        # If graph is too large, prioritize edges near source and target
        import networkx as nx

        # Get nodes within a certain distance from source and target
        source_neighbors = set(nx.ego_graph(G, source, radius=3))
        target_neighbors = set(nx.ego_graph(G, target, radius=3)) if target else set()

        # Prioritize edges between these nodes
        priority_edges = []
        other_edges = []

        for u, v in G.edges():
            if u in source_neighbors or u in target_neighbors or v in source_neighbors or v in target_neighbors:
                priority_edges.append((u, v))
            else:
                other_edges.append((u, v))

        # Take all priority edges plus some random edges up to max_edges
        import random
        if len(priority_edges) < max_edges:
            # Add random edges up to max_edges
            random.shuffle(other_edges)
            remaining = max_edges - len(priority_edges)
            return priority_edges + other_edges[:remaining]
        else:
            # If too many priority edges, take a random subset
            random.shuffle(priority_edges)
            return priority_edges[:max_edges]