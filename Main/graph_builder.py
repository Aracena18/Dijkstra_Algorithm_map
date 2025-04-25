import os
import warnings
from functools import lru_cache
import colorsys

import osmnx as ox
import networkx as nx
from shapely.geometry import Point
from shapely.geometry.linestring import LineString

ox.settings.timeout = 300
MAX_PATHS = 100
MAX_NODES = 10000
PATH_SIMPLIFICATION_TOLERANCE = 0.0001  # Degrees


@lru_cache(maxsize=None)
def _load_graphml_cached(path):
    return ox.load_graphml(path)


def load_or_build_graph(center_point, radius, graphml_path, network_type="drive"):
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
    return [next(gen) for _ in range(k) if True]


class OptimizedGraphBuilder:
    def __init__(self, map_widget, cache_dir='graphs', default_weight='length'):
        self.map_widget = map_widget
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.G_unproj = None
        self.G_proj = None
        self.G_simple = None
        self.edge_coords = []
        self.orig_node = None
        self.dest_node = None
        self.node_coords = {}
        self.highlight_objects = []
        self.edge_objects = []
        self.weight = default_weight
        self.proj_s = None
        self.proj_e = None
        self.path_cache = {}

    def highlight_paths(self, paths, color_func=lambda i: 'blue', width=3):
        """Draw paths with geometry simplification"""
        self.clear_highlights()
        simplified_paths = []

        for path in paths:
            if len(path) < 2:
                continue

            # Convert coordinates to Shapely LineString and simplify
            line = LineString([self.node_coords[n] for n in path])
            simplified = line.simplify(PATH_SIMPLIFICATION_TOLERANCE)

            # Extract coordinates from simplified geometry
            if simplified.geom_type == 'LineString':
                simplified_paths.append(list(simplified.coords))
            elif simplified.geom_type == 'MultiLineString':
                for part in simplified.geoms:
                    simplified_paths.extend(list(part.coords))
            else:
                simplified_paths.append(list(line.coords))

        # Create path objects with simplified coordinates
        self.highlight_objects = [
            self.map_widget.set_path(
                path,
                color=self._ensure_hex_color(color_func(i)),
                width=width
            )
            for i, path in enumerate(simplified_paths)
        ]
        return self.highlight_objects

    def _validate_nodes(self, G, buffer):
        """Validate nodes using projected coordinates"""
        for node, role in [(self.orig_node, "Origin"), (self.dest_node, "Destination")]:
            if node not in G.nodes:
                raise ValueError(f"{role} node not found in graph")

            node_x = G.nodes[node]['x']
            node_y = G.nodes[node]['y']
            input_point = self.proj_s if role == "Origin" else self.proj_e
            dist = ((node_x - input_point.x) ** 2 + (node_y - input_point.y) ** 2) ** 0.5

            if dist > buffer * 2:
                warnings.warn(f"{role} node is {dist:.0f}m from input location")

    def build_graph(self, start, end, use_bbox=False, buffer=1000):
        if len(start) != 2 or len(end) != 2:
            raise ValueError("Start and end points must be (lat, lon) tuples")

        lat1, lon1 = start
        lat2, lon2 = end

        if use_bbox:
            north, south = max(lat1, lat2) + 0.01, min(lat1, lat2) - 0.01
            east, west = max(lon1, lon2) + 0.01, min(lon1, lon2) - 0.01
            fname = os.path.join(self.cache_dir,
                                 f"bbox_{north:.5f}_{south:.5f}_{east:.5f}_{west:.5f}.graphml")
            G0 = _load_graphml_cached(fname) if os.path.exists(fname) else ox.graph_from_bbox(
                north, south, east, west, network_type="drive", simplify=True)
            if len(G0.nodes) > MAX_NODES:
                raise MemoryError("Graph too large for bbox approach")
            if not os.path.exists(fname):
                ox.save_graphml(G0, fname)
        else:
            straight_dist = ox.distance.great_circle(lat1, lon1, lat2, lon2)
            radius = straight_dist / 2 + buffer
            center = ((lat1 + lat2) / 2, (lon1 + lon2) / 2)
            fname = os.path.join(self.cache_dir,
                                 f"pt_{center[0]:.5f}_{center[1]:.5f}_{radius:.0f}.graphml")
            G0 = load_or_build_graph(center, radius, fname)

        G0 = ox.distance.add_edge_lengths(G0)
        Gp = ox.project_graph(G0)
        self.node_coords = {n: (data['y'], data['x']) for n, data in G0.nodes(data=True)}

        # Corrected edge_coords list comprehension
        self.edge_coords = [
            ((G0.nodes[u]['y'], G0.nodes[u]['x']),
             (G0.nodes[v]['y'], G0.nodes[v]['x']))
            for u, v in G0.edges()
        ]

        crs = G0.graph.get('crs')
        ps, pe = Point(lon1, lat1), Point(lon2, lat2)
        self.proj_s = ox.projection.project_geometry(ps, crs=crs)[0]
        self.proj_e = ox.projection.project_geometry(pe, crs=crs)[0]

        self.orig_node = ox.distance.nearest_nodes(Gp, X=self.proj_s.x, Y=self.proj_s.y)
        self.dest_node = ox.distance.nearest_nodes(Gp, X=self.proj_e.x, Y=self.proj_e.y)
        self._validate_nodes(Gp, buffer)

        self.G_proj = Gp
        self.G_simple = ox.convert.to_digraph(Gp, weight=self.weight)
        return self

    def display_graph(self, color='gray', width=1):
        for obj in self.edge_objects:
            try:
                obj.delete()
            except:
                pass
        self.edge_objects = [
            self.map_widget.set_path([u, v], color=color, width=width)
            for u, v in self.edge_coords
        ]
        return self.edge_objects

    def find_k_paths(self, k=5, weight=None):
        if not all([self.G_simple, self.orig_node, self.dest_node]):
            raise RuntimeError("Graph not properly initialized")
        cache_key = (self.orig_node, self.dest_node, k, weight or self.weight)
        if cache_key not in self.path_cache:
            self.path_cache[cache_key] = _enumerate_k_paths(
                self.G_simple, self.orig_node, self.dest_node, k, weight or self.weight)
        return self.path_cache[cache_key]


    def _ensure_hex_color(self, color):
        if color.startswith("hsl"):
            return self._hsl_to_hex(color)
        return color

    def _hsl_to_hex(self, hsl_str):
        try:
            hsl = hsl_str.strip("hsl(%) ").split(",")
            h = float(hsl[0]) / 360
            s = float(hsl[1]) / 100
            l = float(hsl[2]) / 100

            r, g, b = colorsys.hls_to_rgb(h, l, s)
            return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        except:
            return "#FF0000"

    def clear_highlights(self):
        for obj in self.highlight_objects:
            try:
                obj.delete()
            except:
                pass
        self.highlight_objects.clear()
        self.path_cache.clear()

    def show_route(self, start, end, algo="dijkstra", use_bbox=False,
                   buffer=1000, show_all=False, k=None):
        try:
            self.build_graph(start, end, use_bbox, buffer)
            if self.orig_node == self.dest_node:
                warnings.warn("Origin and destination are identical")
                return []

            # redraw the full graph in gray (or any background color you like)
            self.display_graph()

            # always draw paths in blue
            blue_fn = lambda i: "#0000FF"

            if show_all:
                paths = self.find_k_paths(k=k or MAX_PATHS)
                return self.highlight_paths(paths, blue_fn, width=5)
            else:
                try:
                    path_func = (
                        nx.dijkstra_path
                        if algo == "dijkstra"
                        else nx.bellman_ford_path
                    )
                    paths = [path_func(self.G_simple, self.orig_node, self.dest_node, self.weight)]
                except nx.NetworkXNoPath:
                    paths = []
                return self.highlight_paths(paths, blue_fn, width=3)

        except Exception as e:
            self.clear_highlights()
            raise RuntimeError(f"Routing failed: {e}") from e
