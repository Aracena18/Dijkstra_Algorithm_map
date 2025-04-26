import os
import warnings
from functools import lru_cache
import colorsys

import osmnx as ox
import networkx as nx
from shapely.geometry import Point, LineString
from PIL import Image, ImageDraw, ImageTk

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
        self.distance_markers = []
        # hold onto PhotoImage refs so they don’t get GC’d
        self._marker_icons = []

        self.weight = default_weight
        self.proj_s = None
        self.proj_e = None
        self.path_cache = {}

    def clear_highlights(self):
        """Remove all paths, edges and node markers."""
        for obj in self.highlight_objects + self.edge_objects + self.distance_markers:
            try:
                obj.delete()
            except Exception:
                pass
        # also clear our icon refs
        self._marker_icons.clear()
        self.highlight_objects.clear()
        self.edge_objects.clear()
        self.distance_markers.clear()
        self.path_cache.clear()

    def _make_circle_icon(self, color, size=6):
        """Return a small PIL PhotoImage of a filled circle."""
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse((0, 0, size - 1, size - 1), fill=color, outline=color)
        return ImageTk.PhotoImage(img)

    def highlight_paths(self, paths, width=3):
        """Draw the blue path(s) and tiny colored node markers with distance labels."""
        self.clear_highlights()

        # 1) simplify and draw each path in solid blue
        simplified_paths = []
        for path in paths:
            if len(path) < 2:
                continue
            line = LineString([self.node_coords[n] for n in path])
            simp = line.simplify(PATH_SIMPLIFICATION_TOLERANCE)
            if simp.geom_type == 'LineString':
                simplified_paths.append(list(simp.coords))
            else:  # MultiLineString or fallback
                for part in getattr(simp, 'geoms', [line]):
                    simplified_paths.append(list(part.coords))

        for coords in simplified_paths:
            p = self.map_widget.set_path(coords, color="#0000FF", width=width)
            self.highlight_objects.append(p)

        # determine text color based on current map background
        bg = self.map_widget.canvas.cget('bg')
        if isinstance(bg, str) and bg.startswith('#') and len(bg) == 7:
            r_bg = int(bg[1:3], 16)
            g_bg = int(bg[3:5], 16)
            b_bg = int(bg[5:7], 16)
            lum = (0.299 * r_bg + 0.587 * g_bg + 0.114 * b_bg) / 255
            text_color = '#FFFFFF' if lum < 0.5 else '#000000'
        else:
            text_color = '#FFFFFF'

        # 2) for each segment endpoint, drop a tiny icon with distance label
        for idx_path, coords in enumerate(simplified_paths):
            # pick a distinct hue per path for the node color
            hue = (idx_path * 0.618033988749895) % 1.0
            r, g, b = colorsys.hls_to_rgb(hue, 0.5, 1.0)
            node_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            icon = self._make_circle_icon(node_color, size=6)
            self._marker_icons.append(icon)

            for i in range(1, len(coords)):
                prev_lat, prev_lon = coords[i-1]
                curr_lat, curr_lon = coords[i]
                dist_m = ox.distance.great_circle(prev_lat, prev_lon,
                                                  curr_lat, curr_lon)
                label = (f"{dist_m:.0f} m"
                         if dist_m < 1000
                         else f"{dist_m/1000:.2f} km")

                m = self.map_widget.set_marker(
                    curr_lat,
                    curr_lon,
                    text=label,
                    font=("Helvetica", 8, "bold"),
                    icon=icon,
                    icon_anchor="center",
                    text_color=text_color
                )
                self.distance_markers.append(m)

        return self.highlight_objects + self.distance_markers

    def build_graph(self, start, end, use_bbox=False, buffer=1000):
        # … your existing load/project logic unchanged …
        if len(start) != 2 or len(end) != 2:
            raise ValueError("Start and end must be (lat,lon) tuples")
        lat1, lon1 = start; lat2, lon2 = end
        straight = ox.distance.great_circle(lat1, lon1, lat2, lon2)
        radius = straight / 2 + buffer
        center = ((lat1 + lat2)/2, (lon1 + lon2)/2)
        fname = os.path.join(self.cache_dir,
                             f"pt_{center[0]:.5f}_{center[1]:.5f}_{radius:.0f}.graphml")
        G0 = load_or_build_graph(center, radius, fname)
        G0 = ox.distance.add_edge_lengths(G0)
        Gp = ox.project_graph(G0)

        self.node_coords = {n: (d['y'], d['x'])
                            for n, d in G0.nodes(data=True)}

        self.edge_coords = [
            ((G0.nodes[u]['y'], G0.nodes[u]['x']),
             (G0.nodes[v]['y'], G0.nodes[v]['x']))
            for u, v in G0.edges()
        ]

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
            raise RuntimeError("Graph not initialized")
        key = (self.orig_node, self.dest_node, k, weight or self.weight)
        if key not in self.path_cache:
            self.path_cache[key] = _enumerate_k_paths(
                self.G_simple, self.orig_node, self.dest_node,
                k, weight or self.weight)
        return self.path_cache[key]

    def show_route(self, start, end, algo="dijkstra",
                   use_bbox=False, buffer=1000, show_all=False, k=None):
        try:
            self.build_graph(start, end, use_bbox, buffer)
            if self.orig_node == self.dest_node:
                warnings.warn("Origin and destination identical")
                return []

            self.display_graph()

            if show_all:
                paths = self.find_k_paths(k or MAX_PATHS)
                return self.highlight_paths(paths, width=5)
            else:
                fn = (nx.dijkstra_path
                      if algo == "dijkstra"
                      else nx.bellman_ford_path)
                try:
                    paths = [fn(self.G_simple,
                                self.orig_node, self.dest_node,
                                self.weight)]
                except nx.NetworkXNoPath:
                    paths = []
                return self.highlight_paths(paths, width=3)

        except Exception as e:
            self.clear_highlights()
            raise RuntimeError(f"Routing failed: {e}") from e
