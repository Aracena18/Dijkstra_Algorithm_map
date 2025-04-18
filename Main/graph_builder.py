import math
import warnings

import osmnx as ox
import networkx as nx
from shapely.geometry import Point

class GraphBuilder:
    """
    Builds and visualizes a street network graph between two geographic points,
    auto‑scaling the download area so that the points lie in one connected component.
    """

    def __init__(self, map_widget):
        self.map_widget = map_widget
        self.G_unproj = None
        self.G_proj = None
        self.orig_node = None
        self.dest_node = None
        self.highlight_object = None

    @staticmethod
    def haversine(coord1, coord2):
        """
        Compute great-circle distance (meters) between two (lat, lon) pairs using the haversine formula.
        """
        lat1, lon1 = map(math.radians, coord1)
        lat2, lon2 = map(math.radians, coord2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.asin(math.sqrt(a))
        return 6371000 * c  # Earth radius in meters

    def build_graph(self, start, end, use_bbox=False, buffer=1000):
        """
        Download and project the driving network so start/end are connected.

        :param start: (lat, lon)
        :param end:   (lat, lon)
        :param use_bbox: if True, use bounding box instead of buffer radius
        :param buffer: extra meters to add around computed radius or bbox
        """
        if use_bbox:
            # bounding box covers exactly between points plus buffer
            lat1, lon1 = start
            lat2, lon2 = end
            north, south = max(lat1, lat2) + 0.01, min(lat1, lat2) - 0.01
            east,  west  = max(lon1, lon2) + 0.01, min(lon1, lon2) - 0.01
            G0 = ox.graph_from_bbox(north, south, east, west,
                                    network_type="drive", simplify=True)
        else:
            # radius = half the straight‑line distance + buffer
            straight_dist = self.haversine(start, end)
            radius = straight_dist / 2 + buffer
            G0 = ox.graph_from_point(
                ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2),
                dist=radius,
                network_type="drive", simplify=True
            )

        # add metric lengths
        G0 = ox.distance.add_edge_lengths(G0)

        # project and save
        Gp = ox.project_graph(G0)
        self.G_unproj = G0
        self.G_proj = Gp

        # find nearest nodes
        pt_start = Point(start[1], start[0])
        pt_end   = Point(end[1],   end[0])
        proj_start, _ = ox.projection.project_geometry(pt_start, crs=G0.graph['crs'])
        proj_end,   _ = ox.projection.project_geometry(pt_end,   crs=G0.graph['crs'])
        try:
            self.orig_node = ox.distance.nearest_nodes(self.G_proj,
                                                       X=proj_start.x,
                                                       Y=proj_start.y)
            self.dest_node = ox.distance.nearest_nodes(self.G_proj,
                                                       X=proj_end.x,
                                                       Y=proj_end.y)
        except ImportError:
            warnings.warn("scipy not installed; falling back to unprojected search")
            self.orig_node = ox.nearest_nodes(self.G_unproj, start[1], start[0])
            self.dest_node = ox.nearest_nodes(self.G_unproj, end[1],   end[0])

    def find_shortest_path(self, algo="dijkstra"):
        """
        Compute the shortest path between origin and destination.
        Catches NetworkXNoPath if still disconnected.
        """
        try:
            if algo.lower() == "dijkstra":
                return nx.dijkstra_path(self.G_proj,
                                        self.orig_node,
                                        self.dest_node,
                                        weight="length")
            else:
                return nx.bellman_ford_path(self.G_proj,
                                            self.orig_node,
                                            self.dest_node,
                                            weight="length")
        except nx.NetworkXNoPath:
            warnings.warn("No path found between points.")
            return None

    def highlight_path(self, node_path, color="yellow", width=5):
        """
        Draw the path on the TkinterMapView widget.
        """
        if self.highlight_object:
            try:
                self.highlight_object.delete()
            except Exception:
                pass

        coords = [(self.G_unproj.nodes[n]['y'], self.G_unproj.nodes[n]['x'])
                  for n in node_path]
        self.highlight_object = self.map_widget.set_path(
            coords, color=color, width=width, name="highlight"
        )
        return self.highlight_object

    def show_route(self, start, end, algo="dijkstra", use_bbox=False, buffer=1000):
        """
        Convenience: builds the graph, finds the shortest path, and highlights it.
        """
        # build and project
        self.build_graph(start, end, use_bbox=use_bbox, buffer=buffer)

        # compute path
        node_path = self.find_shortest_path(algo=algo)
        if node_path is None:
            return None

        # draw on the map
        return self.highlight_path(node_path)