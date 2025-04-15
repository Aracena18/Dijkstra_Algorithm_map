# locate.py

import tkinter as tk
import logging
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging (can also be configured to write to a file)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LocationHandler:
    def __init__(self, map_viewer, text_output):
        """
        :param map_viewer: The tkintermapview widget to display the map.
        :param text_output: A text widget to output search messages.
        """
        self.map_viewer = map_viewer
        self.text_output = text_output
        self.geolocator = Nominatim(
            user_agent="ShortestPathFinder/1.0 (robertjhonaracenab@gmail.com)"
        )
        self.geocode = RateLimiter(
            self.geolocator.geocode,
            min_delay_seconds=1,
            max_retries=3
        )
        self.cache = {}
        self.cache_lock = threading.Lock()  # Ensure thread-safe cache access.
        self.markers = {}  # Dictionary to store marker references by location type ("FROM", "TO").

        # Use a ThreadPoolExecutor to limit concurrent geocoding threads.
        self.executor = ThreadPoolExecutor(max_workers=4)

    def search_location(self, address, location_type):
        """
        Submit a geocoding task to the thread pool.
        :param address: The address string to search.
        :param location_type: "FROM" or "TO".
        """
        self.executor.submit(self._search_location_thread, address, location_type)

    def _search_location_thread(self, address, location_type):
        try:
            with self.cache_lock:
                if address in self.cache:
                    location = self.cache[address]
                    logging.info("Using cached geocode result for '%s'.", address)
                else:
                    location = self.geocode(address, exactly_one=True)
                    self.cache[address] = location
                    logging.info("Geocoded '%s' successfully.", address)

            if location:
                lat, lon = location.latitude, location.longitude
                self.map_viewer.after(0, lambda: self._update_map(lat, lon, location_type))
                self.map_viewer.after(0, lambda: self._append_text(
                    f"{location_type} location set at ({lat}, {lon}).\n\n"))
            else:
                msg = f"Location {location_type} not found for address: {address}\n\n"
                self.map_viewer.after(0, lambda: self._append_text(msg))
                logging.error(msg)
        except Exception as e:
            msg = f"Error searching for location {location_type}: {e}\n\n"
            self.map_viewer.after(0, lambda: self._append_text(msg))
            logging.exception("Error during geocoding:")

    def _update_map(self, lat, lon, location_type):
        """
        Update the map on the main thread:
         - Remove an existing marker for the same location_type if present.
         - Centers the map, sets the zoom, and adds a new marker.
        """
        # Remove existing marker if it exists for this location type.
        if location_type in self.markers and self.markers[location_type]:
            try:
                self.markers[location_type].delete()  # Remove the marker from the map.
            except Exception as e:
                logging.warning("Unable to delete previous marker: %s", e)

        # Center map and set zoom.
        self.map_viewer.set_position(lat, lon)
        self.map_viewer.set_zoom(16)

        # Choose marker colors based on location type.
        if location_type.upper() == "FROM":
            marker_color_circle = "blue"
            marker_color_outside = "darkblue"
        else:
            marker_color_circle = "red"
            marker_color_outside = "darkred"

        # Place a new marker.
        marker = self.map_viewer.set_marker(
            lat, lon,
            text="",
            marker_color_circle=marker_color_circle,
            marker_color_outside=marker_color_outside
        )
        # Store the marker for later removal.
        self.markers[location_type] = marker

    def _append_text(self, message):
        """
        Append a message to the text output widget.
        """
        self.text_output.insert("end", message)
