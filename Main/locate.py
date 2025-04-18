# locate.py

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable


# Configure logging (can also be configured to write to a file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LocationHandler:
    def __init__(self, map_viewer, text_output):
        """
        :param map_viewer: The tkintermapview widget to display the map.
        :param text_output: A text widget to output search messages.
        """
        self.map_viewer = map_viewer
        self.text_output = text_output

        # 1) Configure geolocator with extended timeout
        self.geolocator = Nominatim(
            user_agent="ShortestPathFinder/1.0 (robertjhonaracenab@gmail.com)",
            timeout=10  # seconds
        )

        # 2) Wrap geocode in a RateLimiter for proper pacing & retries
        self.geocode = RateLimiter(
            self.geolocator.geocode,
            min_delay_seconds=1,
            max_retries=3,
            error_wait_seconds=2,
            swallow_exceptions=False
        )

        # Thread-safe cache for geocode results
        self.cache = {}
        self.cache_lock = threading.Lock()

        # Store markers by type: "FROM", "TO"
        self.markers = {}

        # Executor for background geocoding
        self.executor = ThreadPoolExecutor(max_workers=4)

    def search_location(self, address, location_type):
        """
        Submit a geocoding task to the thread pool.
        :param address: The address string to search.
        :param location_type: "FROM" or "TO".
        """
        self.executor.submit(
            self._search_location_thread,
            address,
            location_type
        )

    def _search_location_thread(self, address, location_type):
        try:
            # Check cache first
            with self.cache_lock:
                if address in self.cache:
                    location = self.cache[address]
                    logging.info("Using cached geocode for '%s'", address)
                else:
                    # Perform geocode with RateLimiter
                    location = self.geocode(
                        address,
                        exactly_one=True,
                        addressdetails=True,
                        language="en"
                    )
                    self.cache[address] = location
                    logging.info("Geocoded '%s' successfully.", address)

            # Handle results
            if location:
                lat, lon = location.latitude, location.longitude
                # Update GUI on main thread
                self.map_viewer.after(
                    0,
                    lambda: self._update_map(lat, lon, location_type)
                )
                self.map_viewer.after(
                    0,
                    lambda: self._append_text(
                        f"{location_type} location set at ({lat}, {lon}).\n\n"
                    )
                )
            else:
                msg = f"No results for '{address}'.\n\n"
                logging.error(msg.strip())
                self.map_viewer.after(0, lambda: self._append_text(msg))

        except GeocoderTimedOut:
            msg = f"⏱ Geocoding timed out for '{address}'. Please retry.\n\n"
            logging.warning(msg.strip())
            self.map_viewer.after(0, lambda: self._append_text(msg))

        except GeocoderUnavailable:
            msg = f"⚠️ Geocoding service unavailable for '{address}'.\n\n"
            logging.warning(msg.strip())
            self.map_viewer.after(0, lambda: self._append_text(msg))

        except Exception as e:
            # Catch-all
            msg = f"❌ Unexpected error for '{address}': {type(e).__name__}\n\n"
            logging.exception("Unexpected error:")
            self.map_viewer.after(0, lambda: self._append_text(msg))

    def _update_map(self, lat, lon, location_type):
        """
        Update the map on the main thread:
         - Remove existing marker if present
         - Center, zoom, and place new marker
        """
        # Remove previous marker if any
        if location_type in self.markers and self.markers[location_type]:
            try:
                self.markers[location_type].delete()
            except Exception as e:
                logging.warning("Error deleting old marker: %s", e)

        # Center and zoom
        self.map_viewer.set_position(lat, lon)
        self.map_viewer.set_zoom(16)

        # Marker colors
        if location_type.upper() == "FROM":
            circle, outside = "blue", "darkblue"
        else:
            circle, outside = "red", "darkred"

        marker = self.map_viewer.set_marker(
            lat,
            lon,
            text="",
            marker_color_circle=circle,
            marker_color_outside=outside
        )
        marker.position = (lat, lon)
        self.markers[location_type] = marker

    def _append_text(self, message):
        """
        Append a message to the text output widget.
        """
        try:
            self.text_output.insert("end", message)
        except Exception:
            logging.warning("Cannot append text; widget may be closed.")
