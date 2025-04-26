# autocomplete.py

import tkinter as tk
import customtkinter as ctk
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import logging
import threading
import sys

# Configure minimal logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


class AutocompleteEntry(ctk.CTkEntry):
    def __init__(self, master=None, debounce_delay=200, **kwargs):
        """
        :param master: The parent widget.
        :param debounce_delay: Delay in milliseconds for debouncing geocoding requests.
        :param kwargs: Other keyword arguments for the CTkEntry.
        """
        super().__init__(master, **kwargs)
        self.debounce_delay = debounce_delay
        self.after_id = None
        self.listbox = None
        self.listbox_frame = None
        self.is_processing = False
        self.listbox_width = None
        self.location_type = None  # Will be set by the main app

        # Color schemes for different themes
        self.themes = {
            "Dark": {
                "bg": "#1a1a1a",
                "fg": "#ffffff",
                "highlight_bg": "#2e2e2e",
                "highlight_fg": "#3a7ebf",
                "border": "#1f538d"
            },
            "Light": {
                "bg": "#f0f0f0",
                "fg": "#333333",
                "highlight_bg": "#e0e0e0",
                "highlight_fg": "#1f538d",
                "border": "#3a7ebf"
            },
            "Barbie": {
                "bg": "#FFE6F3",
                "fg": "#333333",
                "highlight_bg": "#FFCCE5",
                "highlight_fg": "#FF1493",
                "border": "#FF1493"
            }
        }

        # Current theme (defaults to Dark)
        self.current_theme = "Dark"

        # Configure geolocator with optimized settings
        self.geolocator = Nominatim(
            user_agent="ShortestPathFinder/1.0 (robertjhonaracenab@gmail.com)",
            timeout=3
        )
        # Rate limiting to avoid IP bans
        self.geocode = RateLimiter(
            self.geolocator.geocode,
            min_delay_seconds=1.0,
            max_retries=2,
            error_wait_seconds=2.0
        )

        # Cache for geocoding results to reduce API calls
        self.geocode_cache = {}

        # Bind events
        self.bind("<KeyRelease>", self.on_key_release)
        self.bind("<Tab>", self.complete_suggestion)
        self.bind("<FocusOut>", self.delayed_hide_suggestions)
        self.bind("<Escape>", lambda e: self._destroy_listbox())

        # Bind arrow keys for navigation
        self.bind("<Down>", self.navigate_down)
        self.bind("<Up>", self.navigate_up)
        self.bind("<Return>", self.complete_selected)

    def on_key_release(self, event):
        """Handle key release with improved debouncing"""
        # Don't trigger on special keys
        if event.keysym in ('Up', 'Down', 'Left', 'Right', 'Tab', 'Return', 'Escape'):
            return

        # Cancel any pending check
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None

        # Set new timer for debounced check
        self.after_id = self.after(self.debounce_delay, self.check_suggestions)

    def check_suggestions(self):
        """Get suggestions based on the current text"""
        text = self.get().strip()

        # Don't search for very short texts
        if not text or len(text) < 3:
            self._destroy_listbox()
            return

        if self.is_processing:
            return

        # Use threading to prevent UI freezing
        self.is_processing = True
        threading.Thread(target=self._fetch_suggestions, args=(text,), daemon=True).start()

    def _fetch_suggestions(self, text):
        """Fetch suggestions in a background thread"""
        suggestions = []
        try:
            # Check cache first
            if text in self.geocode_cache:
                suggestions = self.geocode_cache[text]
            else:
                # Try to get locations
                try:
                    locations = self.geocode(
                        text, exactly_one=False, addressdetails=True, limit=5
                    )
                    if locations:
                        suggestions = [loc.address for loc in locations]
                        # Cache the results
                        self.geocode_cache[text] = suggestions
                except Exception as ex:
                    # Reduce logging verbosity for common errors
                    if "not found" not in str(ex).lower():
                        logging.warning(f"Geocoding error: {str(ex)}")
        except Exception as ex:
            logging.warning(f"Error in background thread: {str(ex)}")

        # Make sure we're still connected to the UI before updating
        try:
            if self.winfo_exists():
                # Update UI in the main thread
                self.after(0, lambda: self._show_suggestions_main_thread(suggestions))
        except tk.TclError:
            pass  # Widget was destroyed

    def _show_suggestions_main_thread(self, suggestions):
        """Show suggestions in the main thread"""
        try:
            self.show_suggestions(suggestions)
        except Exception as e:
            logging.warning(f"Error showing suggestions: {str(e)}")
        finally:
            self.is_processing = False

    def detect_theme(self):
        """Detect current theme from appearance mode or parent widget"""
        try:
            # Try to get appearance mode from customtkinter
            appearance_mode = ctk.get_appearance_mode()
            if appearance_mode == "Dark":
                return "Dark"

            # Try to detect theme from parent widget colors
            if self.master and hasattr(self.master, 'winfo_toplevel'):
                root = self.winfo_toplevel()
                if hasattr(root, '_current_theme'):
                    return root._current_theme

                # Try to check fg_color attribute
                if hasattr(root, 'cget'):
                    fg_color = root.cget('fg_color')
                    if isinstance(fg_color, str):
                        if '#242424' in fg_color or '#2e2e2e' in fg_color:
                            return "Dark"
                        elif '#f658b8' in fg_color or '#FF1493' in fg_color:
                            return "Barbie"

            # Default to Light if uncertain
            return "Light"
        except Exception:
            return "Light"

    def show_suggestions(self, suggestions):
        """Display the suggestions dropdown with improved styling"""
        self._destroy_listbox()
        if not suggestions:
            return

        # Try to detect current theme
        self.current_theme = self.detect_theme()
        theme_colors = self.themes.get(self.current_theme, self.themes["Light"])

        # Get the root window for proper layering
        parent = self.winfo_toplevel()

        # Create a frame with rounded corners
        self.listbox_frame = tk.Frame(
            parent,
            bg=theme_colors["bg"],
            bd=1,
            highlightthickness=1,
            highlightbackground=theme_colors["border"]
        )

        # Calculate optimal width
        max_length = max([len(s) for s in suggestions])
        self.listbox_width = max(self.winfo_width(), max_length * 8)  # Adjust character width

        # Create styled listbox
        self.listbox = tk.Listbox(
            self.listbox_frame,
            width=0,  # Let frame control width
            height=min(len(suggestions), 5),
            font=("Segoe UI", 11),
            bg=theme_colors["bg"],
            fg=theme_colors["fg"],
            bd=0,
            highlightthickness=0,
            relief="flat",
            selectbackground=theme_colors["highlight_bg"],
            selectforeground=theme_colors["highlight_fg"],
            activestyle="none",
            cursor="hand2"
        )

        # Add suggestions with alternating background colors
        for i, suggestion in enumerate(suggestions):
            self.listbox.insert(tk.END, suggestion)
            if i % 2 == 1:
                # Slightly lighter/darker alternating rows based on theme
                if self.current_theme == "Dark":
                    alt_color = "#232323"
                elif self.current_theme == "Barbie":
                    alt_color = "#FFD9ED"
                else:
                    alt_color = "#f9f9f9"
                self.listbox.itemconfig(i, bg=alt_color)

        # Add scrollbar for many results
        scrollbar = tk.Scrollbar(self.listbox_frame, orient="vertical")
        scrollbar.config(command=self.listbox.yview)
        self.listbox.config(yscrollcommand=scrollbar.set)

        # Bind events
        self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)
        self.listbox.bind("<Enter>", lambda e: self.listbox.config(cursor="hand2"))

        # Position the dropdown precisely below the entry
        x = self.winfo_rootx() - parent.winfo_rootx()
        y = self.winfo_rooty() - parent.winfo_rooty() + self.winfo_height()

        # Place the frame and pack the listbox
        self.listbox_frame.place(x=x, y=y, width=self.listbox_width)
        self.listbox.pack(side="left", fill="both", expand=True)

        # Only show scrollbar if needed
        if len(suggestions) > 5:
            scrollbar.pack(side="right", fill="y")

    def complete_suggestion(self, event):
        """Complete with the currently selected suggestion"""
        if self.listbox and self.listbox.size() > 0:
            selection = self.listbox.curselection()
            index = selection[0] if selection else 0
            selection = self.listbox.get(index)
            self.delete(0, tk.END)
            self.insert(0, selection)
            self._destroy_listbox()
            self.event_generate("<<LocationSelected>>")
            return "break"  # Prevent default tab behavior

    def on_listbox_select(self, event):
        """Handle listbox selection"""
        if self.listbox and self.listbox.curselection():
            index = self.listbox.curselection()[0]
            selection = self.listbox.get(index)
            self.delete(0, tk.END)
            self.insert(0, selection)
            self._destroy_listbox()
            self.event_generate("<<LocationSelected>>")
            self.focus_set()  # Return focus to entry

    def _destroy_listbox(self):
        """Safely destroy the listbox and frame"""
        if self.listbox:
            self.listbox.destroy()
            self.listbox = None
        if self.listbox_frame:
            self.listbox_frame.destroy()
            self.listbox_frame = None

    def navigate_down(self, event):
        """Navigate down in the suggestions"""
        if self.listbox:
            if self.listbox.curselection():
                current = self.listbox.curselection()[0]
                if current < self.listbox.size() - 1:
                    self.listbox.selection_clear(0, tk.END)
                    self.listbox.selection_set(current + 1)
                    self.listbox.activate(current + 1)
                    self.listbox.see(current + 1)
            else:
                self.listbox.selection_set(0)
                self.listbox.activate(0)
            return "break"  # Prevent default navigation

    def navigate_up(self, event):
        """Navigate up in the suggestions"""
        if self.listbox:
            if self.listbox.curselection():
                current = self.listbox.curselection()[0]
                if current > 0:
                    self.listbox.selection_clear(0, tk.END)
                    self.listbox.selection_set(current - 1)
                    self.listbox.activate(current - 1)
                    self.listbox.see(current - 1)
            else:
                self.listbox.selection_set(0)
                self.listbox.activate(0)
            return "break"  # Prevent default navigation

    def complete_selected(self, event):
        """Complete with Enter key"""
        if self.listbox and self.listbox.curselection():
            self.complete_suggestion(event)
            return "break"  # Prevent default Enter behavior

    def delayed_hide_suggestions(self, event):
        """Hide suggestions after a small delay to allow for clicks"""
        self.after(200, self._destroy_listbox)

    def clear_cache(self):
        """Clear the geocoding cache"""
        self.geocode_cache.clear()