# autocomplete.py

import tkinter as tk
import customtkinter as ctk
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AutocompleteEntry(ctk.CTkEntry):
    def __init__(self, master=None, debounce_delay=300, **kwargs):
        """
        :param master: The parent widget.
        :param debounce_delay: Delay in milliseconds for debouncing geocoding requests.
        :param kwargs: Other keyword arguments for the CTkEntry.
        """
        super().__init__(master, **kwargs)
        self.debounce_delay = debounce_delay
        self.after_id = None
        self.listbox = None
        self.listbox_frame = None  # For improved styling container

        self.geolocator = Nominatim(
            user_agent="ShortestPathFinder/1.0 (robertjhonaracenab@gmail.com)"
        )
        self.geocode = RateLimiter(
            self.geolocator.geocode,
            min_delay_seconds=1,
            max_retries=3
        )

        self.bind("<KeyRelease>", self.on_key_release)
        self.bind("<Tab>", self.complete_suggestion)

    def on_key_release(self, event):
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None
        self.after_id = self.after(self.debounce_delay, self.check_suggestions)

    def check_suggestions(self):
        text = self.get().strip()
        suggestions = []
        if text:
            try:
                locations = self.geocode(
                    text, exactly_one=False, addressdetails=True, limit=5
                )
                suggestions = [loc.address for loc in locations] if locations else []
            except Exception as ex:
                logging.exception("Error during geocoding for suggestions: %s", ex)
        self.show_suggestions(suggestions)

    def show_suggestions(self, suggestions):
        self._destroy_listbox()
        if suggestions:
            # Use the top-level window as parent to ensure proper layering.
            parent = self.winfo_toplevel()
            self.listbox_frame = tk.Frame(
                parent, bg="#ffffff", bd=0, highlightthickness=1, highlightbackground="#cccccc"
            )
            self.listbox = tk.Listbox(
                self.listbox_frame,
                width=int(self.cget("width")),
                height=min(len(suggestions), 5),
                font=("Segoe UI", 10),
                bg="#ffffff",
                fg="#000000",
                bd=0,
                highlightthickness=0,
                relief="flat",
                selectbackground="#e0e0e0",
                activestyle="none",
                cursor="hand2"
            )
            for suggestion in suggestions:
                self.listbox.insert(tk.END, suggestion)
            self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)

            # Compute the position relative to the top-level window.
            parent_x = parent.winfo_rootx()
            parent_y = parent.winfo_rooty()
            x = self.winfo_rootx() - parent_x
            y = self.winfo_rooty() - parent_y + self.winfo_height()
            self.listbox_frame.place(x=x, y=y, width=self.winfo_width())
            self.listbox.pack(fill="both", expand=True)
        else:
            self._destroy_listbox()

    def complete_suggestion(self, event):
        if self.listbox and self.listbox.size() > 0:
            index = self.listbox.curselection()[0] if self.listbox.curselection() else 0
            selection = self.listbox.get(index)
            self.delete(0, tk.END)
            self.insert(0, selection)
            self._destroy_listbox()
            self.event_generate("<<LocationSelected>>")
            return "break"

    def on_listbox_select(self, event):
        if self.listbox and self.listbox.curselection():
            index = self.listbox.curselection()[0]
            selection = self.listbox.get(index)
            self.delete(0, tk.END)
            self.insert(0, selection)
            self._destroy_listbox()
            self.event_generate("<<LocationSelected>>")

    def _destroy_listbox(self):
        if self.listbox:
            self.listbox.destroy()
            self.listbox = None
        if self.listbox_frame:
            self.listbox_frame.destroy()
            self.listbox_frame = None