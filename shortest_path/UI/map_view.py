# map_view.py

import customtkinter as ctk
import tkintermapview
from tkinter import TclError
from Main.autocomplete import AutocompleteEntry
from Main.locate import LocationHandler

class MapViewApp:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.default_colors = {
            "Dark": {
                "fg": "#242424", "button": "#1f538d", "hover": "#144870",
                "border": "#1f538d", "entry_fg": "#343638", "text_border_width": 1
            },
            "Light": {
                "fg": "#ebebeb", "button": "#3a7ebf", "hover": "#325882",
                "border": "#3a7ebf", "entry_fg": "#dbdbdb", "text_border_width": 1
            },
            "Barbie": {
                "fg": "#f658b8", "button": "#FF1493", "hover": "#FFB6C1",
                "border": "#FF1493", "entry_fg": "#FFE6F3", "text_border_width": 2
            }
        }

        try:
            self.default_font = ("Winky-Sans", 12, "bold")
        except TclError:
            print("Winky Sans not found, using Helvetica")
            self.default_font = ("Helvetica", 12, "bold")

        self.build_main_window()
        self.selected_algorithm = ctk.StringVar(value="none")
        self.theme_var = ctk.StringVar(value="Dark")
        self.build_widgets()

    def build_main_window(self):
        self.main = ctk.CTk()
        self.main.title("Shortest Path Comparison: Dijkstra vs Bellman-Ford")
        self.main.geometry("1000x600")
        self.main.configure(fg_color=self.default_colors["Dark"]["fg"])
        self.main.protocol("WM_DELETE_WINDOW", self.on_closing)

    def build_widgets(self):
        # Build individual widget sections for improved maintainability.
        self.build_entry_widgets()
        self.build_theme_selector()
        self.build_map_container()
        self.build_buttons()
        self.build_text_output()

        # Initialize location handling.
        self.locator = LocationHandler(self.map_viewer, self.text_output)

    def build_entry_widgets(self):
        colors = self.default_colors["Dark"]
        self.entry_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        self.entry_frame.place(x=293, y=33)

        # "From" input with integrated search button.
        self.from_frame = ctk.CTkFrame(self.entry_frame, fg_color="transparent")
        self.from_frame.pack(pady=5)
        self.entry_from = AutocompleteEntry(
            self.from_frame,
            placeholder_text="From Location",
            width=400,  # Adjust width to accommodate the button
            height=40,
            font=self.default_font,
            border_color=colors["border"],
            border_width=1
        )
        self.entry_from.pack(side="left", fill="x", expand=True)
        self.search_from_button = ctk.CTkButton(
            self.from_frame,
            text="🔍",
            width=40,
            height=40,
            font=self.default_font,
            fg_color=colors["button"],
            hover_color=colors["hover"],
            command=lambda: self.search_from_location()
        )
        self.search_from_button.pack(side="left")

        # "To" input with integrated search button.
        self.to_frame = ctk.CTkFrame(self.entry_frame, fg_color="transparent")
        self.to_frame.pack(pady=5)
        self.entry_to = AutocompleteEntry(
            self.to_frame,
            placeholder_text="To Location",
            width=400,
            height=40,
            font=self.default_font,
            border_color=colors["border"],
            border_width=1
        )
        self.entry_to.pack(side="left", fill="x", expand=True)
        self.search_to_button = ctk.CTkButton(
            self.to_frame,
            text="🔍",
            width=40,
            height=40,
            font=self.default_font,
            fg_color=colors["button"],
            hover_color=colors["hover"],
            command=lambda: self.search_to_location()
        )
        self.search_to_button.pack(side="left")

        # Assign location type and bind events.
        self.entry_from.location_type = "FROM"
        self.entry_to.location_type = "TO"
        self.entry_from.bind("<<LocationSelected>>", self.handle_location_selected)
        self.entry_to.bind("<<LocationSelected>>", self.handle_location_selected)
        self.entry_from.bind("<Return>", lambda event: self.search_from_location())
        self.entry_to.bind("<Return>", lambda event: self.search_to_location())

    def build_theme_selector(self):
        self.theme_selector = ctk.CTkOptionMenu(
            self.main,
            values=["Light", "Dark", "Barbie"],
            variable=self.theme_var,
            command=self.handle_theme,
            width=120, height=30, font=self.default_font
        )
        self.theme_selector.place(x=16, y=15)

    def build_map_container(self):
        self.map_container = ctk.CTkFrame(self.main, fg_color="transparent")
        self.map_container.place(x=264, y=151)
        self.map_frame = ctk.CTkFrame(self.map_container, width=500, height=280)
        self.map_frame.pack()
        self.map_frame.pack_propagate(False)
        self.map_viewer = tkintermapview.TkinterMapView(master=self.map_frame)
        self.map_viewer.pack(fill="both", expand=True)

    def build_buttons(self):
        colors = self.default_colors["Dark"]
        self.play_button = ctk.CTkButton(
            self.main, text="▶", width=40, height=100,
            font=("Helvetica", 20, "bold"),
            fg_color=colors["button"],
            hover_color=colors["hover"],
            state="disabled",
            command=self.run_algorithm
        )
        self.play_button.place(x=203, y=151)
        self.button_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        self.button_frame.place(x=785, y=174)
        self.button_dijkstra = ctk.CTkButton(
            self.button_frame, text="Dijkstra's",
            width=150, height=40, font=self.default_font,
            command=lambda: self.select_algorithm("dijkstra")
        )
        self.button_dijkstra.pack()
        self.button_bellman = ctk.CTkButton(
            self.button_frame, text="Bellman-Ford",
            width=150, height=40, font=self.default_font,
            command=lambda: self.select_algorithm("bellman-ford")
        )
        self.button_bellman.pack(pady=10)

    def build_text_output(self):
        colors = self.default_colors["Dark"]
        self.text_output = ctk.CTkTextbox(
            self.main, width=455, height=120,
            font=self.default_font,
            border_color=colors["border"],
            border_width=colors["text_border_width"]
        )
        self.text_output.place(x=287, y=452)

    def handle_location_selected(self, event):
        widget = event.widget
        location_type = getattr(widget, "location_type", "UNKNOWN")
        if location_type == "FROM":
            self.search_from_location()
        elif location_type == "TO":
            self.search_to_location()

    def handle_theme(self, theme):
        colors = self.default_colors[theme]
        if theme == "Barbie":
            ctk.set_appearance_mode("light")
        else:
            ctk.set_appearance_mode(theme.lower())
        self.main.configure(fg_color=colors["fg"])
        self.map_frame.configure(fg_color=colors["entry_fg"])
        self.map_container.configure(fg_color="transparent")
        self.entry_frame.configure(fg_color="transparent")
        self.button_frame.configure(fg_color="transparent")
        self.theme_selector.configure(
            fg_color=colors["button"],
            button_color=colors["button"],
            button_hover_color=colors["hover"]
        )
        for widget in [self.entry_from, self.entry_to]:
            widget.configure(
                fg_color=colors["entry_fg"],
                border_color=colors["border"],
                border_width=colors["text_border_width"]
            )
        for btn in [self.play_button, self.button_dijkstra, self.button_bellman,
                    self.search_from_button, self.search_to_button]:
            btn.configure(
                fg_color=colors["button"],
                hover_color=colors["hover"]
            )
        self.text_output.configure(
            fg_color=colors["entry_fg"],
            border_color=colors["border"],
            border_width=colors["text_border_width"]
        )

    def select_algorithm(self, algo_name):
        self.selected_algorithm.set(algo_name)
        self.play_button.configure(state="normal")
        theme = self.theme_var.get()
        color_set = {
            "Dark": {"selected": "#1f538d", "unselected": ("gray75", "gray25")},
            "Light": {"selected": "#3a7ebf", "unselected": ("gray75", "gray25")},
            "Barbie": {"selected": "#FF1493", "unselected": "#f658b8"}
        }
        colors = color_set[theme]
        self.button_dijkstra.configure(
            fg_color=colors["selected"] if algo_name == "dijkstra" else colors["unselected"]
        )
        self.button_bellman.configure(
            fg_color=colors["selected"] if algo_name == "bellman-ford" else colors["unselected"]
        )

    def run_algorithm(self):
        algo = self.selected_algorithm.get()
        self.text_output.insert("end", f"Running {algo} algorithm...\n")
        # Place your algorithm processing in a non-blocking thread if needed.

    def search_from_location(self):
        address = self.entry_from.get()
        self.locator.search_location(address, "FROM")

    def search_to_location(self):
        address = self.entry_to.get()
        self.locator.search_location(address, "TO")

    def on_closing(self):
        try:
            self.main.destroy()
        except Exception as e:
            print(f"Error while closing: {e}")

    def run(self):
        self.main.mainloop()

