import customtkinter as ctk
import tkintermapview
from tkinter import TclError
from Main.autocomplete import AutocompleteEntry
from Main.locate import LocationHandler
from Main.graph_builder import OptimizedGraphBuilder


class MapViewApp:
    def __init__(self, api_key: str):
        # Store the Stadia Maps API key
        self.api_key = api_key

        # Configure appearance before any widgets are created
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")

        # Define theme configurations including map tiles (base URLs)
        self.default_colors = {
            "Dark": {
                "fg": "#242424",
                "button": "#1f538d",
                "hover": "#144870",
                "border": "#1f538d",
                "entry_fg": "#343638",
                "text_border_width": 1,
                # Base URL without API key
                "tile_server": "https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}.png",
                "canvas_bg": "#2e2e2e"
            },
            "Light": {
                "fg": "#ebebeb",
                "button": "#3a7ebf",
                "hover": "#325882",
                "border": "#3a7ebf",
                "entry_fg": "#dbdbdb",
                "text_border_width": 1,
                "tile_server": "https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}",
                "canvas_bg": "#ffffff"
            },
            "Barbie": {
                "fg": "#f658b8",
                "button": "#FF1493",
                "hover": "#FFB6C1",
                "border": "#FF1493",
                "entry_fg": "#FFE6F3",
                "text_border_width": 2,
                "tile_server": "https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}",
                "canvas_bg": "#FFE6F3"
            }
        }

        # Font configuration
        try:
            self.default_font = ("Winky-Sans", 12, "bold")
        except TclError:
            self.default_font = ("Helvetica", 12, "bold")

        # Main window setup
        self.main = ctk.CTk()
        self.main.title("Shortest Path Comparison: Dijkstra vs Bellman-Ford")
        self.main.geometry("1000x600")
        self.main.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Theme control variables
        self.selected_algorithm = ctk.StringVar(master=self.main, value="none")
        self.theme_var = ctk.StringVar(master=self.main, value="Dark")
        self.show_all_var = ctk.BooleanVar(master=self.main, value=False)

        # Initialize UI
        current_theme = self.theme_var.get()
        self.main.configure(fg_color=self.default_colors[current_theme]["fg"])
        self.build_widgets()

        # Initialize services (after map_viewer exists)
        self.graph_builder = OptimizedGraphBuilder(self.map_viewer)
        self.locator = LocationHandler(self.map_viewer, self.text_output)

    def build_widgets(self):
        """Build all application widgets"""
        self.build_entry_widgets()
        self.build_theme_selector()
        self.build_option_controls()
        self.build_map_container()
        self.build_buttons()
        self.build_text_output()

    def build_map_container(self):
        """Build map widget container with theme support"""
        # Use the canvas background color instead of transparent
        config = self.default_colors[self.theme_var.get()]

        self.map_container = ctk.CTkFrame(self.main, fg_color=config["canvas_bg"])
        self.map_container.place(x=264, y=151)

        self.map_frame = ctk.CTkFrame(
            self.map_container,
            width=500,
            height=280,
            fg_color=config["canvas_bg"]
        )
        self.map_frame.pack()
        self.map_frame.pack_propagate(False)

        # Initialize map with current theme
        self.map_viewer = tkintermapview.TkinterMapView(
            master=self.map_frame,
            width=500,
            height=280,
            corner_radius=0
        )
        self.map_viewer.pack(fill="both", expand=True)
        self.update_map_theme(self.theme_var.get())

    def update_map_theme(self, theme: str):
        """Update map visuals based on selected theme"""
        config = self.default_colors[theme]
        base_url = config["tile_server"]
        # Append API key if using Stadia Maps dark tiles
        if theme == "Dark" and self.api_key:
            tile_url = f"{base_url}?api_key={self.api_key}"
        else:
            tile_url = base_url
        self.map_viewer.set_tile_server(tile_url, max_zoom=22)
        # Set the map canvas background
        self.map_viewer.canvas.configure(bg=config["canvas_bg"])

    def build_theme_selector(self):
        """Build theme selection dropdown"""
        self.theme_selector = ctk.CTkOptionMenu(
            master=self.main,
            values=list(self.default_colors.keys()),
            variable=self.theme_var,
            command=self.handle_theme,
            width=120,
            height=30,
            font=self.default_font
        )
        self.theme_selector.place(x=16, y=15)

    def handle_theme(self, theme):
        """Handle theme change event"""
        # Update appearance mode
        ctk.set_appearance_mode("dark" if theme == "Dark" else "light")

        # Get theme configuration
        config = self.default_colors[theme]

        # Update main window background
        self.main.configure(fg_color=config["fg"])

        # Update map container & frame backgrounds
        self.map_container.configure(fg_color=config["canvas_bg"])
        self.map_frame.configure(fg_color=config["canvas_bg"])

        # Update map tiles
        self.update_map_theme(theme)

        # Update other UI elements
        for entry in [self.entry_from, self.entry_to]:
            entry.configure(
                fg_color=config["entry_fg"],
                border_color=config["border"],
                border_width=config["text_border_width"]
            )

        for button in [
            self.play_button,
            self.button_dijkstra,
            self.button_bellman,
            self.search_from_button,
            self.search_to_button,
            self.show_graph_button
        ]:
            button.configure(
                fg_color=config["button"],
                hover_color=config["hover"]
            )

        self.text_output.configure(
            fg_color=config["entry_fg"],
            border_color=config["border"],
            border_width=config["text_border_width"]
        )

    def build_entry_widgets(self):
        """Build location entry widgets"""
        config = self.default_colors[self.theme_var.get()]
        self.entry_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        self.entry_frame.place(x=293, y=33)

        # From location entry
        self.from_frame = ctk.CTkFrame(self.entry_frame, fg_color="transparent")
        self.from_frame.pack(pady=5)
        self.entry_from = AutocompleteEntry(
            master=self.from_frame,
            placeholder_text="From Location",
            width=400,
            height=40,
            font=self.default_font,
            border_color=config["border"],
            border_width=1
        )
        self.entry_from.pack(side="left", fill="x", expand=True)
        self.search_from_button = ctk.CTkButton(
            master=self.from_frame,
            text="🔍",
            width=40,
            height=40,
            font=self.default_font,
            fg_color=config["button"],
            hover_color=config["hover"],
            command=self.search_from_location
        )
        self.search_from_button.pack(side="left")

        # To location entry
        self.to_frame = ctk.CTkFrame(self.entry_frame, fg_color="transparent")
        self.to_frame.pack(pady=5)
        self.entry_to = AutocompleteEntry(
            master=self.to_frame,
            placeholder_text="To Location",
            width=400,
            height=40,
            font=self.default_font,
            border_color=config["border"],
            border_width=1
        )
        self.entry_to.pack(side="left", fill="x", expand=True)
        self.search_to_button = ctk.CTkButton(
            master=self.to_frame,
            text="🔍",
            width=40,
            height=40,
            font=self.default_font,
            fg_color=config["button"],
            hover_color=config["hover"],
            command=self.search_to_location
        )
        self.search_to_button.pack(side="left")

        # Event bindings
        self.entry_from.location_type = "FROM"
        self.entry_to.location_type = "TO"
        self.entry_from.bind("<<LocationSelected>>", self.handle_location_selected)
        self.entry_to.bind("<<LocationSelected>>", self.handle_location_selected)
        self.entry_from.bind("<Return>", lambda e: self.search_from_location())
        self.entry_to.bind("<Return>", lambda e: self.search_to_location())

    def build_option_controls(self):
        """Build additional options controls"""
        self.option_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        self.option_frame.place(x=16, y=60)
        self.show_all_checkbox = ctk.CTkCheckBox(
            master=self.option_frame,
            text="Show All Paths",
            variable=self.show_all_var,
            font=self.default_font
        )
        self.show_all_checkbox.pack()

    def build_buttons(self):
        """Build action buttons"""
        config = self.default_colors[self.theme_var.get()]

        # Play button
        self.play_button = ctk.CTkButton(
            master=self.main,
            text="▶",
            width=40,
            height=100,
            font=("Helvetica", 20, "bold"),
            fg_color=config["button"],
            hover_color=config["hover"],
            state="disabled",
            command=self.run_algorithm
        )
        self.play_button.place(x=203, y=151)

        # Algorithm buttons
        self.button_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        self.button_frame.place(x=785, y=174)
        self.button_dijkstra = ctk.CTkButton(
            master=self.button_frame,
            text="Dijkstra's",
            width=150,
            height=40,
            font=self.default_font,
            command=lambda: self.select_algorithm("dijkstra")
        )
        self.button_dijkstra.pack(pady=5)
        self.button_bellman = ctk.CTkButton(
            master=self.button_frame,
            text="Bellman-Ford",
            width=150,
            height=40,
            font=self.default_font,
            command=lambda: self.select_algorithm("bellman-ford")
        )
        self.button_bellman.pack(pady=5)
        self.show_graph_button = ctk.CTkButton(
            master=self.button_frame,
            text="Show Route",
            width=150,
            height=40,
            font=self.default_font,
            fg_color=config["button"],
            hover_color=config["hover"],
            command=self.show_route
        )
        self.show_graph_button.pack(pady=5)

    def build_text_output(self):
        """Build text output widget"""
        config = self.default_colors[self.theme_var.get()]
        self.text_output = ctk.CTkTextbox(
            master=self.main,
            width=455,
            height=120,
            font=self.default_font,
            border_color=config["border"],
            border_width=config["text_border_width"]
        )
        self.text_output.place(x=287, y=452)

    def handle_location_selected(self, event):
        """Handle location selection event"""
        if event.widget.location_type == "FROM":
            self.search_from_location()
        else:
            self.search_to_location()

    def select_algorithm(self, algo_name):
        """Select routing algorithm"""
        self.selected_algorithm.set(algo_name)
        self.play_button.configure(state="normal")
        theme_colors = self.default_colors[self.theme_var.get()]
        selected_color = theme_colors["button"]
        self.button_dijkstra.configure(
            fg_color=selected_color if algo_name == "dijkstra" else theme_colors["border"]
        )
        self.button_bellman.configure(
            fg_color=selected_color if algo_name == "bellman-ford" else theme_colors["border"]
        )

    def run_algorithm(self):
        """Execute selected routing algorithm"""
        marker_from = self.locator.markers.get("FROM")
        marker_to = self.locator.markers.get("TO")
        if not (marker_from and marker_to):
            self.text_output.insert("end", "Both FROM and TO locations must be set.\n")
            return

        start = marker_from.position
        end = marker_to.position

        if self.show_all_var.get():
            self.text_output.insert("end", f"Showing all paths from {start} to {end}...\n")
            self.graph_builder.show_route(start, end, show_all=True)
        else:
            algo = self.selected_algorithm.get()
            if algo == "none":
                self.text_output.insert("end", "Select an algorithm or check 'Show All Paths'.\n")
                return
            self.text_output.insert("end", f"Routing from {start} to {end} using {algo.title()}...\n")
            self.graph_builder.show_route(start, end, algo=algo)

    def show_route(self):
        """Wrapper for route display"""
        self.run_algorithm()

    def search_from_location(self):
        """Handle 'From' location search"""
        self.locator.search_location(self.entry_from.get(), "FROM")

    def search_to_location(self):
        """Handle 'To' location search"""
        self.locator.search_location(self.entry_to.get(), "TO")

    def on_closing(self):
        """Handle window close event"""
        self.main.destroy()

    def run(self):
        """Start application main loop"""
        self.main.mainloop()


if __name__ == '__main__':
    app = MapViewApp()
    app.run()
