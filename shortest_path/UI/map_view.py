import customtkinter as ctk
import tkintermapview
from tkinter import TclError
from Main.autocomplete import AutocompleteEntry
from Main.locate import LocationHandler
from Main.graph_builder import OptimizedGraphBuilder
from Main.Algorithms.algorithmvisualizer import AlgorithmVisualizer  # New import


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
        self.main.geometry("1200x800")  # Increased window size for better layout
        self.main.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Theme control variables
        self.selected_algorithm = ctk.StringVar(master=self.main, value="none")
        self.theme_var = ctk.StringVar(master=self.main, value="Dark")
        self.show_all_var = ctk.BooleanVar(master=self.main, value=False)

        # Track application state
        self.graph_loaded = False
        self.dijkstra_results = None
        self.bellman_ford_results = None

        # Initialize UI
        current_theme = self.theme_var.get()
        self.main.configure(fg_color=self.default_colors[current_theme]["fg"])
        self.build_widgets()

        # Initialize services (after map_viewer exists)
        self.graph_builder = OptimizedGraphBuilder(self.map_viewer)
        self.locator = LocationHandler(self.map_viewer, self.text_output)

        # Initialize algorithm visualizer
        self.algorithm_visualizer = AlgorithmVisualizer(self.main, self.theme_var, self.default_colors)

    def build_widgets(self):
        """Build all application widgets using a grid layout"""
        # Create main frames for organization
        self.left_panel = ctk.CTkFrame(self.main, fg_color="transparent")
        self.center_panel = ctk.CTkFrame(self.main, fg_color="transparent")
        self.right_panel = ctk.CTkFrame(self.main, fg_color="transparent")

        # Place panels in a 3-column layout
        self.left_panel.pack(side="left", fill="y", padx=15, pady=15)
        self.center_panel.pack(side="left", fill="both", expand=True, padx=15, pady=15)
        self.right_panel.pack(side="right", fill="y", padx=15, pady=15)

        # Build components in each panel
        self.build_theme_selector()
        self.build_option_controls()
        self.build_entry_widgets()
        self.build_map_container()
        self.build_buttons()
        self.build_text_output()

    def build_map_container(self):
        """Build map widget container with theme support"""
        # Use the canvas background color instead of transparent
        config = self.default_colors[self.theme_var.get()]

        self.map_container = ctk.CTkFrame(self.center_panel, fg_color=config["canvas_bg"])
        self.map_container.pack(fill="both", expand=True)

        self.map_frame = ctk.CTkFrame(
            self.map_container,
            fg_color=config["canvas_bg"]
        )
        self.map_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Initialize map with current theme
        self.map_viewer = tkintermapview.TkinterMapView(
            master=self.map_frame,
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
        selection_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        selection_frame.pack(fill="x", pady=(0, 15))

        theme_label = ctk.CTkLabel(
            selection_frame,
            text="Theme:",
            font=self.default_font
        )
        theme_label.pack(side="left", padx=(0, 10))

        self.theme_selector = ctk.CTkOptionMenu(
            master=selection_frame,
            values=list(self.default_colors.keys()),
            variable=self.theme_var,
            command=self.handle_theme,
            width=120,
            height=30,
            font=self.default_font
        )
        self.theme_selector.pack(side="left", fill="x", expand=True)

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
            self.run_algorithm_button,
            self.button_dijkstra,
            self.button_bellman,
            self.button_compare,  # Added the new compare button
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

        # Location entries header
        location_header = ctk.CTkLabel(
            self.center_panel,
            text="Route Planning",
            font=("Helvetica", 16, "bold")
        )
        location_header.pack(anchor="w", pady=(0, 10))

        self.entry_frame = ctk.CTkFrame(self.center_panel, fg_color="transparent")
        self.entry_frame.pack(fill="x", pady=(0, 10))

        # From location entry
        self.from_frame = ctk.CTkFrame(self.entry_frame, fg_color="transparent")
        self.from_frame.pack(fill="x", pady=5)

        from_label = ctk.CTkLabel(
            self.from_frame,
            text="From:",
            font=self.default_font,
            width=60
        )
        from_label.pack(side="left", padx=(0, 10))

        self.entry_from = AutocompleteEntry(
            master=self.from_frame,
            placeholder_text="Enter starting location",
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
        self.search_from_button.pack(side="left", padx=(10, 0))

        # To location entry
        self.to_frame = ctk.CTkFrame(self.entry_frame, fg_color="transparent")
        self.to_frame.pack(fill="x", pady=5)

        to_label = ctk.CTkLabel(
            self.to_frame,
            text="To:",
            font=self.default_font,
            width=60
        )
        to_label.pack(side="left", padx=(0, 10))

        self.entry_to = AutocompleteEntry(
            master=self.to_frame,
            placeholder_text="Enter destination",
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
        self.search_to_button.pack(side="left", padx=(10, 0))

        # Event bindings
        self.entry_from.location_type = "FROM"
        self.entry_to.location_type = "TO"
        self.entry_from.bind("<<LocationSelected>>", self.handle_location_selected)
        self.entry_to.bind("<<LocationSelected>>", self.handle_location_selected)
        self.entry_from.bind("<Return>", lambda e: self.search_from_location())
        self.entry_to.bind("<Return>", lambda e: self.search_to_location())

    def build_option_controls(self):
        """Build additional options controls"""
        options_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        options_frame.pack(fill="x", pady=(0, 15))

        options_label = ctk.CTkLabel(
            options_frame,
            text="Options",
            font=("Helvetica", 14, "bold")
        )
        options_label.pack(anchor="w", pady=(0, 10))

        self.show_all_checkbox = ctk.CTkCheckBox(
            master=options_frame,
            text="Show All Paths",
            variable=self.show_all_var,
            font=self.default_font
        )
        self.show_all_checkbox.pack(anchor="w")

    def build_buttons(self):
        """Build action buttons"""
        config = self.default_colors[self.theme_var.get()]

        # Algorithm section
        algo_label = ctk.CTkLabel(
            self.right_panel,
            text="Algorithms",
            font=("Helvetica", 14, "bold")
        )
        algo_label.pack(anchor="w", pady=(0, 10))

        # Algorithm selection buttons (initially disabled)
        self.button_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.button_frame.pack(fill="x")

        self.button_dijkstra = ctk.CTkButton(
            master=self.button_frame,
            text="Dijkstra's",
            width=150,
            height=40,
            font=self.default_font,
            fg_color=config["button"],
            hover_color=config["hover"],
            command=lambda: self.select_algorithm("dijkstra"),
            state="disabled"  # Initially disabled
        )
        self.button_dijkstra.pack(fill="x", pady=5)

        self.button_bellman = ctk.CTkButton(
            master=self.button_frame,
            text="Bellman-Ford",
            width=150,
            height=40,
            font=self.default_font,
            fg_color=config["button"],
            hover_color=config["hover"],
            command=lambda: self.select_algorithm("bellman-ford"),
            state="disabled"  # Initially disabled
        )
        self.button_bellman.pack(fill="x", pady=5)

        # NEW: Compare algorithms button
        self.button_compare = ctk.CTkButton(
            master=self.button_frame,
            text="Compare Algorithms",
            width=150,
            height=40,
            font=self.default_font,
            fg_color=config["button"],
            hover_color=config["hover"],
            command=self.compare_algorithms,
            state="disabled"  # Initially disabled
        )
        self.button_compare.pack(fill="x", pady=5)

        # Action buttons
        action_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        action_frame.pack(fill="x", pady=(20, 0))

        action_label = ctk.CTkLabel(
            action_frame,
            text="Actions",
            font=("Helvetica", 14, "bold")
        )
        action_label.pack(anchor="w", pady=(0, 10))

        # Run Algorithm button - for after selection
        self.run_algorithm_button = ctk.CTkButton(
            master=action_frame,
            text="Run Algorithm ▶",
            width=150,
            height=40,
            font=self.default_font,
            fg_color=config["button"],
            hover_color=config["hover"],
            state="disabled",
            command=self.execute_algorithm
        )
        self.run_algorithm_button.pack(fill="x", pady=5)

        # Show Graph button (renamed from "Show Route")
        self.show_graph_button = ctk.CTkButton(
            master=action_frame,
            text="Show Graph",  # Renamed to better reflect purpose
            width=150,
            height=40,
            font=self.default_font,
            fg_color=config["button"],
            hover_color=config["hover"],
            command=self.show_graph  # New method
        )
        self.show_graph_button.pack(fill="x", pady=5)

    def build_text_output(self):
        """Build text output widget"""
        config = self.default_colors[self.theme_var.get()]

        output_label = ctk.CTkLabel(
            self.center_panel,
            text="Results",
            font=("Helvetica", 14, "bold")
        )
        output_label.pack(anchor="w", pady=(10, 5))

        self.text_output = ctk.CTkTextbox(
            master=self.center_panel,
            height=150,
            font=self.default_font,
            border_color=config["border"],
            border_width=config["text_border_width"]
        )
        self.text_output.pack(fill="x", expand=False)

    def handle_location_selected(self, event):
        """Handle location selection event"""
        if event.widget.location_type == "FROM":
            self.search_from_location()
        else:
            self.search_to_location()

    def select_algorithm(self, algo_name):
        """Select routing algorithm"""
        self.selected_algorithm.set(algo_name)
        self.run_algorithm_button.configure(state="normal")
        theme_colors = self.default_colors[self.theme_var.get()]
        selected_color = theme_colors["hover"]
        unselected_color = theme_colors["button"]
        self.button_dijkstra.configure(
            fg_color=selected_color if algo_name == "dijkstra" else unselected_color
        )
        self.button_bellman.configure(
            fg_color=selected_color if algo_name == "bellman-ford" else unselected_color
        )

    def show_graph(self):
        """Display the graph for the selected route without running algorithm"""
        marker_from = self.locator.markers.get("FROM")
        marker_to = self.locator.markers.get("TO")
        if not (marker_from and marker_to):
            self.text_output.delete("1.0", "end")
            self.text_output.insert("end", "⚠️ Both FROM and TO locations must be set.\n")
            return

        start = marker_from.position
        end = marker_to.position

        # Clear previous output
        self.text_output.delete("1.0", "end")
        self.text_output.insert("end", f"Building graph from {start} to {end}...\n")

        # Build the graph but don't run any algorithm yet
        try:
            # Build the graph and display it
            self.graph_builder.build_graph(start, end)
            self.graph_builder.display_graph()

            # Update UI state
            self.graph_loaded = True
            self.text_output.insert("end",
                                    "Graph loaded successfully. Select an algorithm to find the shortest path.\n")

            # Enable algorithm selection buttons
            self.button_dijkstra.configure(state="normal")
            self.button_bellman.configure(state="normal")

            # Reset results when a new graph is loaded
            self.dijkstra_results = None
            self.bellman_ford_results = None
            self.button_compare.configure(state="disabled")

        except Exception as e:
            self.text_output.insert("end", f"Error building graph: {str(e)}\n")
            self.graph_loaded = False

    def execute_algorithm(self):
        """Execute the selected algorithm on the loaded graph"""
        if not self.graph_loaded:
            self.text_output.delete("1.0", "end")
            self.text_output.insert("end", "⚠️ Please load the graph first using 'Show Graph'.\n")
            return

        algo = self.selected_algorithm.get()
        if algo == "none":
            self.text_output.delete("1.0", "end")
            self.text_output.insert("end", "⚠️ Select an algorithm first.\n")
            return

        self.text_output.delete("1.0", "end")
        self.text_output.insert("end", f"Running {algo.title()} algorithm...\n")

        try:
            # Run the selected algorithm
            if algo == "dijkstra":
                path, explanation, details = self.graph_builder.run_dijkstra()
                self.dijkstra_results = details
            elif algo == "bellman-ford":
                # Fixed method name (it was defined but not called correctly)
                path, explanation, details = self.graph_builder.run_bellman_ford()
                self.bellman_ford_results = details

            # Insert the step-by-step explanation
            if explanation:
                self.text_output.insert("end", explanation)
            else:
                self.text_output.insert("end", "No path found or no explanation available.\n")

            # Enable comparison button if both algorithms have been run
            if self.dijkstra_results and self.bellman_ford_results:
                self.button_compare.configure(state="normal")

        except Exception as e:
            self.text_output.insert("end", f"Error running algorithm: {str(e)}\n")

    def compare_algorithms(self):
        """Open the comparison visualization window"""
        if not (self.dijkstra_results and self.bellman_ford_results):
            self.text_output.delete("1.0", "end")
            self.text_output.insert("end", "⚠️ Run both algorithms first to compare results.\n")
            return

        # Show the comparison visualization
        self.algorithm_visualizer.show_comparison(
            self.graph_builder.G_simple,  # Changed from G to G_simple
            self.dijkstra_results,  # Dijkstra results
            self.bellman_ford_results,  # Bellman-Ford results
            self.locator.markers.get("FROM").position,  # Source coordinates
            self.locator.markers.get("TO").position,  # Target coordinates
        )

    def search_from_location(self):
        """Handle 'From' location search"""
        self.locator.search_location(self.entry_from.get(), "FROM")
        # Reset graph state when locations change
        self.graph_loaded = False
        self.button_dijkstra.configure(state="disabled")
        self.button_bellman.configure(state="disabled")
        self.button_compare.configure(state="disabled")
        self.run_algorithm_button.configure(state="disabled")
        self.dijkstra_results = None
        self.bellman_ford_results = None

    def search_to_location(self):
        """Handle 'To' location search"""
        self.locator.search_location(self.entry_to.get(), "TO")
        # Reset graph state when locations change
        self.graph_loaded = False
        self.button_dijkstra.configure(state="disabled")
        self.button_bellman.configure(state="disabled")
        self.button_compare.configure(state="disabled")
        self.run_algorithm_button.configure(state="disabled")
        self.dijkstra_results = None
        self.bellman_ford_results = None

    def on_closing(self):
        """Handle window close event"""
        self.main.destroy()

    def run(self):
        """Start application main loop"""
        self.main.mainloop()


if __name__ == '__main__':
    app = MapViewApp(api_key="your-stadia-maps-api-key-here")
    app.run()