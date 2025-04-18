import customtkinter as ctk
import tkintermapview
from tkinter import TclError
from Main.autocomplete import AutocompleteEntry
from Main.locate import LocationHandler
from Main.graph_builder import GraphBuilder

class MapViewApp:
    def __init__(self):
        # Configure appearance before any widgets are created
        ctk.set_appearance_mode("Dark")  # or "Light" based on user preference
        ctk.set_default_color_theme("dark-blue")

        # Define default theme colors
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

        # Attempt to use custom font, fallback if unavailable
        try:
            self.default_font = ("Winky-Sans", 12, "bold")
        except TclError:
            self.default_font = ("Helvetica", 12, "bold")

        # Build main window and widgets
        self.build_main_window()
        self.selected_algorithm = ctk.StringVar(value="none")
        self.theme_var = ctk.StringVar(value="Dark")
        self.build_widgets()

        # Initialize GraphBuilder and LocationHandler
        self.graph_builder = GraphBuilder(self.map_viewer)
        self.locator = LocationHandler(self.map_viewer, self.text_output)

    def build_main_window(self):
        self.main = ctk.CTk()
        self.main.title("Shortest Path Comparison: Dijkstra vs Bellman-Ford")
        self.main.geometry("1000x600")
        self.main.configure(fg_color=self.default_colors["Dark"]["fg"])
        self.main.protocol("WM_DELETE_WINDOW", self.on_closing)

    def build_widgets(self):
        self.build_entry_widgets()
        self.build_theme_selector()
        self.build_map_container()
        self.build_buttons()
        self.build_text_output()

    def build_entry_widgets(self):
        colors = self.default_colors["Dark"]
        self.entry_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        self.entry_frame.place(x=293, y=33)

        # "From" location input
        self.from_frame = ctk.CTkFrame(self.entry_frame, fg_color="transparent")
        self.from_frame.pack(pady=5)
        self.entry_from = AutocompleteEntry(
            master=self.from_frame,
            placeholder_text="From Location",
            width=400, height=40,
            font=self.default_font,
            border_color=colors["border"], border_width=1
        )
        self.entry_from.pack(side="left", fill="x", expand=True)
        self.search_from_button = ctk.CTkButton(
            master=self.from_frame,
            text="🔍", width=40, height=40,
            font=self.default_font,
            fg_color=colors["button"], hover_color=colors["hover"],
            command=self.search_from_location
        )
        self.search_from_button.pack(side="left")

        # "To" location input
        self.to_frame = ctk.CTkFrame(self.entry_frame, fg_color="transparent")
        self.to_frame.pack(pady=5)
        self.entry_to = AutocompleteEntry(
            master=self.to_frame,
            placeholder_text="To Location",
            width=400, height=40,
            font=self.default_font,
            border_color=colors["border"], border_width=1
        )
        self.entry_to.pack(side="left", fill="x", expand=True)
        self.search_to_button = ctk.CTkButton(
            master=self.to_frame,
            text="🔍", width=40, height=40,
            font=self.default_font,
            fg_color=colors["button"], hover_color=colors["hover"],
            command=self.search_to_location
        )
        self.search_to_button.pack(side="left")

        # Tag entries for event handling
        self.entry_from.location_type = "FROM"
        self.entry_to.location_type = "TO"
        self.entry_from.bind("<<LocationSelected>>", self.handle_location_selected)
        self.entry_to.bind("<<LocationSelected>>", self.handle_location_selected)
        self.entry_from.bind("<Return>", lambda e: self.search_from_location())
        self.entry_to.bind("<Return>", lambda e: self.search_to_location())

    def build_theme_selector(self):
        self.theme_selector = ctk.CTkOptionMenu(
            master=self.main,
            values=list(self.default_colors.keys()),
            variable=self.theme_var,
            command=self.handle_theme,
            width=120, height=30,
            font=self.default_font
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
        # Play button
        self.play_button = ctk.CTkButton(
            master=self.main,
            text="▶", width=40, height=100,
            font=("Helvetica", 20, "bold"),
            fg_color=colors["button"], hover_color=colors["hover"],
            state="disabled",
            command=self.run_algorithm
        )
        self.play_button.place(x=203, y=151)

        # Algorithm selection & show graph
        self.button_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        self.button_frame.place(x=785, y=174)
        self.button_dijkstra = ctk.CTkButton(
            master=self.button_frame,
            text="Dijkstra's", width=150, height=40,
            font=self.default_font,
            command=lambda: self.select_algorithm("dijkstra")
        )
        self.button_dijkstra.pack(pady=5)
        self.button_bellman = ctk.CTkButton(
            master=self.button_frame,
            text="Bellman-Ford", width=150, height=40,
            font=self.default_font,
            command=lambda: self.select_algorithm("bellman-ford")
        )
        self.button_bellman.pack(pady=5)
        self.show_graph_button = ctk.CTkButton(
            master=self.button_frame,
            text="Show Route", width=150, height=40,
            font=self.default_font,
            fg_color=colors["button"], hover_color=colors["hover"],
            command=self.show_route
        )
        self.show_graph_button.pack(pady=5)

    def build_text_output(self):
        colors = self.default_colors["Dark"]
        self.text_output = ctk.CTkTextbox(
            master=self.main,
            width=455, height=120,
            font=self.default_font,
            border_color=colors["border"], border_width=colors["text_border_width"]
        )
        self.text_output.place(x=287, y=452)

    def handle_location_selected(self, event):
        if event.widget.location_type == "FROM":
            self.search_from_location()
        else:
            self.search_to_location()

    def select_algorithm(self, algo_name):
        self.selected_algorithm.set(algo_name)
        self.play_button.configure(state="normal")
        theme_colors = self.default_colors[self.theme_var.get()]
        selected_color = theme_colors["button"]
        # Highlight selected button
        self.button_dijkstra.configure(
            fg_color=selected_color if algo_name == "dijkstra" else theme_colors["border"]
        )
        self.button_bellman.configure(
            fg_color=selected_color if algo_name == "bellman-ford" else theme_colors["border"]
        )

    def run_algorithm(self):
        marker_from = self.locator.markers.get("FROM")
        marker_to   = self.locator.markers.get("TO")
        if not (marker_from and marker_to):
            self.text_output.insert("end", "Both FROM and TO locations must be set.\n")
            return
        start = marker_from.position
        end   = marker_to.position
        # show route (builds graph, finds path, draws)
        self.text_output.insert("end", f"Routing from {start} to {end}...\n")
        self.graph_builder.show_route(start, end,
                                      algo=self.selected_algorithm.get())

    def show_route(self):
        # treat Show Route button same as run_algorithm
        self.run_algorithm()

    def search_from_location(self):
        self.locator.search_location(self.entry_from.get(), "FROM")

    def search_to_location(self):
        self.locator.search_location(self.entry_to.get(), "TO")

    def handle_theme(self, theme):
        ctk.set_appearance_mode("light" if theme == "Barbie" else theme.lower())
        colors = self.default_colors[theme]
        self.main.configure(fg_color=colors["fg"])
        self.map_frame.configure(fg_color=colors["entry_fg"])
        self.entry_frame.configure(fg_color="transparent")
        self.button_frame.configure(fg_color="transparent")
        for w in [self.entry_from, self.entry_to]:
            w.configure(fg_color=colors["entry_fg"], border_color=colors["border"], border_width=colors["text_border_width"])
        for btn in [self.play_button, self.button_dijkstra, self.button_bellman, self.search_from_button, self.search_to_button, self.show_graph_button]:
            btn.configure(fg_color=colors["button"], hover_color=colors["hover"])
        self.text_output.configure(fg_color=colors["entry_fg"], border_color=colors["border"], border_width=colors["text_border_width"])

    def on_closing(self):
        self.main.destroy()

    def run(self):
        self.main.mainloop()

if __name__ == '__main__':
    app = MapViewApp()
    app.run()
