import pygame
import pygame_gui

class DesignMenu:
    """
    Manages the UI elements for the AI Design Menu.
    """
    def __init__(self, rect, ui_manager):
        self.panel = pygame_gui.elements.UIPanel(
            relative_rect=rect,
            manager=ui_manager,
            visible=True # Start visible, then hide later
        )

        # --- UI Elements for Architecture ---
        self.arch_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 10), (280, 25)),
            text='Hidden Layers (comma-separated):',
            manager=ui_manager,
            container=self.panel
        )
        self.arch_text_entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((10, 40), (280, 30)),
            manager=ui_manager,
            container=self.panel
        )
        self.arch_text_entry.set_text("16,8")

        # --- UI Elements for Inputs ---
        self.whisker_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 80), (180, 25)),
            text='Number of Whiskers:',
            manager=ui_manager,
            container=self.panel
        )
        self.whisker_count_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((240, 80), (50, 25)),
            text='7',
            manager=ui_manager,
            container=self.panel
        )
        self.whisker_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, 110), (280, 20)),
            start_value=7,
            value_range=(1, 7),
            manager=ui_manager,
            container=self.panel
        )

        self.whisker_length_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 140), (180, 25)),
            text='Whisker Length:',
            manager=ui_manager,
            container=self.panel
        )
        self.whisker_length_value_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((240, 140), (50, 25)),
            text='150',
            manager=ui_manager,
            container=self.panel
        )
        self.whisker_length_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, 170), (280, 20)),
            start_value=150,
            value_range=(50, 250),
            manager=ui_manager,
            container=self.panel
        )

        # --- UI Elements for Outputs ---
        self.attack_checkbox = pygame_gui.elements.UICheckBox(
            relative_rect=pygame.Rect((10, 200), (280, 25)),
            text="Enable Attack Action",
            manager=ui_manager,
            container=self.panel
        )

        # --- UI Elements for Perception Types ---
        self.sense_wall_checkbox = pygame_gui.elements.UICheckBox(
            relative_rect=pygame.Rect((10, 230), (140, 25)), text="Sense Walls", manager=ui_manager, container=self.panel)
        self.sense_wall_checkbox.checked = True
        self.sense_enemy_checkbox = pygame_gui.elements.UICheckBox(
            relative_rect=pygame.Rect((150, 230), (140, 25)), text="Sense Enemies", manager=ui_manager, container=self.panel)
        self.sense_enemy_checkbox.checked = True
        self.sense_unit_checkbox = pygame_gui.elements.UICheckBox(
            relative_rect=pygame.Rect((10, 260), (140, 25)), text="Sense Friendlies", manager=ui_manager, container=self.panel)
        self.sense_unit_checkbox.checked = True

        # --- Action Buttons ---
        self.update_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 290), (280, 40)),
            text='Create New Population',
            manager=ui_manager,
            container=self.panel
        )
        self.load_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 340), (280, 40)),
            text='Load Last Saved Brain',
            manager=ui_manager,
            container=self.panel
        )
        self.close_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 390), (280, 40)),
            text='Close',
            manager=ui_manager,
            container=self.panel
        )

        # Now hide the panel after all children are created
        self.panel.hide()

    def show(self):
        self.panel.show()

    def hide(self):
        self.panel.hide()

    def get_architecture_from_input(self, input_nodes, output_nodes):
        """
        Parses the text input and returns a valid MLP architecture list.
        Returns None if the input is invalid.
        """
        try:
            hidden_layers_str = self.arch_text_entry.get_text()
            if not hidden_layers_str:
                hidden_nodes = []
            else:
                hidden_nodes = [int(x.strip()) for x in hidden_layers_str.split(',')]

            if any(n <= 0 for n in hidden_nodes):
                print("Error: All layer sizes must be positive integers.")
                return None

            return [input_nodes] + hidden_nodes + [output_nodes]
        except ValueError:
            print("Error: Invalid input for hidden layers. Please use comma-separated integers.")
            return None

    def get_perceivable_types(self):
        """Returns a list of selected type strings."""
        types = []
        if self.sense_wall_checkbox.checked:
            types.append("wall")
        if self.sense_enemy_checkbox.checked:
            types.append("enemy")
        if self.sense_unit_checkbox.checked:
            types.append("unit")
        # "target" is not a player-selectable option for now
        return types


class SimulationUI:
    """
    Manages the UI elements for the main simulation view, including all sliders.
    """
    def __init__(self, world_width, world_height, ui_manager, initial_params):
        self.manager = ui_manager

        # --- Top-right buttons ---
        self.to_design_menu_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((world_width - 220, 20), (200, 40)),
            text='AI Design Menu', manager=self.manager)
        self.save_brain_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((world_width - 220, 70), (200, 40)),
            text='Save Fittest Brain', manager=self.manager)
        self.to_map_editor_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((world_width - 220, 120), (200, 40)),
            text='Map Editor', manager=self.manager)
        self.back_to_sim_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((world_width - 220, 20), (200, 40)),
            text='Back to Simulation', manager=self.manager, visible=False)

        # --- Bottom-left buttons ---
        self.train_nav_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((20, world_height - 60), (150, 40)),
            text='Train Navigation', manager=self.manager)
        self.train_combat_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((180, world_height - 60), (150, 40)),
            text='Train Combat', manager=self.manager)

        # --- Bottom-right sliders ---
        y_pos = world_height - 165 # Starting y-position for the top slider

        # Population Size Slider
        self.pop_size_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((world_width - 230, y_pos), (120, 30)),
            text='Population:', manager=self.manager)
        self.pop_size_value_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((world_width - 110, y_pos), (50, 30)),
            text=str(initial_params['population_size']), manager=self.manager)
        self.pop_size_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((world_width - 230, y_pos + 25), (200, 20)),
            start_value=initial_params['population_size'], value_range=(10, 500), manager=self.manager)

        y_pos += 55 # Increment for next slider

        # Drawn Units Slider
        self.drawn_units_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((world_width - 230, y_pos), (120, 30)),
            text='Drawn Units:', manager=self.manager)
        self.drawn_units_value_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((world_width - 110, y_pos), (50, 30)),
            text=str(initial_params['drawn_units']), manager=self.manager)
        self.drawn_units_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((world_width - 230, y_pos + 25), (200, 20)),
            start_value=initial_params['drawn_units'],
            value_range=(1, initial_params['population_size']), manager=self.manager)

        y_pos += 55 # Increment for next slider

        # Simulation Length Slider
        self.sim_length_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((world_width - 230, y_pos), (80, 30)),
            text='Steps:', manager=self.manager)
        self.sim_length_value_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((world_width - 160, y_pos), (50, 30)),
            text=str(initial_params['steps_per_generation']), manager=self.manager)
        self.sim_length_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((world_width - 230, y_pos + 25), (200, 20)),
            start_value=initial_params['steps_per_generation'], value_range=(100, 2000), manager=self.manager)

        self.elements = [
            self.to_design_menu_button, self.save_brain_button, self.to_map_editor_button,
            self.train_nav_button, self.train_combat_button,
            self.pop_size_label, self.pop_size_value_label, self.pop_size_slider,
            self.drawn_units_label, self.drawn_units_value_label, self.drawn_units_slider,
            self.sim_length_label, self.sim_length_value_label, self.sim_length_slider
        ]

    def show(self):
        for element in self.elements:
            element.show()

    def hide(self):
        for element in self.elements:
            element.hide()

    def update_drawn_units_range(self, max_value):
        """Dynamically updates the range of the 'Drawn Units' slider."""
        current_val = self.drawn_units_slider.get_current_value()
        self.drawn_units_slider.value_range = (1, max_value)
        # Clamp the current value to the new range
        if current_val > max_value:
            self.drawn_units_slider.set_current_value(max_value)
            self.drawn_units_value_label.set_text(str(max_value))
