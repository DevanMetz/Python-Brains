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

        # --- UI Elements for Outputs ---
        self.attack_checkbox = pygame_gui.elements.UICheckBox(
            relative_rect=pygame.Rect((10, 140), (280, 25)),
            text="Enable Attack Action",
            manager=ui_manager,
            container=self.panel
        )

        # --- UI Elements for Perception Types ---
        self.sense_wall_checkbox = pygame_gui.elements.UICheckBox(
            relative_rect=pygame.Rect((10, 170), (140, 25)), text="Sense Walls", manager=ui_manager, container=self.panel)
        self.sense_wall_checkbox.checked = True
        self.sense_enemy_checkbox = pygame_gui.elements.UICheckBox(
            relative_rect=pygame.Rect((150, 170), (140, 25)), text="Sense Enemies", manager=ui_manager, container=self.panel)
        self.sense_enemy_checkbox.checked = True
        self.sense_unit_checkbox = pygame_gui.elements.UICheckBox(
            relative_rect=pygame.Rect((10, 200), (140, 25)), text="Sense Friendlies", manager=ui_manager, container=self.panel)
        self.sense_unit_checkbox.checked = True

        # --- Action Buttons ---
        self.update_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 230), (280, 40)),
            text='Create New Population',
            manager=ui_manager,
            container=self.panel
        )
        self.load_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 280), (280, 40)),
            text='Load Last Saved Brain',
            manager=ui_manager,
            container=self.panel
        )
        self.close_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 330), (280, 40)),
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
