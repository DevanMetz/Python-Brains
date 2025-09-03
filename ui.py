"""
Manages the user interface components of the application.

This module contains classes that encapsulate the different UI screens and
panels, such as the AI Design Menu and the main Simulation UI. These classes
are responsible for creating, managing, and handling interactions with all the
`pygame_gui` elements like buttons, sliders, and text boxes.
"""
import pygame
import pygame_gui

class DesignMenu:
    """
    Manages the UI elements for the AI Design Menu screen.
    NOTE: This menu is now simplified as the simulation uses a fixed architecture.
    """
    def __init__(self, rect, ui_manager):
        """Initializes the DesignMenu."""
        self.panel = pygame_gui.elements.UIPanel(
            relative_rect=rect,
            manager=ui_manager,
            visible=False
        )

        # --- Simplified UI Elements ---
        self.title_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 10), (280, 30)),
            text='Simulation Controls',
            manager=ui_manager,
            container=self.panel
        )

        self.info_textbox = pygame_gui.elements.UITextBox(
            relative_rect=pygame.Rect((10, 50), (280, 180)),
            html_text="The simulation has been refactored to a simpler, grid-based model.<br><br>"
                      "Units are now tiles that move one step at a time in 8 directions.<br><br>"
                      "The AI architecture and whisker configuration are now fixed. Use the 'Reset Population' button to start a new training run with random brains.",
            manager=ui_manager,
            container=self.panel
        )

        # --- Action Buttons ---
        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 240), (280, 40)),
            text='Reset Population',
            manager=ui_manager,
            container=self.panel
        )
        self.load_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 290), (280, 40)),
            text='Load Last Saved Brain',
            manager=ui_manager,
            container=self.panel
        )
        self.close_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, 340), (280, 40)),
            text='Close',
            manager=ui_manager,
            container=self.panel
        )

    def show(self):
        """Makes the design menu panel visible."""
        self.panel.show()

    def hide(self):
        """Hides the design menu panel."""
        self.panel.hide()


class SimulationUI:
    """
    Manages the UI elements for the main simulation view.

    This class builds and controls the HUD (Heads-Up Display) for the main
    simulation screen, including buttons for navigating to other menus,
    changing training modes, and sliders for controlling simulation parameters
    like population size and simulation speed.
    """
    def __init__(self, world_width, world_height, ui_manager, initial_params):
        """Initializes the SimulationUI.

        Args:
            world_width (int): The width of the game world, used for positioning.
            world_height (int): The height of the game world, used for positioning.
            ui_manager (pygame_gui.UIManager): The main UI manager.
            initial_params (dict): A dictionary containing the initial values
                for the UI sliders, such as 'population_size'.
        """
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
        """Makes all simulation UI elements visible."""
        for element in self.elements:
            element.show()

    def hide(self):
        """Hides all simulation UI elements."""
        for element in self.elements:
            element.hide()

    def update_drawn_units_range(self, max_value):
        """Dynamically updates the range of the 'Drawn Units' slider.

        This is called when the total population size changes, to ensure that
        the user cannot select to draw more units than actually exist.

        Args:
            max_value (int): The new maximum value for the slider, typically
                the new population size.
        """
        current_val = self.drawn_units_slider.get_current_value()
        self.drawn_units_slider.value_range = (1, max_value)
        # Clamp the current value to the new range
        if current_val > max_value:
            self.drawn_units_slider.set_current_value(max_value)
            self.drawn_units_value_label.set_text(str(max_value))
