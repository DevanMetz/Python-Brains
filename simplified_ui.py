"""
This file defines the UI for the simplified simulation.
It uses pygame_gui to create sliders, buttons, and text inputs.
"""
import pygame
import pygame_gui

class SimplifiedUI:
    """A class to manage all UI elements for the simplified simulation."""
    def __init__(self, rect, manager):
        self.manager = manager
        self.rect = rect

        self.panel = pygame_gui.elements.UIPanel(
            relative_rect=rect,
            manager=manager,
            starting_layer_height=1
        )

        # --- UI Elements ---
        y_pos = 10

        # Mode Toggle Button
        self.mode_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 20, 30)),
            text='Enter Editor Mode',
            manager=manager,
            container=self.panel
        )
        y_pos += 40

        # Vision Radius Slider
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 20, 20)),
            text='Vision Radius:', manager=manager, container=self.panel
        )
        y_pos += 25
        self.vision_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 20, 20)),
            start_value=1, value_range=(1, 5), manager=manager, container=self.panel
        )
        self.vision_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((rect.width - 40, y_pos - 25), (30, 20)),
            text='1', manager=manager, container=self.panel
        )
        y_pos += 30

        # Sim Speed Slider
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 20, 20)),
            text='Sim Speed:', manager=manager, container=self.panel
        )
        y_pos += 25
        self.speed_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 20, 20)),
            start_value=1, value_range=(1, 32), manager=manager, container=self.panel
        )
        self.speed_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((rect.width - 40, y_pos - 25), (30, 20)),
            text='1x', manager=manager, container=self.panel
        )
        y_pos += 30

        # Sim Length Slider
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 20, 20)),
            text='Sim Length (steps):', manager=manager, container=self.panel
        )
        y_pos += 25
        self.length_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 20, 20)),
            start_value=100, value_range=(50, 1000), manager=manager, container=self.panel
        )
        self.length_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((rect.width - 50, y_pos - 25), (40, 20)),
            text='100', manager=manager, container=self.panel
        )
        y_pos += 40

        # MLP Arch Text Input
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 20, 20)),
            text='MLP Hidden Layers:', manager=manager, container=self.panel
        )
        y_pos += 25
        self.mlp_arch_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 20, 30)),
            manager=manager, container=self.panel
        )
        self.mlp_arch_input.set_text("16")
        y_pos += 40

        # Apply Button
        self.apply_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 20, 30)),
            text='Apply & Restart',
            manager=manager,
            container=self.panel
        )

    def get_current_settings(self):
        """Returns a dictionary of the current UI settings."""
        return {
            'vision_radius': int(self.vision_slider.get_current_value()),
            'sim_speed': int(self.speed_slider.get_current_value()),
            'sim_length': int(self.length_slider.get_current_value()),
            'mlp_arch_str': self.mlp_arch_input.get_text()
        }

    def update_labels(self):
        """Updates the text labels next to the sliders."""
        self.vision_label.set_text(str(int(self.vision_slider.get_current_value())))
        self.speed_label.set_text(f"{int(self.speed_slider.get_current_value())}x")
        self.length_label.set_text(str(int(self.length_slider.get_current_value())))

    def show_simulation_ui(self):
        self.mode_button.set_text('Enter Editor Mode')
        self.apply_button.show()
        self.vision_slider.show()
        self.speed_slider.show()
        self.length_slider.show()
        self.mlp_arch_input.show()
        # Also show all the labels associated with the controls
        for child in self.panel.get_container().elements:
             if 'label' in child.element_ids or 'text_entry' in str(type(child)):
                child.show()

    def show_editor_ui(self):
        self.mode_button.set_text('Back to Simulation')
        self.apply_button.hide()
        self.vision_slider.hide()
        self.speed_slider.hide()
        self.length_slider.hide()
        self.mlp_arch_input.hide()
        # Also hide all the labels associated with the controls
        for child in self.panel.get_container().elements:
            if 'label' in child.element_ids or 'text_entry' in str(type(child)):
                if child not in [self.mode_button]: # Keep the mode button visible
                     child.hide()
