"""
This file defines the UI for the simplified simulation.
It uses pygame_gui to create sliders, buttons, and text inputs.
"""
import pygame
import pygame_gui
from mlp_visualizer import draw_mlp

class SimplifiedUI:
    """A class to manage all UI elements for the simplified simulation."""
    def __init__(self, rect, manager):
        self.manager = manager
        self.rect = rect

        # --- Main Controls Panel ---
        controls_height = 450
        self.controls_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect(rect.topleft, (rect.width, controls_height)),
            manager=manager, starting_height=1
        )

        y_pos = 10
        self.mode_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 20, 30)),
            text='Enter Editor Mode', manager=manager, container=self.controls_panel
        )
        y_pos += 40

        self.sliders = {}
        self.slider_labels = {}
        def create_slider(name, text, y, min_val, max_val, start_val, label_format):
            pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect((10, y), (rect.width - 20, 20)),
                text=text, manager=manager, container=self.controls_panel)
            y += 25
            slider = pygame_gui.elements.UIHorizontalSlider(
                relative_rect=pygame.Rect((10, y), (rect.width - 20, 20)),
                start_value=start_val, value_range=(min_val, max_val),
                manager=manager, container=self.controls_panel)
            label = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect((rect.width - 55, y - 25), (45, 20)),
                text=label_format.format(start_val), manager=manager, container=self.controls_panel)
            self.sliders[name] = slider
            self.slider_labels[name] = (label, label_format)
            return y + 30
        y_pos = create_slider('vision_radius', 'Vision Radius:', y_pos, 1, 5, 1, '{}')
        y_pos = create_slider('sim_speed', 'Sim Speed:', y_pos, 1, 32, 1, '{}x')
        y_pos = create_slider('sim_length', 'Sim Length:', y_pos, 50, 1000, 100, '{}')
        y_pos = create_slider('population_size', 'Population:', y_pos, 10, 500, 100, '{}')
        y_pos = create_slider('mutation_rate', 'Mutation Rate:', y_pos, 0, 0.2, 0.05, '{:.2f}')
        y_pos += 10
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 20, 20)),
            text='MLP Hidden Layers:', manager=manager, container=self.controls_panel)
        y_pos += 25
        self.mlp_arch_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 20, 30)),
            manager=manager, container=self.controls_panel)
        self.mlp_arch_input.set_text("16")
        y_pos += 40
        self.apply_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 20, 30)),
            text='Apply & Restart', manager=manager, container=self.controls_panel)

        # --- MLP Visualizer Panel ---
        visualizer_height = rect.height - controls_height - 10
        self.visualizer_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((rect.left, rect.top + controls_height + 10), (rect.width, visualizer_height)),
            manager=manager, starting_height=1
        )

    def get_current_settings(self):
        settings = {name: slider.get_current_value() for name, slider in self.sliders.items()}
        settings['mlp_arch_str'] = self.mlp_arch_input.get_text()
        settings['vision_radius'] = int(settings['vision_radius'])
        settings['sim_speed'] = int(settings['sim_speed'])
        settings['sim_length'] = int(settings['sim_length'])
        settings['population_size'] = int(settings['population_size'])
        return settings

    def update_labels(self):
        for name, (label, label_format) in self.slider_labels.items():
            value = self.sliders[name].get_current_value()
            label.set_text(label_format.format(value))

    def draw_fittest_brain(self, brain):
        # Clear the panel first by filling it with its background color
        self.visualizer_panel.image.fill(self.visualizer_panel.ui_theme.get_colour('dark_bg'))
        draw_mlp(self.visualizer_panel.image, brain)

    def show_simulation_ui(self):
        self.mode_button.set_text('Enter Editor Mode')
        for child in self.controls_panel.get_container().elements:
            child.show()

    def show_editor_ui(self):
        self.mode_button.set_text('Back to Simulation')
        for child in self.controls_panel.get_container().elements:
            if child != self.mode_button:
                child.hide()
