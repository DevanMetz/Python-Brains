"""
This file defines the UI for the simplified simulation.
It uses pygame_gui to create sliders, buttons, and text inputs.
"""
import pygame
import pygame_gui
from mlp_visualizer import draw_mlp
from simplified_game import Tile

class SimplifiedUI:
    """A class to manage all UI elements for the simplified simulation."""
    def __init__(self, rect, manager):
        self.manager = manager
        self.simulation_controls = []
        self.editor_controls = []
        self.brush_map = {}

        self.controls_panel = pygame_gui.elements.UIScrollingContainer(
            relative_rect=rect, manager=manager
        )

        y_pos = 10
        self.mode_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            text='Enter Editor Mode', manager=manager, container=self.controls_panel
        )
        y_pos += 40

        # --- Simulation Controls ---
        self.pause_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            text='Pause', manager=manager, container=self.controls_panel
        )
        self.simulation_controls.append(self.pause_button)
        y_pos += 40
        self.restart_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            text='Restart', manager=manager, container=self.controls_panel
        )
        self.simulation_controls.append(self.restart_button)
        y_pos += 40
        self.fast_forward_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            text='Run 10 Gens Fast', manager=manager, container=self.controls_panel
        )
        self.simulation_controls.append(self.fast_forward_button)
        y_pos += 40

        self.sliders = {}
        self.slider_labels = {}

        def create_slider(name, text, y, min_val, max_val, start_val, label_format):
            label_title = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect((10, y), (rect.width - 40, 20)),
                text=text, manager=manager, container=self.controls_panel)
            y += 25
            slider = pygame_gui.elements.UIHorizontalSlider(
                relative_rect=pygame.Rect((10, y), (rect.width - 40, 20)),
                start_value=start_val, value_range=(min_val, max_val),
                manager=manager, container=self.controls_panel)
            label_val = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect((rect.width - 75, y - 25), (45, 20)),
                text=label_format.format(start_val), manager=manager, container=self.controls_panel)
            self.sliders[name] = slider
            self.slider_labels[name] = (label_val, label_format)
            self.simulation_controls.extend([label_title, slider, label_val])
            return y + 30

        y_pos = create_slider('sps', 'Steps Per Second:', y_pos, 1, 1000, 60, '{}')
        y_pos = create_slider('sim_length', 'Sim Length:', y_pos, 50, 1000, 100, '{}')
        y_pos = create_slider('population_size', 'Population:', y_pos, 10, 500, 100, '{}')
        y_pos = create_slider('mutation_rate', 'Mutation Rate:', y_pos, 0, 0.2, 0.05, '{:.2f}')
        y_pos = create_slider('attack_awareness_radius', 'Attack Awareness:', y_pos, 1, 10, 2, '{}')
        y_pos += 40

        mlp_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 20)),
            text='MLP Hidden Layers:', manager=manager, container=self.controls_panel)
        y_pos += 25
        self.mlp_arch_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            manager=manager, container=self.controls_panel)
        self.mlp_arch_input.set_text("16")
        y_pos += 40
        self.apply_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            text='Apply Settings', manager=manager, container=self.controls_panel)
        self.simulation_controls.extend([mlp_label, self.mlp_arch_input, self.apply_button])
        y_pos += 40

        # --- Editor Controls ---
        editor_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 20)),
            text='--- Editor Brushes ---', manager=manager, container=self.controls_panel)
        y_pos += 30
        self.editor_controls.append(editor_label)

        brushes = {
            "Wall": Tile.WALL, "Empty": Tile.EMPTY, "Resource": Tile.RESOURCE,
            "Dropoff": Tile.DROPOFF, "Enemy": Tile.ENEMY
        }
        for text, tile in brushes.items():
            button = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
                text=text, manager=manager, container=self.controls_panel)
            self.editor_controls.append(button)
            self.brush_map[button] = tile
            y_pos += 40

        self.save_map_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            text='Save Map', manager=manager, container=self.controls_panel)
        y_pos += 40
        self.load_map_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            text='Load Map', manager=manager, container=self.controls_panel)
        self.editor_controls.extend([self.save_map_button, self.load_map_button])
        y_pos += 40

        self.controls_panel.set_scrollable_area_dimensions((rect.width - 30, y_pos))
        self.show_simulation_ui() # Start in simulation mode

    def get_current_settings(self):
        settings = {name: slider.get_current_value() for name, slider in self.sliders.items()}
        settings['mlp_arch_str'] = self.mlp_arch_input.get_text()
        return settings

    def update_labels(self):
        for name, (label, label_format) in self.slider_labels.items():
            value = self.sliders[name].get_current_value()
            label.set_text(label_format.format(value))

    def draw_fittest_brain(self, surface, brain, activations=None):
        surface.fill(pygame.Color("#303030"))
        input_labels = [
            "HP", "Carry", "Cargo", "Res_dX", "Res_dY",
            "Base_dX", "Base_dY", "Attacked", "En_dX", "En_dY"
        ]
        output_labels = ["To_Res", "Gather", "To_Base", "Flee", "Idle"]
        draw_mlp(surface, brain, input_labels, output_labels, activations)

    def show_simulation_ui(self):
        self.mode_button.set_text('Enter Editor Mode')
        for control in self.simulation_controls:
            control.show()
        for control in self.editor_controls:
            control.hide()

    def show_editor_ui(self):
        self.mode_button.set_text('Back to Simulation')
        for control in self.simulation_controls:
            control.hide()
        for control in self.editor_controls:
            control.show()
