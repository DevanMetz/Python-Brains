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

        self.controls_panel = pygame_gui.elements.UIScrollingContainer(
            relative_rect=rect, manager=manager
        )

        y_pos = 10
        self.mode_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            text='Enter Editor Mode', manager=manager, container=self.controls_panel
        )
        y_pos += 40
        self.pause_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            text='Pause', manager=manager, container=self.controls_panel
        )
        y_pos += 40
        self.restart_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            text='Restart', manager=manager, container=self.controls_panel
        )
        y_pos += 40
        self.fast_forward_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            text='Run 10 Gens Fast', manager=manager, container=self.controls_panel
        )
        y_pos += 40

        self.sliders = {}
        self.slider_labels = {}
        self.dropdowns = {}

        def create_slider(name, text, y, min_val, max_val, start_val, label_format):
            pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect((10, y), (rect.width - 40, 20)),
                text=text, manager=manager, container=self.controls_panel)
            y += 25
            slider = pygame_gui.elements.UIHorizontalSlider(
                relative_rect=pygame.Rect((10, y), (rect.width - 40, 20)),
                start_value=start_val, value_range=(min_val, max_val),
                manager=manager, container=self.controls_panel)
            label = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect((rect.width - 75, y - 25), (45, 20)),
                text=label_format.format(start_val), manager=manager, container=self.controls_panel)
            self.sliders[name] = slider
            self.slider_labels[name] = (label, label_format)
            return y + 30

        y_pos = create_slider('sps', 'Steps Per Second:', y_pos, 1, 1000, 60, '{}')
        y_pos = create_slider('vision_radius', 'Vision Radius:', y_pos, 1, 15, 5, '{}')
        y_pos = create_slider('sim_length', 'Sim Length:', y_pos, 50, 1000, 100, '{}')
        y_pos = create_slider('population_size', 'Population:', y_pos, 10, 500, 100, '{}')
        y_pos = create_slider('mutation_rate', 'Mutation Rate:', y_pos, 0, 0.2, 0.05, '{:.2f}')

        y_pos += 10
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 20)),
            text='--- Reward Settings ---', manager=manager, container=self.controls_panel)
        y_pos += 30

        y_pos = create_slider('proximity_bonus', 'Proximity Bonus:', y_pos, 0, 1.0, 1.0, '{:.2f}')
        y_pos = create_slider('exploration_bonus', 'Exploration Bonus:', y_pos, 0, 1.0, 0.0, '{:.2f}')

        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 20)),
            text='Proximity Function:', manager=manager, container=self.controls_panel)
        y_pos += 25
        prox_options = ['None', 'Inverse Squared', 'Inverse', 'Exponential', 'Logarithmic']
        self.dropdowns['proximity_func'] = pygame_gui.elements.UIDropDownMenu(
            options_list=prox_options, starting_option=prox_options[1],
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            manager=manager, container=self.controls_panel)
        y_pos += 40

        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 20)),
            text='Exploration Function:', manager=manager, container=self.controls_panel)
        y_pos += 25
        expl_options = ['None', 'Linear', 'Square Root']
        self.dropdowns['exploration_func'] = pygame_gui.elements.UIDropDownMenu(
            options_list=expl_options, starting_option=expl_options[1],
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            manager=manager, container=self.controls_panel)
        y_pos += 40

        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 20)),
            text='Reward System:', manager=manager, container=self.controls_panel)
        y_pos += 25
        reward_options = ['Navigation', 'Resource Collection']
        self.dropdowns['reward_system'] = pygame_gui.elements.UIDropDownMenu(
            options_list=reward_options, starting_option=reward_options[0],
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            manager=manager, container=self.controls_panel)
        y_pos += 50

        self.bank_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 20)),
            text='Bank: 0', manager=manager, container=self.controls_panel)
        y_pos += 30

        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 20)),
            text='--- Other Settings ---', manager=manager, container=self.controls_panel)
        y_pos += 30

        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 20)),
            text='MLP Hidden Layers:', manager=manager, container=self.controls_panel)
        y_pos += 25
        self.mlp_arch_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            manager=manager, container=self.controls_panel)
        self.mlp_arch_input.set_text("16")
        y_pos += 40
        self.save_map_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            text='Save Map', manager=manager, container=self.controls_panel)
        y_pos += 40
        self.load_map_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            text='Load Map', manager=manager, container=self.controls_panel)
        y_pos += 40
        self.apply_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (rect.width - 40, 30)),
            text='Apply Settings', manager=manager, container=self.controls_panel)
        y_pos += 40

        self.controls_panel.set_scrollable_area_dimensions((rect.width - 30, y_pos))

        # --- Editor UI ---
        self.editor_panel = pygame_gui.elements.UIScrollingContainer(
            relative_rect=rect, manager=manager, visible=False
        )
        editor_y = 10
        self.mode_button_editor = pygame_gui.elements.UIButton(
             relative_rect=pygame.Rect((10, editor_y), (rect.width - 40, 30)),
             text='Back to Simulation', manager=manager, container=self.editor_panel
        )
        editor_y += 50

        self.editor_buttons = {}
        button_data = [
            ('wall', 'Place Wall'), ('resource', 'Place Resource'),
            ('empty', 'Eraser'), ('base', 'Set Base'), ('target', 'Set Target')
        ]
        for name, text in button_data:
            self.editor_buttons[name] = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect((10, editor_y), (rect.width - 40, 30)),
                text=text, manager=manager, container=self.editor_panel
            )
            editor_y += 40

        self.editor_panel.set_scrollable_area_dimensions((rect.width - 30, editor_y))


    def get_current_settings(self):
        settings = {name: slider.get_current_value() for name, slider in self.sliders.items()}
        settings['mlp_arch_str'] = self.mlp_arch_input.get_text()
        for name, dropdown in self.dropdowns.items():
            settings[name] = dropdown.selected_option
        return settings

    def update_labels(self, bank_amount=0):
        for name, (label, label_format) in self.slider_labels.items():
            value = self.sliders[name].get_current_value()
            label.set_text(label_format.format(value))
        self.bank_label.set_text(f'Bank: {bank_amount}')

    def draw_fittest_brain(self, surface, brain, activations=None):
        surface.fill(pygame.Color("#303030"))
        input_labels = [
            "W_N", "W_NE", "W_E", "W_SE", "W_S", "W_SW", "W_W", "W_NW", # Wall vision
            "R_N", "R_NE", "R_E", "R_SE", "R_S", "R_SW", "R_W", "R_NW", # Resource vision
            "dX", "dY", "Dist",
            "LM_N", "LM_E", "LM_S", "LM_W", "L_STAY", "L_MINE",
            "Inv"
        ]
        output_labels = ["MOVE_N", "MOVE_E", "MOVE_S", "MOVE_W", "STAY", "MINE"]
        draw_mlp(surface, brain, input_labels, activations, output_labels)

    def show_simulation_ui(self):
        self.controls_panel.show()
        self.editor_panel.hide()

        # Move buttons back to sim panel
        self.save_map_button.change_container(self.controls_panel)
        self.load_map_button.change_container(self.controls_panel)


    def show_editor_ui(self):
        self.controls_panel.hide()
        self.editor_panel.show()

        # Move buttons to editor panel
        self.save_map_button.change_container(self.editor_panel)
        self.load_map_button.change_container(self.editor_panel)
