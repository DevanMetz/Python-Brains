import torch
import pygame
import math

class MLPVisualizer:
    """
    A class to handle the visualization of the MLP.
    It uses hooks to capture layer activations and manages a camera for pan/zoom.
    """
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        self._register_hooks()

        # Camera state
        self.zoom = 1.0
        self.offset = pygame.Vector2(0, 0)
        self.panning = False

        # Font and labels
        self.font = pygame.font.SysFont("Arial", 14)
        self.input_labels = ["Unit X", "Unit Y", "Reward X", "Reward Y", "Obstacle X", "Obstacle Y"]
        self.output_labels = ["Up", "Down", "Left", "Right"]

    def _register_hooks(self):
        """
        Registers a forward hook on each linear layer of the model
        to capture its output activations.
        """
        # We rely on the order of layers defined in the MLP class
        layer_names = ['layer1', 'layer2', 'layer3']
        for name, layer in self.model.named_children():
            if name in layer_names:
                self.hooks.append(layer.register_forward_hook(self._hook_fn(name)))

    def remove_hooks(self):
        """Removes all registered hooks."""
        for hook in self.hooks:
            hook.remove()

    def world_to_screen(self, x, y, surface):
        """Converts world coordinates to screen coordinates within the surface."""
        center_x, center_y = surface.get_width() / 2, surface.get_height() / 2

        # Apply pan and zoom
        screen_x = (x + self.offset.x) * self.zoom + center_x
        screen_y = (y + self.offset.y) * self.zoom + center_y

        return pygame.Vector2(screen_x, screen_y)

    def screen_to_world(self, x, y, surface):
        """Converts screen coordinates to world coordinates."""
        center_x, center_y = surface.get_width() / 2, surface.get_height() / 2

        world_x = (x - center_x) / self.zoom - self.offset.x
        world_y = (y - center_y) / self.zoom - self.offset.y

        return pygame.Vector2(world_x, world_y)

    def handle_event(self, event, panel_rect):
        """Handles user input for panning and zooming."""
        if event.type == pygame.MOUSEWHEEL:
            # Check if mouse is inside the panel
            if panel_rect.collidepoint(pygame.mouse.get_pos()):
                # Get world coordinates before zoom
                mouse_pos_world_before = self.screen_to_world(event.pos[0] - panel_rect.x, event.pos[1] - panel_rect.y, panel_rect.size)

                # Update zoom
                self.zoom *= 1.1 if event.y > 0 else 0.9
                self.zoom = max(0.1, min(self.zoom, 5.0)) # Clamp zoom level

                # Get world coordinates after zoom
                mouse_pos_world_after = self.screen_to_world(event.pos[0] - panel_rect.x, event.pos[1] - panel_rect.y, panel_rect.size)

                # Adjust offset to zoom towards the mouse cursor
                self.offset += mouse_pos_world_before - mouse_pos_world_after

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if panel_rect.collidepoint(event.pos) and event.button == 2: # Middle mouse button
                self.panning = True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 2:
                self.panning = False

        elif event.type == pygame.MOUSEMOTION:
            if self.panning:
                # Adjust offset by mouse movement, scaled by zoom
                self.offset.x += event.rel[0] / self.zoom
                self.offset.y += event.rel[1] / self.zoom

    def draw(self, surface, frame_count):
        """Draws the visualization of the network on the given surface."""
        surface.fill((20, 20, 30))

        # --- Diagnostic Logging ---
        if frame_count % 60 == 0:
            print("\n--- MLP VISUALIZER DIAGNOSTIC (Frame {}) ---".format(frame_count))
            print("Camera State: zoom={:.2f}, offset=({}, {})".format(self.zoom, self.offset.x, self.offset.y))
            print("Activation Data Captured: {}".format(bool(self.activations)))

        layers = [self.model.layer1, self.model.layer2, self.model.layer3]
        layer_sizes = [layers[0].in_features] + [l.out_features for l in layers]

        # Define the world-space dimensions of the network layout
        world_width, world_height = 600, 250 # Adjusted height to fit in the bottom panel

        node_positions = []
        for i, size in enumerate(layer_sizes):
            layer_nodes = []
            layer_x = -world_width / 2 + i * (world_width / (len(layer_sizes) - 1))
            for j in range(size):
                layer_y = -world_height / 2 + j * (world_height / (size - 1)) if size > 1 else 0
                layer_nodes.append(pygame.Vector2(layer_x, layer_y))
            node_positions.append(layer_nodes)

        # --- Draw Connections (Weights) ---
        line_width = max(1, int(1 * self.zoom))
        line_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA) # Optimized: one surface for all lines
        with torch.no_grad():
            for i, layer in enumerate(layers):
                weights = layer.weight.data
                max_weight = weights.abs().max()
                if max_weight == 0: continue # Safeguard

                for j, start_node_world in enumerate(node_positions[i]):
                    for k, end_node_world in enumerate(node_positions[i+1]):
                        start_node_screen = self.world_to_screen(start_node_world.x, start_node_world.y, surface)
                        end_node_screen = self.world_to_screen(end_node_world.x, end_node_world.y, surface)

                        weight = weights[k, j]
                        alpha = int(255 * min(1.0, abs(weight) / max_weight))
                        color = (0, 255, 0, alpha) if weight > 0 else (255, 0, 0, alpha)

                        pygame.draw.line(line_surf, color, start_node_screen, end_node_screen, line_width)

        surface.blit(line_surf, (0,0)) # Optimized: blit once

        # --- Draw Neurons (Activations) ---
        node_radius = max(2, int(5 * self.zoom))

        # Draw input nodes
        for j, pos_world in enumerate(node_positions[0]):
            pos_screen = self.world_to_screen(pos_world.x, pos_world.y, surface)
            if frame_count % 60 == 0 and j == 0: # Log first input neuron
                print("First Input Neuron Screen Coords: ({}, {})".format(pos_screen.x, pos_screen.y))
            try:
                activation = self.activations['layer1_input'][0][j]
            except (KeyError, IndexError):
                activation = 0
            color_val = int(255 * min(1.0, abs(activation)))
            pygame.draw.circle(surface, (color_val, color_val, 0), pos_screen, node_radius)

        # Draw hidden and output nodes
        for i, layer_name in enumerate(['layer1', 'layer2', 'layer3']):
            if layer_name in self.activations:
                activations = self.activations[layer_name][0]
                max_act = activations.abs().max()
                if max_act == 0: continue # Safeguard

                for j, pos_world in enumerate(node_positions[i+1]):
                    pos_screen = self.world_to_screen(pos_world.x, pos_world.y, surface)
                    if frame_count % 60 == 0 and i == 2 and j == 0: # Log first output neuron
                        print("First Output Neuron Screen Coords: ({}, {})".format(pos_screen.x, pos_screen.y))
                        print("Panel Dims: {}".format(surface.get_size()))
                    activation = activations[j]
                    color_val = int(255 * min(1.0, abs(activation) / max_act))
                    pygame.draw.circle(surface, (0, color_val, color_val), pos_screen, node_radius)

        # --- Draw Labels ---
        if self.zoom > 0.5: # Only draw labels if zoomed in enough
            scaled_font_size = int(14 * self.zoom)
            if scaled_font_size < 1: return # Don't draw if font is too small

            scaled_font = pygame.font.SysFont("Arial", scaled_font_size)

            # Input labels
            for i, label_text in enumerate(self.input_labels):
                if i < len(node_positions[0]):
                    pos_world = node_positions[0][i]
                    pos_screen = self.world_to_screen(pos_world.x, pos_world.y, surface)
                    text_surf = scaled_font.render(label_text, True, (200, 200, 200))
                    text_rect = text_surf.get_rect(right=pos_screen.x - (node_radius + 5), centery=pos_screen.y)
                    surface.blit(text_surf, text_rect)

            # Output labels
            for i, label_text in enumerate(self.output_labels):
                if i < len(node_positions[-1]):
                    pos_world = node_positions[-1][i]
                    pos_screen = self.world_to_screen(pos_world.x, pos_world.y, surface)
                    text_surf = scaled_font.render(label_text, True, (200, 200, 200))
                    text_rect = text_surf.get_rect(left=pos_screen.x + (node_radius + 5), centery=pos_screen.y)
                    surface.blit(text_surf, text_rect)

    def _hook_fn(self, name):
        """A closure to create a hook function with a specific name."""
        def hook(module, input, output):
            # Store both input and output of the layer
            self.activations[name + '_input'] = input[0].detach()
            self.activations[name] = output.detach()
        return hook
