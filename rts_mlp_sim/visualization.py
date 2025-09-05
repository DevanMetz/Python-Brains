import torch
import pygame
import math

class MLPVisualizer:
    """
    A class to handle the visualization of the MLP.
    It uses hooks to capture layer activations.
    """
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _hook_fn(self, name):
        """A closure to create a hook function with a specific name."""
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

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

    def draw(self, surface):
        """Draws the visualization of the network on the given surface."""
        surface.fill((20, 20, 30)) # Background color

        # --- Get Layer Info ---
        layers = [self.model.layer1, self.model.layer2, self.model.layer3]
        layer_sizes = [layers[0].in_features] + [l.out_features for l in layers]

        # --- Calculate Layout ---
        width, height = surface.get_size()
        h_margin, v_margin = 50, 50
        layer_spacing = (width - 2 * h_margin) / (len(layer_sizes) - 1)

        node_positions = []
        for i, size in enumerate(layer_sizes):
            layer_nodes = []
            neuron_spacing = (height - 2 * v_margin) / (size - 1) if size > 1 else (height - 2 * v_margin)
            for j in range(size):
                x = h_margin + i * layer_spacing
                y = v_margin + j * neuron_spacing
                layer_nodes.append((x, y))
            node_positions.append(layer_nodes)

        # --- Draw Connections (Weights) ---
        with torch.no_grad():
            for i, layer in enumerate(layers):
                weights = layer.weight.data
                max_weight = weights.abs().max()
                for j, start_node in enumerate(node_positions[i]):
                    for k, end_node in enumerate(node_positions[i+1]):
                        weight = weights[k, j]
                        alpha = int(255 * min(1.0, abs(weight) / max_weight))
                        color = (0, 255, 0, alpha) if weight > 0 else (255, 0, 0, alpha)

                        # Create a temporary surface for the line to handle alpha
                        line_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
                        pygame.draw.line(line_surf, color, start_node, end_node, 1)
                        surface.blit(line_surf, (0,0))

        # --- Draw Neurons (Activations) ---
        # Input layer activations (state)
        input_activations = self.activations.get('layer1_input', torch.zeros(1, layer_sizes[0]))

        all_nodes = [item for sublist in node_positions for item in sublist]
        max_activation = 1.0 # Normalized state, so max is 1

        # Draw input nodes
        for j, pos in enumerate(node_positions[0]):
            try:
                # The input to the first layer is our state vector
                activation = self.activations['layer1_input'][0][j]
            except (KeyError, IndexError):
                activation = 0

            color_val = int(255 * min(1.0, abs(activation)))
            pygame.draw.circle(surface, (color_val, color_val, 0), pos, 5)

        # Draw hidden and output nodes
        for i, layer_name in enumerate(['layer1', 'layer2', 'layer3']):
            if layer_name in self.activations:
                activations = self.activations[layer_name][0]
                max_act = activations.abs().max()
                for j, pos in enumerate(node_positions[i+1]):
                    activation = activations[j]
                    color_val = int(255 * min(1.0, abs(activation) / max_act if max_act > 0 else 0))
                    pygame.draw.circle(surface, (0, color_val, color_val), pos, 5)

    def _hook_fn(self, name):
        """A closure to create a hook function with a specific name."""
        def hook(module, input, output):
            # Store both input and output of the layer
            self.activations[name + '_input'] = input[0].detach()
            self.activations[name] = output.detach()
        return hook
