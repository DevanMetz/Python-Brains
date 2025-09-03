import pygame
import numpy as np

def draw_mlp(surface, brain, input_labels=None):
    """
    Draws a representation of the MLP's structure and weights onto a surface.
    Optionally labels the input nodes.
    """
    if not brain:
        return

    font = pygame.font.SysFont("Arial", 12)
    margin = 20
    width = surface.get_width() - margin * 2
    height = surface.get_height() - margin * 2
    pos = (margin, margin)

    layers = brain.layer_sizes
    num_layers = len(layers)
    max_nodes = max(layers) if layers else 0
    if max_nodes == 0: return

    node_radius = min(15, int(height / (max_nodes * 2)))
    layer_spacing = width / max(1, num_layers - 1)

    # Calculate node positions
    node_positions = []
    for i, layer_size in enumerate(layers):
        layer_nodes = []
        layer_x = pos[0] + (i * layer_spacing) if num_layers > 1 else pos[0] + width / 2

        # Calculate vertical spacing to center the nodes
        total_node_height = layer_size * node_radius * 2 + max(0, layer_size - 1) * 10
        y_start = pos[1] + (height - total_node_height) / 2

        for j in range(layer_size):
            node_y = y_start + j * (node_radius * 2 + 10) + node_radius
            layer_nodes.append((layer_x, node_y))
        node_positions.append(layer_nodes)

    # Draw weights (lines)
    for i in range(len(brain.weights)):
        layer_weights = brain.weights[i]
        # Normalize weights for better visualization
        max_abs_weight = np.max(np.abs(layer_weights))
        if max_abs_weight == 0: max_abs_weight = 1.0

        for j, start_node_pos in enumerate(node_positions[i]):
            for k, end_node_pos in enumerate(node_positions[i+1]):
                weight = layer_weights[j, k]

                # Blue for positive, red for negative
                color = (100, 100, 255) if weight > 0 else (255, 100, 100)

                alpha = int(min(255, 50 + abs(weight) / max_abs_weight * 205))
                color_with_alpha = (*color, alpha)

                thickness = int(max(1, abs(weight) / max_abs_weight * 4))

                pygame.draw.line(surface, color_with_alpha, start_node_pos, end_node_pos, thickness)

    # Draw nodes (circles) and input labels
    for i, layer_nodes in enumerate(node_positions):
        for j, node_pos in enumerate(layer_nodes):
            pygame.draw.circle(surface, (200, 200, 200), (int(node_pos[0]), int(node_pos[1])), node_radius)
            pygame.draw.circle(surface, (50, 50, 50), (int(node_pos[0]), int(node_pos[1])), node_radius, 1)

            # If this is the input layer and we have labels, draw them
            if i == 0 and input_labels and j < len(input_labels):
                label_text = font.render(input_labels[j], True, (200, 200, 200))
                surface.blit(label_text, (node_pos[0] - node_radius - label_text.get_width() - 5, node_pos[1] - label_text.get_height() / 2))


    # Draw labels
    input_text = font.render("Inputs", True, (200, 200, 200))
    output_text = font.render("Outputs", True, (200, 200, 200))
    surface.blit(input_text, (pos[0], pos[1] - 20))
    if num_layers > 1:
        surface.blit(output_text, (pos[0] + width - output_text.get_width(), pos[1] - 20))

    title_text = font.render("Fittest Brain", True, (255, 255, 255))
    surface.blit(title_text, (surface.get_width()/2 - title_text.get_width()/2, 5))
