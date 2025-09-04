"""
This is the main entry point for the simplified, CPU-based simulation.
It uses Pygame for visualization and multiprocessing for parallel simulation.
"""
import pygame
import pygame_gui
import numpy as np
import time
import os
from enum import Enum
from simplified_game import SimplifiedGame, Tile, get_vision_inputs
from simplified_ui import SimplifiedUI

# --- Constants ---
GRID_WIDTH, GRID_HEIGHT = 50, 30
TILE_SIZE = 20
UI_WIDTH = 220
VISUALIZER_HEIGHT = 250
SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE + UI_WIDTH
SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE + VISUALIZER_HEIGHT
FPS = 60
MAPS_DIR = "maps"
DEFAULT_MAP_PATH = os.path.join(MAPS_DIR, "default.csv")
SAVED_MAP_PATH = os.path.join(MAPS_DIR, "saved_map.csv")

# --- Colors ---
BLACK, WHITE = (0, 0, 0), (255, 255, 255)
GRID_COLOR, WALL_COLOR = (40, 40, 40), (100, 100, 100)
UNIT_COLOR, TARGET_COLOR = (0, 150, 255), (0, 255, 0)
RESOURCE_COLOR = (255, 180, 0)

class GameState(Enum):
    SIMULATING, EDITING, PAUSED, FAST_FORWARDING = 1, 2, 3, 4

def draw_game_world(surface, game):
    surface.fill(BLACK)
    for x in range(game.tile_map.grid_width):
        for y in range(game.tile_map.grid_height):
            if game.tile_map.is_wall(x, y):
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(surface, WALL_COLOR, rect)
            elif game.tile_map.is_resource(x, y):
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(surface, RESOURCE_COLOR, rect)


    spawn_center = (int(game.spawn_point[0] * TILE_SIZE + TILE_SIZE / 2),
                    int(game.spawn_point[1] * TILE_SIZE + TILE_SIZE / 2))
    pygame.draw.circle(surface, (0, 100, 200), spawn_center, TILE_SIZE / 2, 2)

    unit_surface = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
    unit_surface.fill((*UNIT_COLOR, 180))
    for unit in game.units:
        rect = pygame.Rect(unit.x * TILE_SIZE, unit.y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        surface.blit(unit_surface, rect.topleft)
    target_rect = pygame.Rect(game.target[0] * TILE_SIZE, game.target[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
    pygame.draw.rect(surface, TARGET_COLOR, target_rect)
    for x in range(0, GRID_WIDTH * TILE_SIZE, TILE_SIZE):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, GRID_HEIGHT * TILE_SIZE))
    for y in range(0, GRID_HEIGHT * TILE_SIZE, TILE_SIZE):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (GRID_WIDTH * TILE_SIZE, y))

def draw_fitness_graph(surface, history, font):
    graph_width, graph_height = 400, 200
    graph_surf = pygame.Surface((graph_width, graph_height))
    graph_surf.fill((20, 20, 20))
    pygame.draw.rect(graph_surf, (80, 80, 80), graph_surf.get_rect(), 1)

    if len(history) < 2:
        text = font.render("Not enough data for graph", True, WHITE)
        graph_surf.blit(text, text.get_rect(center=graph_surf.get_rect().center))
        surface.blit(graph_surf, (10, 140))
        return

    max_fitness = max(history) if history else 1.0
    points = []
    for i, fitness in enumerate(history):
        x = int((i / (len(history) - 1)) * (graph_width - 20)) + 10
        y = graph_height - int((fitness / max_fitness) * (graph_height - 20)) - 10
        points.append((x, y))

    if len(points) > 1:
        pygame.draw.lines(graph_surf, (0, 200, 255), False, points, 2)

    # Draw axes and labels
    pygame.draw.line(graph_surf, WHITE, (10, graph_height - 10), (graph_width - 10, graph_height - 10), 1) # X-axis
    pygame.draw.line(graph_surf, WHITE, (10, 10), (10, graph_height - 10), 1) # Y-axis
    max_fit_text = font.render(f"{max_fitness:.4f}", True, WHITE)
    graph_surf.blit(max_fit_text, (15, 5))
    gen_text = font.render(f"Gen {len(history)}", True, WHITE)
    graph_surf.blit(gen_text, (graph_width - gen_text.get_width() - 5, graph_height - 25))

    surface.blit(graph_surf, (10, 140))

def save_map(game, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savetxt(game.tile_map.static_grid, filepath, delimiter=',', fmt='%d')
    print(f"Map saved to {filepath}")

def load_map(filepath):
    if os.path.exists(filepath): return np.loadtxt(filepath, delimiter=',').astype(int)
    return None

def load_or_create_default_map():
    if os.path.exists(DEFAULT_MAP_PATH): return load_map(DEFAULT_MAP_PATH)
    temp_game = SimplifiedGame(width=GRID_WIDTH, height=GRID_HEIGHT)
    save_map(temp_game, DEFAULT_MAP_PATH)
    return temp_game.tile_map.static_grid

def create_game_config_from_settings(settings):
    config = settings.copy()
    config['perception_radius'] = config.pop('vision_radius', 5)
    config['steps_per_gen'] = config.pop('sim_length', 100)
    config.pop('sps', None)
    return config

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Simplified Brains Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    game_world_surface = pygame.Surface((GRID_WIDTH * TILE_SIZE, GRID_HEIGHT * TILE_SIZE))

    def setup_ui(width, height):
        manager = pygame_gui.UIManager((width, height), "theme.json")
        game_world_rect = pygame.Rect(0, 0, max(1, width - UI_WIDTH), max(1, height - VISUALIZER_HEIGHT))
        controls_panel_rect = pygame.Rect(game_world_rect.right, 0, UI_WIDTH, height)
        visualizer_panel_rect = pygame.Rect(0, game_world_rect.bottom, width, VISUALIZER_HEIGHT)
        ui = SimplifiedUI(rect=controls_panel_rect, manager=manager)
        visualizer_panel = pygame_gui.elements.UIPanel(relative_rect=visualizer_panel_rect, manager=manager)
        return manager, ui, visualizer_panel, game_world_rect

    ui_manager, ui, visualizer_panel, game_world_rect = setup_ui(SCREEN_WIDTH, SCREEN_HEIGHT)
    default_map = load_or_create_default_map()

    initial_settings = ui.get_current_settings()
    game_config = create_game_config_from_settings(initial_settings)
    game = SimplifiedGame(width=GRID_WIDTH, height=GRID_HEIGHT, static_grid=default_map, **game_config)

    step_counter, measured_sps, sps_counter, sps_timer, time_since_last_step = 0, 0, 0, 0.0, 0.0
    ff_generations_to_run, ff_generations_completed = 0, 0
    current_state = GameState.SIMULATING
    running = True

    while running:
        time_delta = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False

            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                ui_manager, ui, visualizer_panel, game_world_rect = setup_ui(event.w, event.h)

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                settings = ui.get_current_settings()
                if event.ui_element == ui.mode_button:
                    current_state = GameState.SIMULATING if current_state == GameState.EDITING else GameState.EDITING
                    ui.show_editor_ui() if current_state == GameState.EDITING else ui.show_simulation_ui()
                elif event.ui_element == ui.pause_button:
                    if current_state == GameState.SIMULATING: current_state = GameState.PAUSED
                    elif current_state == GameState.PAUSED: current_state = GameState.SIMULATING
                elif event.ui_element == ui.restart_button:
                    game.restart()
                    step_counter, current_state = 0, GameState.SIMULATING
                elif event.ui_element == ui.save_map_button: save_map(game, SAVED_MAP_PATH)
                elif event.ui_element == ui.load_map_button:
                    loaded_map = load_map(SAVED_MAP_PATH)
                    if loaded_map is not None:
                        game_config = create_game_config_from_settings(settings)
                        game = SimplifiedGame(width=loaded_map.shape[0], height=loaded_map.shape[1], static_grid=loaded_map, **game_config)
                        step_counter = 0
                elif event.ui_element == ui.apply_button:
                    game.update_settings(create_game_config_from_settings(settings))
                    step_counter = 0
                elif event.ui_element == ui.fast_forward_button:
                    if current_state in [GameState.SIMULATING, GameState.PAUSED]:
                        current_state = GameState.FAST_FORWARDING
                        ff_generations_to_run, ff_generations_completed = 10, 0
                        ui.pause_button.disable(); ui.restart_button.disable()
            ui_manager.process_events(event)

        ui_manager.update(time_delta)
        settings = ui.get_current_settings()
        ui.update_labels(game.bank)

        if current_state == GameState.SIMULATING:
            sps_timer += time_delta
            time_since_last_step += time_delta
            step_interval = 1.0 / settings['sps'] if settings['sps'] > 0 else 0
            while time_since_last_step >= step_interval:
                if step_counter < settings['sim_length']:
                    game.run_simulation_step()
                    step_counter += 1; sps_counter += 1
                else:
                    game.evolve_population(); step_counter = 0
                time_since_last_step -= step_interval

        elif current_state == GameState.EDITING:
            buttons, keys, (mx, my) = pygame.mouse.get_pressed(), pygame.key.get_pressed(), pygame.mouse.get_pos()
            if game_world_rect.collidepoint(mx, my):
                scaled_mx, scaled_my = mx * (game_world_surface.get_width() / game_world_rect.width), my * (game_world_surface.get_height() / game_world_rect.height)
                grid_x, grid_y = int(scaled_mx // TILE_SIZE), int(scaled_my // TILE_SIZE)
                if buttons[0] and (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]): game.spawn_point = (grid_x, grid_y)
                elif buttons[0]: game.tile_map.set_tile(grid_x, grid_y, Tile.WALL)
                elif buttons[2]: game.tile_map.set_tile(grid_x, grid_y, Tile.EMPTY)
                elif buttons[1]: game.target = (grid_x, grid_y)

        elif current_state == GameState.FAST_FORWARDING:
            for _ in range(settings['sim_length']): game.run_simulation_step()
            game.evolve_population()
            step_counter, ff_generations_completed = 0, ff_generations_completed + 1
            if ff_generations_completed >= ff_generations_to_run:
                current_state = GameState.SIMULATING
                ui.pause_button.enable(); ui.restart_button.enable()

        if sps_timer >= 1.0:
            measured_sps, sps_counter, sps_timer = sps_counter, 0, sps_timer - 1.0

        screen.fill(pygame.Color("#202020"))
        draw_game_world(game_world_surface, game)
        screen.blit(pygame.transform.scale(game_world_surface, game_world_rect.size), game_world_rect.topleft)

        if current_state == GameState.FAST_FORWARDING:
            progress_text = f"Fast Forwarding... Gen {game.generation}/{game.generation - ff_generations_completed + ff_generations_to_run}"
            text_surf = font.render(progress_text, True, WHITE, pygame.Color("#404040"))
            screen.blit(text_surf, text_surf.get_rect(center=screen.get_rect().center))

        if game.fittest_brain and game.units:
            unit_for_vis = game.units[0]
            inputs = game._get_unit_inputs(unit_for_vis)
            _, live_activations = game.fittest_brain.forward(inputs)
            ui.draw_fittest_brain(visualizer_panel.image, game.fittest_brain, live_activations)

        # Draw text info
        y_offset = 10
        texts = [f"FPS: {int(clock.get_fps())}", f"SPS: {measured_sps}", f"Generation: {game.generation}",
                 f"Best Fitness: {game.best_fitness:.4f}", f"Avg Fitness: {game.average_fitness:.4f}"]
        prox, expl = game.best_fitness_components
        texts.extend([f"  - Proximity: {prox:.4f}", f"  - Exploration: {expl:.4f}"])
        for text in texts:
            screen.blit(font.render(text, True, WHITE), (10, y_offset))
            y_offset += 20

        # Draw graph on hover
        hover_zone = pygame.Rect(0, 0, 200, y_offset)
        if hover_zone.collidepoint(pygame.mouse.get_pos()):
            draw_fitness_graph(screen, game.fitness_history, font)

        ui_manager.draw_ui(screen)
        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()
