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
# These are now initial values, they will change if the window is resized
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

class GameState(Enum):
    SIMULATING, EDITING, PAUSED, FAST_FORWARDING = 1, 2, 3, 4

def draw_game_world(surface, game):
    surface.fill(BLACK)
    for x in range(game.tile_map.grid_width):
        for y in range(game.tile_map.grid_height):
            if game.tile_map.static_grid[x, y] == Tile.WALL.value:
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(surface, WALL_COLOR, rect)

    # Draw spawn point
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

def save_map(game, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savetxt(filepath, game.tile_map.static_grid, delimiter=',', fmt='%d')
    print(f"Map saved to {filepath}")

def load_map(filepath):
    if os.path.exists(filepath):
        print(f"Loading map from {filepath}")
        return np.loadtxt(filepath, delimiter=',').astype(int)
    return None

def load_or_create_default_map():
    if os.path.exists(DEFAULT_MAP_PATH):
        return load_map(DEFAULT_MAP_PATH)
    else:
        print("No default map found. Creating one.")
        temp_game = SimplifiedGame(width=GRID_WIDTH, height=GRID_HEIGHT)
        save_map(temp_game, DEFAULT_MAP_PATH)
        return temp_game.tile_map.static_grid

def create_game_config_from_settings(settings):
    config = settings.copy()
    if 'vision_radius' in config:
        config['perception_radius'] = config.pop('vision_radius')
    if 'sim_length' in config:
        config['steps_per_gen'] = config.pop('sim_length')
    config.pop('sps', None)
    return config

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Simplified Brains Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    # Fixed size for the simulation surface itself
    game_world_surface = pygame.Surface((GRID_WIDTH * TILE_SIZE, GRID_HEIGHT * TILE_SIZE))

    # UI setup function
    def setup_ui(width, height):
        manager = pygame_gui.UIManager((width, height))

        game_world_rect = pygame.Rect(0, 0, width - UI_WIDTH, height - VISUALIZER_HEIGHT)
        controls_panel_rect = pygame.Rect(game_world_rect.right, 0, UI_WIDTH, height - VISUALIZER_HEIGHT)
        visualizer_panel_rect = pygame.Rect(0, game_world_rect.bottom, width, VISUALIZER_HEIGHT)

        ui = SimplifiedUI(rect=controls_panel_rect, manager=manager)
        visualizer_panel = pygame_gui.elements.UIPanel(relative_rect=visualizer_panel_rect, manager=manager)

        return manager, ui, visualizer_panel, game_world_rect

    ui_manager, ui, visualizer_panel, game_world_rect = setup_ui(SCREEN_WIDTH, SCREEN_HEIGHT)

    default_map = load_or_create_default_map()
    game = SimplifiedGame(width=GRID_WIDTH, height=GRID_HEIGHT, static_grid=default_map)

    step_counter = 0
    current_state = GameState.SIMULATING
    time_since_last_step, sps_counter, sps_timer, measured_sps = 0, 0, 0, 0
    ff_generations_to_run = 0
    ff_generations_completed = 0

    running = True
    while running:
        time_delta = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                ui_manager, ui, visualizer_panel, game_world_rect = setup_ui(event.w, event.h)

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == ui.mode_button:
                    if current_state == GameState.EDITING:
                        current_state = GameState.SIMULATING
                        ui.pause_button.set_text("Pause")
                    else: # SIMULATING or PAUSED
                        current_state = GameState.EDITING
                    ui.show_editor_ui() if current_state == GameState.EDITING else ui.show_simulation_ui()

                elif event.ui_element == ui.pause_button:
                    if current_state == GameState.SIMULATING:
                        current_state = GameState.PAUSED
                        ui.pause_button.set_text("Resume")
                    elif current_state == GameState.PAUSED:
                        current_state = GameState.SIMULATING
                        ui.pause_button.set_text("Pause")

                elif event.ui_element == ui.restart_button:
                    game.restart()
                    step_counter = 0
                    current_state = GameState.SIMULATING
                    ui.pause_button.set_text("Pause")

                elif event.ui_element == ui.save_map_button:
                    save_map(game, SAVED_MAP_PATH)

                elif event.ui_element == ui.load_map_button:
                    loaded_map = load_map(SAVED_MAP_PATH)
                    if loaded_map is not None:
                        current_settings = ui.get_current_settings()
                        game_config = create_game_config_from_settings(current_settings)

                        # Preserve spawn and target from old game instance
                        current_target = game.target
                        current_spawn = game.spawn_point

                        game = SimplifiedGame(width=GRID_WIDTH, height=GRID_HEIGHT, static_grid=loaded_map, **game_config)

                        game.target = current_target
                        game.spawn_point = current_spawn
                        step_counter = 0

                elif event.ui_element == ui.apply_button:
                    settings = ui.get_current_settings()
                    game_config = create_game_config_from_settings(settings)
                    game.update_settings(game_config)
                    step_counter = 0

                elif event.ui_element == ui.fast_forward_button:
                    if current_state in [GameState.SIMULATING, GameState.PAUSED]:
                        current_state = GameState.FAST_FORWARDING
                        ff_generations_to_run = 10
                        ff_generations_completed = 0
                        ui.pause_button.disable()
                        ui.restart_button.disable()

            ui_manager.process_events(event)

        ui_manager.update(time_delta)
        ui.update_labels()

        settings = ui.get_current_settings()

        if current_state == GameState.SIMULATING:
            sps_timer += time_delta

        if current_state == GameState.EDITING:
            buttons = pygame.mouse.get_pressed()
            keys = pygame.key.get_pressed()
            mx, my = pygame.mouse.get_pos()

            if game_world_rect.collidepoint(mx, my):
                scaled_mx = mx * (game_world_surface.get_width() / game_world_rect.width)
                scaled_my = my * (game_world_surface.get_height() / game_world_rect.height)
                grid_x, grid_y = int(scaled_mx // TILE_SIZE), int(scaled_my // TILE_SIZE)

                # Shift + Left Click to set spawn
                if buttons[0] and (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]):
                    game.spawn_point = (grid_x, grid_y)
                # Left Click to draw wall
                elif buttons[0]:
                    game.tile_map.set_tile(grid_x, grid_y, Tile.WALL)
                # Right Click to erase
                elif buttons[2]:
                    game.tile_map.set_tile(grid_x, grid_y, Tile.EMPTY)
                # Middle Click to set target
                elif buttons[1]:
                    game.target = (grid_x, grid_y)

        elif current_state == GameState.FAST_FORWARDING:
            for _ in range(game.steps_per_generation):
                game.run_simulation_step()
            game.evolve_population()
            step_counter = 0
            ff_generations_completed += 1
            if ff_generations_completed >= ff_generations_to_run:
                current_state = GameState.SIMULATING
                ui.pause_button.enable()
                ui.restart_button.enable()

        elif current_state == GameState.SIMULATING:
            time_since_last_step += time_delta
            step_interval = 1.0 / settings['sps'] if settings['sps'] > 0 else 0
            while time_since_last_step >= step_interval:
                if step_counter < game.steps_per_generation:
                    game.run_simulation_step()
                    step_counter += 1
                    sps_counter += 1
                else:
                    game.evolve_population()
                    step_counter = 0
                time_since_last_step -= step_interval

        if sps_timer >= 1.0:
            measured_sps = sps_counter
            sps_counter = 0
            sps_timer -= 1.0

        screen.fill(pygame.Color("#202020"))

        draw_game_world(game_world_surface, game)
        screen.blit(pygame.transform.scale(game_world_surface, game_world_rect.size), game_world_rect.topleft)

        if current_state == GameState.FAST_FORWARDING:
            progress_text = f"Fast Forwarding... Gen {game.generation}/{game.generation - ff_generations_completed + ff_generations_to_run}"
            text_surf = font.render(progress_text, True, WHITE, pygame.Color("#404040"))
            text_rect = text_surf.get_rect(center=screen.get_rect().center)
            screen.blit(text_surf, text_rect)

        if game.fittest_brain and game.units:
            # We need a unit to get the inputs from, let's use the first one.
            # Note: this unit's brain is not necessarily the same as fittest_brain
            unit_for_vis = game.units[0]
            inputs = game._get_unit_inputs(unit_for_vis)
            _, live_activations = game.fittest_brain.forward(inputs)
            ui.draw_fittest_brain(visualizer_panel.image, game.fittest_brain, live_activations)

        fps_text = font.render(f"FPS: {int(clock.get_fps())}", True, WHITE)
        sps_text = font.render(f"SPS: {measured_sps}", True, WHITE)
        best_fitness_text = font.render(f"Best Fitness: {game.best_fitness:.4f}", True, WHITE)
        avg_fitness_text = font.render(f"Avg Fitness: {game.average_fitness:.4f}", True, WHITE)
        screen.blit(fps_text, (10, 10))
        screen.blit(sps_text, (10, 30))
        screen.blit(best_fitness_text, (10, 50))
        screen.blit(avg_fitness_text, (10, 70))

        ui_manager.draw_ui(screen)
        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()
