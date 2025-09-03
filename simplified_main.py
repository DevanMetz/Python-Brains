"""
This is the main entry point for the simplified, CPU-based simulation.
It uses Pygame for visualization and multiprocessing for parallel simulation.
"""
import pygame
import pygame_gui
import multiprocessing
import numpy as np
from enum import Enum
from simplified_game import SimplifiedGame, Tile, process_unit_logic, get_vision_inputs
from simplified_ui import SimplifiedUI

# --- Constants ---
GRID_WIDTH, GRID_HEIGHT = 50, 30
TILE_SIZE = 20
UI_WIDTH = 220
VISUALIZER_HEIGHT = 250
SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE + UI_WIDTH
SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE + VISUALIZER_HEIGHT
FPS = 60

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRID_COLOR = (40, 40, 40)
WALL_COLOR = (100, 100, 100)
UNIT_COLOR = (0, 150, 255)
TARGET_COLOR = (0, 255, 0)

class GameState(Enum):
    SIMULATING = 1
    EDITING = 2

def draw_game_world(surface, game):
    """Draws the entire game world (map and dynamic elements) onto a surface."""
    surface.fill(BLACK)

    for x in range(game.tile_map.grid_width):
        for y in range(game.tile_map.grid_height):
            if game.tile_map.static_grid[x, y] == Tile.WALL.value:
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(surface, WALL_COLOR, rect)

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

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Simplified CPU Simulation")
    clock = pygame.time.Clock()

    game_world_rect = pygame.Rect(0, 0, GRID_WIDTH * TILE_SIZE, GRID_HEIGHT * TILE_SIZE)
    controls_panel_rect = pygame.Rect(game_world_rect.right, 0, UI_WIDTH, SCREEN_HEIGHT - VISUALIZER_HEIGHT)
    visualizer_panel_rect = pygame.Rect(0, game_world_rect.bottom, SCREEN_WIDTH, VISUALIZER_HEIGHT)

    ui_manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))
    ui = SimplifiedUI(rect=controls_panel_rect, manager=ui_manager)
    visualizer_panel = pygame_gui.elements.UIPanel(relative_rect=visualizer_panel_rect, manager=ui_manager)

    game_world_surface = pygame.Surface(game_world_rect.size)
    game = SimplifiedGame(width=GRID_WIDTH, height=GRID_HEIGHT)
    step_counter = 0
    current_state = GameState.SIMULATING

    multiprocessing.set_start_method("spawn", force=True)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    running = True
    while running:
        time_delta = clock.tick(FPS) / 1000.0
        settings = ui.get_current_settings()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == ui.mode_button:
                    current_state = GameState.EDITING if current_state == GameState.SIMULATING else GameState.SIMULATING
                    ui.show_editor_ui() if current_state == GameState.EDITING else ui.show_simulation_ui()
                elif event.ui_element == ui.apply_button:
                    current_map = game.tile_map.static_grid.copy()
                    current_target = game.target
                    game = SimplifiedGame(
                        width=GRID_WIDTH, height=GRID_HEIGHT,
                        population_size=settings['population_size'],
                        mlp_arch_str=settings['mlp_arch_str'],
                        steps_per_gen=settings['sim_length'],
                        mutation_rate=settings['mutation_rate'],
                        static_grid=current_map)
                    game.target = current_target
                    step_counter = 0

            ui_manager.process_events(event)

        ui.update_labels()

        if current_state == GameState.EDITING:
            buttons = pygame.mouse.get_pressed()
            mx, my = pygame.mouse.get_pos()
            if game_world_rect.collidepoint(mx, my):
                grid_x, grid_y = mx // TILE_SIZE, my // TILE_SIZE
                if buttons[0]: game.tile_map.set_tile(grid_x, grid_y, Tile.WALL)
                elif buttons[2]: game.tile_map.set_tile(grid_x, grid_y, Tile.EMPTY)
                elif buttons[1]: game.target = (grid_x, grid_y)

        elif current_state == GameState.SIMULATING:
            for _ in range(settings['sim_speed']):
                if step_counter < game.steps_per_generation:
                    tasks = [(u.id, u.x, u.y, [w.copy() for w in u.brain.weights], [b.copy() for b in u.brain.biases],
                              game.tile_map.static_grid, game.target, game.mlp_arch) for u in game.units]
                    results = pool.map(process_unit_logic, tasks)
                    game.update_simulation_with_results(results)
                    step_counter += 1
                else:
                    game.evolve_population()
                    step_counter = 0
                    break

        screen.fill(pygame.Color("#202020"))
        draw_game_world(game_world_surface, game)
        screen.blit(game_world_surface, game_world_rect.topleft)

        live_activations = None
        if game.fittest_brain and game.units:
            # Get the state of the first unit (which is an elite) for visualization
            fittest_unit = game.units[0]
            vision_inputs = get_vision_inputs(fittest_unit.x, fittest_unit.y, game.tile_map.static_grid)
            dx_to_target = (game.target[0] - fittest_unit.x) / game.tile_map.grid_width
            dy_to_target = (game.target[1] - fittest_unit.y) / game.tile_map.grid_height
            target_inputs = np.array([dx_to_target, dy_to_target])
            inputs = np.concatenate((vision_inputs, target_inputs))

            # Run a shadow forward pass to get activations for visualization
            _, live_activations = game.fittest_brain.forward(inputs)

        ui.draw_fittest_brain(visualizer_panel.image, game.fittest_brain, live_activations)

        ui_manager.update(time_delta)
        ui_manager.draw_ui(screen)
        pygame.display.flip()

    pool.close()
    pool.join()
    pygame.quit()

if __name__ == '__main__':
    main()
