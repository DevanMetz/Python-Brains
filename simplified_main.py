"""
This is the main entry point for the simplified, CPU-based simulation.
It uses Pygame for visualization and multiprocessing for parallel simulation.
"""
import pygame
import pygame_gui
import multiprocessing
from enum import Enum
from simplified_game import SimplifiedGame, Tile, process_unit_logic
from simplified_ui import SimplifiedUI

# --- Constants ---
GRID_WIDTH, GRID_HEIGHT = 40, 30
TILE_SIZE = 20
UI_WIDTH = 200
SCREEN_WIDTH, SCREEN_HEIGHT = GRID_WIDTH * TILE_SIZE + UI_WIDTH, GRID_HEIGHT * TILE_SIZE
FPS = 60
POPULATION_SIZE = 100

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRID_COLOR = (40, 40, 40)
WALL_COLOR = (100, 100, 100)
UNIT_COLOR = (0, 150, 255)
TARGET_COLOR = (0, 255, 0)
VISION_COLOR = (255, 255, 255, 50) # Faint white for vision circle

class GameState(Enum):
    SIMULATING = 1
    EDITING = 2

def draw_tile_map(screen, game):
    """Draws the static part of the map (walls and grid)."""
    for x in range(game.tile_map.grid_width):
        for y in range(game.tile_map.grid_height):
            if game.tile_map.static_grid[x, y] == Tile.WALL.value:
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(screen, WALL_COLOR, rect)

    map_width_pixels = game.tile_map.grid_width * TILE_SIZE
    map_height_pixels = game.tile_map.grid_height * TILE_SIZE
    for x in range(0, map_width_pixels, TILE_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, map_height_pixels))
    for y in range(0, map_height_pixels, TILE_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (map_width_pixels, y))

def draw_dynamic_elements(screen, game, settings):
    """Draws the units and the target with transparency and vision circles."""
    # Draw units with transparency
    unit_surface = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
    unit_surface.fill((*UNIT_COLOR, 180)) # 180/255 alpha
    for unit in game.units:
        rect = pygame.Rect(unit.x * TILE_SIZE, unit.y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        screen.blit(unit_surface, rect.topleft)

    # Draw vision circle for the first unit to avoid clutter
    if game.units:
        first_unit = game.units[0]
        vision_radius_px = settings['vision_radius'] * TILE_SIZE
        # Ensure radius is at least 1 to be visible
        if vision_radius_px > 0:
            center_px = (first_unit.x * TILE_SIZE + TILE_SIZE // 2,
                         first_unit.y * TILE_SIZE + TILE_SIZE // 2)

            # To draw a transparent circle, we must draw it on a separate surface.
            circle_surface_size = vision_radius_px * 2 + 4 # Add padding for line width
            circle_surface = pygame.Surface((circle_surface_size, circle_surface_size), pygame.SRCALPHA)

            pygame.draw.circle(
                circle_surface,
                VISION_COLOR,
                (circle_surface_size // 2, circle_surface_size // 2),
                vision_radius_px,
                width=2
            )
            screen.blit(circle_surface, (center_px[0] - circle_surface_size // 2, center_px[1] - circle_surface_size // 2))

    # Draw target (fully opaque)
    target_rect = pygame.Rect(game.target[0] * TILE_SIZE, game.target[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
    pygame.draw.rect(screen, TARGET_COLOR, target_rect)

def main():
    """Main function to run the simulation."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Simplified CPU Simulation")
    clock = pygame.time.Clock()

    ui_manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))
    ui_panel_rect = pygame.Rect(GRID_WIDTH * TILE_SIZE, 0, UI_WIDTH, SCREEN_HEIGHT)
    ui = SimplifiedUI(rect=ui_panel_rect, manager=ui_manager)

    game = SimplifiedGame(width=GRID_WIDTH, height=GRID_HEIGHT, population_size=POPULATION_SIZE)
    step_counter = 0
    current_state = GameState.SIMULATING

    multiprocessing.set_start_method("spawn", force=True)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    running = True
    while running:
        time_delta = clock.tick(FPS) / 1000.0
        settings = ui.get_current_settings()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == ui.mode_button:
                    if current_state == GameState.SIMULATING:
                        current_state = GameState.EDITING
                        ui.show_editor_ui()
                    else:
                        current_state = GameState.SIMULATING
                        ui.show_simulation_ui()
                elif event.ui_element == ui.apply_button:
                    game = SimplifiedGame(
                        width=GRID_WIDTH, height=GRID_HEIGHT,
                        population_size=POPULATION_SIZE,
                        mlp_arch_str=settings['mlp_arch_str'],
                        perception_radius=settings['vision_radius'],
                        steps_per_gen=settings['sim_length']
                    )
                    step_counter = 0

            ui_manager.process_events(event)

        ui.update_labels()

        if current_state == GameState.EDITING:
            buttons = pygame.mouse.get_pressed()
            mx, my = pygame.mouse.get_pos()
            grid_x, grid_y = mx // TILE_SIZE, my // TILE_SIZE

            if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                if buttons[0]: # Left-click to draw
                    game.tile_map.set_tile(grid_x, grid_y, Tile.WALL)
                elif buttons[2]: # Right-click to erase
                    game.tile_map.set_tile(grid_x, grid_y, Tile.EMPTY)
                elif buttons[1]: # Middle-click to move target
                    game.target = (grid_x, grid_y)

        elif current_state == GameState.SIMULATING:
            for _ in range(settings['sim_speed']):
                if step_counter < game.steps_per_generation:
                    tasks = []
                    for unit in game.units:
                        brain_weights = [w.copy() for w in unit.brain.weights]
                        brain_biases = [b.copy() for b in unit.brain.biases]
                        tasks.append((
                            unit.id, unit.x, unit.y,
                            brain_weights, brain_biases,
                            game.tile_map.static_grid,
                            game.tile_map.dynamic_grid,
                            game.target,
                            game.mlp_arch,
                            game.perception_radius
                        ))
                    results = pool.map(process_unit_logic, tasks)
                    game.update_simulation_with_results(results)
                    step_counter += 1
                else:
                    game.evolve_population()
                    step_counter = 0
                    break

        ui_manager.update(time_delta)
        screen.fill(BLACK)
        draw_tile_map(screen, game)
        draw_dynamic_elements(screen, game, settings)
        ui_manager.draw_ui(screen)
        pygame.display.flip()

    pool.close()
    pool.join()
    pygame.quit()

if __name__ == '__main__':
    main()
