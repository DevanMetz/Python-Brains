"""
This is the main entry point for the simplified, CPU-based simulation.
It uses Pygame for visualization and multiprocessing for parallel simulation.
"""
import pygame
import multiprocessing
from simplified_game import SimplifiedGame, Tile, process_unit_logic

# --- Constants ---
GRID_WIDTH, GRID_HEIGHT = 40, 30
TILE_SIZE = 20
SCREEN_WIDTH, SCREEN_HEIGHT = GRID_WIDTH * TILE_SIZE, GRID_HEIGHT * TILE_SIZE
FPS = 60
POPULATION_SIZE = 100

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRID_COLOR = (40, 40, 40)
WALL_COLOR = (100, 100, 100)
UNIT_COLOR = (0, 150, 255)
TARGET_COLOR = (0, 255, 0)

def draw_tile_map(screen, game):
    """Draws the static part of the map (walls and grid)."""
    for x in range(game.tile_map.grid_width):
        for y in range(game.tile_map.grid_height):
            if game.tile_map.static_grid[x, y] == Tile.WALL.value:
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(screen, WALL_COLOR, rect)

    # Draw grid lines
    for x in range(0, SCREEN_WIDTH, TILE_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, TILE_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (SCREEN_WIDTH, y))

def draw_dynamic_elements(screen, game):
    """Draws the units and the target."""
    # Draw units
    for unit in game.units:
        rect = pygame.Rect(unit.x * TILE_SIZE, unit.y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(screen, UNIT_COLOR, rect)

    # Draw target
    target_rect = pygame.Rect(game.target[0] * TILE_SIZE, game.target[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
    pygame.draw.rect(screen, TARGET_COLOR, target_rect)


def main():
    """Main function to run the simulation."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Simplified CPU Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    game = SimplifiedGame(width=GRID_WIDTH, height=GRID_HEIGHT, population_size=POPULATION_SIZE)
    step_counter = 0

    # --- Multiprocessing Setup ---
    # Use 'spawn' to be safe on all platforms (macOS, Windows)
    multiprocessing.set_start_method("spawn", force=True)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Simulation Logic ---
        if step_counter < game.steps_per_generation:
            # Prepare arguments for all units
            tasks = []
            for unit in game.units:
                # Need to pass copies of brain parameters, not the objects themselves
                brain_weights = [w.copy() for w in unit.brain.weights]
                brain_biases = [b.copy() for b in unit.brain.biases]
                tasks.append((
                    unit.id, unit.x, unit.y,
                    brain_weights, brain_biases,
                    game.tile_map.static_grid,
                    game.tile_map.dynamic_grid,
                    game.target,
                    game.mlp_arch
                ))

            # Run simulation step in parallel
            results = pool.map(process_unit_logic, tasks)

            # Update game state with results
            game.update_simulation_with_results(results)

            step_counter += 1
        else:
            # End of generation, evolve the population
            game.evolve_population()
            step_counter = 0

        # --- Drawing ---
        screen.fill(BLACK)
        draw_tile_map(screen, game)
        draw_dynamic_elements(screen, game)

        # --- UI Text ---
        gen_text = font.render(f"Generation: {game.generation}", True, WHITE)
        step_text = font.render(f"Step: {step_counter}/{game.steps_per_generation}", True, WHITE)
        screen.blit(gen_text, (10, 10))
        screen.blit(step_text, (10, 30))

        pygame.display.flip()
        clock.tick(FPS)

    # --- Cleanup ---
    pool.close()
    pool.join()
    pygame.quit()

if __name__ == '__main__':
    main()
