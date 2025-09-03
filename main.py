import pygame
import sys
from trainer import TrainingSimulation

# --- Constants ---
WORLD_WIDTH, WORLD_HEIGHT = 800, 600
FPS = 60
STEPS_PER_GENERATION = 400

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def main():
    # --- Pygame Setup ---
    pygame.init()
    screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
    pygame.display.set_caption("MLP Training Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)

    # --- Simulation Setup ---
    trainer = TrainingSimulation(population_size=50, world_size=(WORLD_WIDTH, WORLD_HEIGHT))
    step_counter = 0
    best_fitness = 0

    # --- Main Loop ---
    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Simulation Update ---
        if step_counter < STEPS_PER_GENERATION:
            trainer.run_generation_step()
            step_counter += 1
        else:
            # A generation has completed
            best_fitness = trainer.evolve_population()
            step_counter = 0
            print(f"Generation: {trainer.generation}, Best Fitness: {best_fitness:.2f}")

        # --- Drawing ---
        screen.fill(BLACK)

        # Draw all world objects (walls, target)
        for obj in trainer.world_objects:
            obj.draw(screen)

        # Draw all units in the population
        for unit in trainer.population:
            unit.draw(screen)

        # Draw info text
        gen_text = font.render(f"Generation: {trainer.generation}", True, WHITE)
        fitness_text = font.render(f"Best Fitness: {best_fitness:.2f}", True, WHITE)
        screen.blit(gen_text, (20, 20))
        screen.blit(fitness_text, (20, 50))

        # --- Update Display ---
        pygame.display.flip()
        clock.tick(FPS)

    # --- Cleanup ---
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
