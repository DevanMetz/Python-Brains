import pygame
import pygame_gui
import sys
from enum import Enum
from trainer import TrainingSimulation, TrainingMode
from ui import DesignMenu
from map import TileMap, Tile

# --- Constants ---
WORLD_WIDTH, WORLD_HEIGHT = 800, 600
FPS = 60
# This is now a variable, not a constant
# STEPS_PER_GENERATION = 400
SPEED_LEVELS = [1, 2, 4, 8, 16, 32]
TILE_SIZE = 20

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class GameState(Enum):
    SIMULATION = 1
    DESIGN_MENU = 2
    MAP_EDITOR = 3

def main():
    # --- Pygame Setup ---
    pygame.init()
    screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
    pygame.display.set_caption("MLP Training Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)

    # --- UI Manager Setup ---
    ui_manager = pygame_gui.UIManager((WORLD_WIDTH, WORLD_HEIGHT))

    # --- Map Setup ---
    tile_map = TileMap(WORLD_WIDTH, WORLD_HEIGHT, TILE_SIZE)

    # --- Simulation Setup ---
    trainer = TrainingSimulation(
        population_size=50,
        world_size=(WORLD_WIDTH, WORLD_HEIGHT),
        tile_map=tile_map
    )
    step_counter = 0
    steps_per_generation = 400 # Now a variable
    best_fitness = 0
    speed_index = 0
    simulation_speed = SPEED_LEVELS[speed_index]

    # --- Game State ---
    current_state = GameState.SIMULATION

    # --- UI Elements ---
    to_design_menu_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((WORLD_WIDTH - 220, 20), (200, 40)),
        text='AI Design Menu', manager=ui_manager)

    save_brain_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((WORLD_WIDTH - 220, 70), (200, 40)),
        text='Save Fittest Brain', manager=ui_manager)

    to_map_editor_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((WORLD_WIDTH - 220, 120), (200, 40)),
        text='Map Editor', manager=ui_manager)

    back_to_sim_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((WORLD_WIDTH - 220, 20), (200, 40)),
        text='Back to Simulation', manager=ui_manager, visible=False)


    design_menu = DesignMenu(
        rect=pygame.Rect((WORLD_WIDTH / 2 - 150, WORLD_HEIGHT / 2 - 230), (300, 460)),
        ui_manager=ui_manager)

    # --- Training Mode Buttons ---
    train_nav_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((20, WORLD_HEIGHT - 60), (150, 40)),
        text='Train Navigation', manager=ui_manager)
    train_combat_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((180, WORLD_HEIGHT - 60), (150, 40)),
        text='Train Combat', manager=ui_manager)

    # --- Simulation Length Slider ---
    sim_length_label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((WORLD_WIDTH - 230, WORLD_HEIGHT - 55), (80, 30)),
        text='Steps:', manager=ui_manager)
    sim_length_value_label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((WORLD_WIDTH - 160, WORLD_HEIGHT - 55), (50, 30)),
        text=str(steps_per_generation), manager=ui_manager)
    sim_length_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((WORLD_WIDTH - 230, WORLD_HEIGHT - 30), (200, 20)),
        start_value=steps_per_generation, value_range=(100, 2000), manager=ui_manager)


    # --- Main Loop ---
    running = True
    while running:
        time_delta = clock.tick(FPS) / 1000.0

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    speed_index = min(speed_index + 1, len(SPEED_LEVELS) - 1)
                    simulation_speed = SPEED_LEVELS[speed_index]
                elif event.key == pygame.K_DOWN:
                    speed_index = max(speed_index - 1, 0)
                    simulation_speed = SPEED_LEVELS[speed_index]

            if current_state == GameState.MAP_EDITOR:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2: # Middle mouse
                    mx, my = pygame.mouse.get_pos()
                    trainer.target.position.x = mx
                    trainer.target.position.y = my

            if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == design_menu.whisker_slider:
                    num_whiskers = int(event.value)
                    design_menu.whisker_count_label.set_text(str(num_whiskers))
                elif event.ui_element == design_menu.whisker_length_slider:
                    length = int(event.value)
                    design_menu.whisker_length_value_label.set_text(str(length))
                elif event.ui_element == sim_length_slider:
                    steps_per_generation = int(event.value)
                    sim_length_value_label.set_text(str(steps_per_generation))

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == to_design_menu_button:
                    current_state = GameState.DESIGN_MENU
                elif event.ui_element == to_map_editor_button:
                    current_state = GameState.MAP_EDITOR
                elif event.ui_element == back_to_sim_button:
                    current_state = GameState.SIMULATION
                elif event.ui_element == design_menu.update_button:
                    num_whiskers = int(design_menu.whisker_slider.get_current_value())
                    whisker_length = int(design_menu.whisker_length_slider.get_current_value())
                    attack_enabled = design_menu.attack_checkbox.is_checked
                    perceivable_types = design_menu.get_perceivable_types()
                    # whiskers * types + velocity + angle + target_dx + target_dy
                    num_inputs = num_whiskers * len(perceivable_types) + 2 + 2
                    num_outputs = 3 if attack_enabled else 2
                    new_arch = design_menu.get_architecture_from_input(input_nodes=num_inputs, output_nodes=num_outputs)
                    if new_arch:
                        trainer.rebuild_with_new_architecture(new_arch, num_whiskers, perceivable_types, whisker_length)
                        best_fitness = 0
                        current_state = GameState.SIMULATION
                elif event.ui_element == design_menu.close_button:
                    current_state = GameState.SIMULATION
                elif event.ui_element == save_brain_button:
                    trainer.save_fittest_brain()
                elif event.ui_element == design_menu.load_button:
                    trainer.load_brain_from_file()
                    current_state = GameState.SIMULATION
                elif event.ui_element == train_nav_button:
                    if trainer.training_mode != TrainingMode.NAVIGATE:
                        trainer.training_mode = TrainingMode.NAVIGATE
                        trainer.evolve_population()
                elif event.ui_element == train_combat_button:
                    if trainer.training_mode != TrainingMode.COMBAT:
                        trainer.training_mode = TrainingMode.COMBAT
                        trainer.evolve_population()

            ui_manager.process_events(event)

        # --- Continuous Input for Editor ---
        if current_state == GameState.MAP_EDITOR:
            buttons = pygame.mouse.get_pressed()
            if buttons[0] or buttons[2]:
                mx, my = pygame.mouse.get_pos()
                grid_x = mx // TILE_SIZE
                grid_y = my // TILE_SIZE
                tile_type = Tile.WALL if buttons[0] else Tile.EMPTY
                tile_map.set_tile(grid_x, grid_y, tile_type)

        ui_manager.update(time_delta)

        # --- Game State Logic and UI Visibility ---
        if current_state == GameState.DESIGN_MENU:
            to_design_menu_button.hide()
            save_brain_button.hide()
            to_map_editor_button.hide()
            back_to_sim_button.hide()
            train_nav_button.hide()
            train_combat_button.hide()
            sim_length_label.hide()
            sim_length_slider.hide()
            sim_length_value_label.hide()
            design_menu.show()
        elif current_state == GameState.SIMULATION:
            design_menu.hide()
            back_to_sim_button.hide()
            to_design_menu_button.show()
            save_brain_button.show()
            to_map_editor_button.show()
            train_nav_button.show()
            train_combat_button.show()
            sim_length_label.show()
            sim_length_slider.show()
            sim_length_value_label.show()

            for _ in range(simulation_speed):
                if step_counter < steps_per_generation:
                    trainer.run_generation_step()
                    step_counter += 1
                else:
                    best_fitness = trainer.evolve_population()
                    step_counter = 0
                    print(f"Generation: {trainer.generation}, Best Fitness: {best_fitness:.2f}")
                    break
        elif current_state == GameState.MAP_EDITOR:
            to_design_menu_button.hide()
            save_brain_button.hide()
            to_map_editor_button.hide()
            train_nav_button.hide()
            train_combat_button.hide()
            sim_length_label.hide()
            sim_length_slider.hide()
            sim_length_value_label.hide()
            design_menu.hide()
            back_to_sim_button.show()

        # --- Drawing ---
        screen.fill(BLACK)
        if current_state == GameState.SIMULATION:
            tile_map.draw(screen) # Draw map behind units
            for obj in trainer.world_objects: obj.draw(screen)
            for unit in trainer.population: unit.draw(screen)
            for proj in trainer.projectiles: proj.draw(screen)

            gen_text = font.render(f"Generation: {trainer.generation}", True, WHITE)
            fitness_text = font.render(f"Best Fitness: {best_fitness:.2f}", True, WHITE)
            mode_text = font.render(f"Mode: {'COMBAT' if trainer.training_mode == TrainingMode.COMBAT else 'NAVIGATE'}", True, WHITE)
            speed_text = font.render(f"Speed: {simulation_speed}x", True, WHITE)
            fps_text = font.render(f"FPS: {clock.get_fps():.0f}", True, WHITE)
            screen.blit(gen_text, (20, 20))
            screen.blit(fitness_text, (20, 50))
            screen.blit(mode_text, (20, 80))
            screen.blit(speed_text, (20, 110))
            screen.blit(fps_text, (20, 140))
        elif current_state == GameState.MAP_EDITOR:
            tile_map.draw(screen)
            trainer.target.draw(screen)

        ui_manager.draw_ui(screen)
        pygame.display.flip()

    # --- Cleanup ---
    trainer.cleanup()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
