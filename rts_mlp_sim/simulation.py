import pygame
import sys
import os
import argparse

# --- Headless Mode Setup ---
# This allows the simulation to run in an environment without a screen
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode for testing.")
parser.add_argument("--screenshot", type=str, default=None, help="Take a screenshot of the initial state and save to this file.")
args = parser.parse_args()

if args.headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"


import torch
import torch.optim as optim
import random
from collections import deque
from agent import MLP
from visualization import MLPVisualizer

import pygame_gui


# --- Constants ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600
SIMULATION_PANEL_WIDTH = 800
MLP_PANEL_WIDTH = SCREEN_WIDTH - SIMULATION_PANEL_WIDTH
BACKGROUND_COLOR = (0, 0, 0)  # Black

# Colors for our objects
UNIT_COLOR = (0, 255, 0)      # Green
REWARD_COLOR = (255, 255, 0)   # Yellow
OBSTACLE_COLOR = (255, 0, 0)   # Red
UNIT_SPEED = 5

# --- RL Constants ---
MAX_FRAMES_PER_EPISODE = 500
STATE_SIZE = 6  # unit (x,y), reward (x,y), obstacle (x,y)
ACTION_SIZE = 4  # up, down, left, right
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# --- Agent and Environment Setup ---
model = MLP(STATE_SIZE, ACTION_SIZE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_SIZE)
epsilon = EPSILON_START

def get_state(unit, reward, obstacle):
    """Gathers the state of the game into a tensor."""
    state = [
        unit.x / SIMULATION_PANEL_WIDTH, unit.y / SCREEN_HEIGHT,
        reward.x / SIMULATION_PANEL_WIDTH, reward.y / SCREEN_HEIGHT,
        obstacle.x / SIMULATION_PANEL_WIDTH, obstacle.y / SCREEN_HEIGHT
    ]
    return torch.FloatTensor(state).unsqueeze(0)

def select_action(state):
    """Selects an action using an epsilon-greedy policy."""
    global epsilon
    if random.random() < epsilon:
        return random.randrange(ACTION_SIZE)  # Explore
    with torch.no_grad():
        # Exploit: select the action with the highest Q-value
        return model(state).max(1)[1].view(1, 1).item()

def main():
    """Main function to run the simulation."""
    global epsilon
    pygame.init()
    window_surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("RTS MLP Simulation")
    background_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    background_surface.fill(pygame.Color('#101010'))

    # --- GUI Setup ---
    theme_path = os.path.join(os.path.dirname(__file__), 'theme.json')
    ui_manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT), theme_path)
    sim_panel = pygame_gui.elements.UIPanel(
        relative_rect=pygame.Rect((0, 0), (SIMULATION_PANEL_WIDTH, SCREEN_HEIGHT)),
        manager=ui_manager,
        anchors={'left': 'left', 'right': 'left', 'top': 'top', 'bottom': 'bottom'}
    )
    mlp_panel = pygame_gui.elements.UIPanel(
        relative_rect=pygame.Rect((SIMULATION_PANEL_WIDTH, 0), (MLP_PANEL_WIDTH, SCREEN_HEIGHT)),
        manager=ui_manager,
        anchors={'left': 'left', 'right': 'right', 'top': 'top', 'bottom': 'bottom'}
    )
    # Note: True resizability requires handling pygame.VIDEORESIZE events and updating panel rects,
    # which is complex. The anchoring provides basic resizing with the main window.

    mlp_visualizer = MLPVisualizer(model)
    clock = pygame.time.Clock()

    # --- Game Objects (coordinates are relative to the sim_panel) ---
    unit = pygame.Rect(100, 300, 25, 25)
    reward = pygame.Rect(600, 300, 20, 20) # Adjusted for 800 width
    obstacle = pygame.Rect(350, 250, 50, 100) # Adjusted for 800 width

    episode, total_reward, frame_count = 0, 0, 0
    running = True
    while running:
        time_delta = clock.tick(60) / 1000.0
        frame_count += 1

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.VIDEORESIZE:
                # This event is crucial for making the layout truly resizable
                ui_manager.set_window_resolution((event.w, event.h))
                background_surface = pygame.Surface((event.w, event.h))
                background_surface.fill(pygame.Color('#101010'))
            ui_manager.process_events(event)

        # --- Agent's Turn ---
        current_state = get_state(unit, reward, obstacle)
        action = select_action(current_state)

        if action == 0: unit.y -= UNIT_SPEED
        elif action == 1: unit.y += UNIT_SPEED
        elif action == 2: unit.x -= UNIT_SPEED
        elif action == 3: unit.x += UNIT_SPEED

        # --- Observe and Train ---
        reward_val = -0.1
        done = False
        if unit.colliderect(reward):
            reward_val, done = 10.0, True
        elif unit.colliderect(obstacle) or unit.left < 0 or unit.right > SIMULATION_PANEL_WIDTH or unit.top < 0 or unit.bottom > SCREEN_HEIGHT:
            reward_val, done = -10.0, True
        if frame_count >= MAX_FRAMES_PER_EPISODE:
            reward_val, done = -5.0, True

        next_state = get_state(unit, reward, obstacle)
        memory.append((current_state, action, reward_val, next_state, done))
        train_step()

        # --- Episode Management ---
        total_reward += reward_val
        if done:
            episode += 1
            print(f"Episode: {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}")
            total_reward, frame_count = 0, 0
            unit.x, unit.y = 100, 300

        if epsilon > EPSILON_END:
            epsilon *= EPSILON_DECAY

        # --- Drawing ---
        ui_manager.update(time_delta)
        window_surface.blit(background_surface, (0, 0))

        # Draw simulation on its panel
        sim_panel.image.fill(BACKGROUND_COLOR)
        pygame.draw.rect(sim_panel.image, UNIT_COLOR, unit)
        pygame.draw.rect(sim_panel.image, REWARD_COLOR, reward)
        pygame.draw.rect(sim_panel.image, OBSTACLE_COLOR, obstacle)

        # Draw MLP viz on its panel
        mlp_visualizer.draw(mlp_panel.image)

        ui_manager.draw_ui(window_surface)
        pygame.display.update()

    mlp_visualizer.remove_hooks()
    pygame.quit()
    sys.exit()

def train_step():
    """Performs a single training step on a batch of experiences."""
    if len(memory) < BATCH_SIZE:
        return # Not enough memories to train

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.cat(states)
    actions = torch.LongTensor(actions).view(-1, 1)
    rewards = torch.FloatTensor(rewards).view(-1, 1)
    next_states = torch.cat(next_states)
    dones = torch.FloatTensor(dones).view(-1, 1)

    # Get current Q values
    current_q_values = model(states).gather(1, actions)

    # Get next Q values
    next_q_values = model(next_states).max(1)[0].detach().view(-1, 1)

    # Compute target Q values
    target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

    # Compute loss
    loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def take_screenshot(filename):
    """Initializes the full UI, draws one frame, and saves it to a file."""
    print(f"Taking screenshot of initial state and saving to {filename}...")
    pygame.init()
    window_surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    # --- GUI Setup ---
    theme_path = os.path.join(os.path.dirname(__file__), 'theme.json')
    ui_manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT), theme_path)
    sim_panel = pygame_gui.elements.UIPanel(
        relative_rect=pygame.Rect((0, 0), (SIMULATION_PANEL_WIDTH, SCREEN_HEIGHT)),
        manager=ui_manager)
    mlp_panel = pygame_gui.elements.UIPanel(
        relative_rect=pygame.Rect((SIMULATION_PANEL_WIDTH, 0), (MLP_PANEL_WIDTH, SCREEN_HEIGHT)),
        manager=ui_manager)

    # --- Model and Visualizer ---
    mlp_visualizer = MLPVisualizer(model)

    # --- Game Objects ---
    unit = pygame.Rect(100, 300, 25, 25)
    reward = pygame.Rect(600, 300, 20, 20)
    obstacle = pygame.Rect(350, 250, 50, 100)

    # --- Trigger one forward pass to populate hooks ---
    initial_state = get_state(unit, reward, obstacle)
    with torch.no_grad():
        model(initial_state)

    # --- Drawing ---
    # Draw simulation
    sim_panel.image.fill(BACKGROUND_COLOR)
    pygame.draw.rect(sim_panel.image, UNIT_COLOR, unit)
    pygame.draw.rect(sim_panel.image, REWARD_COLOR, reward)
    pygame.draw.rect(sim_panel.image, OBSTACLE_COLOR, obstacle)

    # Draw MLP visualization
    mlp_visualizer.draw(mlp_panel.image)

    # Update and draw the entire UI
    ui_manager.update(0)
    ui_manager.draw_ui(window_surface)

    # Save the final surface
    pygame.image.save(window_surface, filename)
    mlp_visualizer.remove_hooks()
    pygame.quit()
    print("Screenshot saved.")

if __name__ == '__main__':
    if args.screenshot:
        take_screenshot(args.screenshot)
    else:
        main()
