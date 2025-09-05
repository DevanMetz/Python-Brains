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

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
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
        unit.x / SCREEN_WIDTH, unit.y / SCREEN_HEIGHT,
        reward.x / SCREEN_WIDTH, reward.y / SCREEN_HEIGHT,
        obstacle.x / SCREEN_WIDTH, obstacle.y / SCREEN_HEIGHT
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
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("RTS MLP Simulation")
    clock = pygame.time.Clock()

    unit = pygame.Rect(100, 300, 25, 25)
    reward = pygame.Rect(700, 300, 20, 20)
    obstacle = pygame.Rect(400, 250, 50, 100)

    # --- Logging and Episode Tracking ---
    episode = 0
    total_reward = 0
    frame_count = 0
    running = True

    while running:
        frame_count += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Agent's Turn ---
        # 1. Get current state
        current_state = get_state(unit, reward, obstacle)

        # 2. Select an action
        action = select_action(current_state)

        # 3. Perform action
        if action == 0: # Up
            unit.y -= UNIT_SPEED
        elif action == 1: # Down
            unit.y += UNIT_SPEED
        elif action == 2: # Left
            unit.x -= UNIT_SPEED
        elif action == 3: # Right
            unit.x += UNIT_SPEED

        # 4. Observe reward and next state
        reward_val = -0.1  # Small penalty for each step to encourage speed
        done = False
        if unit.colliderect(reward):
            reward_val = 10.0  # Big reward for reaching the goal
            done = True
        elif unit.colliderect(obstacle) or unit.left < 0 or unit.right > SCREEN_WIDTH or unit.top < 0 or unit.bottom > SCREEN_HEIGHT:
            reward_val = -10.0 # Big penalty for crashing
            done = True

        if frame_count >= MAX_FRAMES_PER_EPISODE:
            reward_val = -5.0 # Penalty for timing out
            done = True

        next_state = get_state(unit, reward, obstacle)

        # 5. Store experience in memory
        memory.append((current_state, action, reward_val, next_state, done))

        # 6. Train the model
        train_step()

        # Update reward tracking
        total_reward += reward_val

        # Handle end of episode
        if done:
            episode += 1
            print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}, Frames: {frame_count}")
            total_reward = 0 # Reset for next episode
            frame_count = 0 # Reset frame count

            unit.x, unit.y = 100, 300 # Reset unit
            # Optionally, reset obstacle and reward positions for more robust training
            # reward.x = random.randint(50, SCREEN_WIDTH - 50)
            # reward.y = random.randint(50, SCREEN_HEIGHT - 50)

        # --- Drawing ---
        screen.fill(BACKGROUND_COLOR)
        pygame.draw.rect(screen, UNIT_COLOR, unit)
        pygame.draw.rect(screen, REWARD_COLOR, reward)
        pygame.draw.rect(screen, OBSTACLE_COLOR, obstacle)
        pygame.display.flip()

        # Decay epsilon
        if epsilon > EPSILON_END:
            epsilon *= EPSILON_DECAY

        clock.tick(60) # Limit frame rate

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
    """Initializes the environment, draws one frame, and saves it to a file."""
    print(f"Taking screenshot and saving to {filename}...")
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    # Create the objects in their initial state
    unit = pygame.Rect(100, 300, 25, 25)
    reward = pygame.Rect(700, 300, 20, 20)
    obstacle = pygame.Rect(400, 250, 50, 100)

    # Draw the scene
    screen.fill(BACKGROUND_COLOR)
    pygame.draw.rect(screen, UNIT_COLOR, unit)
    pygame.draw.rect(screen, REWARD_COLOR, reward)
    pygame.draw.rect(screen, OBSTACLE_COLOR, obstacle)

    # Save the buffer to a file
    pygame.image.save(screen, filename)
    pygame.quit()
    print("Screenshot saved.")

if __name__ == '__main__':
    if args.screenshot:
        take_screenshot(args.screenshot)
    else:
        main()
