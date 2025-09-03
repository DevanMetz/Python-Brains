# Software Design Specification

This document is a living specification for the MLP-based RTS game. It will be updated as the project evolves.

## 1. Core Concept
The project is a Real-Time Strategy (RTS) game where the player does not directly control their units. Instead, the player designs, trains, and assigns "brains" to the units. These brains are Multilayer Perceptrons (MLPs).

## 2. Gameplay Loop
1.  **Design:** The player designs an MLP architecture (number of layers, nodes, etc.) and specifies its inputs and outputs.
2.  **Train:** The player trains the MLP in a dedicated training simulation using techniques like genetic algorithms or reinforcement learning. The goal is to teach the MLP a specific behavior (e.g., resource gathering, attacking).
3.  **Assign:** The player creates a library of these trained MLPs.
4.  **Deploy:** In the main game, the player assigns these trained brains to individual units, who will then act autonomously based on their learned behaviors.

## 3. Unit AI and Perception
-   **Brains:** The core of each unit's AI is an MLP.
-   **Perception:** Units perceive the world through a "whisker" system.
    -   Whiskers are rays cast out from the unit in a forward-facing arc.
    -   They can detect different types of objects (enemies, resources, walls, friendlies).
    -   The data from these whiskers (e.g., distance to a detected object) forms the primary input to the unit's MLP.
-   **Inputs/Outputs:**
    -   **Inputs:** Whisker data, internal state (e.g., health, position).
    -   **Outputs:** Actions (e.g., move direction, attack, collect resources).

## 4. Technology Stack
-   **Language:** Python
-   **Graphics/Game Loop:** Pygame
-   **Numerical Computation:** NumPy

## 5. Current Milestone: "Hello, World!" of AI
The initial goal is to create a minimal viable product that demonstrates the core training loop.
-   **Objective:** Train a single unit to move towards a designated target on the map.
-   **Components:**
    -   A single `Unit` instance.
    -   A `Target` object.
    -   A `TrainingSimulation` that uses a genetic algorithm to evolve the unit's MLP brain.
    -   A visual representation of the simulation running.

## 6. Architecture V2: The AI Design Toolkit

The second major iteration introduces the core gameplay loop: player-driven AI design.

### 6.1. Game State Management
The application is managed by a state machine in `main.py` that can switch between two primary states:
-   `GameState.SIMULATION`: The view where the training simulation runs.
-   `GameState.DESIGN_MENU`: The UI view for creating and configuring AI brains.

### 6.2. UI (`pygame-gui`)
The project uses the `pygame-gui` library to handle UI elements. A `DesignMenu` class in `ui.py` encapsulates the elements and logic for the design menu.

### 6.3. AI Design Features
The AI Design Menu allows the player to configure:
-   **MLP Architecture**: The player can define the number and size of hidden layers for the neural network.
-   **Sensory Inputs**: The player can choose the number of "whiskers" the unit uses to see, which dynamically changes the size of the MLP's input layer.

### 6.4. Save/Load System
A system for persisting AI brains has been implemented.
-   **Location**: Brains are saved in the `saved_brains/` directory.
-   **Format**:
    -   The brain's architecture (layer sizes, whisker count) is saved to a `.json` file.
    -   The trained weights and biases of the MLP are saved to a `.npz` file using NumPy.
-   **Functionality**: The player can save the best-performing brain from a training session and load it back later to continue training or use it.
