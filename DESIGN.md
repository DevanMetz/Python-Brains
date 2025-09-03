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
-   **Typed Perception:** Units perceive the world through a "typed whisker" system.
    -   Whiskers are rays cast out from the unit in a forward-facing arc.
    -   They can detect different *types* of objects (e.g., "wall", "enemy", "unit"). The player can configure which types a brain can "see".
    -   For each whisker, a separate input neuron is created for each perceivable object type.
    -   The value of the input is `1.0 - (distance / max_whisker_length)`, allowing the AI to know both the type and proximity of a detected object.
-   **Inputs/Outputs:**
    -   **Inputs:** The primary inputs come from the typed whisker system. The total number of whisker inputs is `num_whiskers * num_perceivable_types`. Additionally, there are 2 inputs for the unit's internal state: current velocity and current angle.
    -   **Outputs:** The MLP has 2 or 3 outputs, activated by `tanh`:
        1.  `Turn`: Controls the unit's rotation.
        2.  `Move`: Controls the unit's forward acceleration.
        3.  `Attack` (Optional): If enabled in the design menu, this output controls when the unit fires a projectile.

## 4. Technology Stack
-   **Language:** Python
-   **Graphics/Game Loop:** Pygame
-   **UI:** `pygame-gui`
-   **Numerical Computation:** NumPy

## 5. Architecture V1: Foundational Prototype
The initial goal was to create a minimal viable product demonstrating the core training loop for navigation.
-   **Objective:** Train a single unit to move towards a designated target.
-   **Components:** A `Unit`, a `Target`, and a `TrainingSimulation` using a genetic algorithm.

## 6. Architecture V2: The AI Design Toolkit
The second major iteration introduced the core gameplay loop: player-driven AI design.

### 6.1. Game State Management
The application is managed by a state machine in `main.py` that switches between `GameState.SIMULATION` and `GameState.DESIGN_MENU`.

### 6.2. UI and Dynamic AI Configuration
The `DesignMenu` (`ui.py`) allows the player to configure:
-   **MLP Architecture**: The number and size of hidden layers.
-   **Sensory Inputs**: The number of whiskers and the object types they can perceive. This dynamically changes the size of the MLP's input layer.
-   **Outputs**: Whether to enable the "Attack" action, which changes the size of the output layer.

### 6.3. Save/Load System
A system for persisting AI brains was implemented.
-   **Location**: `saved_brains/`
-   **Format**: A `.json` file for architecture metadata (layer sizes, whisker config) and a `.npz` file for weights and biases.

## 7. Architecture V3: Combat & Advanced Training
The third iteration introduced combat mechanics and more advanced training options.

### 7.1. Combat System
-   **Projectiles:** Units with an enabled attack output can fire `Projectile` objects. These are simple, straight-moving entities with a limited lifespan.
-   **Enemies:** A stationary `Enemy` class was introduced. It has health and can be damaged by projectiles.
-   **Damage Tracking:** Units track the total damage they have dealt, which is used for fitness calculation.

### 7.2. Dual Training Modes
The `TrainingSimulation` now supports two distinct modes, selectable in the UI:
-   **`TrainingMode.NAVIGATE`**: The original mode. Fitness is calculated based on the unit's proximity to a `Target` object.
-   **`TrainingMode.COMBAT`**: A new mode. Fitness is calculated based on the damage dealt to the `Enemy`, with a small bonus for being close to the enemy. This allows for the evolution of aggressive, combat-oriented brains.
