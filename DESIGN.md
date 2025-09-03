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
