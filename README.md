# Python Brains: An MLP-based RTS AI Trainer

This project is a simulation environment for designing and training "brains" for units in a Real-Time Strategy (RTS) game. Instead of directly controlling units, you design a Multi-Layer Perceptron (MLP) — a type of neural network — and train it using a genetic algorithm to perform specific tasks.

## Features

-   **Dynamic AI Creation**: Use the in-game menu to define the architecture of your AI brains, including the number of hidden layers, the number of sensory "whiskers," and the types of objects the AI can perceive.
-   **Dual Training Modes**:
    -   **Navigation**: Train your AI to seek a target, rewarding it for getting closer over time.
    -   **Combat**: Train your AI to fight a stationary enemy, rewarding it for dealing damage.
-   **Genetic Algorithm**: New generations of brains are created by evolving the most successful brains from the previous generation through elitism, crossover, and mutation.
-   **Typed Whisker Perception**: Units "see" the world using rays (whiskers). This system can distinguish between different object types (walls, enemies, other units), providing rich input for the AI.
-   **Save & Load Brains**: Save the best-performing brain from a training session and load it back later to continue training or refine its architecture.

## Requirements

-   Python 3.x
-   NumPy
-   Pygame
-   Pygame GUI

These packages are listed in `requirements.txt`.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python main.py
    ```

## How to Use

The application has two main screens: the **Simulation View** and the **AI Design Menu**.

-   **AI Design Menu**:
    -   Click the "AI Design Menu" button to open this screen.
    -   Here you can define the MLP's hidden layers (e.g., `16,8`), the number of whiskers, whether to enable the "Attack" action, and what object types the AI can sense.
    -   Click "Create New Population" to apply your new design and return to the simulation.
    -   You can also "Load Last Saved Brain" from this menu.

-   **Simulation View**:
    -   This is the main view where you can watch your population of units learn.
    -   Use the "Train Navigation" and "Train Combat" buttons at the bottom to switch between the two fitness modes. The simulation will reset with the new goal.
    -   Use the "Save Fittest Brain" button to save the best-performing brain of the current population to the `saved_brains/` directory.
