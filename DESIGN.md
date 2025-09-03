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

## 8. Architecture V4: Performance Optimization
As the complexity of the simulation and the size of the population grew, the single-threaded nature of the training loop in `trainer.py` became a significant performance bottleneck. The application was unable to utilize modern multi-core CPUs effectively. This version introduces a parallel processing architecture to address this.

### 8.1. Identifying the Bottleneck
The primary bottleneck was located in the `run_generation_step` method within the `TrainingSimulation` class. This method iterated through every unit in the population sequentially, performing two computationally expensive tasks for each:
1.  **Input Calculation (`get_inputs`):** Ray-casting whiskers and checking for collisions with all other objects in the world.
2.  **Neural Network Inference (`brain.forward`):** Performing a forward pass through the MLP.
Since the state of each unit within a single simulation step is independent of the others, this loop was an ideal candidate for parallelization.

### 8.2. Parallel Processing with `multiprocessing`
To solve the bottleneck, the simulation loop was re-architected to use Python's built-in `multiprocessing` module.

-   **Process Pool:** A `multiprocessing.Pool` is now initialized in the `TrainingSimulation` constructor, creating a pool of persistent worker processes, typically one for each available CPU core.
-   **Worker Function:** The core logic for updating a single unit was extracted from the `Unit` class and placed into a standalone `run_single_unit_step` function in `trainer.py`. This "worker function" is what gets distributed to the pool.
-   **Data Serialization:** Because `multiprocessing` requires data to be "pickled" to be sent between processes, game objects like `Unit` and `Target` were given a `to_dict()` method to convert their state into a simple, serializable dictionary. Complex objects that are not easily pickled (like `pygame.Vector2`) are converted to basic types (like tuples) before being sent.
-   **Efficient State Sharing:** To avoid the high overhead of sending the large `TileMap` object to every worker for every task, it is now passed once to each worker process upon its creation using the pool's `initializer` function.

This change allows the simulation to scale with the number of CPU cores, leading to a dramatic reduction in the time required to train a generation of brains.

### 8.3. Spatial Partitioning with a Quadtree
The next major bottleneck identified was in the collision detection logic within `get_unit_inputs`. For each unit, the original implementation performed a raycast check against every other object in the simulation, leading to O(n²) complexity that did not scale well with a larger number of units.

To address this, a Quadtree was implemented to spatially partition the game world.

-   **Implementation:** A `QuadTree` class was built from scratch in `quadtree.py`. It recursively subdivides the world into four quadrants, storing objects in its leaf nodes.
-   **Optimized Perception:** Before a worker process is tasked with calculating a unit's inputs, the main process now queries the Quadtree for objects that are physically near the unit. Only this small, localized list of objects is passed to the worker, dramatically reducing the number of collision checks required for the unit's "whiskers". This change also significantly reduces the amount of data that needs to be pickled and sent to each process.
-   **Optimized Projectile Collisions:** The projectile update loop was also modified to use the Quadtree. Instead of checking against all possible targets, projectiles now query the Quadtree to find nearby objects, making the combat simulation more efficient.
-   **Debug Visualization:** A toggleable debug view was added to render the Quadtree's boundaries, allowing for easy visual verification of its state and behavior during runtime.

### 8.4. GPU Acceleration with OpenCL
While `multiprocessing` effectively utilized multiple CPU cores, the MLP forward pass and whisker calculations remained candidates for further optimization. To leverage the massive parallelism of modern GPUs, the PyOpenCL library was integrated as a more hardware-agnostic solution than the previous CuPy implementation.

-   **Unified Perception Kernel:** The entire perception system (whisker-object and whisker-wall intersection checks) was offloaded to a single, unified OpenCL kernel in `math_utils_opencl.py`. This kernel, `unified_perception_kernel`, computes all intersections for all whiskers of a unit in a single, parallel GPU operation. This approach is highly efficient as it minimizes the number of kernel launches and data transfers between the CPU and GPU.
-   **Conditional Perception:** The unified kernel and its Python wrapper were enhanced with flags (`detect_walls`, `detect_circles`) to conditionally disable parts of the perception calculation. If a brain is configured not to see walls, for example, the wall-detection part of the kernel is skipped entirely, further optimizing performance for specialized brains.
-   **Initial MLP Acceleration:** The `MLP.forward()` method was offloaded to the GPU. However, the initial implementation was naive, transferring data to and from the GPU for each layer in the network, creating a new performance bottleneck.

### 8.5. Advanced OpenCL MLP Optimization
The naive OpenCL implementation of the forward pass was found to be inefficient due to excessive data transfers between the host (CPU) and the device (GPU). A new, highly optimized approach was implemented to resolve this.

-   **Zero-Copy Forward Pass:** The `MLPOpenCL.forward` method was re-architected. Instead of transferring intermediate results for each layer back to the CPU, the new implementation uses a "ping-pong" strategy with two pre-allocated buffers that reside on the GPU for the duration of a forward pass.
-   **Ping-Pong Buffer Strategy:** A custom OpenCL kernel, `forward_layer`, was written. The `forward` method uses two intermediate buffers to "ping-pong" between. The output of one layer (written to one buffer) becomes the input for the next layer (read from that same buffer) without ever leaving the GPU.
-   **Efficient Buffer Caching:** The `_gpu_brain_cache` in the `trainer.py` worker processes was upgraded. When a brain is first encountered, the worker now allocates the persistent `weights`, `biases`, and the intermediate buffers on the GPU. These buffers are reused for all subsequent forward passes of that same brain, minimizing overhead.

This advanced optimization ensures that for a given MLP forward pass, there is only one data transfer from the CPU to the GPU (the initial sensory inputs) and one transfer back (the final action outputs), maximizing the performance gains from GPU acceleration.

## 9. Architecture V5: Rendering Performance Optimization
While the previous optimizations focused on the simulation's computational bottlenecks (MLP forward pass and collision detection), the application's frame rate could still suffer with very large populations due to the rendering load. This version introduces an optimization to decouple the simulation rate from the rendering rate.

-   **Decoupled Simulation and Rendering:** The core idea is to continue simulating the entire population of units but only render a small, user-defined subset of them. This drastically reduces the number of draw calls per frame, improving visual performance (FPS) without sacrificing the accuracy or scale of the underlying simulation.
-   **Drawing the Fittest:** To provide the most useful visual information, the system is designed to draw the top N fittest units from the preceding generation. This allows the user to visually track the progress of the most successful individuals.
-   **New UI Controls:** To manage this feature, the main simulation UI was updated with two new sliders:
    -   **Population Size:** Allows the user to dynamically change the total number of units in the simulation.
    -   **Drawn Units:** Allows the user to control how many of the fittest units are rendered on screen.
-   **UI Refactoring:** To accommodate these new controls cleanly, the simulation-related UI elements were refactored from `main.py` into a dedicated `SimulationUI` class in `ui.py`.
