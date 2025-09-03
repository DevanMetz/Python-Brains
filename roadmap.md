# Project Roadmap: MLP-RTS Game

This document outlines the development milestones for the AI-driven Real-Time Strategy game.

---

## ✅ Milestone 1: Foundational Prototype *(Complete)*

This milestone established the core engine and proof-of-concept for the game.

-   **[✓] Simulation Engine:** A Pygame-based world to simulate units and objects.
-   **[✓] Genetic Algorithm Trainer:** A system to train AI brains over generations.
-   **[✓] Basic Unit AI:** Units with MLP brains can be trained to navigate towards a target.
-   **[✓] Core Documentation:** Initial `DESIGN.md` and `AGENTS.md` created.

---

## ✅ Milestone 2: The AI Design Toolkit *(Complete)*

This milestone focused on creating the core gameplay loop: empowering the player to design their own AIs.

-   **[✓] UI for AI Design:** A menu allows players to define the MLP architecture (hidden layers, nodes).
-   **[✓] Dynamic Sensory Inputs:** Players can customize the number of "whiskers" a unit uses, which dynamically alters the AI's input layer.
-   **[✓] State Management:** The application now handles multiple states (e.g., simulation vs. design menu).
-   **[✓] Save/Load Brains:** A system to save trained brains (architecture + weights) and load them back into the simulation.

---

## ➡️ Milestone 3: Advanced AI - Perception & Combat *(Proposed Next Step)*

This milestone will introduce the first elements of conflict and more complex AI behaviors.

-   **[ ] Implement "Attack" Action:** Add an "attack" output to the MLP and introduce a basic, damageable enemy target.
-   **[ ] Typed Perception (Smarter Whiskers):** Enhance the sensory system so whiskers can report the *type* of object detected (e.g., Wall, Enemy, Friendly), not just the distance. This allows for more specialized AI.
-   **[ ] Combat Training Scenario:** Create a new training environment where the fitness function rewards combat effectiveness (e.g., damage dealt to the enemy).

---

## 🚀 Future Milestones *(Potential Ideas)*

These are potential future directions to expand the game based on the established foundation.

-   **[ ] Dynamic Enemies & Base Defense:** Introduce enemy units that move and attack. Create scenarios where the player must train defensive AIs.
-   **[ ] Resource Gathering & Economy:** Add resource nodes and a "collect" action. Train AIs to manage an economy and build new units.
-   **[ ] Multi-Unit & Cooperative AI:** Design scenarios that require multiple units with different, specialized brains to work together to solve a complex objective.
-   **[ ] Advanced Training Methods:** Explore and implement alternative training paradigms, such as Reinforcement Learning (Q-learning, PPO), as an option for the player.
