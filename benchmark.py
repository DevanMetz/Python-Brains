import time
import numpy as np
from trainer import TrainingSimulation
from map import TileMap, Tile

POPULATION_SIZE = 500
NUM_STEPS = 1000

def run_benchmark():
    """
    Runs a benchmark of the core simulation loop to measure performance.
    """
    print("--- Starting Benchmark ---")
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Simulation steps: {NUM_STEPS}")

    # 1. Setup the simulation environment
    tile_map = TileMap(800, 600, 20)
    # Add some walls to make the simulation more realistic
    for y in range(10, 20):
        tile_map.set_tile(20, y, Tile.WALL)

    trainer = TrainingSimulation(
        population_size=POPULATION_SIZE,
        world_size=(800, 600),
        tile_map=tile_map
    )

    # 2. Run the simulation and time it
    print("Starting simulation loop...")
    start_time = time.time()

    for i in range(NUM_STEPS):
        if (i + 1) % 100 == 0:
            print(f"  Step {i+1}/{NUM_STEPS}")
        trainer.run_generation_step()

    end_time = time.time()
    total_time = end_time - start_time
    steps_per_second = NUM_STEPS / total_time

    print("\n--- Benchmark Results ---")
    print(f"Total time for {NUM_STEPS} steps: {total_time:.2f} seconds")
    print(f"Steps per second: {steps_per_second:.2f}")
    print("--------------------------")

    # 3. Important: Clean up the multiprocessing pool
    trainer.cleanup()

    return steps_per_second

if __name__ == "__main__":
    run_benchmark()
