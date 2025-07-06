#!/usr/bin/env python3
"""
Test script for Sokoban RL environment
"""

from sokoban_rl import SokobanEnv
import numpy as np

def test_environment():
    """Test the Sokoban environment"""
    print("Testing Sokoban RL Environment...")
    
    # Create environment
    env = SokobanEnv(level_idx=0)
    
    # Test reset
    state = env.reset()
    print(f"State shape: {state.shape}")
    print(f"State dtype: {state.dtype}")
    
    # Test rendering
    print("\nInitial state:")
    env.render()
    
    # Test random actions
    print("Testing random actions...")
    for step in range(10):
        action = np.random.randint(0, 4)
        state, reward, done, info = env.step(action)
        
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        print(f"Step {step+1}: Action={action_names[action]}, Reward={reward:.2f}, Done={done}")
        
        if done:
            print("Episode finished!")
            break
    
    print("\nFinal state:")
    env.render()
    
    # Test state representation
    print(f"Final state shape: {state.shape}")
    print("State channels:")
    print("  Channel 0 (Walls):", np.sum(state[0]))
    print("  Channel 1 (Empty):", np.sum(state[1]))
    print("  Channel 2 (Storage):", np.sum(state[2]))
    print("  Channel 3 (Boxes):", np.sum(state[3]))
    print("  Channel 4 (Boxes on storage):", np.sum(state[4]))
    print("  Channel 5 (Player):", np.sum(state[5]))
    
    print("\nEnvironment test completed successfully!")

if __name__ == "__main__":
    test_environment() 