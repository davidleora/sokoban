#!/usr/bin/env python3
"""
Visualization script to watch the trained RL agent play Sokoban
"""

import pygame
import time
import torch
from sokoban_rl import SokobanEnv, DQN

# Initialize pygame
pygame.init()

# Colors
COLORS = {
    '@': (167, 135, 255),  # Player
    '+': (158, 119, 255),  # Player on storage
    '$': (255, 201, 126),  # Box
    '*': (150, 255, 127),  # Box on storage
    '.': (156, 229, 255),  # Storage
    '#': (255, 147, 209),  # Wall
    ' ': (255, 255, 190),  # Empty
}

class SokobanVisualizer:
    def __init__(self, cell_size=30):
        self.cell_size = cell_size
        self.screen = None
        self.clock = pygame.time.Clock()
        
    def setup_display(self, level):
        """Setup pygame display based on level size"""
        height = len(level)
        width = max(len(row) for row in level)
        
        screen_width = width * self.cell_size
        screen_height = height * self.cell_size
        
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Sokoban RL Agent")
        
    def render_level(self, level):
        """Render the current level state"""
        if self.screen is None:
            self.setup_display(level)
        
        # Fill background
        self.screen.fill(COLORS[' '])
        
        # Draw cells
        for y, row in enumerate(level):
            for x, cell in enumerate(row):
                if cell != ' ':
                    rect = pygame.Rect(
                        x * self.cell_size, 
                        y * self.cell_size, 
                        self.cell_size, 
                        self.cell_size
                    )
                    pygame.draw.rect(self.screen, COLORS[cell], rect)
                    
                    # Draw border
                    pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
                    
                    # Draw text
                    font = pygame.font.Font(None, 24)
                    text = font.render(cell, True, (255, 255, 255))
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)
        
        pygame.display.flip()
        
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True


def load_trained_agent(model_path, state_shape, action_size):
    """Load a trained DQN agent"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create network with correct dimensions
    input_height, input_width = state_shape[1], state_shape[2]
    network = DQN(state_shape[0], action_size, input_height, input_width).to(device)
    
    # Load weights
    try:
        network.load_state_dict(torch.load(model_path, map_location=device))
        network.eval()
        print(f"Loaded trained model from {model_path}")
        return network, device
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the agent first.")
        return None, device


def visualize_agent(model_path="sokoban_dqn_model.pth", level_idx=0, delay=1.0):
    """Visualize the trained agent playing Sokoban"""
    
    # Create environment
    env = SokobanEnv(level_idx)
    state = env.reset()
    
    # Load trained agent
    network, device = load_trained_agent(model_path, state.shape, env.action_space)
    if network is None:
        return
    
    # Create visualizer
    visualizer = SokobanVisualizer()
    
    # Game loop
    running = True
    step = 0
    total_reward = 0
    
    print(f"Starting visualization for level {level_idx}")
    print("Press ESC to quit")
    
    while running:
        # Handle events
        running = visualizer.handle_events()
        
        # Render current state
        visualizer.render_level(env.level)
        
        # Get action from agent
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = network(state_tensor)
            action = q_values.argmax().item()
        
        # Take action
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        print(f"Step {step}: Action={action_names[action]}, Reward={reward:.2f}, Total={total_reward:.2f}")
        
        if done:
            if env._is_complete():
                print(f"Level completed in {step} steps! Total reward: {total_reward:.2f}")
            else:
                print(f"Episode ended after {step} steps. Total reward: {total_reward:.2f}")
            
            # Wait a bit before restarting
            time.sleep(2)
            
            # Reset for next episode
            state = env.reset()
            step = 0
            total_reward = 0
        else:
            state = next_state
        
        # Control speed
        time.sleep(delay)
        visualizer.clock.tick(60)
    
    pygame.quit()


def manual_control(level_idx=0):
    """Allow manual control of the game for comparison"""
    env = SokobanEnv(level_idx)
    state = env.reset()
    
    visualizer = SokobanVisualizer()
    
    running = True
    step = 0
    total_reward = 0
    
    print(f"Manual control for level {level_idx}")
    print("Use arrow keys to move, ESC to quit")
    
    while running:
        # Render current state
        visualizer.render_level(env.level)
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                action = None
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset level
                    state = env.reset()
                    step = 0
                    total_reward = 0
                    print("Level reset")
                
                if action is not None:
                    next_state, reward, done, info = env.step(action)
                    total_reward += reward
                    step += 1
                    
                    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                    print(f"Step {step}: Action={action_names[action]}, Reward={reward:.2f}, Total={total_reward:.2f}")
                    
                    if done:
                        if env._is_complete():
                            print(f"Level completed in {step} steps! Total reward: {total_reward:.2f}")
                        else:
                            print(f"Episode ended after {step} steps. Total reward: {total_reward:.2f}")
                        
                        # Reset for next episode
                        state = env.reset()
                        step = 0
                        total_reward = 0
                    else:
                        state = next_state
        
        visualizer.clock.tick(60)
    
    pygame.quit()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        manual_control()
    else:
        visualize_agent() 