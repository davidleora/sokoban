#!/usr/bin/env python3
"""
Visualization script to watch the trained PPO agent play Sokoban
"""

import pygame
import time
import torch
from torch.distributions import Categorical
from sokoban_ppo import SokobanEnv, PPONet

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
        
        screen_width = max(width * self.cell_size, 800)
        screen_height = height * self.cell_size + 120
        
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Sokoban PPO Agent")
        
    def render_level(self, level, step=0, reward=0, action_probs=None, value=0.0, is_winning=False):
        """Render the current level state with info"""
        if self.screen is None:
            self.setup_display(level)
        
        # Fill background - green when winning
        bg_color = (200, 255, 200) if is_winning else COLORS[' ']
        self.screen.fill(bg_color)
        
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
        
        # Draw info panel
        info_y = len(level) * self.cell_size + 10
        font_large = pygame.font.Font(None, 32)
        font_small = pygame.font.Font(None, 24)
        
        # First row: Basic info
        step_text = font_large.render(f"Step: {step}", True, (0, 0, 0))
        reward_text = font_large.render(f"Reward: {reward:.2f}", True, (0, 0, 0))
        value_text = font_large.render(f"Value: {value:.2f}", True, (0, 0, 0))
        
        self.screen.blit(step_text, (10, info_y))
        self.screen.blit(reward_text, (150, info_y))
        self.screen.blit(value_text, (300, info_y))
        
        # Winning message
        if is_winning:
            win_text = font_large.render("🎉 LEVEL COMPLETED!", True, (0, 150, 0))
            text_rect = win_text.get_rect(center=(self.screen.get_width()//2, info_y))
            self.screen.blit(win_text, text_rect)
        
        # Second row: Action probabilities with colors
        if action_probs is not None:
            action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            action_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0)]
            
            for i, (name, prob, color) in enumerate(zip(action_names, action_probs, action_colors)):
                prob_text = font_small.render(f"{name}: {prob:.3f}", True, color)
                x_pos = 10 + i * 120
                if x_pos + 100 <= self.screen.get_width():  # Only draw if fits
                    self.screen.blit(prob_text, (x_pos, info_y + 35))
        
        pygame.display.flip()
        
    def handle_events(self):
        """Handle pygame events"""
        keys = pygame.key.get_pressed()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, 1.0, False  # running, speed, paused
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False, 1.0, False
        
        # Speed control
        speed = 1.0
        if keys[pygame.K_PLUS] or keys[pygame.K_EQUALS]:
            speed = 0.1  # Faster
        elif keys[pygame.K_MINUS]:
            speed = 2.0  # Slower
        
        # Pause control
        paused = keys[pygame.K_SPACE]
        
        return True, speed, paused


def load_trained_ppo_agent(model_path, state_shape, action_size):
    """Load a trained PPO agent"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create network
    network = PPONet(state_shape[0], state_shape[1], action_size).to(device)
    
    # Load weights
    try:
        network.load_state_dict(torch.load(model_path, map_location=device))
        network.eval()
        print(f"Loaded trained PPO model from {model_path}")
        return network, device
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the PPO agent first.")
        return None, device


def visualize_ppo_agent(model_path="sokoban_ppo_model.pth", level_idx=0, delay=1.0):
    """Visualize the trained PPO agent playing Sokoban"""
    
    # Create environment
    env = SokobanEnv(level_idx)
    state = env.reset()
    
    # Load trained agent
    network, device = load_trained_ppo_agent(model_path, state.shape, env.action_space)
    if network is None:
        return
    
    # Create visualizer
    visualizer = SokobanVisualizer()
    
    # Game loop
    running = True
    step = 0
    total_reward = 0
    speed = delay
    paused = False
    
    print(f"Starting PPO visualization for level {level_idx}")
    print("Controls:")
    print("  ESC - Quit")
    print("  SPACE - Pause/Resume")
    print("  + - Faster")
    print("  - - Slower")
    
    while running:
        # Handle events
        running, speed, paused = visualizer.handle_events()
        
        if not paused:
            # Get action from PPO agent
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                policy, value = network(state_tensor)
                dist = Categorical(policy)
                action = dist.sample().item()
                action_probs = policy.squeeze().cpu().numpy()
            
            # Render current state with action probabilities
            visualizer.render_level(env.level, step, total_reward, action_probs, value.item())
            time.sleep(speed)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
            
            action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            print(f"Step {step}: Action={action_names[action]}, Reward={reward:.2f}, "
                  f"Total={total_reward:.2f}, Value={value.item():.2f}")
            
            # IMPORTANT: Render the state AFTER the action
            if done and env._is_complete():
                # Show the winning state
                visualizer.render_level(env.level, step, total_reward, action_probs, value.item(), True)
                print(f"✅ Level completed in {step} steps! Total reward: {total_reward:.2f}")
                time.sleep(3)  # Show winning state longer
                
                state = env.reset()
                step = 0
                total_reward = 0
            elif done:
                print(f"❌ Episode ended after {step} steps. Total reward: {total_reward:.2f}")
                time.sleep(2)
                
                state = env.reset()
                step = 0
                total_reward = 0
            else:
                state = next_state
        else:
            visualizer.render_level(env.level, step, total_reward)
            
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
        visualizer.render_level(env.level, step, total_reward)
        
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize trained PPO Sokoban agent')
    parser.add_argument('--model', default='sokoban_ppo_model.pth', 
                       help='Path to trained model')
    parser.add_argument('--level', type=int, default=0, 
                       help='Level index to play')
    parser.add_argument('--delay', type=float, default=1.0, 
                       help='Delay between actions (seconds)')
    parser.add_argument('--manual', action='store_true', 
                       help='Manual control mode')
    
    args = parser.parse_args()
    
    if args.manual:
        manual_control(args.level)
    else:
        visualize_ppo_agent(args.model, args.level, args.delay) 