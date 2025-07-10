#!/usr/bin/env python3
"""
Pygame Zero visualization for PPO agent - matches original sokoban.py style
"""

import copy
import time
import torch
from torch.distributions import Categorical
from sokoban_ppo import SokobanEnv, PPONet
from padded_levels import levels

# Global variables (Pygame Zero style)
env = None
agent_network = None
device = None
current_level_idx = 2
step_count = 0
total_reward = 0.0
last_action_probs = None
last_value = 0.0
auto_play = True
delay_time = 1.0
last_step_time = 0
reset_pending_until = None  # time until which we keep showing solved/failed state
WIDTH = 800
HEIGHT = 600

# Game symbols (same as original)
player = '@'
player_on_storage = '+'
box = '$'
box_on_storage = '*'
storage = '.'
wall = '#'
empty = ' '

def load_ppo_agent(model_path="sokoban_ppo_model.pth"):
    """Load trained PPO agent"""
    global agent_network, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy env to get state shape
    dummy_env = SokobanEnv(0)
    state_shape = dummy_env.reset().shape
    
    # Create and load network
    agent_network = PPONet(state_shape[0], state_shape[1], dummy_env.action_space).to(device)
    
    try:
        agent_network.load_state_dict(torch.load(model_path, map_location=device))
        agent_network.eval()
        print(f"Loaded PPO model from {model_path}")
        return True
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train first.")
        return False

def setup_env():
    """Setup environment and screen size"""
    global env, WIDTH, HEIGHT, last_step_time
    
    env = SokobanEnv(current_level_idx)
    state = env.reset()
    # Ensure initial state is shown for at least 'delay_time' seconds before first move
    last_step_time = time.time()
    
    # Calculate screen size (same as original)
    cell_size = 23
    level_width = max(len(row) for row in env.level)
    level_height = len(env.level)
    WIDTH = max(level_width * cell_size, 800)
    HEIGHT = level_height * cell_size + 120  # Extra space for info

def get_ppo_action():
    """Get action from PPO agent"""
    global last_action_probs, last_value
    
    if agent_network is None:
        return 0  # Default action
    
    state = env._get_state()
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    with torch.no_grad():
        policy, value = agent_network(state_tensor)
        dist = Categorical(policy)
        action = dist.sample().item()
        last_action_probs = policy.squeeze().cpu().numpy()
        last_value = value.item()
    
    return action

def on_key_down(key):
    """Handle keyboard input (Pygame Zero style)"""
    global current_level_idx, auto_play, delay_time, step_count, total_reward
    
    if key == keys.SPACE:
        auto_play = not auto_play
        print(f"Auto-play: {'ON' if auto_play else 'OFF'}")
    
    elif key == keys.R:
        # Reset current level
        setup_env()
        step_count = 0
        total_reward = 0.0
        print("Level reset")
    
    elif key == keys.N:
        # Next level
        current_level_idx = (current_level_idx + 1) % len(levels)
        setup_env()
        step_count = 0
        total_reward = 0.0
        print(f"Switched to level {current_level_idx}")
    
    elif key == keys.P:
        # Previous level
        current_level_idx = (current_level_idx - 1) % len(levels)
        setup_env()
        step_count = 0
        total_reward = 0.0
        print(f"Switched to level {current_level_idx}")
    
    elif key == keys.PLUS or key == keys.EQUALS:
        delay_time = max(0.1, delay_time - 0.2)
        print(f"Speed up: {delay_time:.1f}s delay")
    
    elif key == keys.MINUS:
        delay_time = min(3.0, delay_time + 0.2)
        print(f"Speed down: {delay_time:.1f}s delay")

def update():
    """Update game state (Pygame Zero style)"""
    global step_count, total_reward, last_step_time, reset_pending_until
    
    if not auto_play or agent_network is None or env is None:
        return

    current_time = time.time()

    # If we are in delay period after episode ends, wait until time passes then reset level
    if reset_pending_until is not None:
        if current_time < reset_pending_until:
            return  # keep showing final state
        else:
            env.reset()
            last_step_time = time.time()
            step_count = 0
            total_reward = 0.0
            reset_pending_until = None
            return
    
    if current_time - last_step_time < delay_time:
        return
    
    last_step_time = current_time
    
    # Get action from PPO agent
    action = get_ppo_action()
    
    # Execute action
    state, reward, done, info = env.step(action)
    step_count += 1
    total_reward += reward
    
    # Print step info
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    print(f"Step {step_count}: {action_names[action]}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
    
    # Handle episode end
    if done:
        if env._is_complete():
            print(f"🎉 Level {current_level_idx} completed in {step_count} steps!")
            reset_pending_until = time.time() + 2.0  # 2-second delay to show winning state
        else:
            print(f"❌ Episode ended after {step_count} steps")
            reset_pending_until = time.time() + 1.0  # 1-second delay to show final state
        return  # Skip further processing until next frame after delay

def draw():
    """Draw game state (Pygame Zero style - same as original sokoban.py)"""
    if env is None:
        screen.fill((100, 100, 100))
        screen.draw.text("Loading PPO Agent...", (10, 10), color="white", fontsize=32)
        return
    
    # Fill background (green if completed, normal otherwise)
    if env._is_complete():
        screen.fill((200, 255, 200))
    else:
        screen.fill((255, 255, 190))
    
    # Draw cells (exactly like original sokoban.py)
    for y, row in enumerate(env.level):
        for x, cell in enumerate(row):
            if cell != empty:
                cell_size = 23
                
                colors = {
                    player: (167, 135, 255),
                    player_on_storage: (158, 119, 255),
                    box: (255, 201, 126),
                    box_on_storage: (150, 255, 127),
                    storage: (156, 229, 255),
                    wall: (255, 147, 209),
                }
                
                screen.draw.filled_rect(
                    Rect(
                        (x * cell_size, y * cell_size),
                        (cell_size, cell_size)
                    ),
                    colors[cell]
                )
                
                screen.draw.text(
                    cell,
                    (x * cell_size, y * cell_size),
                    color=(255, 255, 255)
                )
    
    # Draw info panel
    info_y = len(env.level) * 23 + 10
    
    # Basic info
    screen.draw.text(f"Level: {current_level_idx}", (10, info_y), color="black", fontsize=24)
    screen.draw.text(f"Step: {step_count}", (150, info_y), color="black", fontsize=24)
    screen.draw.text(f"Reward: {total_reward:.2f}", (250, info_y), color="black", fontsize=24)
    screen.draw.text(f"Value: {last_value:.2f}", (400, info_y), color="black", fontsize=24)
    
    # Auto-play status
    status = "AUTO" if auto_play else "PAUSED"
    color = "green" if auto_play else "red"
    screen.draw.text(f"Mode: {status}", (550, info_y), color=color, fontsize=24)
    
    # Action probabilities
    if last_action_probs is not None:
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        action_colors = ['red', 'green', 'blue', 'orange']
        
        for i, (name, prob) in enumerate(zip(action_names, last_action_probs)):
            x_pos = 10 + i * 150
            if x_pos + 120 <= WIDTH:
                screen.draw.text(f"{name}: {prob:.3f}", (x_pos, info_y + 30), 
                               color=action_colors[i], fontsize=20)
    
    # Controls help
    screen.draw.text("Controls: SPACE=Play/Pause, R=Reset, N/P=Next/Prev Level, +/-=Speed", 
                    (10, info_y + 60), color="gray", fontsize=16)

# Initialize when module loads
if load_ppo_agent():
    setup_env()
    print("PPO visualization ready!")
    print("Controls:")
    print("  SPACE - Toggle auto-play")
    print("  R - Reset level")
    print("  N/P - Next/Previous level") 
    print("  +/- - Speed control")
else:
    print("Failed to load PPO agent") 