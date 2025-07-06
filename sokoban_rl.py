import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt

# Import the levels from the original sokoban.py
from sokoban import levels

class SokobanEnv:
    """Sokoban environment wrapper for RL training"""
    
    def __init__(self, level_idx=0):
        self.level_idx = level_idx
        self.original_level = None
        self.level = None
        self.max_steps = 200  # Prevent infinite episodes
        self.step_count = 0
        
        # Define symbols
        self.player = '@'
        self.player_on_storage = '+'
        self.box = '$'
        self.box_on_storage = '*'
        self.storage = '.'
        self.wall = '#'
        self.empty = ' '
        
        # Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = 4
        
        self.reset()
        
    def reset(self):
        """Reset the environment to initial state"""
        self.original_level = copy.deepcopy(levels[self.level_idx])
        self.level = copy.deepcopy(self.original_level)
        self.step_count = 0
        
        # Count initial boxes and storage locations
        self.initial_boxes = self._count_boxes()
        self.storage_locations = self._get_storage_locations()
        
        return self._get_state()
    
    def _get_player_position(self):
        """Find player position"""
        for y, row in enumerate(self.level):
            for x, cell in enumerate(row):
                if cell == self.player or cell == self.player_on_storage:
                    return x, y
        return None, None
    
    def _count_boxes(self):
        """Count number of boxes not on storage"""
        count = 0
        for row in self.level:
            for cell in row:
                if cell == self.box:
                    count += 1
        return count
    
    def _get_storage_locations(self):
        """Get all storage locations"""
        locations = []
        for y, row in enumerate(self.level):
            for x, cell in enumerate(row):
                if cell == self.storage or cell == self.player_on_storage or cell == self.box_on_storage:
                    locations.append((x, y))
        return locations
    
    def _get_state(self):
        """Convert game state to numerical representation"""
        # Create a numerical representation of the board
        height = len(self.level)
        width = max(len(row) for row in self.level)
        
        # Create multiple channels for different elements
        state = np.zeros((6, height, width), dtype=np.float32)
        
        for y, row in enumerate(self.level):
            for x, cell in enumerate(row):
                if x < width:
                    if cell == self.wall:
                        state[0, y, x] = 1.0  # Wall channel
                    elif cell == self.empty:
                        state[1, y, x] = 1.0  # Empty channel
                    elif cell == self.storage:
                        state[2, y, x] = 1.0  # Storage channel
                    elif cell == self.box:
                        state[3, y, x] = 1.0  # Box channel
                    elif cell == self.box_on_storage:
                        state[2, y, x] = 1.0  # Storage channel
                        state[4, y, x] = 1.0  # Box on storage channel
                    elif cell == self.player:
                        state[5, y, x] = 1.0  # Player channel
                    elif cell == self.player_on_storage:
                        state[2, y, x] = 1.0  # Storage channel
                        state[5, y, x] = 1.0  # Player channel
        
        return state
    
    def step(self, action):
        """Execute action and return (state, reward, done, info)"""
        self.step_count += 1
        
        # Get current player position
        player_x, player_y = self._get_player_position()
        
        # Define movement directions
        directions = {
            0: (0, -1),  # UP
            1: (0, 1),   # DOWN
            2: (-1, 0),  # LEFT
            3: (1, 0)    # RIGHT
        }
        
        dx, dy = directions[action]
        
        # Check if move is valid
        new_x, new_y = player_x + dx, player_y + dy
        
        # Check bounds
        if (new_y < 0 or new_y >= len(self.level) or 
            new_x < 0 or new_x >= len(self.level[new_y])):
            return self._get_state(), -0.1, False, {}  # Small penalty for invalid move
        
        current = self.level[player_y][player_x]
        adjacent = self.level[new_y][new_x]
        
        # Check if there's a box to push
        beyond = ''
        if (0 <= new_y + dy < len(self.level) and 
            0 <= new_x + dx < len(self.level[new_y + dy])):
            beyond = self.level[new_y + dy][new_x + dx]
        
        reward = 0
        moved = False
        
        # Define state transitions
        next_adjacent = {
            self.empty: self.player,
            self.storage: self.player_on_storage,
        }
        next_current = {
            self.player: self.empty,
            self.player_on_storage: self.storage,
        }
        next_beyond = {
            self.empty: self.box,
            self.storage: self.box_on_storage,
        }
        next_adjacent_push = {
            self.box: self.player,
            self.box_on_storage: self.player_on_storage,
        }
        
        # Try to move
        if adjacent in next_adjacent:
            # Simple move
            self.level[player_y][player_x] = next_current[current]
            self.level[new_y][new_x] = next_adjacent[adjacent]
            reward = -0.01  # Small penalty for each step
            moved = True
            
        elif beyond in next_beyond and adjacent in next_adjacent_push:
            # Push box
            self.level[player_y][player_x] = next_current[current]
            self.level[new_y][new_x] = next_adjacent_push[adjacent]
            self.level[new_y + dy][new_x + dx] = next_beyond[beyond]
            
            # Reward for pushing box onto storage
            if beyond == self.storage:
                reward = 1.0
            else:
                reward = -0.01
            moved = True
            
        elif adjacent == self.wall:
            # Hit wall
            reward = -0.1
        else:
            # Invalid move
            reward = -0.1
        
        # Check if level is complete
        done = self._is_complete()
        if done:
            reward = 10.0  # Large reward for completing level
        
        # Check if too many steps
        if self.step_count >= self.max_steps:
            done = True
            reward = -1.0  # Penalty for taking too long
        
        return self._get_state(), reward, done, {'moved': moved}
    
    def _is_complete(self):
        """Check if level is complete (no boxes left)"""
        for row in self.level:
            for cell in row:
                if cell == self.box:
                    return False
        return True
    
    def render(self):
        """Print the current state"""
        for row in self.level:
            print(''.join(row))
        print()


class DQN(nn.Module):
    """Deep Q-Network for Sokoban"""
    
    def __init__(self, input_channels=6, action_size=4, input_height=8, input_width=8):
        super(DQN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Calculate the flattened size after conv layers
        # After conv1 + pool: (H/2, W/2)
        # After conv2 + pool: (H/4, W/4)
        # After conv3: (H/4, W/4)
        conv_output_height = input_height // 4
        conv_output_width = input_width // 4
        self.flatten_size = 64 * conv_output_height * conv_output_width
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class DQNAgent:
    """DQN Agent for Sokoban"""
    
    def __init__(self, state_shape, action_size, learning_rate=0.001):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.batch_size = 32
        self.target_update_freq = 100
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Neural networks - pass the actual input dimensions
        input_height, input_width = state_shape[1], state_shape[2]
        self.q_network = DQN(state_shape[0], action_size, input_height, input_width).to(self.device)
        self.target_network = DQN(state_shape[0], action_size, input_height, input_width).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert to numpy arrays first, then to tensors for efficiency
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([e[4] for e in batch])).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_agent(episodes=1000, level_idx=0):
    """Train the DQN agent"""
    env = SokobanEnv(level_idx)
    
    # Get state shape
    sample_state = env.reset()
    state_shape = sample_state.shape
    
    agent = DQNAgent(state_shape, env.action_space)
    
    scores = []
    solved_episodes = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        scores.append(total_reward)
        
        # Check if level was solved
        if env._is_complete():
            solved_episodes.append(episode)
            print(f"Episode {episode}: Solved! Total reward: {total_reward:.2f}, Steps: {steps}")
        
        # Train the agent
        agent.replay()
        
        # Update target network
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()
        
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, scores, solved_episodes


def test_agent(agent, level_idx=0, render=True):
    """Test the trained agent"""
    env = SokobanEnv(level_idx)
    state = env.reset()
    
    total_reward = 0
    steps = 0
    
    # Set agent to evaluation mode (no exploration)
    agent.epsilon = 0
    
    if render:
        print("Initial state:")
        env.render()
    
    while True:
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if render:
            print(f"Step {steps}, Action: {['UP', 'DOWN', 'LEFT', 'RIGHT'][action]}, Reward: {reward:.2f}")
            env.render()
        
        if done:
            break
    
    solved = env._is_complete()
    print(f"Test completed! Solved: {solved}, Total reward: {total_reward:.2f}, Steps: {steps}")
    
    return solved, total_reward, steps


if __name__ == "__main__":
    print("Training DQN agent on Sokoban...")
    
    # Train the agent
    agent, scores, solved_episodes = train_agent(episodes=2000, level_idx=0)
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(solved_episodes, [1] * len(solved_episodes), 'ro', markersize=3)
    plt.title('Solved Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Solved')
    
    plt.tight_layout()
    plt.show()
    
    # Test the trained agent
    print("\nTesting trained agent...")
    test_agent(agent, level_idx=0, render=True)
    
    # Save the trained model
    torch.save(agent.q_network.state_dict(), 'sokoban_dqn_model.pth')
    print("Model saved as 'sokoban_dqn_model.pth'") 