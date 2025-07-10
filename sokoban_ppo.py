import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# Import the levels from the original sokoban.py
from padded_levels import levels

class SokobanEnv:
    """Sokoban environment for PPO training"""
    
    def __init__(self, level_idx=0):
        self.level_idx = level_idx
        self.original_level = None
        self.level = None
        self.max_steps = 200
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
        """Reset the environment"""
        self.original_level = copy.deepcopy(levels[self.level_idx])
        self.level = copy.deepcopy(self.original_level)
        self.step_count = 0
        return self._get_state()
    
    def _get_player_position(self):
        """Find player position"""
        for y, row in enumerate(self.level):
            for x, cell in enumerate(row):
                if cell == self.player or cell == self.player_on_storage:
                    return x, y
        return None, None
    
    def _get_state(self):
        """Convert game state to numerical representation"""
        height = len(self.level)
        width = max(len(row) for row in self.level)
        
        # Simple 1-channel representation
        state = np.zeros((height, width), dtype=np.float32)
        
        for y, row in enumerate(self.level):
            for x, cell in enumerate(row):
                if x < width:
                    if cell == self.wall:
                        state[y, x] = 1.0
                    elif cell == self.empty:
                        state[y, x] = 0.0
                    elif cell == self.storage:
                        state[y, x] = 0.3
                    elif cell == self.box:
                        state[y, x] = 0.5
                    elif cell == self.box_on_storage:
                        state[y, x] = 0.8
                    elif cell == self.player:
                        state[y, x] = 0.6
                    elif cell == self.player_on_storage:
                        state[y, x] = 0.9
        
        return state
    
    def step(self, action):
        """Execute action"""
        self.step_count += 1
        
        player_x, player_y = self._get_player_position()
        
        # Movement directions
        directions = {
            0: (0, -1),  # UP
            1: (0, 1),   # DOWN
            2: (-1, 0),  # LEFT
            3: (1, 0)    # RIGHT
        }
        
        dx, dy = directions[action]
        new_x, new_y = player_x + dx, player_y + dy
        
        # Check bounds
        if (new_y < 0 or new_y >= len(self.level) or 
            new_x < 0 or new_x >= len(self.level[new_y])):
            return self._get_state(), -0.1, False, {}
        
        current = self.level[player_y][player_x]
        adjacent = self.level[new_y][new_x]
        
        # Check box push
        beyond = ''
        if (0 <= new_y + dy < len(self.level) and 
            0 <= new_x + dx < len(self.level[new_y + dy])):
            beyond = self.level[new_y + dy][new_x + dx]
        
        reward = 0
        
        # State transitions (same as original sokoban.py)
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
            reward = -0.01
            
        elif beyond in next_beyond and adjacent in next_adjacent_push:
            # Push box
            self.level[player_y][player_x] = next_current[current]
            self.level[new_y][new_x] = next_adjacent_push[adjacent]
            self.level[new_y + dy][new_x + dx] = next_beyond[beyond]
            
            if beyond == self.storage:
                reward = 1.0  # Box on storage
            else:
                reward = -0.01
        else:
            # Invalid move
            reward = -0.1
        
        # Check completion
        done = self._is_complete()
        if done:
            reward = 10.0
        
        # Time limit
        if self.step_count >= self.max_steps:
            done = True
            reward = -1.0
        
        return self._get_state(), reward, done, {}
    
    def _is_complete(self):
        """Check if level is complete"""
        for row in self.level:
            for cell in row:
                if cell == self.box:
                    return False
        return True


class PPONet(nn.Module):
    """Simple PPO network for Sokoban"""
    
    def __init__(self, input_height=8, input_width=8, action_size=4):
        super(PPONet, self).__init__()
        
        # Simple CNN
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Calculate flattened size
        conv_output_size = input_height * input_width * 32
        
        # Shared layers
        self.fc_shared = nn.Linear(conv_output_size, 128)
        
        # Actor (policy) head
        self.actor = nn.Linear(128, action_size)
        
        # Critic (value) head
        self.critic = nn.Linear(128, 1)
        
    def forward(self, x):
        # Add channel dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))
        
        # Policy and value outputs
        policy = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        
        return policy, value


class PPOAgent:
    """PPO Agent for Sokoban"""
    
    def __init__(self, state_shape, action_size, lr=3e-4):
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network
        self.net = PPONet(state_shape[0], state_shape[1], action_size).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        
        # PPO parameters
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        print(f"Using device: {self.device}")
    
    def act(self, state):
        """Select action"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy, value = self.net(state_tensor)
            dist = Categorical(policy)
            action = dist.sample()
            
        return action.item(), dist.log_prob(action).item(), value.item()
    
    def evaluate(self, states, actions):
        """Evaluate states and actions"""
        policy, values = self.net(states)
        dist = Categorical(policy)
        
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return action_log_probs, values.squeeze(), entropy
    
    def update(self, states, actions, rewards, old_log_probs, values, dones):
        """PPO update"""
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Calculate advantages
        advantages = []
        returns = []
        
        advantage = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else values[i]
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value - values[i]
            advantage = delta + self.gamma * self.gae_lambda * advantage * (1 - dones[i])
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[i])
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        for _ in range(4):  # Update epochs
            # Evaluate current policy
            new_log_probs, new_values, entropy = self.evaluate(states, actions)
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = F.mse_loss(new_values, returns)
            entropy_loss = -entropy.mean()
            
            total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()


def train_ppo(episodes=1000, level_idx=0):
    """Train PPO agent"""
    env = SokobanEnv(level_idx)
    
    # Get state shape
    sample_state = env.reset()
    state_shape = sample_state.shape
    
    agent = PPOAgent(state_shape, env.action_space)
    
    scores = []
    solved_episodes = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_log_probs = []
        episode_values = []
        episode_dones = []
        
        total_reward = 0
        steps = 0
        
        while True:
            action, log_prob, value = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob)
            episode_values.append(value)
            episode_dones.append(done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        scores.append(total_reward)
        
        # Check if solved
        if env._is_complete():
            solved_episodes.append(episode)
        
        # Update agent
        agent.update(episode_states, episode_actions, episode_rewards, 
                    episode_log_probs, episode_values, episode_dones)
        
        # Progress report
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")
    
    return agent, scores, solved_episodes


if __name__ == "__main__":
    print("Training PPO agent on Sokoban...")
    
    # Train the agent
    agent, scores, solved_episodes = train_ppo(episodes=10000, level_idx=0)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Training Scores (PPO)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(solved_episodes, [1] * len(solved_episodes), 'go', markersize=3)
    plt.title('Solved Episodes (PPO)')
    plt.xlabel('Episode')
    plt.ylabel('Solved')
    
    plt.tight_layout()
    plt.show()
    
    # Save model
    torch.save(agent.net.state_dict(), 'sokoban_ppo_model.pth')
    print("PPO model saved as 'sokoban_ppo_model.pth'") 