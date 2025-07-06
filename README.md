# Sokoban Reinforcement Learning Implementation

This project implements a Deep Q-Network (DQN) agent to learn how to play Sokoban puzzle games using reinforcement learning.

## Overview

The implementation includes:
- **Environment Wrapper**: Converts the Sokoban game into an RL environment
- **DQN Agent**: Deep Q-Network with convolutional neural network for state representation
- **Training Loop**: Complete training pipeline with experience replay and target networks
- **Visualization**: Tools to watch the agent play and analyze performance

## Architecture

### State Representation
The game state is converted to a 6-channel tensor representing:
- Channel 0: Walls
- Channel 1: Empty spaces
- Channel 2: Storage locations
- Channel 3: Boxes
- Channel 4: Boxes on storage
- Channel 5: Player position

### Neural Network
- **Input**: 6-channel convolutional layers
- **Architecture**: 3 conv layers + 3 fully connected layers
- **Output**: Q-values for 4 actions (UP, DOWN, LEFT, RIGHT)

### Reward System
- **+10.0**: Complete the level
- **+1.0**: Push box onto storage location
- **-0.01**: Regular movement step
- **-0.1**: Invalid move (hit wall)
- **-1.0**: Episode timeout

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the original `sokoban.py` file in the same directory.

## Usage

### 1. Test the Environment
First, verify the environment works correctly:
```bash
python test_environment.py
```

### 2. Train the Agent
Train a DQN agent on the first level:
```bash
python sokoban_rl.py
```

This will:
- Train for 10000 episodes
- Save the trained model as `sokoban_dqn_model.pth`
- Show training progress plots
- Test the final agent

### 3. Visualize the Trained Agent
Watch the trained agent play:
```bash
python visualize_agent.py
```

Or play manually for comparison:
```bash
python visualize_agent.py manual
```

## Key Features

### Experience Replay
- Stores experiences in a replay buffer
- Samples random batches for training
- Breaks correlation between consecutive experiences

### Target Network
- Separate target network for stable training
- Updated every 100 episodes
- Reduces moving target problem

### Adaptive Reward Shaping
- Immediate feedback for good/bad actions
- Sparse completion rewards
- Step penalties to encourage efficiency

## Training Tips

### Hyperparameters
- **Learning Rate**: 0.001 (Adam optimizer)
- **Batch Size**: 32
- **Memory Size**: 10,000 experiences
- **Discount Factor**: 0.99
- **Epsilon Decay**: 0.9995

### Common Issues
1. **Agent gets stuck**: Increase exploration or add negative rewards for repetitive actions
2. **Training unstable**: Reduce learning rate or increase target network update frequency
3. **Poor performance**: Try different reward shaping or network architecture

### Monitoring Training
Watch for these indicators:
- **Increasing average scores**: Agent is learning
- **Solved episodes**: Track completion rate
- **Epsilon decay**: Ensure proper exploration schedule

## Performance Expectations

### Training Progress
- **Episodes 0-500**: Random exploration, negative scores
- **Episodes 500-1000**: Learning basic movements
- **Episodes 1000-1500**: Developing strategies
- **Episodes 1500+**: Consistent solving (for simple levels)

### Level Difficulty
- **Simple levels (0-10)**: Usually solvable within 2000 episodes
- **Medium levels**: May require 5000+ episodes
- **Complex levels**: Might need curriculum learning

## Files Description

- `sokoban_rl.py`: Main RL implementation
- `test_environment.py`: Environment testing script
- `visualize_agent.py`: Visualization tools
- `requirements.txt`: Python dependencies
- `README_RL.md`: This documentation

## Future Improvements

1. **Curriculum Learning**: Start with simple levels
2. **Transfer Learning**: Use pre-trained features
3. **Hierarchical RL**: Break down complex actions
4. **Multi-Agent**: Train multiple agents simultaneously
5. **Imitation Learning**: Learn from human demonstrations
