import argparse
import numpy as np
import torch
from typing import List

from sokoban_ppo import SokobanEnv, PPOAgent, levels  # reuse environment & network


def train_ppo_multi_levels(episodes: int = 1000, level_indices: List[int] | None = None):
    """Train PPO agent across multiple levels in parallel (vectorised loop).

    Args:
        episodes: Number of outer update cycles.
        level_indices: Which levels to include. Defaults to first 5 levels.
    Returns:
        agent : trained PPOAgent
        score_history : list of mean reward per update cycle
    """
    if level_indices is None:
        level_indices = list(range(5))

    envs = [SokobanEnv(idx) for idx in level_indices]
    num_envs = len(envs)

    sample_state = envs[0].reset()
    state_shape = sample_state.shape

    agent = PPOAgent(state_shape, envs[0].action_space)

    # states list keeps last observation for each env
    states = [env.reset() for env in envs]
    score_history: list[float] = []

    for episode in range(episodes):
        traj_states, traj_actions = [], []
        traj_rewards, traj_log_probs, traj_values, traj_dones = [], [], [], []

        done_flags = [False] * num_envs
        ep_rewards_per_env = [0.0] * num_envs

        while not all(done_flags):
            for i, env in enumerate(envs):
                if done_flags[i]:
                    continue

                action, log_prob, value = agent.act(states[i])
                next_state, reward, done, _ = env.step(action)

                traj_states.append(states[i])
                traj_actions.append(action)
                traj_rewards.append(reward)
                traj_log_probs.append(log_prob)
                traj_values.append(value)
                traj_dones.append(done)

                states[i] = next_state
                ep_rewards_per_env[i] += reward

                if done:
                    done_flags[i] = True
                    states[i] = env.reset()

        score_history.append(sum(ep_rewards_per_env) / num_envs)

        agent.update(
            traj_states,
            traj_actions,
            traj_rewards,
            traj_log_probs,
            traj_values,
            traj_dones,
        )

        if episode % 50 == 0:
            avg = np.mean(score_history[-50:])
            print(f"[Multi-Level] Episode {episode}, Avg Reward (last 50): {avg:.2f}")

    return agent, score_history


def main():
    parser = argparse.ArgumentParser(description="Multi-level PPO training for Sokoban")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of update cycles")
    parser.add_argument("--levels", type=int, nargs="*", default=None,
                        help="List of level indices to train on (default: first 5 levels)")
    parser.add_argument("--model", default="sokoban_ppo_multilevel.pth", help="Path to save model")

    args = parser.parse_args()

    if args.levels is None:
        level_indices = list(range(5))
    else:
        level_indices = args.levels

    print(f"Training multi-level PPO on levels {level_indices} for {args.episodes} episodes …")

    agent, history = train_ppo_multi_levels(args.episodes, level_indices)

    torch.save(agent.net.state_dict(), args.model)
    print(f"Done. Model saved to {args.model}")


if __name__ == "__main__":
    main() 