import os
from dataclasses import dataclass

import tyro
import jax
import jax.numpy as jnp
from frozenlake.env import FrozenLake
from frozenlake.viewer import FrozenLakeViewer

from utils.path import get_algorithm_dir_path

@dataclass
class Args:
    """ set rollout parameters"""

    seed: int = 1
    """seed of the experiment"""
    max_episode_steps: int = 100
    """the number of steps after which an episode is truncated"""
    render: bool = False
    """if toggled, save the animation of the simulated episode as a gif in data/Frozenlake"""

# jumanji's env.reset(key) returns (state, timestep), and env.step(state, action) takes no key. Reward comes from timestep.reward and done from timestep.last()
def rollout(env, key, num_steps=100):

    # initialize lists to store the sequence of states and rewards
    state_seq, reward_seq = [], []

    # jit reset and step functions
    reset_fn, step_fn = jax.jit(env.reset), jax.jit(env.step)

    # reset environment
    key, key_reset = jax.random.split(key, 2)
    state, timestep = reset_fn(key_reset)

    for _ in range(num_steps):
        state_seq.append(state)

        # sample random action
        key, key_act = jax.random.split(key, 2)
        action = env.action_space_sample(key_act)

        # make step
        state, timestep = step_fn(state, action)
        reward_seq.append(timestep.reward)

        # break if done
        if timestep.last():
            break

    # keep the final state so that the whole episode can be visualized
    state_seq.append(state)

    return state_seq, reward_seq

def main():

    # load arguments
    args = tyro.cli(Args)

    # Make environment
    env = FrozenLake(time_limit=args.max_episode_steps)

    # initialize jax key
    key = jax.random.key(args.seed)

    # run rollout
    state_seq, reward_seq = rollout(env, key, args.max_episode_steps)

    # compute cumulative rewards
    cum_rewards = jnp.cumsum(jnp.array(reward_seq))

    print(f"Trajectory's length: {len(reward_seq)}")
    print(f"Cumulative rewards: {cum_rewards[-1]}")

    # visualize episode
    if args.render:
        file_path = os.path.join(
            get_algorithm_dir_path("FrozenLake", "random"),
            f"random_policy_seed{args.seed}.gif",
        )
        viewer = FrozenLakeViewer("Frozen Lake")
        viewer.animate(state_seq, interval=200, save_path=file_path)

if __name__ == '__main__':
    main()
