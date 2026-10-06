import importlib
import os
from dataclasses import dataclass
from typing import Literal

import tyro
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from frozenlake.env import FrozenLake
from frozenlake.viewer import FrozenLakeViewer

from utils.path import get_algorithm_dir_path

@dataclass
class Args:
    """ set rollout parameters of the trained (greedy) policy"""

    seed: int = 1
    """seed of the experiment"""
    max_episode_steps: int = 100
    """the number of steps after which an episode is truncated"""
    algorithm_name: Literal["dqn", "dqn_vqc"] = "dqn"
    """the algorithm that trained the policy: dqn (neural network) or dqn_vqc (variational quantum circuit)"""
    render: bool = False
    """if toggled, save the animation of the simulated episode as a gif in data/FrozenLake/[algorithm_name]"""


# jumanji's env.reset(key) returns (state, timestep), and env.step(state, action) takes no key. Reward comes from timestep.reward and done from timestep.last()
def rollout(env, q_values, q_params, get_obs, key, num_steps=100):

    state_seq, reward_seq = [], []

    # reset environment
    key, key_reset = jax.random.split(key, 2)
    state, timestep = env.reset(key_reset)

    for _ in range(num_steps):
        state_seq.append(state)

        # greedy action w.r.t. the trained q-value function
        action = q_values(q_params, get_obs(env, state)).argmax()

        # make step
        state, timestep = env.step(state, action)
        reward_seq.append(timestep.reward)

        # break if done
        if timestep.last():
            break

    # keep the final state so that the whole episode can be visualized
    state_seq.append(state)

    return state_seq, reward_seq

ALGORITHM_MODULES = {"dqn": "algorithms.dqn_frozen_lake", "dqn_vqc": "algorithms.dqn_vqc_frozen_lake"}


def main():

    # load arguments
    args = tyro.cli(Args)

    # the dqn_vqc module enables jax x64 when imported, so import before anything creates jax arrays
    algorithm = importlib.import_module(ALGORITHM_MODULES[args.algorithm_name])

    # Make environment
    env = FrozenLake(time_limit=args.max_episode_steps)

    # load the trained q-value function; each algorithm has its own way of loading and encoding the state
    if args.algorithm_name == "dqn":
        q_network, q_params = algorithm.load_q_network(env)
        q_values = q_network.apply
    else:
        q_params = algorithm.load_q_value()
        q_values = algorithm.q_values
    get_obs = algorithm.get_obs

    # initialize jax key
    key = jax.random.key(args.seed)

    # run rollout
    state_seq, reward_seq = rollout(env, q_values, q_params, get_obs, key, args.max_episode_steps)

    # compute cumulative rewards
    cum_rewards = jnp.cumsum(jnp.array(reward_seq))

    print(f"Trajectory's length: {len(reward_seq)}")
    print(f"Cumulative rewards: {cum_rewards[-1]}")

    # visualize episode
    if args.render:
        file_path = os.path.join(
            get_algorithm_dir_path(algorithm.ENV_NAME, args.algorithm_name),
            f"trained_policy_seed{args.seed}.gif",
        )
        viewer = FrozenLakeViewer("Frozen Lake")
        viewer.animate(state_seq, interval=200, save_path=file_path)

if __name__ == '__main__':
    main()
