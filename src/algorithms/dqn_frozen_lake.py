# DQN for the jumanji Frozenlake environment, adapted from dqn.py (gymnax version)
import json
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState

from frozenlake import FrozenLake

from utils.path import get_q_value_dir_path

ENV_NAME = "FrozenLake"
ALGORITHM_NAME = "dqn"


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""

    #map_name: str = "4x4"
    #"""the map of the Frozen Lake, one of MAPS ("4x4" or "8x8")"""
    max_episode_steps: int = 100
    """the number of steps after which an episode is truncated"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 1000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    stats_window: int = 100
    """the number of recent episodes used for the running average of return and length"""


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


class ReplayBuffer:
    """Minimal circular replay buffer stored in numpy arrays."""

    def __init__(self, buffer_size, obs_shape):
        self.buffer_size = buffer_size
        self.observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, action, reward, done):
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.buffer_size
        self.full = self.full or self.pos == 0

    def sample(self, batch_size, rng):
        upper_bound = self.buffer_size if self.full else self.pos
        idx = rng.integers(0, upper_bound, size=batch_size)
        return (
            self.observations[idx],
            self.actions[idx],
            self.next_observations[idx],
            self.rewards[idx],
            self.dones[idx],
        )


def get_obs(env, env_state):
    """One-hot encoding of the agent position."""
    index = env_state.player_position.row * env.num_cols + env_state.player_position.col
    return jax.nn.one_hot(index, env.num_rows * env.num_cols)


def save_q_value(q_params, args):
    """Save the q-value network parameters and the arguments of the run in data/[env]/[algorithm]/q-value."""
    dir_path = get_q_value_dir_path(ENV_NAME, ALGORITHM_NAME)
    with open(os.path.join(dir_path, "params.msgpack"), "wb") as f:
        f.write(flax.serialization.to_bytes(q_params))
    with open(os.path.join(dir_path, "args.json"), "w") as f:
        json.dump(asdict(args), f, indent=4)
    print(f"q-value network saved to {dir_path}")


def load_q_value(env):
    """Load the q-value network parameters saved by `save_q_value`."""
    dir_path = get_q_value_dir_path(ENV_NAME, ALGORITHM_NAME)
    q_network = QNetwork(action_dim=env.action_spec().num_values)
    # the initialization only provides the parameters structure, they are overwritten below
    key = jax.random.key(0)
    env_state, _ = env.reset(key)
    q_params = q_network.init(key, get_obs(env, env_state))
    with open(os.path.join(dir_path, "params.msgpack"), "rb") as f:
        q_params = flax.serialization.from_bytes(q_params, f.read())
    return q_network, q_params


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def main():

    # load args
    args = tyro.cli(Args)

    # seeding
    np_rng = np.random.default_rng(args.seed)
    key = jax.random.key(args.seed)
    key, q_key, reset_key = jax.random.split(key, 3)

    # make environment
    env = FrozenLake(time_limit=args.max_episode_steps)
    num_actions = env.action_spec().num_values

    # reset environment
    env_state, _ = env.reset(reset_key)
    obs = get_obs(env, env_state)

    # define neural network to approximate the q-value function
    q_network = QNetwork(action_dim=num_actions)
    q_params = q_network.init(q_key, obs)
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_params,
        target_params=q_params,
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    # make replay buffer
    rb = ReplayBuffer(args.buffer_size, obs.shape)

    @jax.jit
    def act_and_step(params, obs, env_state, epsilon, key):
        """Epsilon-greedy action selection followed by an environment step."""
        key_explore, key_act, key_reset = jax.random.split(key, 3)
        q_values = q_network.apply(params, obs)
        action = jnp.where(
            jax.random.uniform(key_explore) < epsilon,
            jax.random.randint(key_act, (), 0, num_actions),
            q_values.argmax(axis=-1),
        )
        next_env_state, timestep = env.step(env_state, action)
        # the env truncates the episode at its time_limit: truncated timesteps keep a unit discount
        terminated = timestep.last() & (timestep.discount == 0)
        truncated = timestep.last() & (timestep.discount != 0)
        final_obs = get_obs(env, next_env_state)

        # jumanji envs do not auto-reset like gymnax: reset manually when done
        reset_env_state, _ = env.reset(key_reset)
        next_env_state = jax.tree.map(
            lambda r, s: jnp.where(terminated | truncated, r, s), reset_env_state, next_env_state
        )
        next_obs = get_obs(env, next_env_state)
        return action, next_obs, next_env_state, timestep.reward, terminated, truncated, final_obs

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones):
        q_next_target = q_network.apply(q_state.target_params, next_observations)  # (batch_size, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * args.gamma * q_next_target

        def mse_loss(params):
            q_pred = q_network.apply(params, observations)  # (batch_size, num_actions)
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions]  # (batch_size,)
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(q_state.params)
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state

    start_time = time.time()
    episodic_returns = []
    episodic_return, episodic_length = 0.0, 0
    recent_returns = deque(maxlen=args.stats_window)
    recent_lengths = deque(maxlen=args.stats_window)

    # start training
    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e,
                                  args.exploration_fraction * args.total_timesteps, global_step)
        key, step_key = jax.random.split(key)
        action, next_obs, env_state, reward, terminated, truncated, final_obs = jax.device_get(
            act_and_step(q_state.params, obs, env_state, epsilon, step_key)
        )

        # record episode statistics
        episodic_return += reward
        episodic_length += 1
        if terminated or truncated:
            episodic_returns.append(episodic_return)
            recent_returns.append(episodic_return)
            recent_lengths.append(episodic_length)
            if len(episodic_returns) % 100 == 0:
                print(
                    f"global_step={global_step}, episode={len(episodic_returns)}, "
                    f"episodic_return={episodic_return}, episodic_length={episodic_length}, "
                    f"avg_return={np.mean(recent_returns):.2f}, avg_length={np.mean(recent_lengths):.2f}"
                )
            episodic_return, episodic_length = 0.0, 0

        # save data to replay buffer; the true next obs is `final_obs` since we auto-reset
        rb.add(obs, final_obs, action, reward, terminated)

        # CRUCIAL step easy to overlook
        obs = next_obs

        # training
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                loss, old_val, q_state = update(q_state, *rb.sample(args.batch_size, np_rng))

            # update target network
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau)
                )

    print(f"SPS={int(args.total_timesteps / (time.time() - start_time))}")

    # save q-value network
    save_q_value(q_state.params, args)
    return q_state, episodic_returns


if __name__ == "__main__":
    main()
