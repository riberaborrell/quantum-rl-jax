# DQN for the jumanji Frozenlake environment, adapted from dqn.py (gymnax version)
import sys
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState

from frozenlake import FrozenLake

from algorithms.dqn_utils import ReplayBuffer, load_q_value, make_epsilon_schedule, make_optimizer, save_q_value
from models.neural_networks import QNetwork

ENV_NAME = "FrozenLake"
ALGORITHM_NAME = "dqn"


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""

    # frozenlake env parameters
    goal_reward: float = 1.0
    """reward for reaching the goal"""
    hole_reward: float = -0.2
    """reward for falling into a hole"""
    step_reward: float = -0.01
    """reward for any other step"""
    max_episode_steps: int = 100
    """the number of steps after which an episode is truncated"""

    # neural network architecture and optimizer
    hidden_dims: Sequence[int] = (32, 16)
    """the dimensions of the hidden layers"""
    optimizer: Literal["sgd", "adam", "rmsprop"] = "adam"
    """the optimizer of the neural network parameters"""
    learning_rate: float = 5e-4
    """the learning rate of the chosen optimizer"""
    clip_grad_norm: float | None = None
    """if set, the maximum global norm of the gradients; larger gradients are rescaled to this norm"""

    # dqn parameters
    total_timesteps: int | None = 20000
    """total timesteps of the experiments"""
    max_episodes: int | None = 1000
    """if set, stop training after this number of episodes (in addition to `total_timesteps`)"""
    buffer_size: int = 1000
    """the replay memory buffer size"""
    gamma: float = 0.95
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    learning_starts: int = 1000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    stats_window: int = 50
    """the number of recent episodes used for the running average of return and length"""

    # exploration
    start_e: float = 0.999
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    epsilon_schedule: Literal["constant", "linear", "exponential"] = "linear"
    """"linear" and "exponential": epsilon decays from `start_e` to `end_e` over the first
    `exploration_fraction` of the timesteps; "constant": epsilon stays at `start_e`"""


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def get_obs(env, env_state):
    """One-hot encoding of the agent position."""
    index = env_state.player_position.row * env.num_cols + env_state.player_position.col
    return jax.nn.one_hot(index, env.num_rows * env.num_cols)


def load_q_network(env):
    """Load the q-value network and the trained parameters saved by `save_q_value`."""
    q_network = QNetwork(action_dim=env.action_spec().num_values)
    key = jax.random.key(0)
    env_state, _ = env.reset(key)
    # eval_shape only provides the structure, shapes and dtypes to restore into, without computing the init
    q_params = jax.eval_shape(lambda: q_network.init(key, get_obs(env, env_state)))
    return q_network, load_q_value(q_params, ENV_NAME, ALGORITHM_NAME)


def main():

    # load args
    args = tyro.cli(Args)

    # seeding
    np_rng = np.random.default_rng(args.seed)
    key = jax.random.key(args.seed)
    key, q_key, reset_key = jax.random.split(key, 3)

    # make environment
    env = FrozenLake(
        goal_reward=args.goal_reward,
        hole_reward=args.hole_reward,
        step_reward=args.step_reward,
        time_limit=args.max_episode_steps,
    )
    num_actions = env.action_spec().num_values

    # reset environment
    env_state, _ = env.reset(reset_key)
    obs = get_obs(env, env_state)

    # define neural network to approximate the q-value function
    q_network = QNetwork(action_dim=num_actions, hidden_dims=args.hidden_dims)
    q_params = q_network.init(q_key, obs)
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_params,
        target_params=q_params,
        tx=make_optimizer(args.optimizer, args.learning_rate, args.clip_grad_norm),
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
    print(f"{'global_step':>11} | {'episode':>7} | {'ep_return':>9} | {'ep_length':>9} | {'avg_return':>10} | {'avg_length':>10} | {'epsilon':>7}")

    # start training
    epsilon_schedule = make_epsilon_schedule(
        args.epsilon_schedule, args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps
    )
    for global_step in range(args.total_timesteps):
        epsilon = float(epsilon_schedule(global_step))
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
            if len(episodic_returns) % 10 == 0:
                print(
                    f"{global_step:>11d} | {len(episodic_returns):>7d} | {episodic_return:>+9.2f} | "
                    f"{episodic_length:>9d} | {np.mean(recent_returns):>+10.2f} | {np.mean(recent_lengths):>10.2f} | {epsilon:>7.3f}"
                )
            episodic_return, episodic_length = 0.0, 0
            if len(episodic_returns) == args.max_episodes:
                break

        # save data to replay buffer; the true next obs is `final_obs` since we auto-reset
        rb.add(obs, final_obs, action, reward, terminated)

        # update observation. CRUCIAL step!
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

    # save q-value network
    save_q_value(q_state.params, ENV_NAME, ALGORITHM_NAME)
    return q_state, episodic_returns


if __name__ == "__main__":
    main()
