# DQN for gymnax environments, adapted from cleanrl's dqn_jax.py
# (https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_jaxpy)
import tyro
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import flax
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from algorithms.dqn_utils import EPSILON_SCHEDULES, ReplayBuffer, make_optimizer
from models.neural_networks import QNetwork


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""

    # gymnax env parameters
    env_id: str = "CartPole-v1"
    """the id of the gymnax environment"""

    # neural network architecture and optimizer
    optimizer: Literal["sgd", "adam", "rmsprop"] = "adam"
    """the optimizer of the neural network parameters"""
    learning_rate: float = 2.5e-4
    """the learning rate of the chosen optimizer"""

    # dqn parameters
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    max_episodes: int | None = None
    """if set, stop training after this number of episodes (in addition to `total_timesteps`)"""
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
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    stats_window: int = 100
    """the number of recent episodes used for the running average of return and length"""

    # exploration
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    epsilon_schedule: Literal["constant", "linear", "exponential"] = "linear"
    """"linear" and "exponential": epsilon decays from `start_e` to `end_e` over the first
    `exploration_fraction` of the timesteps; "constant": epsilon stays at `start_e`"""


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def main():

    # load args
    args = tyro.cli(Args)

    # seeding
    np_rng = np.random.default_rng(args.seed)
    key = jax.random.key(args.seed)
    key, q_key, reset_key = jax.random.split(key, 3)

    # make environment
    env, env_params = gymnax.make(args.env_id)
    action_space = env.action_space(env_params)
    assert isinstance(action_space, gymnax.environments.spaces.Discrete), "only discrete action space is supported"

    # reset environment
    obs, env_state = env.reset(reset_key, env_params)

    # define neural network to approximate the q-value function 
    q_network = QNetwork(action_dim=action_space.n)
    q_params = q_network.init(q_key, obs)
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_params,
        target_params=q_params,
        tx=make_optimizer(args.optimizer, args.learning_rate),
    )

    # make replay buffer
    rb = ReplayBuffer(args.buffer_size, obs.shape)

    @jax.jit
    def act_and_step(params, obs, env_state, epsilon, key):
        """Epsilon-greedy action selection followed by an environment step."""
        key_explore, key_act, key_step = jax.random.split(key, 3)
        q_values = q_network.apply(params, obs)
        action = jnp.where(
            jax.random.uniform(key_explore) < epsilon,
            action_space.sample(key_act),
            q_values.argmax(axis=-1),
        )
        next_obs, env_state, reward, terminated, truncated, info = env.step(
            key_step, env_state, action, env_params
        )
        return action, next_obs, env_state, reward, terminated, truncated, info["final_observation"]

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
    episode_counter = 0
    episodic_return, episodic_length = 0.0, 0
    recent_returns = deque(maxlen=args.stats_window)
    recent_lengths = deque(maxlen=args.stats_window)
    print(f"{'global_step':>11} | {'episode':>7} | {'ep_return':>9} | {'ep_length':>9} | {'avg_return':>10} | {'avg_length':>10} | {'epsilon':>7}")

    # start training
    for global_step in range(args.total_timesteps):
        epsilon = EPSILON_SCHEDULES[args.epsilon_schedule](
            args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step
        )
        key, step_key = jax.random.split(key)
        action, next_obs, env_state, reward, terminated, truncated, final_obs = jax.device_get(
            act_and_step(q_state.params, obs, env_state, epsilon, step_key)
        )

        # record episode statistics
        episodic_return += reward
        episodic_length += 1
        if terminated or truncated:
            episodic_returns.append(episodic_return)
            episode_counter += 1
            recent_returns.append(episodic_return)
            recent_lengths.append(episodic_length)
            if len(episodic_returns) % 100 == 0:
                print(
                    f"{global_step:>11d} | {len(episodic_returns):>7d} | {episodic_return:>+9.2f} | "
                    f"{episodic_length:>9d} | {np.mean(recent_returns):>+10.2f} | {np.mean(recent_lengths):>10.2f} | {epsilon:>7.3f}"
                )
            episodic_return, episodic_length = 0.0, 0
            if episode_counter == args.max_episodes:
                break

        # save data to replay buffer; gymnax auto-resets, so the true next obs is `final_obs`
        rb.add(obs, final_obs, action, reward, terminated)

        # CRUCIAL step easy to overlook
        obs = next_obs

        # training
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                loss, old_val, q_state = update(q_state, *rb.sample(args.batch_size, np_rng))

                if global_step % 100 == 0:
                    pass
                    #print(
                    #    f"global_step={global_step}, td_loss={loss:.4f}, q_values={old_val.mean():.4f}"
                    #)

            # update target network
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau)
                )

    return q_state, episodic_returns


if __name__ == "__main__":
    main()
