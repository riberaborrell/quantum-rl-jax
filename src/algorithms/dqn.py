# DQN for gymnax environments, adapted from cleanrl's dqn_jax.py
# (https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_jaxpy)
import tyro
import time
from collections import deque
from dataclasses import dataclass

import flax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""
    env_id: str = "CartPole-v1"
    """the id of the gymnax environment"""
    total_timesteps: int = 500000
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
    learning_starts: int = 10000
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
        tx=optax.adam(learning_rate=args.learning_rate),
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
            episode_counter += 1
            recent_returns.append(episodic_return)
            recent_lengths.append(episodic_length)
            if len(episodic_returns) % 100 == 0:
                print(
                    f"global_step={global_step}, episode={episode_counter}, "
                    f"episodic_return={episodic_return}, episodic_length={episodic_length}, "
                    f"avg_return={np.mean(recent_returns):.2f}, avg_length={np.mean(recent_lengths):.2f}"
            )
            episodic_return, episodic_length = 0.0, 0

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
