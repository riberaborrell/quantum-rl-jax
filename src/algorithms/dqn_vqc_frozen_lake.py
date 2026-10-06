# VQ-DQN (variational quantum circuit DQN) for the jumanji Frozenlake environment.
# JAX + PennyLane adaptation of Var-QuantumCircuits-DeepRL (Chen et al., https://arxiv.org/abs/1907.00397),
# built on top of dqn_frozen_lake.py: only the q-value approximator changes.
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pennylane as qml
import tyro
from frozenlake.env import FrozenLake
from jaxtyping import Array, Float

from algorithms import dqn_utils
from algorithms.dqn_frozen_lake import TrainState
from algorithms.dqn_utils import ReplayBuffer, make_epsilon_schedule, make_optimizer
from models.var_quantum_circuits import QValueParams, init_q_value, make_q_values

# enable JAX x64
jax.config.update("jax_enable_x64", True)

ENV_NAME = "FrozenLake"
ALGORITHM_NAME = "dqn_vqc"

# 16 states are binary encoded in 4 qubits, which also gives the 4 actions
NUM_QUBITS = 4
device = qml.device("default.qubit", wires=NUM_QUBITS)
q_values = make_q_values(device)


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

    # variational quantum circuit architecture and optimizer
    num_layers: int = 4
    """number of variational layers of the circuit"""
    optimizer: Literal["sgd", "adam"] = "adam"
    """the optimizer of the circuit parameters"""
    learning_rate: float = 1e-2
    """the learning rate of the optimizer"""
    clip_grad_norm: float | None = None
    """if set, the maximum global norm of the gradients; larger gradients are rescaled to this norm"""

    # dqn parameters
    total_timesteps: int = 25000
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
    exploration_fraction: float = 0.25
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    epsilon_schedule: Literal["constant", "linear", "exponential"] = "linear"
    """"linear" and "exponential": epsilon decays from `start_e` to `end_e` over the first
    `exploration_fraction` of the timesteps; "constant": epsilon stays at `start_e`"""



def get_obs(env: FrozenLake, env_state) -> Float[Array, "qubits"]:
    """Binary encoding (most significant bit first) of the index of the agent position."""
    index = env_state.player_position.row * env.num_cols + env_state.player_position.col
    return ((index >> jnp.arange(NUM_QUBITS - 1, -1, -1)) & 1).astype(jnp.float32)


def load_q_value() -> QValueParams:
    """Load the circuit parameters saved by `dqn_utils.save_q_value`; the number of layers is read from the checkpoint."""
    structure = dqn_utils.get_q_value_structure(ENV_NAME, ALGORITHM_NAME)
    return dqn_utils.load_q_value(structure, ENV_NAME, ALGORITHM_NAME)


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
    assert num_actions == NUM_QUBITS, "one qubit per action is required"

    # reset environment
    env_state, _ = env.reset(reset_key)
    obs = get_obs(env, env_state)

    # variational circuit to approximate the q-value function
    q_params = init_q_value(q_key, args.num_layers, NUM_QUBITS)
    q_state = TrainState.create(
        apply_fn=q_values,
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
        action = jnp.where(
            jax.random.uniform(key_explore) < epsilon,
            jax.random.randint(key_act, (), 0, num_actions),
            q_values(params, obs).argmax(axis=-1),
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
        q_next_target = q_values(q_state.target_params, next_observations).max(axis=-1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * args.gamma * q_next_target

        def mse_loss(params):
            q_pred = q_values(params, observations)[jnp.arange(observations.shape[0]), actions]
            return ((q_pred - next_q_value) ** 2).mean()

        loss_value, grads = jax.value_and_grad(mse_loss)(q_state.params)
        return loss_value, q_state.apply_gradients(grads=grads)

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
            if len(episodic_returns) % 100 == 0:
                print(
                    f"{global_step:>11d} | {len(episodic_returns):>7d} | {episodic_return:>+9.2f} | "
                    f"{episodic_length:>9d} | {np.mean(recent_returns):>+10.2f} | {np.mean(recent_lengths):>10.2f} | {epsilon:>7.3f}"
                )
            episodic_return, episodic_length = 0.0, 0
            if len(episodic_returns) == args.max_episodes:
                break

        # save data to replay buffer; the true next obs is `final_obs` since we auto-reset
        rb.add(obs, final_obs, action, reward, terminated)
        obs = next_obs

        # training
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                loss, q_state = update(q_state, *rb.sample(args.batch_size, np_rng))

            # update target circuit
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau)
                )

    # save q-value circuit
    dqn_utils.save_q_value(q_state.params, ENV_NAME, ALGORITHM_NAME)
    return q_state, episodic_returns




if __name__ == "__main__":
    main()
