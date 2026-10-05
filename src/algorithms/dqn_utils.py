import json
import os
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Literal

import flax
import numpy as np
import optax

from utils.path import get_q_value_dir_path


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


def make_optimizer(name: Literal["sgd", "adam", "rmsprop"], learning_rate: float) -> optax.GradientTransformation:
    """Optax optimizer with the given name and learning rate."""
    return {"sgd": optax.sgd, "adam": optax.adam, "rmsprop": optax.rmsprop}[name](learning_rate)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def exponential_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    """Geometric decay from `start_e` at t=0 to `end_e` at t=`duration`, constant afterwards."""
    return max(start_e * (end_e / start_e) ** (t / duration), end_e)


def constant_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    """Constant `start_e`; same signature as the decaying schedules so they are interchangeable."""
    return start_e


EPSILON_SCHEDULES: dict[str, Callable[[float, float, int, int], float]] = {
    "constant": constant_schedule,
    "linear": linear_schedule,
    "exponential": exponential_schedule,
}


def save_q_value(q_params: Any, args: Any, env_name: str, algorithm_name: str) -> None:
    """Save the q-value parameters and the (dataclass) arguments of the run in data/[env]/[algorithm]/q-value."""
    dir_path = get_q_value_dir_path(env_name, algorithm_name)
    with open(os.path.join(dir_path, "params.msgpack"), "wb") as f:
        f.write(flax.serialization.to_bytes(q_params))
    with open(os.path.join(dir_path, "args.json"), "w") as f:
        json.dump(asdict(args), f, indent=4)
    print(f"q-value saved to {dir_path}")


def load_q_value(q_params: Any, env_name: str, algorithm_name: str) -> Any:
    """Load the q-value parameters saved by `save_q_value`.

    Args:
        q_params: Parameters with the same structure as the saved ones, e.g. a fresh initialization.
            Their values are discarded.
        env_name: Name of the environment the parameters were trained on.
        algorithm_name: Name of the algorithm that trained the parameters.

    Returns:
        The saved parameters.
    """
    dir_path = get_q_value_dir_path(env_name, algorithm_name)
    with open(os.path.join(dir_path, "params.msgpack"), "rb") as f:
        return flax.serialization.from_bytes(q_params, f.read())

