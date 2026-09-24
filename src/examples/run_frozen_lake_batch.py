from functools import partial
from dataclasses import dataclass
import tyro
import jax
import jax.numpy as jnp
import Gridworld

@dataclass
class Args:
    """ set rollout parameters"""

    seed: int = 1
    """seed of the experiment"""
    num_steps: int = 100
    """number of time steps of the rollout"""
    batch_size: int = 1024
    """number of trajectories to run in parallel"""


def rollout(env, num_actions, key_input, num_steps=100):
    """Rollout a jitted jumanji episode with lax.scan (auto-resets on episode end)."""

    key_reset, key_episode = jax.random.split(key_input)
    state, timestep = env.reset(key_reset)

    def policy_step(carry, _):
        """lax.scan compatible step transition in jax env."""
        state, key = carry
        key, key_act, key_reset = jax.random.split(key, 3)
        action = jax.random.randint(key_act, (), 0, num_actions)
        next_state, timestep = env.step(state, action)
        done = timestep.last()

        # jumanji envs do not auto-reset like gymnax: reset manually when done
        reset_state, _ = env.reset(key_reset)
        carry_state = jax.tree.map(
            lambda r, s: jnp.where(done, r, s), reset_state, next_state
        )
        carry = (carry_state, key)
        return carry, (state.elf_position, action, timestep.reward, next_state.elf_position, done)

    # Scan over episode step loop
    _, scan_out = jax.lax.scan(
        policy_step,
        (state, key_episode),
        None,
        length=num_steps,
    )
    pos, action, reward, next_pos, done = scan_out
    return pos, action, reward, next_pos, done

def main():

    # load arguments
    args = tyro.cli(Args)

    # Make environment
    env = Gridworld.Frozenlake()

    # initialize jax key
    key = jax.random.key(args.seed)
    keys_batch = jax.random.split(key, args.batch_size)

    # jumanji specs cannot be built inside jit, so get the number of actions here
    num_actions = env.action_spec().num_values

    # run rollout
    rollout_batch = jax.jit(
        jax.vmap(partial(rollout, env, num_actions, num_steps=args.num_steps))
    )
    pos, action, reward, next_pos, done = rollout_batch(keys_batch)

    # compute statistics
    n_episodes = done.sum()
    print(f"Batch shapes: reward {reward.shape}, done {done.shape}")
    print(f"Completed episodes: {n_episodes}")
    print(f"Mean episode length: {done.size / n_episodes:.2f}")
    print(f"Mean return per episode: {reward.sum() / n_episodes:.2f}")

if __name__ == '__main__':
    main()
