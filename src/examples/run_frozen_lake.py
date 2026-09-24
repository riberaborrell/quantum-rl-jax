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


# jumanji's env.reset(key) returns (state, timestep), and env.step(state, action) takes no key. Reward comes from timestep.reward and done from timestep.last()
def rollout(env, key, num_steps=100):

    state_seq, reward_seq = [], []

    # reset environment
    key, key_reset = jax.random.split(key, 2)
    state, timestep = env.reset(key_reset)

    for _ in range(num_steps):
        state_seq.append(state)

        # sample random action
        key, key_act = jax.random.split(key, 2)
        action = jax.random.randint(key_act, (), 0, env.action_spec().num_values)

        # make step
        state, timestep = env.step(state, action)
        reward_seq.append(timestep.reward)

        # break if done
        if timestep.last():
            break

    return state_seq, reward_seq

def main():

    # load arguments
    args = tyro.cli(Args)

    # Make environment
    env = Gridworld.Frozenlake()

    # initialize jax key
    key = jax.random.key(args.seed)

    # run rollout
    state_seq, reward_seq = rollout(env, key, args.num_steps)

    # compute cumulative rewards
    cum_rewards = jnp.cumsum(jnp.array(reward_seq))

    print(f"Trajectory's length: {len(state_seq)}")
    print(f"Cumulative rewards: {cum_rewards[-1]}")

if __name__ == '__main__':
    main()
