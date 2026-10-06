from collections.abc import Callable, Sequence

import flax.linen as nn
from jaxtyping import Array, Float


class QNetwork(nn.Module):
    """MLP that maps an observation to the q-value of each action.

    Attributes:
        action_dim: Number of actions, i.e. the output dimension.
        hidden_dims: Width of each hidden layer; its length sets the number of hidden layers.
        activation: Activation applied after every hidden layer (not to the output).
    """

    action_dim: int
    hidden_dims: Sequence[int] = (32, 16)
    activation: Callable[[Array], Array] = nn.relu

    @nn.compact
    def __call__(self, x: Float[Array, "... obs_dim"]) -> Float[Array, "... action_dim"]:
        for hidden_dim in self.hidden_dims:
            x = self.activation(nn.Dense(hidden_dim)(x))
        return nn.Dense(self.action_dim)(x)
