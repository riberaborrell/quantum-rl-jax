from collections.abc import Callable

import jax
import jax.numpy as jnp
import pennylane as qml
from jaxtyping import Array, Float

QValueParams = dict[str, Float[Array, "..."]]


def make_q_values(
    device: qml.devices.Device,
) -> Callable[[QValueParams, Float[Array, "... qubits"]], Float[Array, "... qubits"]]:
    """Build the q-value function of a variational quantum circuit running on the given device.

    The observation is angle-encoded with one feature per qubit, so it needs as many features as the
    device has wires. The ⟨Z⟩ of each qubit is the q-value of one action, so there is one action per wire.

    Args:
        device: PennyLane device; its number of wires sets the number of qubits.

    Returns:
        Function mapping the parameters and a single observation or a batch of them to q-values.
    """
    num_qubits = len(device.wires)

    @qml.qnode(device, interface="jax")
    def circuit(weights: Float[Array, "layers qubits 3"], obs: Float[Array, "qubits"]) -> list:

        # data embedding
        for wire in range(num_qubits):
            qml.RX(jnp.pi * obs[wire], wires=wire)
            qml.RZ(jnp.pi * obs[wire], wires=wire)

        # trainable ansatz
        for layer_weights in weights:
            for wire in range(num_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
            for wire in range(num_qubits):
                qml.Rot(*layer_weights[wire], wires=wire)

        return [qml.expval(qml.PauliZ(wire)) for wire in range(num_qubits)]

    def q_values(params: QValueParams, obs: Float[Array, "... qubits"]) -> Float[Array, "... qubits"]:
        apply = lambda o: jnp.stack(circuit(params["weights"], o)) + params["bias"]
        return apply(obs) if obs.ndim == 1 else jax.vmap(apply)(obs)

    return q_values


def init_q_value(key: Array, num_layers: int, num_qubits: int) -> QValueParams:
    """Initialize the circuit weights close to zero and the output bias at zero."""
    return {
        "weights": 0.01 * jax.random.normal(key, (num_layers, num_qubits, 3)),
        "bias": jnp.zeros(num_qubits),
    }
