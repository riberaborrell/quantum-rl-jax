import jax
import jax.numpy as jnp
import pennylane as qml
import pytest

from models.var_quantum_circuits import init_q_value, make_q_values

# enable JAX x64
jax.config.update("jax_enable_x64", True)

NUM_QUBITS = 2

@pytest.fixture(scope="module")
def q_values():
    return make_q_values(qml.device("default.qubit", wires=NUM_QUBITS))


@pytest.fixture
def observations():
    return jax.random.uniform(jax.random.PRNGKey(0), (5, NUM_QUBITS), minval=-1.0, maxval=1.0)


def test_init_q_value_shapes_and_values():
    params = init_q_value(jax.random.PRNGKey(0), num_layers=4, num_qubits=NUM_QUBITS)

    assert params["weights"].shape == (4, NUM_QUBITS, 3)
    assert params["bias"].shape == (NUM_QUBITS,)
    assert jnp.all(params["bias"] == 0)
    assert jnp.max(jnp.abs(params["weights"])) < 0.1  # small init keeps the circuit near its fixed point


def test_init_q_value_is_deterministic_in_key():
    a = init_q_value(jax.random.PRNGKey(1), 2, NUM_QUBITS)
    b = init_q_value(jax.random.PRNGKey(1), 2, NUM_QUBITS)
    c = init_q_value(jax.random.PRNGKey(2), 2, NUM_QUBITS)

    assert jnp.array_equal(a["weights"], b["weights"])
    assert not jnp.array_equal(a["weights"], c["weights"])


def test_output_shape_single_and_batch(q_values, observations):
    params = init_q_value(jax.random.PRNGKey(0), 2, NUM_QUBITS)

    assert q_values(params, observations[0]).shape == (NUM_QUBITS,)
    assert q_values(params, observations).shape == (len(observations), NUM_QUBITS)


def test_no_layers_gives_cosine_of_encoding(q_values, observations):
    # RX(pi x) takes <Z> to cos(pi x) and RZ does not change it
    params = {"weights": jnp.zeros((0, NUM_QUBITS, 3)), "bias": jnp.zeros(NUM_QUBITS)}

    assert jnp.allclose(q_values(params, observations), jnp.cos(jnp.pi * observations), atol=1e-6)


def test_bias_is_added_to_outputs(q_values, observations):
    params = init_q_value(jax.random.PRNGKey(0), 2, NUM_QUBITS)
    bias = jnp.array([0.5, -1.0])

    shifted = q_values({**params, "bias": bias}, observations)

    assert jnp.allclose(shifted, q_values(params, observations) + bias, atol=1e-6)


def test_outputs_bounded_by_expectation_range(q_values, observations):
    params = {
        "weights": jax.random.uniform(jax.random.PRNGKey(3), (3, NUM_QUBITS, 3), maxval=2 * jnp.pi),
        "bias": jnp.zeros(NUM_QUBITS),
    }

    assert jnp.all(jnp.abs(q_values(params, observations)) <= 1 + 1e-6)


def test_gradients_are_finite_and_nonzero(q_values, observations):
    params = init_q_value(jax.random.PRNGKey(0), 2, NUM_QUBITS)

    grads = jax.grad(lambda p: q_values(p, observations).sum())(params)

    assert grads["weights"].shape == params["weights"].shape
    assert jnp.all(jnp.isfinite(grads["weights"])) and jnp.any(grads["weights"] != 0)
    assert jnp.allclose(grads["bias"], len(observations))  # d(sum)/d(bias) counts each batch row


def test_jit_matches_eager(q_values, observations):
    params = init_q_value(jax.random.PRNGKey(0), 2, NUM_QUBITS)

    assert jnp.allclose(jax.jit(q_values)(params, observations), q_values(params, observations), atol=1e-6)

