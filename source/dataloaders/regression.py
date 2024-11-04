import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import flax.linen as nn


class LinearSystem:
    def __init__(
        self,
        seed: int = 42,
        d_input: int = 1,
        d_hidden: int = 10,
        d_output: int = 1,
        T: int = 100,
        parametrization: str = "diagonal_complex",
        input_type: str = "gaussian",
        input_mean: float = 0.0,
        input_std: float = 1.0,
        noise_std: float = 0.0,
        normalization: str = "none",
        **kwargs
    ):
        key = jax.random.PRNGKey(seed)
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.input_mean = input_mean
        self.input_std = input_std
        self.noise_std = noise_std
        self.input_type = input_type
        self.T = T

        if parametrization == "diagonal_complex":
            # Diagonal complex parametrization of the state space model
            # Enables controlling min and max norm of the eigenvalues of A
            key_nu, key_theta, key_B, key_C, key_D = jax.random.split(key, 5)
            nu = jax.random.uniform(
                key_nu, shape=(d_hidden,), minval=kwargs["min_nu"], maxval=kwargs["max_nu"]
            )
            theta = jax.random.uniform(
                key_theta,
                shape=(d_hidden,),
                minval=-kwargs["max_phase"],
                maxval=kwargs["max_phase"],
            )
            self.A = jnp.eye(d_hidden) * jnp.diag(nu * jnp.exp(1j * theta))
            self.B = jax.random.normal(
                key_B, shape=(d_hidden, d_input), dtype=jnp.complex64
            ) / jnp.sqrt(2 * d_hidden)
            self.C = jax.random.normal(
                key_C, shape=(d_output, d_hidden), dtype=jnp.complex64
            ) / jnp.sqrt(2 * d_hidden)
            self.D = jax.random.normal(
                key_D, shape=(d_output, d_input), dtype=jnp.float32
            ) / jnp.sqrt(d_output)
        elif parametrization == "diagonal_real":
            # Same as diagonal complex, but restrict the eigenvalues to be symmetric
            raise NotImplementedError
        elif parametrization == "controllable":
            # Controllable canonical form of the state space model
            # https://en.wikipedia.org/wiki/State-space_representation#Canonical_realizations
            assert d_input == 1
            assert d_output == 1
            raise NotImplementedError
        elif parametrization == "dense":
            # Vanilla parametrization of the state space model
            # All parameters are sampled from a normal distribution
            key_A, key_B, key_C, key_D = jax.random.split(key, 4)
            self.A = jax.random.normal(key_A, shape=(d_hidden, d_hidden)) / jnp.sqrt(d_hidden)
            with jax.default_device(jax.devices("cpu")[0]):  # eigval decomposition has to be on CPU
                l, V = jnp.linalg.eig(self.A)
                l = (
                    kwargs["min_nu"] + (kwargs["max_nu"] - kwargs["min_nu"]) * nn.tanh(jnp.abs(l))
                ) * jnp.exp(1.0j * jnp.angle(l) * kwargs["max_phase"] / jnp.pi)
                self.A = (V @ jnp.diag(l) @ jnp.linalg.inv(V)).real
            self.B = jax.random.normal(key_B, shape=(d_hidden, d_input)) / jnp.sqrt(d_input)
            self.C = jax.random.normal(key_C, shape=(d_output, d_hidden)) / jnp.sqrt(d_hidden)
            self.D = jax.random.normal(key_D, shape=(d_output, d_input)) / jnp.sqrt(d_input)

            # If needed, normalize B to make sure no eigenvalues dominate the output
            if normalization == "L2":
                self.B = V @ jnp.diag(jnp.sqrt(1 - jnp.abs(l) ** 2)) * jnp.linalg.inv(V) @ self.B
                self.B = self.B.real
            elif normalization == "L1":
                self.B = V @ jnp.diag(1 - jnp.abs(l)) * jnp.linalg.inv(V) @ self.B
                self.B = self.B.real
        else:
            raise ValueError

        # If we don't use B, C, D, we set them to Id, Id, 0
        if not kwargs["use_B_C_D"]:
            assert d_input == d_hidden
            assert d_hidden == d_output
            self.B = jnp.eye(d_hidden)
            self.C = jnp.eye(d_output)
            self.D = jnp.zeros_like(self.D)

    @partial(jax.jit, static_argnums=(0,))
    def sample(self, key):
        key_inputs, key_noise = jax.random.split(key, 2)
        inputs = (
            self.input_mean
            + jax.random.normal(key_inputs, shape=(self.T, self.d_input)) * self.input_std
        )
        if self.input_type == "gaussian":
            inputs = inputs
        elif self.input_type == "constant":
            inputs = jnp.ones_like(inputs) * inputs[0]
        noise = self.noise_std * jax.random.normal(key_noise, shape=(self.T, self.d_output))

        def _step(h, x):
            new_h = self.A @ h + self.B @ x
            return new_h, new_h

        init_hiddens = jnp.zeros((self.d_hidden,), dtype=self.A.dtype)
        hiddens = jax.lax.scan(_step, init_hiddens, inputs)[1]
        outputs = jax.vmap(lambda h, x, n: (self.C @ h).real + self.D @ x + n)(
            hiddens, inputs, noise
        )
        masks = jnp.ones(outputs.shape[:-1])
        return inputs, outputs, masks

    def stats_lambdas(self):
        with jax.default_device(jax.devices("cpu")[0]):
            eigvals = jnp.linalg.eigvals(self.A)
            return jnp.abs(eigvals).min(), jnp.abs(eigvals).max()


class RegressionDataLoader:
    def __init__(self, dataset, rng, batch_size, size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.key = jax.random.PRNGKey(rng)
        self.size = size

    def __iter__(self):
        # Sample a new random seed everytime we initialize an iterator
        self.key, _ = jax.random.split(self.key)
        for key in jax.random.split(self.key, self.size):
            inputs, targets, masks = jax.vmap(self.dataset.sample)(
                jax.random.split(key, self.batch_size)
            )
            yield inputs, targets, masks

    def __len__(self):
        return self.size
