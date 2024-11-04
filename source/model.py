from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

parallel_scan = jax.lax.associative_scan


# Parallel scan operations
@jax.vmap
def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def matrix_init(key, shape, dtype=jnp.float32, normalization=1):
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / jnp.sqrt(shape[0]) / normalization


def nu_init(key, shape, r_min, r_max, max_phase, dtype=jnp.float32, param="exp"):
    key_nu, key_theta = jax.random.split(key, 2)
    u_nu = jax.random.uniform(key=key_nu, shape=shape, dtype=dtype)
    nu = jnp.sqrt(u_nu * (r_max**2 - r_min**2) + r_min**2)
    u_theta = jax.random.uniform(
        key_theta, shape=shape, dtype=dtype, minval=-max_phase, maxval=max_phase
    )
    theta = jnp.mod(u_theta, 2 * jnp.pi)
    lamb = nu * jnp.exp(1j * theta)
    if param == "polar":
        out = nu, theta
    elif param == "exp":
        out = jnp.log(-jnp.log(nu)), jnp.log(theta)
    elif param == "scaled_exp":
        out = jnp.log(-jnp.log(nu)), theta / (1 - nu)
    elif param == "S4":
        out = jnp.log(lamb).real, jnp.log(lamb).imag
    else:
        out = lamb.real, lamb.imag
    return jnp.stack(out, axis=0)


def gamma_init(key, lamb, param="exp"):
    nu, theta = lamb
    if param == "exp" or param == "scaled_exp":
        nu = jnp.exp(-jnp.exp(nu))
        return jnp.log(jnp.sqrt(1 - jnp.abs(nu) ** 2))
    else:
        diag_lambda = nu * jnp.exp(1j * theta)
        return jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2)


class LRUBase(nn.Module):
    """
    LRU module in charge of the recurrent processing.
    Implementation following the one of Orvieto et al. 2023.
    """

    d_input: int  # input dimension
    d_hidden: int  # hidden state dimension
    d_output: int  # output dimension
    min_nu: float = 0.0  # smallest lambda norm
    max_nu: float = 1.0  # largest lambda norm
    max_phase: float = jnp.pi  # max phase lambda
    which_gamma: str = "learned"  # which gamma
    parametrization: str = "exp"  # which parametrization to use
    use_B_C_D: bool = True  # use B, C, D

    def setup(self):
        assert self.parametrization in ["exp", "polar", "scaled_exp", "default"]
        self.lamb = self.param(
            "lambda",
            partial(
                nu_init,
                r_min=self.min_nu,
                r_max=self.max_nu,
                max_phase=self.max_phase,
                param=self.parametrization,
            ),
            (self.d_hidden,),
        )

        assert self.which_gamma in ["learned", "L2", "L1", "none"]
        if self.which_gamma == "learned":
            self.gamma = self.param(
                "gamma",
                partial(gamma_init, param=self.parametrization),
                (self.lamb[0], self.lamb[1]),
            )

        if self.use_B_C_D:
            self.B_re = self.param("B_re", matrix_init, (self.d_hidden, self.d_input))
            self.B_im = self.param("B_im", matrix_init, (self.d_hidden, self.d_input))
            self.C_re = self.param("C_re", matrix_init, (self.d_output, self.d_hidden))
            self.C_im = self.param("C_im", matrix_init, (self.d_output, self.d_hidden))
            self.D = self.param("D", matrix_init, (self.d_output, self.d_input))
        else:
            assert self.d_input == self.d_hidden
            assert self.d_hidden == self.d_output

    def get_diag(self):
        if self.parametrization == "polar":
            return self.lamb[0] * jnp.exp(1j * self.lamb[1])
        elif self.parametrization == "exp":
            return jnp.exp(-jnp.exp(self.lamb[0]) + 1j * jnp.exp(self.lamb[1]))
        elif self.parametrization == "scaled_exp":
            nu = jnp.exp(-jnp.exp(self.lamb[0]))
            return nu * jnp.exp(1j * jax.lax.stop_gradient(1 - jnp.abs(nu)) * self.lamb[1])
        else:
            return self.lamb[0] + 1j * self.lamb[1]

    def __call__(self, inputs):
        """Forward pass of a LRU: h_t+1 = lambda * h_t + B x_t+1, y_t = Re[C h_t + D x_t]"""
        # Compute diagonal terms
        diag_lambda = self.get_diag()
        Lambda_elements = jnp.repeat(diag_lambda[None, ...], inputs.shape[0], axis=0)

        # Compute inputs
        if self.use_B_C_D:
            B_norm = self.B_re + 1j * self.B_im
            if self.which_gamma == "learned":
                gamma = jnp.exp(self.gamma) if self.parametrization == "exp" else self.gamma
                B_norm = jnp.diag(gamma) @ B_norm
            elif self.which_gamma == "L2":
                B_norm = jnp.diag(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2 + 1e-6)) @ B_norm
            elif self.which_gamma == "L1":
                B_norm = jnp.diag(1 - jnp.abs(diag_lambda)) @ B_norm
            Bu_elements = jax.vmap(lambda x: B_norm @ x)(inputs)
        else:
            Bu_elements = inputs + 1j * jnp.zeros_like(inputs)

        # Compute hidden states
        _, hidden_states = parallel_scan(binary_operator_diag, (Lambda_elements, Bu_elements))

        # Use them to compute the output of the module
        if self.use_B_C_D:
            C = self.C_re + 1j * self.C_im
            outputs = jax.vmap(lambda h, x: (C @ h).real + self.D @ x)(hidden_states, inputs)
            return outputs
        else:
            return hidden_states.real


LRU = nn.vmap(
    LRUBase,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "cache": 0, "prime": None},
    split_rngs={"params": False},
    axis_name="batch",
)


# NOTE: this is a non conventional initialization of S4, it is here to reflect the structure of
# the teacher in the linear system identification task
def delta_init(key, shape, scheme="default", nu_min=0, nu_max=1, dtype=jnp.float32):
    inv_softplus = lambda x: jnp.log(jnp.exp(x) - 1)
    if scheme == "default":
        u_nu = jax.random.uniform(key, shape=shape, dtype=dtype)
        nu = jnp.sqrt(u_nu * (nu_max**2 - nu_min**2) + nu_min**2)
        return inv_softplus(-jnp.log(nu))
    elif scheme == "delta_1":
        return inv_softplus(1 * jnp.ones(shape, dtype=dtype))


def A_init(key, shape, Delta, scheme="default", nu_min=0.0, nu_max=1.0, dtype=jnp.float32):
    if scheme == "default":
        theta = jax.random.uniform(key, shape=shape, dtype=dtype) * 2 * jnp.pi / Delta
        return jnp.stack([-jnp.ones(shape, dtype=dtype), theta])
    elif scheme == "delta_1":
        key_nu, key_theta = jax.random.split(key)
        u_nu = jax.random.uniform(key_nu, shape=shape, dtype=dtype)
        nu = jnp.sqrt(u_nu * (nu_max**2 - nu_min**2) + nu_min**2)
        theta = jax.random.uniform(key_theta, shape=shape, dtype=dtype) * 2 * jnp.pi
        return jnp.stack([jnp.log(nu), theta])


class S4Base(nn.Module):
    d_model: int  # input / output dimension (D)
    hiddens_per_input: int = 16  # number of hidden states per input (N)
    shared_Delta: bool = True  # same Delta for all hidden states per input dimension
    min_nu: float = 0.0  # smallest lambda norm
    max_nu: float = 1.0  # largest lambda norm
    max_phase: float = jnp.pi  # max phase lambda
    init_scheme: str = "default"  # initialization scheme

    def setup(self):
        # NOTE: deviation from usual architecture, should be D
        if self.shared_Delta:
            self.Delta = self.param(
                "Delta",
                partial(
                    delta_init, nu_min=self.min_nu, nu_max=self.max_nu, scheme=self.init_scheme
                ),
                (self.d_model,),
            )  # (D)
        else:
            self.Delta = self.param(
                "Delta",
                partial(
                    delta_init, nu_min=self.min_nu, nu_max=self.max_nu, scheme=self.init_scheme
                ),
                (self.hiddens_per_input,),
            )  # (N)
        self.A = self.param(
            "A",
            partial(
                A_init,
                Delta=self.Delta,
                nu_min=self.min_nu,
                nu_max=self.max_nu,
                scheme=self.init_scheme,
            ),
            (self.d_model, self.hiddens_per_input),
        )  # (D, N) complex
        self.B_re = self.param(
            "B_re", partial(matrix_init, normalization=1), (self.d_model, self.hiddens_per_input)
        )
        self.B_im = self.param(
            "B_im", partial(matrix_init, normalization=1), (self.d_model, self.hiddens_per_input)
        )
        self.C_re = self.param("C_re", matrix_init, (self.d_model, self.hiddens_per_input))
        self.C_im = self.param("C_im", matrix_init, (self.d_model, self.hiddens_per_input))
        self.D = self.param("D", matrix_init, (self.d_model,))

    def __call__(self, x):
        B = self.B_re + 1j * self.B_im
        C = self.C_re + 1j * self.C_im
        A = self.A[0] + 1j * self.A[1]

        Delta = nn.softplus(self.Delta)
        if self.shared_Delta:
            Lambda_elements = jnp.repeat(
                jnp.exp(Delta[:, None] * A)[None, ...], x.shape[0], axis=0
            )  # L D N
            Bu_elements = jnp.einsum("D, D N, L D -> L D N", Delta, B, x)  # L D N
        else:
            Lambda_elements = jnp.repeat(
                jnp.exp(Delta[None, :] * A)[None, ...], x.shape[0], axis=0
            )  # L D N
            Bu_elements = jnp.einsum("N, D N, L D -> L D N", Delta, B, x)  # L D N
        _, hidden_states = parallel_scan(binary_operator_diag, (Lambda_elements, Bu_elements))
        out = jnp.einsum("L D N, D N -> L D", hidden_states, C) + self.D[None, :] * x
        return out.real


S4 = nn.vmap(
    S4Base,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "cache": 0, "prime": None},
    split_rngs={"params": False},
    axis_name="batch",
)


def recurrent_matrix_init(rng, shape, min_nu, max_nu, max_phase):
    A = jax.random.normal(rng, shape) / jnp.sqrt(shape[0])
    with jax.default_device(jax.devices("cpu")[0]):  # eigval decomposition has to be on CPU
        l, V = jnp.linalg.eig(A)
        l = (min_nu + (max_nu - min_nu) * nn.tanh(jnp.abs(l))) * jnp.exp(
            1.0j * jnp.angle(l) * max_phase / jnp.pi
        )
        A = V @ jnp.diag(l) @ jnp.linalg.inv(V)
    return A.real


class RNNBase(nn.Module):
    d_hidden: int  # hidden state dimension
    d_output: int  # output dimension
    min_nu: int = 0.0  # smallest norm
    max_nu: int = 1.0  # largest norm
    max_phase: float = jnp.pi  # max phase lambda
    A_init: str = "normal"  # initialization of A
    force_complex: bool = False  # force complex like behavior
    use_B_C_D: bool = False  # use B, C, D

    def setup(self):
        if self.force_complex:
            assert self.d_hidden == 2
            init = partial(
                nu_init, r_min=self.min_nu, r_max=self.max_nu, max_phase=jnp.pi, param="default"
            )
            self.lamb = self.param("lambda", init, (1,))
        else:
            self.A = self.param(
                "A",
                partial(
                    recurrent_matrix_init,
                    min_nu=self.min_nu,
                    max_nu=self.max_nu,
                    max_phase=self.max_phase,
                ),
                (self.d_hidden, self.d_hidden),
            )
        if self.use_B_C_D:
            self.B = nn.Dense(self.d_hidden, use_bias=False, name="B")
            self.C = nn.Dense(self.d_output, use_bias=False, name="C")
            self.D = nn.Dense(self.d_output, use_bias=False, name="D")

    def __call__(self, x):
        def _step(h, x):
            if self.force_complex:
                A = jnp.array(
                    [
                        [self.lamb[0, 0], -self.lamb[1, 0]],
                        [self.lamb[1, 0], self.lamb[0, 0]],
                    ]
                )
            else:
                A = self.A
            new_h = A @ h + x
            return new_h, new_h

        if self.use_B_C_D:
            inp = self.B(x)
        else:
            inp = x
        h = jax.lax.scan(_step, jnp.zeros((self.d_hidden,)), inp)[1]
        if self.use_B_C_D:
            return self.C(h) + self.D(x)
        else:
            return h


RNN = nn.vmap(
    RNNBase,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "cache": 0, "prime": None},
    split_rngs={"params": False},
    axis_name="batch",
)


class MultiHeadRNN(nn.Module):
    d_hidden: int  # hidden state dimension
    d_output: int  # output dimension
    min_nu: int = 0.0  # smallest norm
    max_nu: int = 1.0  # largest norm
    max_phase: float = jnp.pi  # max phase lambda
    A_init: str = "normal"  # initialization of A
    n_heads: int = 1  # number of heads
    force_complex: bool = False  # force complex like behavior

    @nn.compact
    def __call__(self, x):
        assert self.d_hidden % self.n_heads == 0
        inputs = x
        x = nn.Dense(
            self.d_hidden,
            name="B",
            use_bias=False,
        )(x)
        x = x.reshape(x.shape[0], x.shape[1], self.n_heads, self.d_hidden // self.n_heads)
        x = nn.vmap(
            RNN,
            in_axes=2,
            out_axes=2,
            variable_axes={"params": 0, "cache": 0, "prime": 0},
            split_rngs={"params": True},
        )(
            self.d_hidden // self.n_heads,
            None,  # no need to specify output dimension here
            self.min_nu,
            self.max_nu,
            self.max_phase,
            self.A_init,
            self.force_complex,
        )(
            x
        )
        x = jnp.reshape(x, (x.shape[0], x.shape[1], -1))
        return nn.Dense(self.d_output, name="C", use_bias=False)(x) + nn.Dense(
            self.d_output, name="D", use_bias=False
        )(inputs)


class SequenceLayer(nn.Module):
    """Single layer, with one LRU module, GLU, dropout and batch/layer norm"""

    lru: LRUBase  # lru module
    d_model: int  # model size
    dropout: float = 0.0  # dropout probability
    norm: str = "layer"  # which normalization to use
    training: bool = True  # in training mode (dropout in trainign mode only)
    pre_norm: bool = True  # whether to apply normalization before or after the LRU

    def setup(self):
        """Initializes the ssm, layer norm and dropout"""
        self.seq = self.lru()
        self.out1 = nn.Dense(self.d_model)
        self.out2 = nn.Dense(self.d_model)
        if self.norm in ["layer"]:
            self.normalization = nn.LayerNorm()
        else:
            self.normalization = nn.BatchNorm(
                use_running_average=not self.training, axis_name="batch"
            )
        self.drop = nn.Dropout(self.dropout, broadcast_dims=[0], deterministic=not self.training)

    def __call__(self, x):
        skip = x
        if self.pre_norm:
            x = self.normalization(x)  # pre normalization
        x = self.seq(x)  # call LRU
        x = self.drop(nn.gelu(x))
        x = self.out1(x) * jax.nn.sigmoid(self.out2(x))  # GLU
        x = self.drop(x)
        x = x + skip  # skip connection
        if not self.pre_norm:
            x = self.normalization(x)
        return x


class StackedEncoderModel(nn.Module):
    """Encoder containing several SequenceLayer"""

    lru: LRUBase
    d_model: int
    n_layers: int
    dropout: float = 0.0
    training: bool = True
    norm: str = "batch"
    pre_norm: bool = True

    def setup(self):
        self.encoder = nn.Dense(self.d_model)
        self.layers = [
            SequenceLayer(
                lru=self.lru,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
                norm=self.norm,
                pre_norm=self.pre_norm,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, inputs):
        x = self.encoder(inputs)  # embed input in latent space
        for layer in self.layers:
            x = layer(x)  # apply each layer
        return x


class FullModel(nn.Module):
    """Stacked encoder with pooling and eventually softmax"""

    lru: nn.Module
    d_output: int
    d_model: int
    n_layers: int
    dropout: float = 0.0
    training: bool = True
    pooling: str = "mean"  # pooling mode
    norm: str = "batch"  # type of normaliztion
    multidim: int = 1  # number of outputs
    classification: bool = True
    pre_norm: bool = True

    def setup(self):
        self.encoder = StackedEncoderModel(
            lru=self.lru,
            d_model=self.d_model,
            n_layers=self.n_layers,
            dropout=self.dropout,
            training=self.training,
            norm=self.norm,
            pre_norm=self.pre_norm,
        )
        self.decoder = nn.Dense(self.d_output * self.multidim, use_bias=False)

    def __call__(self, x):
        x = self.encoder(x)
        if self.pooling in ["mean"]:
            x = jnp.mean(x, axis=0)  # mean pooling across time
        elif self.pooling in ["last"]:
            x = x[-1]  # just take last
        elif self.pooling in ["none"]:
            x = x  # do not pool at all
        x = self.decoder(x)

        if self.multidim > 1:
            x = x.reshape(-1, self.d_output, self.multidim)

        # Softmax if classification
        if self.classification:
            return nn.log_softmax(x, axis=-1)
        else:
            return x


class CellWrapper(nn.Module):
    """Wrapper for recurrent cell to process an entire sequence at once"""

    cell: nn.Module

    def setup(self):
        self.d_hidden = self.cell.d_hidden
        self.cell = self.cell()

    def __call__(self, x):
        init_h = jnp.zeros((x.shape[0], self.d_hidden))
        return jax.lax.scan(self.cell.__call__, x, init_h)


# Batched version of the different models

# Here we call vmap to parallelize across a batch of input sequences
BatchFullModel = nn.vmap(
    FullModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "batch_stats": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
    axis_name="batch",
)

BatchRecurrentLayer = nn.vmap(
    CellWrapper,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "cache": 0, "prime": None},
    split_rngs={"params": False},
    axis_name="batch",
)
