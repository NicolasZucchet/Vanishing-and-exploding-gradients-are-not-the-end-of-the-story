import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from jax.flatten_util import ravel_pytree
from .train_helpers import loss_fn


def analyze_hessian(model, params, X, Y, masks, pre_factor=None, log_scale=False):
    p, unravel = ravel_pytree(params)
    safe_unravel = lambda p: jax.tree_util.tree_map(lambda x: x.astype(p.dtype), unravel(p))

    @jax.jit
    def loss(p):
        pred = model.apply(safe_unravel(p), X)
        return loss_fn(pred, Y, masks)

    print("Loss", loss(p))

    H = jax.hessian(loss)(p)

    fig, ax = plt.subplots(1, 3, figsize=(18, 4), dpi=400)

    sizes = jnp.array([np.prod(x.shape) for x in jax.tree.flatten(unravel(p))[0]])
    cum_sizes = jnp.array([0] + list(sizes.cumsum()))

    if log_scale:
        norm = LogNorm()
        im = ax[0].imshow(jnp.abs(H), norm=norm)
        ax[0].set_title("Hessian (abs)")
    else:
        norm = Normalize(vmin=jnp.min(H), vmax=jnp.max(H))
        im = ax[0].imshow(H, norm=norm, interpolation="none")
        ax[0].set_title("Hessian")

    plt.colorbar(ax=ax[0], mappable=im)
    ax[0].set_xticks(cum_sizes - 0.5, [f"" for s in cum_sizes])
    ax[0].set_yticks(cum_sizes - 0.5, [f"" for s in cum_sizes])
    print("Param groups", jax.tree.flatten(unravel(p))[1])

    norm = Normalize(vmin=-1, vmax=1)
    im = ax[1].imshow(jnp.linalg.eigh(H).eigenvectors, cmap="RdBu", norm=norm)
    ax[1].set_xlabel("Index eigval")
    ax[1].set_yticks(cum_sizes - 0.5, [f"" for s in cum_sizes])
    plt.colorbar(ax=ax[1], mappable=im)
    ax[1].set_title("Eigenvectors")

    ax[2].plot(jnp.linalg.eigh(H).eigenvalues)
    ax[2].set_xlabel("Index eigval")
    ax[2].set_ylabel("Magnitude eigval")
    ax[2].set_title("Eigenvalues")
    if log_scale:
        ax[2].set_yscale("log")


def plot_history_scale(opt, hist_opt_state, params):
    ones = jax.tree_map(lambda x: jnp.ones_like(x), params)
    scale, _ = opt.update(ones, hist_opt_state, params)
