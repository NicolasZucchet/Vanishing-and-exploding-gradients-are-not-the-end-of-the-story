from functools import partial
import jax
import jax.numpy as jnp
from jax.nn import one_hot
from tqdm import tqdm
from flax.training import train_state
import optax
from typing import Any


def map_nested_fn(fn):
    """
    Recursively apply `fn to the key-value pairs of a nested dict / pytree.
    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {k: (map_fn(v) if hasattr(v, "keys") else fn(k, v)) for k, v in nested_dict.items()}

    return map_fn


def loss_fn(preds, targets, masks):
    return 0.5 * jnp.sum((preds - targets) ** 2 * masks[:, :, None]) / jnp.sum(masks)


@partial(jax.jit, static_argnums=(5, 6))
def train_step(params, opt_state, inputs, labels, masks, opt, model):
    """Performs a single training step given a batch of data"""

    def _loss(params):
        logits = model.apply(params, inputs)
        return loss_fn(logits, labels, masks), logits

    (loss, preds), grads = jax.value_and_grad(_loss, has_aux=True)(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    metrics = {
        "Grad norm": optax.global_norm(grads),
        "Param norm": optax.global_norm(params),
        "Training loss": loss,
    }

    return params, opt_state, metrics, preds


def train_epoch(params, opt_state, opt, model, trainloader):
    """
    Training function for an epoch that loops over batches.
    """
    metrics = {"Grad norm": [], "Param norm": [], "Training loss": []}
    for batch in trainloader:
        inputs, labels, masks = batch
        params, opt_state, step_metrics, preds = train_step(
            params, opt_state, inputs, labels, masks, opt, model
        )
        for k, v in step_metrics.items():
            metrics[k].append(v.item())

    average_metrics = {k: jnp.mean(jnp.array(v)) for k, v in metrics.items()}
    # Return average loss over batches
    return params, opt_state, average_metrics


@partial(jax.jit, static_argnums=(4,))
def eval_step(inputs, targets, masks, params, model):
    logits = model.apply(params, inputs)
    losses = loss_fn(logits, targets, masks)

    return jnp.mean(losses), logits


def validate(params, model, testloader):
    """Validation function that loops over batches"""
    losses = jnp.array([])

    for batch in testloader:
        inputs, labels, masks = batch
        loss, _ = eval_step(inputs, labels, masks, params, model)
        losses = jnp.append(losses, loss)
    return jnp.mean(losses)


def compute_n_params(params):
    def fn_is_complex(x):
        return x.dtype in [jnp.complex64, jnp.complex128]

    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(
        params
    )
    return sum(jax.tree_util.tree_leaves(param_sizes))


def norm(pytree):
    leaves = jax.tree_util.tree_leaves(pytree)
    return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))


def dot_product(pytree1, pytree2):
    leaves1 = jax.tree_util.tree_leaves(pytree1)
    leaves2 = jax.tree_util.tree_leaves(pytree2)
    return sum(jnp.vdot(x, y) for x, y in zip(leaves1, leaves2))


@partial(jax.jit, static_argnums=0)
def hvp(loss, x, v):
    return jax.grad(lambda x: dot_product(jax.grad(loss)(x), v))(x)


def compute_L_hessian(params, model, inputs, outputs, masks, loss=None):
    if loss is None:
        loss = loss_fn
    _loss = lambda params: loss(model.apply(params, inputs), outputs, masks)
    est_eigvect = jax.tree_map(lambda x: jnp.ones(x.shape), params)
    est_eigvect = jax.tree_util.tree_map(lambda x: x / norm(est_eigvect), est_eigvect)
    est_eigval = 0
    values = [est_eigval]
    for _ in range(50):
        new_est_eigvect = hvp(_loss, params, est_eigvect)
        est_eigval = dot_product(est_eigvect, new_est_eigvect)
        values.append(est_eigval)
        new_est_eigvect = jax.tree_util.tree_map(
            lambda x: x / norm(new_est_eigvect), new_est_eigvect
        )
        est_eigvect = new_est_eigvect

    return est_eigval


def compute_directional_sharpness(params, model, inputs, outputs, masks, loss=None):
    if loss is None:
        loss = loss_fn
    _loss = lambda params: loss(model.apply(params, inputs), outputs, masks)
    grads = jax.grad(_loss)(params)
    Hg = hvp(_loss, params, grads)
    norm_grads = norm(grads)
    return dot_product(grads, Hg) / (norm_grads**2)
