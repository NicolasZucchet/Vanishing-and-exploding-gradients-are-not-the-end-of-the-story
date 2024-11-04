from functools import partial
from jax import random
import wandb
from .train_helpers import (
    train_epoch,
    validate,
    compute_n_params,
    compute_L_hessian,
    compute_directional_sharpness,
)
from .dataloading import Datasets
from .model import BatchRecurrentLayer, LRU, S4, MultiHeadRNN
from lru.analysis import analyze_hessian
from tqdm import tqdm
import jax.numpy as jnp
import jax
import optax
import time


def train(args):
    """
    Main function to train over a certain number of epochs
    """

    assert (
        args.dataset == "linear-system-identification"
    ), "This repo is tailored to linear system identification."

    lr = args.lr_base

    # Set randomness...
    print("[*] Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, model_rng, train_rng = random.split(key, num=3)

    # Get dataset creation function
    create_dataset_fn = Datasets[args.dataset]

    kwargs = {"d_input": args.d_input, "d_output": args.d_output}
    if args.dataset == "linear-system-identification":
        if args.force_lsi_min_nu_model:
            args.lsi_min_nu = args.min_nu
            args.lsi_max_phase = args.max_phase
        param = args.lsi_parametrization
        # Put all args that start with "lsi_[param]" into kwargs
        name_args = "lsi_" + param
        for arg in vars(args):
            if arg.startswith(name_args):
                kwargs[arg[len(name_args) + 1 :]] = getattr(args, arg)
            elif arg.startswith("lsi_"):  # we add the other ones
                kwargs[arg[4:]] = getattr(args, arg)

    if args.use_wandb:
        # Make wandb config dictionary
        wandb.init(
            project=args.wandb_project,
            job_type="model_training",
            config=vars(args),
            entity=args.wandb_entity,
        )

    # Create dataset...
    init_rng, key = random.split(init_rng, num=2)
    (
        trainloader,
        valloader,
        testloader,
        aux_dataloaders,
        n_classes,
        seq_len,
        in_dim,
        train_size,
    ) = create_dataset_fn(
        args.dir_name,
        seed=args.jax_seed,
        batch_size=args.batch_size,
        n_batches_per_epoch=args.n_batches_per_epoch,
        **kwargs,
    )
    print(f"[*] Starting training on `{args.dataset}` =>> Initializing...")

    if args.dataset == "linear-system-identification":
        dataset = trainloader.dataset
        min_lambda, max_lambda = dataset.stats_lambdas()
        wandb.run.summary["min_lambda_data"] = min_lambda
        wandb.run.summary["max_lambda_data"] = max_lambda

    # In this setting, the model is just the reccurent layer
    if "LRU" in args.model:
        model_cls = partial(
            LRU,
            d_input=args.d_input,
            d_hidden=args.d_hidden,
            d_output=args.d_output,
            min_nu=args.min_nu,
            max_nu=args.max_nu,
            max_phase=args.max_phase,
            which_gamma=args.lru_which_gamma,
            parametrization=args.lru_param,
            use_B_C_D=args.use_B_C_D,
        )
    if args.model == "S4":
        model_cls = partial(
            S4,
            d_model=args.d_input,
            hiddens_per_input=args.d_hidden,
            min_nu=args.min_nu,
            max_nu=args.max_nu,
            shared_Delta=args.s4_shared_Delta,
            init_scheme=args.s4_init_scheme,
        )
    elif args.model == "RNN":
        model_cls = partial(
            MultiHeadRNN,
            d_hidden=args.d_hidden,
            d_output=args.d_output,
            # min_nu=args.min_nu, better results without
            max_nu=args.max_nu,
            max_phase=args.max_phase,
            n_heads=args.rnn_n_heads,
            force_complex=args.rnn_force_complex,
        )

    model = model_cls()

    # Initialize model
    params = model.init(model_rng, jnp.ones((args.batch_size, seq_len, in_dim)))
    n_params = compute_n_params(params)
    wandb.run.summary["Number params"] = n_params
    print("[*] Number of parameters:", n_params)

    # Initialize optimizer
    if args.lr_schedule == "constant":
        schedule = optax.constant_schedule(args.lr_base)
    elif args.lr_schedule == "cosine":
        full_training_length = args.epochs * args.n_batches_per_epoch
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=args.lr_min,
            peak_value=args.lr_base,
            warmup_steps=args.warmup_end * args.n_batches_per_epoch,
            decay_steps=full_training_length,
            end_value=args.lr_min,
        )
    if args.optimizer == "adamw":
        opt = optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        opt = optax.adam(learning_rate=schedule)
    elif args.optimizer == "sgd":
        opt = optax.sgd(learning_rate=schedule)
    elif args.optimizer == "sgd_nesterov":
        opt = optax.sgd(learning_rate=schedule, nesterov=True, momentum=0.9)
    elif args.optimizer == "sgd_momentum":
        opt = optax.sgd(learning_rate=schedule, momentum=0.9)
    else:
        raise NotImplementedError()
    opt_state = opt.init(params)

    # Training loop over epochs
    for i in tqdm(range(args.epochs)):
        if args.use_wandb:
            if args.compute_hessian > 0 and i % args.compute_hessian == 0:
                inputs, labels, masks = next(iter(trainloader))
                inputs, labels, masks = next(iter(trainloader))
                dir_sharpness = compute_directional_sharpness(params, model, inputs, labels, masks)
                inputs, labels, masks = inputs[:32], labels[:32], masks[:32]
                L = compute_L_hessian(params, model, inputs, labels, masks)
                metrics = {"L": jnp.max(L), "Directional sharpness": dir_sharpness}
            else:
                metrics = {}

        params, opt_state, epoch_metrics = train_epoch(params, opt_state, opt, model, trainloader)

        if args.use_wandb:
            metrics = {
                **metrics,
                **epoch_metrics,
            }

            # Metrics for adaptive learning rate optimizers
            ones = jax.tree_map(lambda x: jnp.ones_like(x), params)
            effective_lr, _ = opt.update(ones, opt_state, params)
            relu = lambda x: jnp.maximum(0.0, x)
            mean_lr = jax.tree_map(
                lambda x: jnp.power(10, jnp.mean(jnp.log10(relu(-x) + 1e-8))).item(), effective_lr
            )
            for k, v in mean_lr.items():
                metrics = {f"Mean LR/{k}": v, **metrics}
            min_lr = jax.tree_map(lambda x: jnp.min(-x), effective_lr)
            for k, v in min_lr.items():
                metrics = {f"Min LR/{k}": v, **metrics}
            std_lr = jax.tree_map(
                lambda x: jnp.power(10, jnp.std(jnp.log10(relu(-x) + 1e-8))).item(), effective_lr
            )
            for k, v in std_lr.items():
                metrics = {f"Std LR/{k}": v, **metrics}

            param_norm = jax.tree_map(lambda x: jnp.linalg.norm(x).item(), params)
            for k, v in param_norm.items():
                metrics = {f"Param norm/{k}": v, **metrics}

            wandb.log(metrics)

        wandb.run.summary["Training loss"] = epoch_metrics["Training loss"]

        if jnp.isnan(epoch_metrics["Training loss"]):
            print("[*] Training loss is NaN. Exiting...")
            break

    # Save the effective lr if needed
    # jnp.save(wandb.run.id + "_lr.npy", effective_lr)

    # Final computations: Hessian and validation accuracy
    if args.compute_hessian >= 0:
        t = time.time()
        inputs, labels, masks = next(iter(trainloader))
        dir_sharpness = compute_directional_sharpness(params, model, inputs, labels, masks)
        wandb.run.summary["Directional sharpness"] = dir_sharpness
        inputs, labels, masks = inputs[:32], labels[:32], masks[:32]
        L = compute_L_hessian(params, model, inputs, labels, masks)
        wandb.run.summary["L end of training"] = jnp.max(L)

        print(f"[*] Eigval calculation in {time.time() - t:.2f} seconds")

    val_loss = validate(params, model, testloader)
    wandb.run.summary["Validation loss"] = val_loss
