import argparse
from source.train import train
from source.dataloading import Datasets
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_wandb", type=str2bool, default=True, help="log with wandb?")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="Identification-linear-systems-with-RNNs",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        help="wandb entity name, e.g. username",
    )
    parser.add_argument(
        "--dir_name",
        type=str,
        default="./cache_dir",
        help="name of directory where data is cached",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset", type=str, choices=Datasets.keys(), default="linear-system-identification"
    )
    parser.add_argument("--d_input", type=int, default=1)
    parser.add_argument("--d_output", type=int, default=1)
    parser.add_argument("--jax_seed", type=int, default=1919, help="seed randomness")
    # Parameters specific to linear system identification (lsi)
    parser.add_argument("--force_lsi_min_nu_model", type=str2bool, default=False)
    parser.add_argument("--lsi_parametrization", type=str, default="dense")
    parser.add_argument("--lsi_d_hidden", type=int, default=10)
    parser.add_argument("--lsi_T", type=int, default=100)
    parser.add_argument("--lsi_input_type", type=str, default="gaussian")
    parser.add_argument("--lsi_input_mean", type=float, default=0.0)
    parser.add_argument("--lsi_input_std", type=float, default=1.0)
    parser.add_argument("--lsi_noise_std", type=float, default=0.0)
    parser.add_argument("--lsi_min_nu", type=float, default=0.0)
    parser.add_argument("--lsi_max_nu", type=float, default=1.0)
    parser.add_argument("--lsi_max_phase", type=float, default=3.14)
    parser.add_argument("--lsi_use_B_C_D", type=str2bool, default=True)
    parser.add_argument("--lsi_normalization", type=str, default="none")

    # Model Parameters
    parser.add_argument("--model", type=str, default="LRU", help="Model to use")
    parser.add_argument("--d_hidden", type=int, default=30, help="Latent size of recurent unit")
    parser.add_argument("--min_nu", type=float, default=0.0, help="|lambda_min|")
    parser.add_argument("--max_nu", type=float, default=1.0, help="|lambda_max|")
    parser.add_argument("--max_phase", type=float, default=3.14, help="Maximum phase")
    parser.add_argument("--lru_param", type=str, default="exp")
    parser.add_argument("--lru_which_gamma", type=str, default="learned")
    parser.add_argument("--s4_shared_Delta", type=str2bool, default=True)
    parser.add_argument("--s4_init_scheme", type=str, default="default")
    parser.add_argument("--rnn_n_heads", type=int, default=1, help="Number of heads for RNNs")
    parser.add_argument(
        "--rnn_force_complex", type=str2bool, default=False, help="Complex behavior for RNNs"
    )
    parser.add_argument("--use_B_C_D", type=str2bool, default=True, help="Use B, C, D")

    # Optimization Parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--n_batches_per_epoch", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100, help="Max number of epochs")
    parser.add_argument("--lr_base", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--lr_min", type=float, default=1e-7, help="Minimum learning rate")
    parser.add_argument("--lr_schedule", type=str, default="constant", help="Lr schedule")
    parser.add_argument("--warmup_end", type=int, default=0, help="When to end linear warmup")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay value")
    parser.add_argument("--compute_hessian", type=int, default=-1, help="Freq to compute hessian")

    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    train(parser.parse_args())
