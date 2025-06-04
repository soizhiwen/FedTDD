import os
import json
import warnings
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import torch
import flwr as fl

from Federated.server import get_fedtdd_fn
from Federated.client import get_client_fn
from Federated.centralized_client import get_centralized_client_fn
from Federated.local_client import get_local_client_fn
from Federated.utils import partition_features, plot_metrics, increment_path
from Utils.io_utils import load_yaml_config, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=1,
        help="Specifies the artificial data partition.",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=1,
        help="Specifies number of global rounds.",
    )
    parser.add_argument(
        "--pis_alpha",
        type=float,
        default=1.0,
        help="Specifies the alpha for PIS.",
    )
    parser.add_argument(
        "--centralized",
        action="store_true",
        default=False,
        help="Run centralized training.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Run local training.",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        default=False,
        help="Fine-tune or not.",
    )
    parser.add_argument(
        "--cudnn_deterministic",
        action="store_true",
        default=False,
        help="set cudnn.deterministic True",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path of config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./OUTPUT",
        help="Directory to save the results",
    )
    parser.add_argument(
        "--num_pub_feats",
        type=int,
        default=1,
        help="Specifies the number of public features.",
    )
    parser.add_argument(
        "--pub_ratio",
        type=float,
        default=0.5,
        help="Ratio of public dataset to the total dataset.",
    )
    parser.add_argument(
        "--missing_ratio",
        type=float,
        default=0.0,
        help="Ratio of Missing Values.",
    )
    parser.add_argument(
        "--proportion",
        type=float,
        default=0.8,
        help="Proportion of client data split.",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="iid_even",
        choices=["iid_even", "iid_random", "non_iid_order", "non_iid_unorder"],
        help="Type of data partitioning",
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=2,
        help="Number of CPUs to assign to a virtual client",
    )
    parser.add_argument(
        "--num_gpus",
        type=float,
        default=0.5,
        help="Ratio of GPU memory to assign to a virtual client",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed for initializing training.",
    )

    args = parser.parse_args()
    args.save_dir = increment_path(f"{args.output}/{args.name}", sep="_", mkdir=True)

    with open(f"{args.save_dir}/args.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    return args


def main():
    args = parse_args()
    config = load_yaml_config(args.config_file)

    # Generate clients and public features groups
    args.feats_groups = partition_features(
        config["model"]["params"]["feature_size"],
        args.num_clients,
        args.num_pub_feats,
        args.save_dir,
    )
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        seed_everything(args.seed, args.cudnn_deterministic)

    strategy = get_fedtdd_fn(config, args)

    if args.centralized:
        client_fn = get_centralized_client_fn(config, args)
    elif args.local:
        client_fn = get_local_client_fn(config, args)
    else:
        client_fn = get_client_fn(config, args)

    client_resources = {
        "num_cpus": args.num_cpus,
        "num_gpus": args.num_gpus,
    }

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=1 if args.centralized else args.num_clients,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    plot_metrics(history, args.save_dir)


if __name__ == "__main__":
    main()
