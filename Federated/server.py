import copy

from Federated import FedTDD


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,
        "local_epochs": 7500 if server_round < 2 else 5000,
        "size_every": 2001,
        "metric_iterations": 5,
    }
    return config


def get_fedtdd_fn(config, args):
    config = copy.deepcopy(config)
    num_clients = 1 if args.centralized else args.num_clients

    return FedTDD(
        config,
        args,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=fit_config,
    )
