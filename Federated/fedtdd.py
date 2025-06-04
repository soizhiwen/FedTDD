from typing import Callable, Optional, Union

import numpy as np
import numpy.random as npr
from flwr.common import (
    EvaluateRes,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

from engine.solver import Trainer, cycle
from Federated.utils import load_data_partitions
from Data.build_dataloader import build_dataloader_fed
from Utils.io_utils import instantiate_from_config


class FedTDD(FedAvg):
    def __init__(
        self,
        config,
        args,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        args.client_id = "server"
        self.args = args
        self.config = config
        data_root = config["dataloader"]["params"]["data_root"]

        # Initialize public dataset
        self.pub_data, self.total_seq = load_data_partitions(
            path=data_root,
            num_clients=args.num_clients,
            pub_ratio=args.pub_ratio,
            split_mode=args.split_mode,
        )
        self.pub_data = self.pub_data[..., args.feats_groups[-1]]

        # Dataloader for public dataset
        self.pub_dl_info = build_dataloader_fed(
            config=config,
            name=args.client_id,
            data=self.pub_data,
            train_size=1.0,
            proportion=1.0,
            missing_ratio=0.0,
            seed=args.seed,
            save2npy=False,
            save_dir=args.save_dir,
        )

        # Initialize teacher model
        config["model"]["params"]["feature_size"] = self.pub_data.shape[-1]
        t_model = instantiate_from_config(config["model"]).to(args.device)

        self.trainer = Trainer(
            config=config,
            args=args,
            model=t_model,
            t_model=t_model,
            dataloader=self.pub_dl_info,
        )

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedTDD(accept_failures={self.accept_failures})"
        return rep

    def get_t_parameters(self):
        t_model_params = [
            val.cpu().numpy() for _, val in self.trainer.t_model.state_dict().items()
        ]
        t_ema_params = [
            val.cpu().numpy() for _, val in self.trainer.t_ema.state_dict().items()
        ]
        return t_model_params + t_ema_params

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Pretrain teacher model on public dataset."""
        if self.args.centralized or self.args.local:
            return ndarrays_to_parameters([])

        self.trainer.dl = cycle(self.pub_dl_info["dataloader"])
        self.trainer.train(is_teacher=True)
        self.trainer.save()

        # Get teacher model parameters
        t_params = self.get_t_parameters()
        t_params = ndarrays_to_parameters(t_params)
        return t_params

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[int, dict[str, Scalar]]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.args.centralized or self.args.local:
            pass
        elif self.args.freeze:
            self.trainer.load()
        else:
            # Aggregate datasets
            # impt_aggregated = [
            #     parameters_to_ndarrays(res.parameters)[0] for _, res in results
            # ]
            synth_aggregated = [
                parameters_to_ndarrays(res.parameters)[1] for _, res in results
            ]
            # impt_aggregated = np.vstack(impt_aggregated)  # Unnorm 3D
            synth_aggregated = np.vstack(synth_aggregated)  # Unnorm 3D

            pub_ds = self.pub_dl_info["dataset"]
            pub_aggregated = self.pub_data.reshape(-1, pub_ds.window, pub_ds.var_num)

            # Select a random subset of each aggregated data
            total = self.total_seq // 10
            total_pub = len(pub_aggregated)
            # total_impt = len(impt_aggregated)
            total_synth = len(synth_aggregated)

            # PIS 1 : 0 to 1 : 0 to 1
            # num_pub = np.min([total, total_pub, total_impt, total_synth])
            num_pub = np.min([total, total_pub, total_synth])
            # num_impt = num_pub * (server_round - 1) / (self.args.num_rounds - 1)
            # num_impt = int(np.ceil(num_impt * self.args.pis_alpha))
            # num_synth = num_impt
            num_synth = num_pub * (server_round - 1) / (self.args.num_rounds - 1)
            num_synth = int(np.ceil(num_synth * self.args.pis_alpha))

            # Set seed for reproducibility
            st0 = npr.get_state()
            npr.seed(server_round)

            data_aggregated = []
            if num_pub > 0:
                pub_idxs = npr.choice(total_pub, num_pub, replace=False)
                data_aggregated.append(np.vstack(pub_aggregated[pub_idxs]))

            # if num_impt > 0:
            #     impt_idxs = npr.choice(total_impt, num_impt, replace=False)
            #     data_aggregated.append(np.vstack(impt_aggregated[impt_idxs]))

            if num_synth > 0:
                synth_idxs = npr.choice(total_synth, num_synth, replace=False)
                data_aggregated.append(np.vstack(synth_aggregated[synth_idxs]))

            npr.set_state(st0)
            data_aggregated = np.vstack(data_aggregated)  # Unnorm 2D

            # Build dataloader for aggregated dataset
            agg_dl_info = build_dataloader_fed(
                config=self.config,
                name=self.args.client_id,
                data=data_aggregated,
                train_size=1.0,
                proportion=1.0,
                missing_ratio=0.0,
                seed=self.args.seed,
                save2npy=False,
                save_dir=self.args.save_dir,
            )

            # Fine-tune teacher model
            # max_epochs = num_pub + num_impt + num_synth
            max_epochs = num_pub + num_synth
            self.trainer.train_num_steps = max_epochs
            self.trainer.dl = cycle(agg_dl_info["dataloader"])
            lr_params = {
                "start_lr": 1.0e-8,
                "patience": max_epochs,
                "warmup_lr": 8.0e-5,
                "warmup": (max_epochs // 20) + 1,
            }
            self.trainer.load(lr_params=lr_params)
            self.trainer.train(is_teacher=True)
            self.trainer.save()

        # Get teacher model parameters
        t_params = self.get_t_parameters()
        t_params = ndarrays_to_parameters(t_params)

        metrics_aggregated = {}
        for client, fit_res in results:
            metrics_aggregated[int(client.cid)] = fit_res.metrics

        return t_params, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[dict[int, Optional[float]], dict[int, dict[str, Scalar]]]:
        """Aggregate evaluation losses using weighted average."""
        return None, {}
