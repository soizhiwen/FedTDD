import os
import copy
import warnings
import argparse
from typing import Any

import numpy as np
import numpy.random as npr
import flwr as fl
from flwr.common import NDArrays, Scalar
from sklearn.metrics import mean_squared_error

from engine.solver import Trainer, cycle
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Data.build_dataloader import build_dataloader_fed, build_dataloader_cond_fed
from Utils.io_utils import instantiate_from_config
from Federated.utils import (
    load_data_partitions,
    write_csv,
    context_fid,
    cross_corr,
    discriminative,
    predictive,
)

warnings.filterwarnings("ignore")


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, config: dict[str, Any], args: argparse.Namespace) -> None:
        self.config = config
        self.args = args

        self.save_dir = args.save_dir
        self.client_id = args.client_id
        self.exist_feats = args.feats_groups[args.client_id]
        self.pub_feats = args.feats_groups[-1]
        self.pvt_feats = sorted(set(self.exist_feats) ^ set(self.pub_feats))

        # Get indices of public features in exist features
        self.pub_idxs = [self.exist_feats.index(f) for f in self.pub_feats]
        self.pvt_idxs = [self.exist_feats.index(f) for f in self.pvt_feats]

        # Load original data
        data, _ = load_data_partitions(
            path=config["dataloader"]["params"]["data_root"],
            num_clients=args.num_clients,
            partition_id=args.client_id,
            pub_ratio=args.pub_ratio,
            split_mode=args.split_mode,
        )
        data = data[..., self.exist_feats]

        # Shuffle data
        st0 = np.random.get_state()
        np.random.seed(args.client_id)
        self.window = config["dataloader"]["params"]["window"]
        data = data.reshape(-1, self.window, len(self.exist_feats))
        npr.shuffle(data)
        data = np.concatenate(data)
        np.random.set_state(st0)

        # Dataloader for exist dataset
        dl_info = build_dataloader_fed(
            config=config,
            name=args.client_id,
            data=data,
            proportion=args.proportion,
            missing_ratio=args.missing_ratio,
            seed=args.client_id,
            pvt_unmask_idxs=self.pvt_idxs,
            save_dir=args.save_dir,
        )
        self.filename = f"{dl_info['dataset'].name}_{dl_info['dataset'].window}"

        # Initialize model
        config["model"]["params"]["feature_size"] = len(self.exist_feats)
        model = instantiate_from_config(config["model"]).to(args.device)

        self.trainer = Trainer(
            config=config,
            args=args,
            model=model,
            t_model=model,
            dataloader=dl_info,
        )

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        server_round = config["server_round"]
        size_every = config["size_every"]
        metric_iterations = config["metric_iterations"]

        # Load train data
        union_data = np.load(f"{self.save_dir}/real/{self.filename}_train.npy")
        len_union_data = len(union_data)
        union_data = union_data.reshape(-1, len(self.exist_feats))

        # Impute the private features
        if self.pvt_idxs and server_round > 1:
            # Dataloader for private imputation
            pvt_impt_dl_info = build_dataloader_cond_fed(
                config=self.config,
                name=self.args.client_id,
                data=union_data,
                train_size=1.0,
                proportion=self.args.proportion,
                missing_ratio=self.args.missing_ratio,
                seed=self.args.client_id,
                pvt_unmask_idxs=self.pvt_idxs,
                save2npy=False,
                save_dir=self.save_dir,
            )

            # Impute the private imputation dataset
            pvt_impt_dl, pvt_impt_ds = (
                pvt_impt_dl_info["dataloader"],
                pvt_impt_dl_info["dataset"],
            )
            coef = self.config["dataloader"]["coefficient"]
            stepsize = self.config["dataloader"]["step_size"]
            sampling_steps = self.config["dataloader"]["sampling_steps"]

            # Impute using student model
            self.trainer.load()
            pvt_impt, _, pvt_impt_masks = self.trainer.restore(
                pvt_impt_dl,
                [pvt_impt_ds.window, pvt_impt_ds.var_num],
                coef,
                stepsize,
                sampling_steps,
            )

            # Unnormalize imputed data
            if pvt_impt_ds.auto_norm:
                pvt_impt = unnormalize_to_zero_to_one(pvt_impt)

            # Calculate MSE for private imputation
            real = np.load(f"{pvt_impt_ds.dir}/norm_{self.filename}_train.npy")
            pvt_impt_masks = pvt_impt_masks.astype(bool)[..., self.pvt_idxs]
            pvt_impt_mse = mean_squared_error(
                real[..., self.pvt_idxs][~pvt_impt_masks],
                pvt_impt[..., self.pvt_idxs][~pvt_impt_masks],
            )
            self.write_client_results({"pvt_impt_mse": pvt_impt_mse}, server_round)

            # Inverse transform imputed data
            pvt_impt = pvt_impt_ds.scaler.inverse_transform(
                pvt_impt.reshape(-1, pvt_impt_ds.var_num)
            ).reshape(pvt_impt.shape)

            # Save imputed data
            impt_dir = f"{self.save_dir}/imputation"
            os.makedirs(impt_dir, exist_ok=True)
            np.save(f"{impt_dir}/{self.filename}.npy", pvt_impt)

            # Replace all union cell with imputed data
            union_data[..., self.pvt_idxs] = np.concatenate(
                pvt_impt[..., self.pvt_idxs]
            )

        # Dataloader for union dataset
        union_dl_info = build_dataloader_fed(
            config=self.config,
            name=self.args.client_id,
            data=(
                union_data
                if server_round > 1
                else union_data[
                    : int(np.ceil(len_union_data * self.args.proportion)) * self.window
                ]
            ),
            train_size=1.0,
            proportion=self.args.proportion if server_round > 1 else 1.0,
            missing_ratio=self.args.missing_ratio,
            seed=self.args.client_id,
            pvt_unmask_idxs=self.pvt_idxs,
            save2npy=False,
            save_dir=self.save_dir,
        )

        # Train student model on union dataset
        self.trainer.train_num_steps = config["local_epochs"]
        self.trainer.dl = cycle(union_dl_info["dataloader"])
        if server_round > 1:
            self.trainer.load()
        train_loss = self.trainer.train()
        self.trainer.save()

        metrics = {"train_loss": float(train_loss)}
        if server_round == self.args.num_rounds:
            # Generate synthetic data
            union_ds = union_dl_info["dataset"]
            seq_len, feat_dim = union_ds.window, union_ds.var_num

            self.trainer.load()
            synth = self.trainer.sample(
                len(union_ds), size_every, shape=[seq_len, feat_dim]
            )

            if union_ds.auto_norm:
                synth = unnormalize_to_zero_to_one(synth)

            # Save synthetic data
            synth_dir = f"{self.save_dir}/synthetic"
            os.makedirs(synth_dir, exist_ok=True)
            np.save(f"{synth_dir}/norm_{self.filename}.npy", synth)

            unnorm_synth = union_ds.scaler.inverse_transform(
                synth.reshape(-1, feat_dim)
            ).reshape(synth.shape)
            np.save(f"{synth_dir}/{self.filename}.npy", unnorm_synth)

            # Compute metrics for all features
            real = np.load(f"{union_ds.dir}/norm_{self.filename}_test.npy")

            all_ctx_fid = context_fid(real, synth, metric_iterations)
            all_cross_corr = cross_corr(real, synth, metric_iterations)
            all_discriminative = discriminative(real, synth, metric_iterations)
            all_predictive = predictive(real, synth, metric_iterations)

            metrics["all_context_fid"] = float(all_ctx_fid[0])
            metrics["all_context_fid_sigma"] = float(all_ctx_fid[1])
            metrics["all_cross_corr"] = float(all_cross_corr[0])
            metrics["all_cross_corr_sigma"] = float(all_cross_corr[1])
            metrics["all_discriminative"] = float(all_discriminative[0])
            metrics["all_discriminative_sigma"] = float(all_discriminative[1])
            metrics["all_predictive"] = float(all_predictive[0])
            metrics["all_predictive_sigma"] = float(all_predictive[1])

        self.write_client_results(metrics, server_round)

        return [], 1, metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        return 0.0, 1, {}

    def write_client_results(
        self,
        results: dict[str, float],
        server_round: int,
    ) -> None:
        """Write client results to disk."""
        for k, v in results.items():
            cid = (
                f"{self.client_id} {self.exist_feats}"
                if self.exist_feats
                else self.client_id
            )
            fields = [server_round, v, cid]
            write_csv(fields, f"clients_{k}", self.save_dir)


def get_local_client_fn(config, args):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """
    config = copy.deepcopy(config)

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""
        args.client_id = int(cid)
        return FlowerClient(config, args).to_client()

    return client_fn
