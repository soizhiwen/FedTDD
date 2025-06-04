import os
import copy
import warnings
import argparse
from typing import Any
from collections import OrderedDict

import torch
import numpy as np
import numpy.random as npr
import flwr as fl
from flwr.common import NDArrays, Scalar
from sklearn.metrics import mean_squared_error

from engine.solver import Trainer, cycle
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Data.build_dataloader import build_dataloader_fed, build_dataloader_cond_fed
from Utils.metric_utils import visualization
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

        # Initialize teacher model
        config["model"]["params"]["feature_size"] = len(self.pub_feats)
        t_model = instantiate_from_config(config["model"]).to(args.device)
        self.t_model_params_len = len(t_model.state_dict().values())

        # Initialize model
        config["model"]["params"]["feature_size"] = len(self.exist_feats)
        model = instantiate_from_config(config["model"]).to(args.device)

        self.trainer = Trainer(
            config=config,
            args=args,
            model=model,
            t_model=t_model,
            dataloader=dl_info,
        )

    def set_parameters(self, parameters: NDArrays) -> None:
        model_params = parameters[: self.t_model_params_len]
        params_dict = zip(self.trainer.t_model.state_dict().keys(), model_params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.trainer.t_model.load_state_dict(state_dict, strict=True)

        ema_params = parameters[self.t_model_params_len :]
        params_dict = zip(self.trainer.t_ema.state_dict().keys(), ema_params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.trainer.t_ema.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        server_round = config["server_round"]
        size_every = config["size_every"]
        metric_iterations = config["metric_iterations"]

        # Set teacher model parameters
        self.set_parameters(parameters)

        # Dataloader for public imputation
        train_data = np.load(f"{self.save_dir}/real/{self.filename}_train.npy")
        train_data = train_data.reshape(-1, len(self.exist_feats))
        pub_impt_dl_info = build_dataloader_cond_fed(
            config=self.config,
            name=self.args.client_id,
            data=train_data,
            train_size=1.0,
            proportion=self.args.proportion,
            missing_ratio=self.args.missing_ratio,
            seed=self.args.client_id,
            pvt_unmask_idxs=self.pvt_idxs,
            target_idxs=self.pub_idxs,
            save2npy=False,
            save_dir=self.save_dir,
        )

        # Impute the public imputation dataset
        pub_impt_dl, pub_impt_ds = (
            pub_impt_dl_info["dataloader"],
            pub_impt_dl_info["dataset"],
        )
        coef = self.config["dataloader"]["coefficient"]
        stepsize = self.config["dataloader"]["step_size"]
        sampling_steps = self.config["dataloader"]["sampling_steps"]

        # Impute using teacher model
        pub_impt, _, pub_impt_masks = self.trainer.restore(
            pub_impt_dl,
            [pub_impt_ds.window, len(self.pub_feats)],
            coef,
            stepsize,
            sampling_steps,
            is_teacher=True,
        )

        # Unnormalize imputed data
        if pub_impt_ds.auto_norm:
            pub_impt = unnormalize_to_zero_to_one(pub_impt)

        # Calculate MSE for public imputation
        real = np.load(f"{pub_impt_ds.dir}/norm_{self.filename}_train.npy")
        pub_impt_masks = pub_impt_masks.astype(bool)
        pub_impt_mse = mean_squared_error(
            real[..., self.pub_idxs][~pub_impt_masks], pub_impt[~pub_impt_masks]
        )
        self.write_client_results({"pub_impt_mse": pub_impt_mse}, server_round)

        # Transform imputed data into zero filling data
        zeros_pub_impt = np.zeros(
            (len(pub_impt), pub_impt_ds.window, pub_impt_ds.var_num)
        )
        zeros_pub_impt[..., self.pub_idxs] = pub_impt

        # Inverse transform imputed data
        pub_impt = pub_impt_ds.scaler.inverse_transform(
            zeros_pub_impt.reshape(-1, pub_impt_ds.var_num)
        ).reshape(zeros_pub_impt.shape)
        pub_impt = pub_impt[..., self.pub_idxs]

        # Save imputed data
        impt_dir = f"{self.save_dir}/imputation"
        os.makedirs(impt_dir, exist_ok=True)
        np.save(f"{impt_dir}/{self.filename}.npy", pub_impt)

        # Replace public missing cells with imputed data
        union_data = train_data.copy()
        union_data[..., self.pub_idxs] = np.concatenate(pub_impt)

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
                pub_unmask_idxs=self.pub_idxs,
                pvt_unmask_idxs=self.pvt_idxs,
                save2npy=False,
                save_dir=self.save_dir,
            )

            # Impute the private imputation dataset
            pvt_impt_dl, pvt_impt_ds = (
                pvt_impt_dl_info["dataloader"],
                pvt_impt_dl_info["dataset"],
            )

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
            pvt_impt_masks = pvt_impt_masks.astype(bool)
            pvt_impt_mse = mean_squared_error(
                real[~pvt_impt_masks], pvt_impt[~pvt_impt_masks]
            )
            self.write_client_results({"pvt_impt_mse": pvt_impt_mse}, server_round)

            # Inverse transform imputed data
            pvt_impt = pvt_impt_ds.scaler.inverse_transform(
                pvt_impt.reshape(-1, pvt_impt_ds.var_num)
            ).reshape(pvt_impt.shape)

            # Save imputed data
            np.save(f"{impt_dir}/{self.filename}.npy", pvt_impt)

            # Replace all union cell with imputed data
            union_data = np.concatenate(pvt_impt)

        # Dataloader for union dataset
        union_dl_info = build_dataloader_fed(
            config=self.config,
            name=self.args.client_id,
            data=(
                union_data
                if server_round > 1
                else union_data[
                    : int(np.ceil(len(pub_impt) * self.args.proportion)) * self.window
                ]
            ),
            train_size=1.0,
            proportion=self.args.proportion if server_round > 1 else 1.0,
            missing_ratio=self.args.missing_ratio if server_round > 1 else 0.0,
            seed=self.args.client_id,
            pub_unmask_idxs=self.pub_idxs if server_round > 1 else None,
            pvt_unmask_idxs=self.pvt_idxs if server_round > 1 else None,
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

        # Aggregate datasets
        unnorm_synth = unnorm_synth[: len(union_ds)]
        unnorm_synth = unnorm_synth[..., self.pub_idxs]
        data_aggregated = [pub_impt, unnorm_synth]  # Unnorm 3D

        metrics = {"train_loss": float(train_loss)}

        if server_round == self.args.num_rounds:
            # Compute metrics for all features
            analysis_types = ("pca", "tsne", "kernel")
            real = np.load(f"{union_ds.dir}/norm_{self.filename}_test.npy")

            for analysis in analysis_types:
                try:
                    visualization(
                        ori_data=real,
                        generated_data=synth,
                        analysis=analysis,
                        compare=real.shape[0],
                        name=f"{union_ds.name}_all",
                        save_dir=self.save_dir,
                    )
                except Exception as e:
                    print(f"Error: {e}")

            try:
                all_ctx_fid = context_fid(real, synth, metric_iterations)
                metrics["all_context_fid"] = float(all_ctx_fid[0])
                metrics["all_context_fid_sigma"] = float(all_ctx_fid[1])
            except Exception as e:
                print(f"Error: {e}")

            try:
                all_cross_corr = cross_corr(real, synth, metric_iterations)
                metrics["all_cross_corr"] = float(all_cross_corr[0])
                metrics["all_cross_corr_sigma"] = float(all_cross_corr[1])
            except Exception as e:
                print(f"Error: {e}")

            try:
                all_discriminative = discriminative(real, synth, metric_iterations)
                metrics["all_discriminative"] = float(all_discriminative[0])
                metrics["all_discriminative_sigma"] = float(all_discriminative[1])
            except Exception as e:
                print(f"Error: {e}")

            try:
                all_predictive = predictive(real, synth, metric_iterations)
                metrics["all_predictive"] = float(all_predictive[0])
                metrics["all_predictive_sigma"] = float(all_predictive[1])
            except Exception as e:
                print(f"Error: {e}")

        self.write_client_results(metrics, server_round)

        return data_aggregated, 1, metrics

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


def get_client_fn(config, args):
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
