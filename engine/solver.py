import os
import copy
import time
import torch
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, t_model, dataloader, logger=None):
        super().__init__()
        self.model = model
        self.t_model = t_model
        self.device = self.model.betas.device
        self.train_num_steps = config["solver"]["max_epochs"]
        self.gradient_accumulate_every = config["solver"]["gradient_accumulate_every"]
        self.save_cycle = config["solver"]["save_cycle"]
        self.dl = cycle(dataloader["dataloader"])
        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger

        self.results_folder = Path(
            f"{self.args.save_dir}/{config['solver']['results_folder']}"
        )
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config["solver"].get("base_lr", 1.0e-4)
        ema_decay = config["solver"]["ema"]["decay"]
        ema_update_every = config["solver"]["ema"]["update_interval"]

        self.opt = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=start_lr,
            betas=[0.9, 0.96],
        )
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(
            self.device
        )

        self.t_opt = Adam(
            filter(lambda p: p.requires_grad, self.t_model.parameters()),
            lr=start_lr,
            betas=[0.9, 0.96],
        )
        self.t_ema = EMA(
            self.t_model, beta=ema_decay, update_every=ema_update_every
        ).to(self.device)

        self.scheduler = config["solver"]["scheduler"]
        sc_cfg = self.scheduler
        sc_cfg["params"]["optimizer"] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        t_sc_cfg = self.scheduler
        t_sc_cfg["params"]["optimizer"] = self.t_opt
        self.t_sch = instantiate_from_config(t_sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, verbose=False):
        save_dir = f"{self.results_folder}/checkpoint_{self.args.client_id}.pt"
        if self.logger is not None and verbose:
            self.logger.log_info(f"Save current model to {save_dir}")
        data = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "t_model": self.t_model.state_dict(),
            "t_ema": self.t_ema.state_dict(),
        }
        torch.save(data, save_dir)

    def load(self, lr_params=None, verbose=False):
        save_dir = f"{self.results_folder}/checkpoint_{self.args.client_id}.pt"
        if self.logger is not None and verbose:
            self.logger.log_info(f"Resume from {save_dir}")
        device = self.device
        data = torch.load(save_dir, map_location=device)
        self.model.load_state_dict(data["model"])
        self.ema.load_state_dict(data["ema"])
        self.t_model.load_state_dict(data["t_model"])
        self.t_ema.load_state_dict(data["t_ema"])

        if lr_params:
            self.t_opt = Adam(
                filter(lambda p: p.requires_grad, self.t_model.parameters()),
                lr=lr_params["start_lr"],
                betas=[0.9, 0.96],
            )
            t_sc_cfg = copy.deepcopy(self.scheduler)
            t_sc_cfg["params"]["patience"] = lr_params["patience"]
            t_sc_cfg["params"]["min_lr"] = lr_params["start_lr"]
            t_sc_cfg["params"]["warmup_lr"] = lr_params["warmup_lr"]
            t_sc_cfg["params"]["warmup"] = lr_params["warmup"]
            t_sc_cfg["params"]["optimizer"] = self.t_opt
            self.t_sch = instantiate_from_config(t_sc_cfg)

    def train(self, is_teacher=False):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info(
                "{}: start training...".format(self.args.name), check_primary=False
            )

        total_losses = 0.0
        with tqdm(initial=step, total=self.train_num_steps, disable=True) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.0
                for _ in range(self.gradient_accumulate_every):
                    data, mask = next(self.dl)
                    data = data.to(device)
                    if is_teacher:
                        loss = self.t_model(data, target=data)
                    else:
                        mask = mask.to(device)
                        loss = self.model(data, target=data, mask=mask)

                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                total_losses += total_loss
                pbar.set_description(f"loss: {total_loss:.6f}")

                if is_teacher:
                    clip_grad_norm_(self.t_model.parameters(), 1.0)
                    self.t_opt.step()
                    self.t_sch.step(total_loss)
                    self.t_opt.zero_grad()
                    self.t_ema.update()
                else:
                    clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    self.sch.step(total_loss)
                    self.opt.zero_grad()
                    self.ema.update()
                self.step += 1
                step += 1

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                        # self.logger.log_info('saved in {}'.format(str(self.results_folder / f'checkpoint-{self.milestone}.pt')))

                    if self.logger is not None and self.step % self.log_frequency == 0:
                        # info = '{}: train'.format(self.args.name)
                        # info = info + ': Epoch {}/{}'.format(self.step, self.train_num_steps)
                        # info += ' ||'
                        # info += '' if loss_f == 'none' else ' Fourier Loss: {:.4f}'.format(loss_f.item())
                        # info += '' if loss_r == 'none' else ' Reglarization: {:.4f}'.format(loss_r.item())
                        # info += ' | Total Loss: {:.6f}'.format(total_loss)
                        # self.logger.log_info(info)
                        self.logger.add_scalar(
                            tag="train/loss",
                            scalar_value=total_loss,
                            global_step=self.step,
                        )

                pbar.update(1)

        print("training complete")
        if self.logger is not None:
            self.logger.log_info(
                "Training done, time: {:.2f}".format(time.time() - tic)
            )

        return total_losses / self.train_num_steps

    def sample(self, num, size_every, shape=None, is_teacher=False):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info("Begin to sample...")
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1
        ema = self.t_ema if is_teacher else self.ema

        for _ in range(num_cycle):
            sample = ema.ema_model.generate_mts(batch_size=size_every)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info(
                "Sampling done, time: {:.2f}".format(time.time() - tic)
            )
        return samples

    def restore(
        self,
        raw_dataloader,
        shape=None,
        coef=1e-1,
        stepsize=1e-1,
        sampling_steps=50,
        is_teacher=False,
    ):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info("Begin to restore...")
        model_kwargs = {}
        model_kwargs["coef"] = coef
        model_kwargs["learning_rate"] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])
        model = self.t_model if is_teacher else self.model
        ema = self.t_ema if is_teacher else self.ema

        for idx, (x, t_m) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)
            if sampling_steps == model.num_timesteps:
                sample = ema.ema_model.sample_infill(
                    shape=x.shape,
                    target=x * t_m,
                    partial_mask=t_m,
                    model_kwargs=model_kwargs,
                )
            else:
                sample = ema.ema_model.fast_sample_infill(
                    shape=x.shape,
                    target=x * t_m,
                    partial_mask=t_m,
                    model_kwargs=model_kwargs,
                    sampling_timesteps=sampling_steps,
                )

            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])

        if self.logger is not None:
            self.logger.log_info(
                "Imputation done, time: {:.2f}".format(time.time() - tic)
            )
        return samples, reals, masks
        # return samples
