import copy
import torch
from Utils.io_utils import instantiate_from_config


def build_dataloader_fed(
    config,
    name,
    data,
    proportion,
    missing_ratio,
    seed,
    train_size=None,
    pub_unmask_idxs=None,
    pvt_unmask_idxs=None,
    save2npy=True,
    save_dir="",
):
    config = copy.deepcopy(config)
    batch_size = config["dataloader"]["batch_size"]
    jud = config["dataloader"]["shuffle"]

    config["dataloader"]["params"]["name"] += f"_{name}"
    config["dataloader"]["params"]["dataset"] = data
    config["dataloader"]["params"]["proportion"] = proportion
    config["dataloader"]["params"]["missing_ratio"] = missing_ratio
    config["dataloader"]["params"]["seed"] = seed
    config["dataloader"]["params"]["pub_unmask_idxs"] = pub_unmask_idxs
    config["dataloader"]["params"]["pvt_unmask_idxs"] = pvt_unmask_idxs
    config["dataloader"]["params"]["save2npy"] = save2npy
    config["dataloader"]["params"]["output_dir"] = save_dir
    if train_size is not None:
        config["dataloader"]["params"]["train_size"] = train_size

    dataset = instantiate_from_config(config["dataloader"])

    if len(dataset) < batch_size:
        batch_size = len(dataset) // config["solver"]["gradient_accumulate_every"]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=jud,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        drop_last=jud,
    )

    dataload_info = {"dataloader": dataloader, "dataset": dataset}

    return dataload_info


def build_dataloader_cond_fed(
    config,
    name,
    data,
    proportion,
    missing_ratio,
    seed,
    train_size=None,
    pub_unmask_idxs=None,
    pvt_unmask_idxs=None,
    target_idxs=None,
    save2npy=True,
    save_dir="",
):
    config = copy.deepcopy(config)
    batch_size = config["dataloader"]["sample_size"]

    config["dataloader"]["params"]["name"] += f"_{name}"
    config["dataloader"]["params"]["dataset"] = data
    config["dataloader"]["params"]["proportion"] = proportion
    config["dataloader"]["params"]["missing_ratio"] = missing_ratio
    config["dataloader"]["params"]["seed"] = seed
    config["dataloader"]["params"]["pub_unmask_idxs"] = pub_unmask_idxs
    config["dataloader"]["params"]["pvt_unmask_idxs"] = pvt_unmask_idxs
    config["dataloader"]["params"]["target_idxs"] = target_idxs
    config["dataloader"]["params"]["save2npy"] = save2npy
    config["dataloader"]["params"]["output_dir"] = save_dir
    if train_size is not None:
        config["dataloader"]["params"]["train_size"] = train_size

    dataset = instantiate_from_config(config["dataloader"])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )

    dataload_info = {"dataloader": dataloader, "dataset": dataset}

    return dataload_info


def build_dataloader_only_fed(config, dataset):
    config = copy.deepcopy(config)
    batch_size = config["dataloader"]["batch_size"]
    jud = config["dataloader"]["shuffle"]

    if len(dataset) < batch_size:
        batch_size = len(dataset) // config["solver"]["gradient_accumulate_every"]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=jud,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        drop_last=jud,
    )

    return dataloader
