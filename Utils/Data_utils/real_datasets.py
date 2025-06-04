import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import (
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)
from Utils.masking_utils import noise_mask


class CustomDataset(Dataset):
    def __init__(
        self,
        name,
        data_root=None,
        dataset=None,
        window=64,
        train_size=0.8,
        proportion=0.8,
        save2npy=True,
        neg_one_to_one=True,
        seed=123,
        output_dir="./OUTPUT",
        target_idxs=None,
        pub_unmask_idxs=None,
        pvt_unmask_idxs=None,
        predict_length=None,
        missing_ratio=None,
        style="separate",
        distribution="geometric",
        mean_mask_length=3,
    ):
        super(CustomDataset, self).__init__()
        self.name, self.pred_len, self.missing_ratio = (
            name,
            predict_length,
            missing_ratio,
        )
        self.style, self.distribution, self.mean_mask_length = (
            style,
            distribution,
            mean_mask_length,
        )
        self.data_root, self.dataset = data_root, dataset
        if self.dataset is not None:
            self.rawdata, self.scaler = self.fit_data(self.dataset)
        elif self.data_root is not None:
            self.rawdata, self.scaler = self.read_data(self.data_root, self.name)
        else:
            raise NotImplementedError()
        self.dir = os.path.join(output_dir, "real")
        os.makedirs(self.dir, exist_ok=True)

        self.window = window
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.sample_num_total = max(self.len - self.window + 1, 0)
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.data = self.__normalize(self.rawdata)
        train_1, train_2, test = self.__getsamples(self.data, train_size, proportion)

        self.samples, self.masking = [], []
        if len(train_1) > 0:
            self.samples.append(train_1)
            unmask_idxs = []
            if pub_unmask_idxs is not None:
                unmask_idxs += pub_unmask_idxs
            if pvt_unmask_idxs is not None:
                unmask_idxs += pvt_unmask_idxs
            self.masking.append(self.mask_row_data(train_1, sorted(unmask_idxs), seed))

        if len(train_2) > 0:
            self.samples.append(train_2)
            self.masking.append(self.mask_row_data(train_2, pub_unmask_idxs, seed))

        self.samples = np.concatenate(self.samples)
        self.masking = np.concatenate(self.masking)
        self.save(self.samples, test, self.masking)

        if target_idxs is not None:
            self.samples = self.samples[..., target_idxs]
            self.masking = self.masking[..., target_idxs]

        self.samples *= self.masking
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, train_size, proportion):
        if self.dataset is not None:
            x = data.reshape(-1, self.window, self.var_num)
        elif self.data_root is not None:
            x = np.zeros((self.sample_num_total, self.window, self.var_num))
            for i in range(self.sample_num_total):
                start = i
                end = i + self.window
                x[i, :, :] = data[start:end, :]
        else:
            raise NotImplementedError()

        train_data, test_data = self.divide(x, train_size)
        train_data_1, train_data_2 = self.divide(train_data, proportion)
        return train_data_1, train_data_2, test_data

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)

    @staticmethod
    def divide(data, ratio):
        size = data.shape[0]
        regular_train_num = int(np.ceil(size * ratio))
        id_rdm = np.arange(size)

        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=""):
        """Reads a single .csv"""
        df = pd.read_csv(filepath, header=0)
        if name == "etth":
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler

    @staticmethod
    def fit_data(data):
        """Transform data to MinMaxScaler"""
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler

    def save(self, train, test, mask):
        if self.save2npy:
            # Save samples
            np.save(
                os.path.join(self.dir, f"{self.name}_{self.window}_train.npy"),
                self.unnormalize(train),
            )
            np.save(
                os.path.join(self.dir, f"{self.name}_{self.window}_test.npy"),
                self.unnormalize(test),
            )
            if self.auto_norm:
                np.save(
                    os.path.join(self.dir, f"norm_{self.name}_{self.window}_train.npy"),
                    unnormalize_to_zero_to_one(train),
                )
                np.save(
                    os.path.join(self.dir, f"norm_{self.name}_{self.window}_test.npy"),
                    unnormalize_to_zero_to_one(test),
                )
            else:
                np.save(
                    os.path.join(self.dir, f"norm_{self.name}_{self.window}_train.npy"),
                    train,
                )
                np.save(
                    os.path.join(self.dir, f"norm_{self.name}_{self.window}_test.npy"),
                    test,
                )

            # Save masking
            np.save(
                os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"),
                mask,
            )

    def mask_row_data(self, data, unmask_idxs=None, seed=2023):
        masks = np.ones_like(data)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(data.shape[0]):
            x = data[idx, :, :]  # (seq_length, feat_dim) array
            mask = np.random.choice(
                np.array([True, False]),
                size=x.shape[0],
                replace=True,
                p=(1 - self.missing_ratio, self.missing_ratio),
            )  # (seq_length) boolean array
            mask = np.repeat(mask, x.shape[1]).reshape(x.shape)
            if unmask_idxs:
                mask[..., unmask_idxs] = True
            masks[idx, :, :] = mask

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(
                x,
                self.missing_ratio,
                self.mean_mask_length,
                self.style,
                self.distribution,
            )  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(
                os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks
            )

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        x = self.samples[ind]  # (seq_length, feat_dim) array
        m = self.masking[ind]  # (seq_length, feat_dim) boolean array
        return torch.from_numpy(x).float(), torch.from_numpy(m)

    def __len__(self):
        return self.sample_num


class fMRIDataset(CustomDataset):
    def __init__(self, proportion=1.0, **kwargs):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=""):
        """Reads a single .csv"""
        data = io.loadmat(filepath + "/sim4.mat")["ts"]
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
