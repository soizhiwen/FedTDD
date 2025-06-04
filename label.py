import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def label_data(
    path,
    window_size=24,
    n_clusters=5,
    step_size=1,
    drop_first=False,
    seed=42,
):
    if path.suffix == ".mat":
        data = io.loadmat(path)["ts"]
    elif path.suffix == ".csv":
        data = pd.read_csv(path)
        if drop_first:
            data.drop(data.columns[0], axis=1, inplace=True)
        data = data.to_numpy()
    else:
        raise NotImplementedError("Only .mat and .csv files are supported")

    if len(data.shape) == 2:
        subsets = [
            data[i : i + window_size]
            for i in range(0, len(data) - window_size + 1, step_size)
        ]
    elif len(data.shape) == 3:
        subsets = data
    else:
        raise NotImplementedError("Only 2D and 3D data are supported")

    flatten_subsets = [np.array(subset).flatten() for subset in subsets]
    flatten_subsets = MinMaxScaler().fit_transform(flatten_subsets)
    kmeans_models = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
    ).fit(flatten_subsets)
    labels = kmeans_models.labels_
    print(np.unique(labels, return_counts=True))
    labeled_dataset = [[subsets[i], labels[i]] for i in range(len(subsets))]
    return labeled_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Label data")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path of data file",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=5,
        help="stock_data=5, energy_data=6",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=24,
        help="Window size for labeling data",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
        help="Step size for labeling data",
    )
    parser.add_argument(
        "--drop_first",
        action="store_true",
        default=False,
        help="Drop the first column of the dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = Path(args.path)
    dataset = label_data(
        path=path,
        window_size=args.window_size,
        n_clusters=args.num_clusters,
        step_size=args.step_size,
        drop_first=args.drop_first,
        seed=args.seed,
    )
    dataset = np.array(dataset, dtype=object)
    np.save(f"{path.parent}/labeled_{path.stem}.npy", dataset)
