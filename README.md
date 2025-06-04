# [ECML PKDD 2025] FedTDD
This is the repository for Federated Time Series Generation on Feature and Temporally Misaligned Data. The backbone model of FedTDD is based on [Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS).

## Getting Started
Follow the steps below to set up the environment.

```shell
cd FedTDD/
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Label the data.

```shell
bash run_label.sh
```

Run all experiments.

```shell
bash run.sh
```