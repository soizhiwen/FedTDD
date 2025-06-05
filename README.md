# [ECML PKDD 2025] FedTDD: Federated Time Series Generation on Feature and Temporally Misaligned Data

> **Abstact:**
> Distributed time series data presents a challenge for federated learning, as clients often possess different feature sets and have misaligned time steps. Existing federated time series models are limited by the assumption of perfect temporal or feature alignment across clients. In this paper, we propose FedTDD, a novel federated time series diffusion model that jointly learns a synthesizer across clients. At the core of FedTDD is a novel data distillation and aggregation framework that reconciles the differences between clients by imputing the misaligned timesteps and features. In contrast to traditional federated learning, FedTDD learns the correlation across clients' time series through the exchange of local synthetic outputs instead of model parameters. A coordinator iteratively improves a global distiller network by leveraging shared knowledge from clients through the exchange of synthetic data. As the distiller becomes more refined over time, it subsequently enhances the quality of the clients' local feature estimates, allowing each client to then improve its local imputations for missing data using the latest, more accurate distiller. Experimental results on five datasets demonstrate FedTDD's effectiveness compared to centralized training, and the effectiveness of sharing synthetic outputs to transfer knowledge of local time series. Notably, FedTDD achieves 79.4% and 62.8% improvement over local training in Context-FID and Correlational scores.

This is the repository for [Federated Time Series Generation on Feature and Temporally Misaligned Data](https://arxiv.org/abs/2410.21072). The backbone model of FedTDD is based on [Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS).

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
