# Stock
python label.py --path ./Data/datasets/stock_data.csv --num_clusters 5

# ETTh
python label.py --path ./Data/datasets/ETTh.csv --num_clusters 6 --drop_first

# Energy
python label.py --path ./Data/datasets/energy_data.csv --num_clusters 6

# fMRI
python label.py --path ./Data/datasets/fMRI/sim4.mat --num_clusters 10

# MuJoCo
python label.py --path ./Data/datasets/mujoco.mat --num_clusters 10