#!/bin/bash

# Hyperparameters
CONFIGS=("stocks" "etth" "mujoco" "energy" "fmri")
NUM_CLIENTS=(10)
PUB_RATIOS=(0.5)
MISSING_RATIOS=(0.5)
PROPORTIONS=(0.5)
SPLIT_MODES=("iid_random")
PIS_ALPHAS=(0.1 0.5 1.0)

# Assign number of public features to each config
declare -A NUM_PUB_FEATS
NUM_PUB_FEATS["stocks"]=3
NUM_PUB_FEATS["etth"]=4
NUM_PUB_FEATS["mujoco"]=7
NUM_PUB_FEATS["energy"]=14
NUM_PUB_FEATS["fmri"]=25

# Assign number of CPUs to each client
declare -A NUM_CPUS
NUM_CPUS[3]=8
NUM_CPUS[10]=5
NUM_CPUS["centralized"]=24

# Assign number of GPUs to each client
declare -A NUM_GPUS
NUM_GPUS[3]=0.3
NUM_GPUS[10]=0.18
NUM_GPUS["centralized"]=0.9

for CONFIG in "${CONFIGS[@]}"; do
    for N_CLIENTS in "${NUM_CLIENTS[@]}"; do
        for PUB_RATIO in "${PUB_RATIOS[@]}"; do
            for MISSING_RATIO in "${MISSING_RATIOS[@]}"; do
                for PROPORTION in "${PROPORTIONS[@]}"; do
                    for SPLIT_MODE in "${SPLIT_MODES[@]}"; do

                        # Run Centralized
                        ./run_sim.sh --name "${CONFIG}_NC_${N_CLIENTS}_${SPLIT_MODE}_PR_${PUB_RATIO}_MR_${MISSING_RATIO}_PROP_${PROPORTION}_centralized" \
                            --config $CONFIG --num_clients $N_CLIENTS --split_mode $SPLIT_MODE --num_pub_feats ${NUM_PUB_FEATS[$CONFIG]} \
                            --pub_ratio $PUB_RATIO --missing_ratio $MISSING_RATIO --proportion $PROPORTION --centralized \
                            --num_cpus ${NUM_CPUS["centralized"]} --num_gpus ${NUM_GPUS["centralized"]}

                        # Run Local
                        ./run_sim.sh --name "${CONFIG}_NC_${N_CLIENTS}_${SPLIT_MODE}_PR_${PUB_RATIO}_MR_${MISSING_RATIO}_PROP_${PROPORTION}_local" \
                            --config $CONFIG --num_clients $N_CLIENTS --split_mode $SPLIT_MODE --num_pub_feats ${NUM_PUB_FEATS[$CONFIG]} \
                            --pub_ratio $PUB_RATIO --missing_ratio $MISSING_RATIO --proportion $PROPORTION --local \
                            --num_cpus ${NUM_CPUS[$N_CLIENTS]} --num_gpus ${NUM_GPUS[$N_CLIENTS]}

                        # Run Pre-trained
                        ./run_sim.sh --name "${CONFIG}_NC_${N_CLIENTS}_${SPLIT_MODE}_PR_${PUB_RATIO}_MR_${MISSING_RATIO}_PROP_${PROPORTION}_freeze" \
                            --config $CONFIG --num_clients $N_CLIENTS --split_mode $SPLIT_MODE --num_pub_feats ${NUM_PUB_FEATS[$CONFIG]} \
                            --pub_ratio $PUB_RATIO --missing_ratio $MISSING_RATIO --proportion $PROPORTION --freeze \
                            --num_cpus ${NUM_CPUS[$N_CLIENTS]} --num_gpus ${NUM_GPUS[$N_CLIENTS]}

                        for PIS_ALPHA in "${PIS_ALPHAS[@]}"; do

                            # Run FedTDD
                            ./run_sim.sh --name "${CONFIG}_NC_${N_CLIENTS}_${SPLIT_MODE}_PR_${PUB_RATIO}_MR_${MISSING_RATIO}_PROP_${PROPORTION}_PIS_${PIS_ALPHA}" \
                                --config $CONFIG --num_clients $N_CLIENTS --split_mode $SPLIT_MODE --num_pub_feats ${NUM_PUB_FEATS[$CONFIG]} \
                                --pub_ratio $PUB_RATIO --missing_ratio $MISSING_RATIO --proportion $PROPORTION --pis_alpha $PIS_ALPHA \
                                --num_cpus ${NUM_CPUS[$N_CLIENTS]} --num_gpus ${NUM_GPUS[$N_CLIENTS]}

                        done
                    done
                done
            done
        done
    done
done
