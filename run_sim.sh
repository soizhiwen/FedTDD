#!/bin/bash
# chmod +x run_sim.sh

usage() {
    echo "Usage: $0 --config config --num_clients num_clients --split_mode split_mode --num_pub_feats num_pub_feats --pub_ratio pub_ratio --missing_ratio missing_ratio --proportion proportion --num_cpus num_cpus --num_gpus num_gpus [--name name] [--pis_alpha pis_alpha] [--centralized] [--local] [--freeze]"
    exit 1
}

NAME=""
PIS_ALPHA=""
CENTRALIZED=""
LOCAL=""
FREEZE=""

# Parse command
while [[ "$#" -gt 0 ]]; do
    case $1 in
    --config)
        CONFIG="$2"
        shift 2
        ;;
    --num_clients)
        NUM_CLIENTS="$2"
        shift 2
        ;;
    --split_mode)
        SPLIT_MODE="$2"
        shift 2
        ;;
    --num_pub_feats)
        NUM_PUB_FEATS="$2"
        shift 2
        ;;
    --pub_ratio)
        PUB_RATIO="$2"
        shift 2
        ;;
    --missing_ratio)
        MISSING_RATIO="$2"
        shift 2
        ;;
    --proportion)
        PROPORTION="$2"
        shift 2
        ;;
    --num_cpus)
        NUM_CPUS="$2"
        shift 2
        ;;
    --num_gpus)
        NUM_GPUS="$2"
        shift 2
        ;;
    --name)
        NAME="--name $2"
        shift 2
        ;;
    --pis_alpha)
        PIS_ALPHA="--pis_alpha $2"
        shift 2
        ;;
    --centralized)
        CENTRALIZED="--centralized"
        shift 1
        ;;
    --local)
        LOCAL="--local"
        shift 1
        ;;
    --freeze)
        FREEZE="--freeze"
        shift 1
        ;;
    *)
        usage
        ;;
    esac
done

# Check if all arguments are provided
if [ -z "$CONFIG" ] || [ -z "$NUM_CLIENTS" ] || [ -z "$SPLIT_MODE" ] || [ -z "$NUM_PUB_FEATS" ] || [ -z "$PUB_RATIO" ] || [ -z "$MISSING_RATIO" ] || [ -z "$PROPORTION" ] || [ -z "$NUM_CPUS" ] || [ -z "$NUM_GPUS" ]; then
    usage
fi

python sim.py $NAME --num_clients $NUM_CLIENTS --num_rounds 5 \
    --config_file ./Config/$CONFIG.yaml --split_mode $SPLIT_MODE \
    --num_pub_feats $NUM_PUB_FEATS --pub_ratio $PUB_RATIO \
    --missing_ratio $MISSING_RATIO --proportion $PROPORTION \
    $PIS_ALPHA $CENTRALIZED $LOCAL $FREEZE --cudnn_deterministic \
    --num_cpus $NUM_CPUS --num_gpus $NUM_GPUS
