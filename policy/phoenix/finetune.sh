#!/bin/bash
# Phoenix-MWA post-training on a new embodiment (fmc3_phoenix.py: upstream online-RGB recipe, new soft prompt slot 18).
#   bash policy/phoenix/finetune.sh <dataset_dir> <run_id> [Hydra overrides...]
# Stages ($PHOENIX_STAGES, default "registry text train"):
#   registry  versioned Robot37 registry with slot 18 assigned -> <out>/embodiment_registry.json
#   text      T5 cache of the dataset's prompts (upstream tools/text_cache.py, one GPU) -> <out>/text_cache
#   train     upstream scripts/train.py through train_fmc3.py, accelerate + DeepSpeed ZeRO-1 (upstream config)
# Multi-node: run "registry text" once, then "train" on every node with the same <run_id>, NNODES, NODE_RANK and
# MASTER_ADDR. Stats: <out>/dataset_stats.json must exist before the train stage (upstream's normalization-stats
# format; this repository has no stats tool, see the README). Weights only, no full-state resume in this release.
# Task config: $PHOENIX_TASK (default fmc3_posttrain = absolute joints; fmc3_posttrain_relative = the arm joints
# relative to s[t], stats from relative_stats.py). robosyn_transforms comes from $ROBOSYN_TRANSFORMS_DIR
# (default: this directory).
# Paths: $PHOENIX_WS is the workspace root that holds the upstream checkout, the venv, the weights and the outputs;
# each of them has its own variable below.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${1:?usage: finetune.sh <dataset_dir> <run_id> [overrides...]}"
RUN_ID="${2:?run_id}"
shift 2
WS="${PHOENIX_WS:?set PHOENIX_WS to your workspace root}"
PHX="${PHOENIX_CHECKOUT:-$WS/Phoenix-MWA}"
PY="${PHOENIX_PYTHON:-$WS/venvs/phoenix/bin/python}"
W="${PHOENIX_WEIGHTS:-$WS/models/phoenix}"
OUT="${PHOENIX_OUTPUT_ROOT:-$WS/train_out/phoenix}/$RUN_ID"
STAGES="${PHOENIX_STAGES:-registry text train}"
mkdir -p "$OUT"

export PYTHONPATH="$PHX/src:$PHX:${ROBOSYN_TRANSFORMS_DIR:-$HERE}" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true TORCH_COMPILE_ENABLED=false HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WAM_PRETRAIN_CHECKPOINT="${WAM_PRETRAIN_CHECKPOINT:-$W/pretrain/pretrain_256013.pt}"
export DINO_ENCODER_CHECKPOINT="$W/encoders/dino110k.pt" CSVAE_CHECKPOINT="$W/encoders/csvae.pt"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$W/wan_base}"  # Wan2.2-TI2V-5B DiT, VAE, umT5, tokenizer (links)
export FMC3_DATA_ROOT="$(realpath -s "$DATA")" FMC3_NORM_STATS="$OUT/dataset_stats.json" FMC3_TEXT_CACHE="$OUT/text_cache"
export EMBODIMENT_REGISTRY="$OUT/embodiment_registry.json"
COMMON=("task=${PHOENIX_TASK:-fmc3_posttrain}" "hydra.searchpath=[file://$HERE/configs]")
cd "$PHX"

if [[ " $STAGES " == *" registry "* ]]; then
    "$PY" -c "import sys; sys.path.insert(0, '$HERE'); import fmc3_phoenix as f; from pathlib import Path; \
print(f.write_registry(Path('$PHX/configs/embodiment_registry/local_robot37_arm18_capacity28_v1.json'), Path('$EMBODIMENT_REGISTRY')))"
fi
if [[ " $STAGES " == *" text "* ]]; then
    CUDA_VISIBLE_DEVICES="${TEXT_GPU:-0}" "$PY" "$HERE/train_fmc3.py" tools/text_cache.py "${COMMON[@]}"
fi
if [[ " $STAGES " == *" train "* ]]; then
    [[ -f "$FMC3_NORM_STATS" && -f "$EMBODIMENT_REGISTRY" ]] || { echo "missing stats or registry in $OUT" >&2; exit 1; }
    NNODES="${NNODES:-1}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}" NODE_RANK="${NODE_RANK:-0}"
    "$PY" -m accelerate.commands.launch \
        --config_file scripts/accelerate_configs/accelerate_joint96_posttrain_zero1.yaml \
        --num_machines "$NNODES" --num_processes $((NNODES * NPROC_PER_NODE)) --machine_rank "$NODE_RANK" \
        --main_process_ip "${MASTER_ADDR:-127.0.0.1}" --main_process_port "${MASTER_PORT:-24491}" \
        --deepspeed_multinode_launcher standard "$HERE/train_fmc3.py" scripts/train.py "${COMMON[@]}" \
        "output_dir=$OUT/run" "wandb.enabled=false" "$@"
fi
