#!/bin/bash
# ----------------------------------------------------------------------------
# Build the Phoenix-MWA interpreter used by phoenix_worker.py (deploy_policy.py starts it) and finetune.sh, and
# download the weights inference needs.
#
#   bash policy/phoenix/setup_env.sh [--skip-venv] [--download-inference] [--with-t5] [--download-pretrain]
#
# The venv lives outside the repo ($WS_ROOT/venvs/phoenix, override with PHOENIX_VENV), where deploy_policy.py looks
# for it when phoenix_python is empty: Python 3.11, torch 2.7.1 / torchvision 0.22.1 for CUDA 12.8 (upstream's pin),
# then the upstream post-train checkout (policy/phoenix/Phoenix-MWA) installed editable. TORCH_INDEX_URL (default the PyTorch cu128
# index) and uv's UV_DEFAULT_INDEX choose other mirrors; TORCH_FIND_LINKS takes torch from a find-links mirror instead.
# For example: UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
#              TORCH_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cu128/
#
# Weights land in $PHOENIX_MODEL_ROOT (default $WS_ROOT/models/phoenix), laid out as upstream's README does:
#   --download-inference  encoders/ (csvae.pt + dino110k.pt, HF AnyverseDynamics/Phoenix-Encoders, 2.8 GB) and wan_base/
#                         (DIFFSYNTH_MODEL_BASE_PATH: Wan2.2_VAE.safetensors, 1.4 GB, only ModelScope has the converted
#                         file; the umT5 tokenizer from HF Wan-AI/Wan2.1-T2V-1.3B at upstream's pinned revision)
#   --with-t5             also the umT5-XXL text encoder (11 GB): only prompts missing from a run's text_cache/ need it
#   --download-pretrain   pretrain_256013.pt (HF AnyverseDynamics/Phoenix-Pretrain, 12 GB), the post-training init
# A trained run (config.yaml, dataset_stats.json, embodiment_registry.json, text_cache/, checkpoints/weights/) is your
# own and comes separately: deploy_policy.yml checkpoint_dir points at it, phoenix_encoders / phoenix_wan_base at the two
# dirs above. Downloads resume where they stopped; reruns skip finished files.
# ----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WS_ROOT="${WS_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
VENV_DIR="${PHOENIX_VENV:-$WS_ROOT/venvs/phoenix}"
MODEL_ROOT="${PHOENIX_MODEL_ROOT:-$WS_ROOT/models/phoenix}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
MODELSCOPE="${MODELSCOPE:-https://www.modelscope.cn}"
TOKENIZER_REVISION=37ec512624d61f7aa208f7ea8140a131f93afc9a  # upstream README's pin of Wan-AI/Wan2.1-T2V-1.3B
UPSTREAM="$SCRIPT_DIR/Phoenix-MWA"

SKIP_VENV=0
DOWNLOADS=()
for arg in "$@"; do
    case "$arg" in
        --skip-venv) SKIP_VENV=1 ;;
        --download-inference) DOWNLOADS+=(inference) ;;
        --with-t5) DOWNLOADS+=(t5) ;;
        --download-pretrain) DOWNLOADS+=(pretrain) ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

if [[ ! -f "$UPSTREAM/pyproject.toml" ]]; then
    git -C "$REPO_ROOT" submodule update --init policy/phoenix/Phoenix-MWA
fi

PY="$VENV_DIR/bin/python"
if [[ "$SKIP_VENV" == 0 ]]; then
    command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }
    if [[ ! -x "$PY" ]]; then
        uv venv --python 3.11 "$VENV_DIR"
    fi

    echo "[1/3] torch 2.7.1 + torchvision 0.22.1 (CUDA 12.8)"
    if [[ -n "${TORCH_FIND_LINKS:-}" ]]; then
        uv pip install --python "$PY" --find-links "$TORCH_FIND_LINKS" torch==2.7.1+cu128 torchvision==0.22.1+cu128
    else
        uv pip install --python "$PY" --index-url "$TORCH_INDEX_URL" torch==2.7.1 torchvision==0.22.1
    fi

    echo "[2/3] the post-train submodule, editable"
    uv pip install --python "$PY" -e "$UPSTREAM"

    echo "[3/3] import check (what phoenix_worker.py imports)"
    (cd "$UPSTREAM" && "$PY" - <<'PYEOF'
import torch, wam.model.factory, wam.inference.frame_joint, wam.training.checkpoint
import eval.robotwin.current_visual_adapter
print(f"torch {torch.__version__} (cuda {torch.version.cuda}, available={torch.cuda.is_available()})")
PYEOF
    )
    echo "Phoenix venv ready: $VENV_DIR"
fi

HF="$VENV_DIR/bin/hf"

modelscope_get() {  # <repo> <dest dir> [file...]: every file sha256-checked against ModelScope's listing
    local repo="$1" dest="$2" path sha
    shift 2
    mkdir -p "$dest"
    python3 - "$MODELSCOPE/api/v1/models/$repo/repo/files?Recursive=true" "$@" > "$dest/.modelscope_files" <<'PYEOF'
import json, sys, urllib.request
wanted = set(sys.argv[2:])
for f in json.load(urllib.request.urlopen(sys.argv[1]))["Data"]["Files"]:
    if f.get("Type", "blob") == "blob" and (not wanted or f["Path"] in wanted):
        print(f["Path"], f["Sha256"])
PYEOF
    while read -r path sha; do
        if [[ -f "$dest/$path" ]]; then
            echo "  have $dest/$path"
            continue
        fi
        mkdir -p "$(dirname "$dest/$path")"
        curl -fL --retry 5 --retry-delay 10 -C - -o "$dest/$path.part" \
            "$MODELSCOPE/models/$repo/resolve/master/$path" < /dev/null
        echo "$sha  $dest/$path.part" | sha256sum -c --quiet - || { echo "sha256 mismatch: $dest/$path.part" >&2; exit 1; }
        mv "$dest/$path.part" "$dest/$path"
    done < "$dest/.modelscope_files"
    rm -f "$dest/.modelscope_files"
}

WAN="$MODEL_ROOT/wan_base"
for item in "${DOWNLOADS[@]}"; do
    case "$item" in
        inference)
            echo "encoders -> $MODEL_ROOT/encoders"
            "$HF" download AnyverseDynamics/Phoenix-Encoders csvae.pt dino110k.pt --local-dir "$MODEL_ROOT/encoders"
            echo "wan_base -> $WAN"
            modelscope_get DiffSynth-Studio/Wan-Series-Converted-Safetensors \
                "$WAN/DiffSynth-Studio/Wan-Series-Converted-Safetensors" Wan2.2_VAE.safetensors
            "$HF" download Wan-AI/Wan2.1-T2V-1.3B --revision "$TOKENIZER_REVISION" --include 'google/umt5-xxl/*' \
                --local-dir "$WAN/Wan-AI/Wan2.1-T2V-1.3B"
            ;;
        t5)
            modelscope_get DiffSynth-Studio/Wan-Series-Converted-Safetensors \
                "$WAN/DiffSynth-Studio/Wan-Series-Converted-Safetensors" models_t5_umt5-xxl-enc-bf16.safetensors
            ;;
        pretrain)
            "$HF" download AnyverseDynamics/Phoenix-Pretrain pretrain_256013.pt --local-dir "$MODEL_ROOT/pretrain"
            ;;
    esac
done
[[ ${#DOWNLOADS[@]} == 0 ]] || echo "weights in $MODEL_ROOT: phoenix_encoders $MODEL_ROOT/encoders, phoenix_wan_base $WAN"
