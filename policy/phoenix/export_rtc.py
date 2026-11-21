#!/usr/bin/env python
"""Put a Training RTC run (upstream prometheus phoenix.train) together as a checkpoint_dir for rtc_worker.py.

    python policy/phoenix/export_rtc.py <train.yaml> <out_dir> [--steps 1000 2000 ...] [--image-crop center_square]

<train.yaml> is the recipe the run was trained with (phoenix.training_rtc_recipe.v1). From it:
    model.yaml           its model_definition (rtc_init.py's phoenix_model_rtc.yaml)
    stvae.pt             the encoder next to that definition (rtc_init.py's)
    normalization.json   data.normalization (sha256 checked against the recipe and the weights)
    text_cache/          data.text_cache_index's directory (text_index.json + the T5 rows)
    checkpoints/step_NNNNNN/weights.pt   from output_dir (hard links when on the same disk), all steps by default
    EXPORT.json          where it came from, and --image-crop: center_square when the training set was built with
                         build_rtc_set.py --crop-square (its CONVERTED.json records it)
Plain JSON/YAML reading only: runs with any Python that has PyYAML or OmegaConf.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_yaml(path: Path) -> dict:
    try:
        import yaml

        return yaml.safe_load(path.read_text())
    except ImportError:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


def place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recipe", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--steps", type=int, nargs="*", help="default: every saved step")
    p.add_argument("--image-crop", choices=("none", "center_square"), default="none",
                   help="the training views' crop (build_rtc_set.py --crop-square: center_square)")  # fmt: skip
    args = p.parse_args()

    recipe = load_yaml(args.recipe)
    if recipe.get("schema") != "phoenix.training_rtc_recipe.v1":
        raise SystemExit(f"{args.recipe} is not a Training RTC recipe")
    data = recipe["data"]
    definition = Path(recipe["model_definition"])
    normalization = Path(data["normalization"])
    if sha256(normalization) != data["normalization_sha256"]:
        raise SystemExit(f"{normalization} differs from the recipe's sha256")
    text_dir = Path(data["text_cache_index"]).parent
    saved = {int(d.name.removeprefix("step_")): d / "weights.pt"
             for d in Path(recipe["output_dir"]).glob("checkpoints/step_*") if (d / "weights.pt").is_file()}  # fmt: skip
    steps = args.steps or sorted(saved)
    missing = [s for s in steps if s not in saved]
    if missing or not steps:
        raise SystemExit(f"steps {missing or steps} not saved in {recipe['output_dir']} (saved: {sorted(saved)})")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    place(definition, out / "model.yaml")
    place(definition.parent / "stvae.pt", out / "stvae.pt")
    place(normalization, out / "normalization.json")
    for item in text_dir.iterdir():
        if item.is_file():
            place(item, out / "text_cache" / item.name)
    for step in steps:
        place(saved[step], out / "checkpoints" / f"step_{step:06d}" / "weights.pt")
    shutil.copy2(args.recipe, out / "train.yaml")
    (out / "EXPORT.json").write_text(json.dumps({
        "recipe": str(args.recipe.resolve()), "output_dir": recipe["output_dir"], "steps": steps,
        "normalization_sha256": data["normalization_sha256"],
        "init": recipe["initialization"]["checkpoint"], "init_sha256": recipe["initialization"]["sha256"],
        "image_crop": args.image_crop,
    }, indent=1))  # fmt: skip
    print(f"{out}: steps {steps}")


if __name__ == "__main__":
    main()
