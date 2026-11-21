#!/usr/bin/env python
"""Weights for upstream prometheus's Training RTC from what Hugging Face publishes and post-train runs.

    PYTHONPATH=<prometheus>:<prometheus>/src python rtc_init.py --source <post-train slot .pt or pretrain_256013.pt> \
        --encoders <dir with csvae.pt, dino110k.pt> --wan-base <DIFFSYNTH_MODEL_BASE_PATH> --out <dir>

prometheus (823f009) loads its frozen encoder as an ST-VAE v1 checkpoint ("spatial_transformer_temporal_vae") and
checks every initial checkpoint's encoder identity against it. Hugging Face has the encoder only as csvae.pt, the
same tensors migrated to "causal_semantic_vae" v2 (its provenance: "Model tensors unchanged"), and pretrain_256013.pt
and post-train weights record that csvae identity. Here, without touching upstream code:

1. <out>/stvae.pt = csvae.pt with model_type / format_version set back to ST-VAE v1, nothing else. Checked:
   post-train's CS-VAE encoder and prometheus's ST-VAE encoder give bit-identical latents (video and current frame)
   for the same input.
2. <out>/init.pt = the source's tensors with the encoder identity prometheus derives for that encoder. The model is
   built from prometheus's configs/phoenix_model.yaml; every identity field of the source must equal the model's once
   csvae is read as stvae (the source's own encoder file hash aside: that file is the renamed one), else nothing is
   written; then prometheus's own strict loader restores every tensor. The soft prompt (registry, slot 18) is kept.
3. <out>/phoenix_model_rtc.yaml = prometheus's configs/phoenix_model.yaml with mot_attention_backend flex: the
   published template pairs sdpa with mot_flex_masked_batches true, which its own MoT refuses ("flex_masked_batches
   requires attention_backend='flex'"); it carries flex training kernel options, and the post-train runs train with
   flex too. Inference switches to sdpa and flex_masked_batches off.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def as_stvae(value):
    """The csvae spelling of an identity read as stvae (keys and strings)."""
    if isinstance(value, dict):
        return {k.replace("csvae", "stvae"): as_stvae(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_stvae(v) for v in value]
    return value.replace("csvae", "stvae") if isinstance(value, str) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--encoders", type=Path, required=True)
    parser.add_argument("--wan-base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch

    args.out.mkdir(parents=True, exist_ok=True)
    stvae = args.out / "stvae.pt"
    if not stvae.exists():
        payload = torch.load(args.encoders / "csvae.pt", map_location="cpu", weights_only=True)
        if (payload.get("model_type"), payload.get("format_version")) != ("causal_semantic_vae", 2):
            raise SystemExit("csvae.pt is not the published causal_semantic_vae v2 file")
        payload["model_type"], payload["format_version"] = "spatial_transformer_temporal_vae", 1
        torch.save(payload, stvae)
    os.environ.update(PHOENIX_STVAE_CHECKPOINT=str(stvae), PHOENIX_ENCODER_CHECKPOINT=str(args.encoders / "dino110k.pt"),
                      DIFFSYNTH_MODEL_BASE_PATH=str(args.wan_base))  # fmt: skip

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from phoenix.checkpoint import _joint96_identity, joint96_checkpoint_metadata
    from phoenix.train import ROOT
    from phoenix.training import initialize_from_pretrained
    import wam.model.factory as factory
    from wam.model.modules.codecs.dino_stvae.provenance import bundled_config
    from wam.utils.config_resolvers import register_default_resolvers
    from wam.utils.torch24_compat import install_safe_serialization_compat

    register_default_resolvers()
    install_safe_serialization_compat()
    payload = torch.load(args.source, map_location="cpu", mmap=True, weights_only=True)
    incoming = copy.deepcopy(payload["joint96_metadata"])
    registry = incoming["soft_prompt"]["registry"]
    definition = OmegaConf.load(ROOT / "configs/phoenix_model.yaml")
    definition.model.mot_attention_backend = "flex"
    OmegaConf.save(definition, args.out / "phoenix_model_rtc.yaml")
    model_cfg = OmegaConf.create(OmegaConf.to_container(definition.model, resolve=True))
    model_cfg.joint_visual_codec = bundled_config(OmegaConf.to_container(model_cfg.joint_visual_codec, resolve=True))
    original = factory.load_wan22_ti2v_5b_components

    def checkpoint_components(**kwargs):  # as prometheus's build_model: the DiT comes from the checkpoint
        kwargs["skip_dit_load_from_pretrain"] = True
        return original(**kwargs)

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "registry.json"
        path.write_text(json.dumps(registry))
        model_cfg.action_dit_config.embodiment_registry_path = str(path)
        factory.load_wan22_ti2v_5b_components = checkpoint_components
        try:
            model = instantiate(model_cfg, model_dtype=torch.bfloat16, device="cpu")
        finally:
            factory.load_wan22_ti2v_5b_components = original
    expected = joint96_checkpoint_metadata(model)
    expected["soft_prompt"] = copy.deepcopy(incoming["soft_prompt"])  # the source's prompt table and slot

    # every identity field must agree once csvae is read as stvae; the encoder file hash is the renamed file's
    want = _joint96_identity(expected)
    have = as_stvae({k: v for k, v in incoming.items()})
    have["schema"] = "phoenix.local_joint96.weights.v1"
    have["model_paths"]["joint_visual_encoder"]["stvae"]["stvae_checkpoint_sha256"] = want["stvae_checkpoint_sha256"]
    have["model_paths"]["joint_visual_encoder"]["source_root"] = expected["model_paths"]["joint_visual_encoder"].get(
        "source_root"
    )
    got = _joint96_identity(have)
    from wam.utils.runtime_contract import semantic_contract

    diff = sorted(k for k in set(want) | set(got) if semantic_contract(want.get(k)) != semantic_contract(got.get(k)))
    report = {"source": str(args.source), "source_sha256": sha256(args.source), "stvae_sha256": sha256(stvae),
              "identity_fields": sorted(want), "differing": {k: {"source": got.get(k), "model": want.get(k)} for k in diff}}  # fmt: skip
    (args.out / "rtc_init_report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    if diff:
        raise SystemExit(f"identity differs beyond the csvae/stvae spelling: {diff} (see rtc_init_report.json)")
    migrated = dict(payload)
    migrated["joint96_metadata"] = expected
    initialize_from_pretrained(model, migrated)  # prometheus's strict loader: every MoT / encoder tensor
    torch.save(migrated, args.out / "init.pt")
    print(json.dumps({"init": str(args.out / "init.pt"), "stvae": str(stvae), "checked_fields": len(want)}))


if __name__ == "__main__":
    main()
