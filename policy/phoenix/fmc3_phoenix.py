"""Phoenix-MWA post-training on a new embodiment (here a CobotMagic: dual PiPER arms, 14-D joints) with a new soft
prompt; upstream left unchanged.

The release's online-RGB recipe (task=robotwin_posttrain) is the one whose data matches this robot's: LeRobot v2.1, three
480x640 cameras, native joint14 (left 6 joints + gripper, right 6 + gripper) mapped to canonical80 by ConcatLeftAlign.
It is pinned to the RoboTwin export and to the ARX5 prompt slot 2. A new robot configuration gets a new soft prompt,
with its joints filled into their canonical slots: this robot takes the unused Robot37 slot 18, exactly as upstream's RoboCasa GR1 recipe activates it (activate_robocasa_slot /
load_robocasa_delta80_slot18_initialization): a versioned registry assigns slot 18, and initialization inherits every
pretrained weight except that prompt row, which must be empty in the parent and gets a fresh N(0, 0.02) row.

install() swaps four names before scripts/train.py runs (the guards compare reader names as strings, so the reader
keeps its upstream name):
  wam.datasets.robotwin_joint96_robot37.RobotwinJoint96OnlineDataset  -> Fmc3Joint96OnlineDataset (any v2.1 root,
      all episodes, prompt-hashed text cache from tools/text_cache.py, same upstream T5 padding)
  wam.training.robotwin_robot37_softprompt.validate_robotwin_prompt / wrap_robotwin_prompt -> slot EMBODIMENT_ID
  wam.trainer.load_joint96_initialization -> load_slot_initialization (from the pretrained parent: a fresh slot 18 row;
      from a run made with this file: continue_own_run, every weight as saved)

fmc3.main_camera_mask_prob (configs/task/fmc3_posttrain.yaml: 0.1, 0 = off): that fraction of training
samples has the main camera (cam_high, slot 0) marked missing through upstream's augmentation_camera_mask, the model's
own main-camera augmentation input (main slot only): the view goes in as zeros and its video loss is not counted.
Upstream merges it into camera_is_pad, so the model knows a step without the top camera the way deploy_policy's
phoenix_stall_fix mask sends it (pad_cameras: a zero latent, its tokens out of attention). phoenix_stall_fix cover is
not that: its black frame is encoded and attended as a real view, which no training shows.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset

EMBODIMENT_ID = 18
EMBODIMENT_NAME = "fmc3_cobotmagic_piper_joint14"
PROMPT_KEY = "mixtures.action.embodiment_prompt.embedding.weight"
MAIN_MASK_KEY = "augmentation_camera_mask"  # bool[3] per sample, [B,3] per batch: main, left wrist, right wrist


def activate_slot(source: dict) -> dict:
    """Version the Robot37 registry while assigning the unused slot 18 to this robot (cf. activate_robocasa_slot)."""
    from wam.embodiment_registry import EmbodimentRegistry

    EmbodimentRegistry.validate(source)
    if (source["num_embodiments"] != 28 or source["slot_status"][EMBODIMENT_ID] != "reserved"
            or source["embodiments"][EMBODIMENT_ID] != f"reserved_{EMBODIMENT_ID}"):  # fmt: skip
        raise ValueError(f"needs the unused Robot37 slot {EMBODIMENT_ID}")
    result = copy.deepcopy(source)
    result["version"] += f".{EMBODIMENT_NAME}_slot{EMBODIMENT_ID}.v1"
    result["embodiments"][EMBODIMENT_ID] = EMBODIMENT_NAME
    result["slot_status"][EMBODIMENT_ID] = "assigned"
    result["assigned_count"] += 1
    result["profile_to_embodiment_id"][EMBODIMENT_NAME] = EMBODIMENT_ID
    EmbodimentRegistry.validate(result)
    return result


def write_registry(parent: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(activate_slot(json.loads(parent.read_text())), indent=1) + "\n")
    return out


def load_slot_initialization(model, path, *, expected_step=None):
    """Strictly inherit every pretrained weight except the newly activated prompt row (cf. RoboCasa slot 18)."""
    from wam.embodiment_registry import EmbodimentRegistry
    from wam.training.checkpoint import _joint96_identity
    from wam.training.checkpoint import _load_joint96_checkpoint
    from wam.training.checkpoint import joint96_checkpoint_metadata
    from wam.utils.runtime_contract import semantic_contract

    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError("slot initialization requires an absolute checkpoint file")
    payload = torch.load(path, mmap=True, map_location="cpu", weights_only=True)
    metadata = joint96_checkpoint_metadata(model)
    migrated = _joint96_identity(payload.get("joint96_metadata"))
    if (migrated.get("action_dim"), migrated.get("proprio_dim")) != (80, 80):
        raise ValueError("needs an unchanged canonical80 parent")
    if migrated["soft_prompt"]["registry"]["embodiments"][EMBODIMENT_ID] == EMBODIMENT_NAME:
        return continue_own_run(model, path, payload)
    prompt_identity = migrated["soft_prompt"]
    prompt_identity["registry"] = activate_slot(prompt_identity["registry"])
    prompt_identity["registry_sha256"] = EmbodimentRegistry.content_hash(prompt_identity["registry"])
    if semantic_contract(migrated) != semantic_contract(_joint96_identity(metadata)):
        raise ValueError("slot transfer changed an unrelated model identity")
    if bool(torch.count_nonzero(payload["mot"][PROMPT_KEY][EMBODIMENT_ID])):
        raise ValueError(f"parent prompt slot {EMBODIMENT_ID} is not empty; refusing to overwrite it")
    prompt = payload["mot"][PROMPT_KEY].clone()
    prompt[EMBODIMENT_ID] = model.mot.state_dict()[PROMPT_KEY][EMBODIMENT_ID].to(prompt)
    _load_joint96_checkpoint(
        model, {**payload, "mot": {**payload["mot"], PROMPT_KEY: prompt}, "joint96_metadata": metadata}
    )
    model.model_paths["posttrain_initialization"] = {"path": str(path), "step": payload.get("step")}
    return {"checkpoint": str(path), "pretrain_step": payload.get("step"), "new_global_step": 0, "weights_only": True,
            "new_prompt_slot": EMBODIMENT_ID, "new_prompt_name": EMBODIMENT_NAME,
            "new_prompt_initialization": "fresh_normal_std0.02_after_empty_parent_check"}  # fmt: skip


def continue_own_run(model, path, payload):
    """A weights checkpoint of a run made with this file (slot 18 already assigned) as the parent: every weight as
    saved, the prompt row included, through upstream's strict loader (same encoder, registry and conditioning identity, every
    tensor). Weights only, as upstream's posttrain initialization: a fresh optimizer and schedule from step 0. Keep the
    parent run's normalization stats, or its actions are read with another scale."""
    from wam.training.checkpoint import _load_joint96_checkpoint

    if not bool(torch.count_nonzero(payload["mot"][PROMPT_KEY][EMBODIMENT_ID])):
        raise ValueError(f"checkpoint assigns slot {EMBODIMENT_ID} but its prompt row is empty")
    _load_joint96_checkpoint(model, payload)
    model.model_paths["posttrain_initialization"] = {"path": str(path), "step": payload.get("step"), "continued": True}
    return {"checkpoint": str(path), "pretrain_step": payload.get("step"), "new_global_step": 0, "weights_only": True,
            "continued_own_run": True, "prompt_slot": EMBODIMENT_ID}  # fmt: skip


def validate_prompt(cfg):
    get = lambda k, d=None: OmegaConf.select(cfg, k, default=d)  # noqa: E731
    if not get("joint96_posttrain.enabled") or get("local_robot_joint96.enabled") or get("resume") is not None \
            or get("local_robot_softprompt.init_checkpoint") is not None:  # fmt: skip
        raise ValueError("needs fresh weights-only initialization")
    if get("joint96_posttrain.robotwin_softprompt.embodiment_id") != EMBODIMENT_ID:
        raise ValueError(f"this robot uses prompt slot {EMBODIMENT_ID}")
    if (get("model.action_dit_config.num_embodiments"), get("model.action_dit_config.soft_prompt_tokens")) != (28, 8):
        raise ValueError("keep the complete 28-slot / 8-token prompt table")


class PromptDataset(Dataset):
    """Every sample carries this robot's prompt id (cf. upstream RobotwinPromptDataset, which pins slot 2), and with
    ``main_camera_mask_prob`` > 0 its main-camera mask (MAIN_MASK_KEY)."""

    def __init__(self, dataset, registry, main_camera_mask_prob=0.0):
        if not 0.0 <= float(main_camera_mask_prob) < 1.0:
            raise ValueError(f"main_camera_mask_prob must be in [0, 1), got {main_camera_mask_prob}")
        self.main_camera_mask_prob = float(main_camera_mask_prob)
        if (
            registry.content["embodiments"][EMBODIMENT_ID] != EMBODIMENT_NAME
            or EMBODIMENT_ID not in registry.assigned_ids
        ):
            raise ValueError(f"registry does not assign slot {EMBODIMENT_ID} to {EMBODIMENT_NAME}")
        self.dataset = dataset
        self.embodiment_registry = registry
        if hasattr(dataset, "fixed_validation_panel"):
            self.fixed_validation_panel = copy.deepcopy(dataset.fixed_validation_panel)
            for row in self.fixed_validation_panel:
                row["embodiment_id"] = EMBODIMENT_ID

    def __getattr__(self, name):
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = {**self.dataset[index], "embodiment_id": torch.tensor(EMBODIMENT_ID, dtype=torch.long)}
        if self.main_camera_mask_prob > 0.0:  # every sample has one, so a batch stacks to bool[B,3]
            masked = bool(torch.rand(()) < self.main_camera_mask_prob)
            sample[MAIN_MASK_KEY] = torch.tensor([masked, False, False])
        return sample

    def collate_fn(self, samples):
        masks = [s[MAIN_MASK_KEY] for s in samples if MAIN_MASK_KEY in s]
        samples = [{k: v for k, v in s.items() if k != MAIN_MASK_KEY} for s in samples]  # not for upstream's collate
        collate = getattr(self.dataset, "collate_fn", None)
        if collate is None:
            from torch.utils.data import default_collate

            batch = default_collate(samples)
        else:
            batch = collate(samples)
        batch["embodiment_id"] = torch.stack([s["embodiment_id"] for s in samples])
        if masks:
            batch[MAIN_MASK_KEY] = torch.stack(masks)
        return batch


def wrap_prompt(model, train, val, cfg, registry):
    if val is None or train is val:
        raise ValueError("needs distinct training and validation readers")
    model.soft_prompt_training_contract = {"domain": "fmc3_cobotmagic", "embodiment_id": EMBODIMENT_ID,
                                           "embodiment_name": EMBODIMENT_NAME,
                                           "action_space": "native_joint14_to_canonical80", "cached_visuals": False}  # fmt: skip
    mask = float(OmegaConf.select(cfg, "fmc3.main_camera_mask_prob", default=0.0) or 0.0)
    if mask > 0.0:
        print(f"[fmc3] main camera (cam_high) marked missing in {mask:.0%} of the training samples", flush=True)
    return PromptDataset(train, registry, main_camera_mask_prob=mask), PromptDataset(val, registry)  # val: as is


def install():
    from wam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
    import wam.datasets.robotwin_joint96_robot37 as reader
    import wam.trainer
    import wam.training.robotwin_robot37_softprompt as prompt

    upstream = reader.RobotwinJoint96OnlineDataset

    class Fmc3Joint96OnlineDataset(upstream):
        """Upstream's online-RGB reader for any one LeRobot v2.1 root (all episodes), prompt-hashed T5 cache."""

        def __init__(self, *, text_cache_source_root=None, **kwargs):
            if kwargs.get("domain") != "robotwin" or "visual_cache" in kwargs:
                raise ValueError("online RGB reader only; no visual cache")
            self.text_cache_source_root = text_cache_source_root
            self._text_cache = None
            super(upstream, self).__init__(**kwargs)  # skip only the RoboTwin-export checks
            datasets = self.lerobot_dataset.multi_dataset._datasets  # noqa: SLF001
            if len(datasets) != 1:
                raise ValueError("expected one data root")
            self._raw_dataset = datasets[0]

        def _get_cached_text_context(self, prompt_text):
            # tools/text_cache.py output; the base reader already zero-fills padding and returns an all-visible mask,
            # the same as upstream's official_text_padding
            return RobotVideoDataset._get_cached_text_context(self, prompt_text)  # noqa: SLF001

    reader.RobotwinJoint96OnlineDataset = Fmc3Joint96OnlineDataset
    prompt.validate_robotwin_prompt = validate_prompt
    prompt.wrap_robotwin_prompt = wrap_prompt
    wam.trainer.load_joint96_initialization = load_slot_initialization
