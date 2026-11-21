"""Action transforms for a 14-D dual-arm joint layout, plugged into upstream's WAMProcessor
(``action_state_transforms``).

Importable as ``robosyn_transforms`` from Hydra configs: finetune.sh puts this directory on PYTHONPATH and
phoenix_worker.py adds it to sys.path.
"""

from __future__ import annotations

# CobotMagic 14-D layout: left joint1-6, left gripper, right joint1-6, right gripper
JOINT_DIMS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)


class RelativeJointTransform:
    """Joint targets relative to the current state s[t] (the openpi DeltaActions convention the pi0.5 models use);
    the other dims (grippers) stay absolute.

    The processor sees two layouts, and s[t] sits in a different place in each:
      training sample       state [S, D] = memory frames, then the ``num_frames`` window starting at s[t];
                            action [H, D] = a[t : t+H]           -> s[t] = state[S - num_frames]
      posttrain_stats.py    state [T, 1, D], action [T, H, D] (one window per episode step) -> s[t] = state[:, 0]
    At inference the batch has no action and passes through; the inference worker adds s[t] back to the joint dims
    of the denormalized chunk.
    """

    def __init__(self, keys, dims=JOINT_DIMS, num_frames: int = 33):
        self.keys = list(keys)
        self.dims = [int(d) for d in dims]
        self.num_frames = int(num_frames)

    def _current(self, state):
        if state.ndim == 3 and state.shape[-2] == 1:  # per-step windows (stats)
            return state
        return state[..., state.shape[-2] - self.num_frames, :]

    def _apply(self, batch, sign: float):
        if "action" not in batch:  # closed-loop inference: state only
            return batch
        for key in self.keys:
            action = batch["action"][key].clone()
            current = self._current(batch["state"][key])
            action[..., self.dims] += sign * current[..., self.dims]
            batch["action"][key] = action
        return batch

    def forward(self, batch):
        return self._apply(batch, -1.0)

    def backward(self, batch):
        return self._apply(batch, 1.0)
