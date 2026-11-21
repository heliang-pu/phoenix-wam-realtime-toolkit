# ----------------------------------------------------------------------------
# Chunk joins of the Phoenix adapter by inpainting (phoenix_smooth inpaint<h>, inpaint<h>soft).
#
# Synchronous chunks join where the next chunk, drawn from a new observation, starts somewhere the last one was not
# going: the large steps of a rollout sit mostly at the joins, not inside a chunk. With inpainting each chunk is drawn
# with its first steps held to the steps of the previous chunk that were not executed (32 - phoenix_steps of them), so
# it starts where the last one was going: the synchronous chunk inpainting of arXiv 2512.06951 (predict 30, execute
# 26, keep 4), here at inference only (phoenix_worker.InpaintScheduler), with no retraining.
#   inpaint<h>        the first h steps held exactly, the rest of the chunk free
#   inpaint<h>soft    the rest of the steps not executed held too, by RTC's soft mask (arXiv 2506.07339, eq. 5)
#   inpaint<h>soft<n> the soft mask over the next n steps only
# A model that hardly re-plans after a held start only moves the jump to step h with a hard hold; the soft mask
# spreads it over the overlap. A smoother composes with it: inpaint2soft+gauss1 (or +lipo1). Reads only the model's
# own chunks.
# ----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

ARM_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
# the choices a UI offers: steps held exactly (0: off) and soft steps after them (None: all)
HOLD_CHOICES = (0, 1, 2, 3, 4, 6, 8)
SOFT_CHOICES = (0, 2, 4, 6, 8, 12, None)
PATTERN = re.compile(r"inpaint(\d*)(?:(soft)(\d*))?")


@dataclass(frozen=True)
class Inpaint:
    steps: int = 4  # held exactly
    soft: bool = False  # the other steps not executed held by RTC's soft mask
    soft_steps: int | None = None  # with soft: over this many steps after the held ones (None: all not executed)

    @property
    def key(self) -> str:
        if not self.soft:
            return f"inpaint{self.steps}"
        return f"inpaint{self.steps}soft{'' if self.soft_steps is None else self.soft_steps}"

    def describe(self) -> dict:
        return {"steps": self.steps, "soft": self.soft, "soft_steps": self.soft_steps if self.soft else 0}


def compose(steps: int, soft_steps: int | None) -> Inpaint | None:
    """The inpainting of ``steps`` held exactly (0: off) and ``soft_steps`` after them (0: none, None: all)."""
    if int(steps) <= 0:
        return None
    if soft_steps is not None and int(soft_steps) <= 0:
        return Inpaint(int(steps))
    return Inpaint(int(steps), True, None if soft_steps is None else int(soft_steps))


def split(value) -> tuple[Inpaint | None, str]:
    """A phoenix_smooth value -> (its inpainting or None, the rest for chunk_smoother): "inpaint4+gauss1" ->
    (Inpaint(4), "gauss1"); a bare "inpaint" holds 4 steps; "inpaint2soft4": 2 held, 4 soft."""
    inpaint, rest = None, []
    for part in (text.strip() for text in str(value or "none").lower().split("+")):
        match = PATTERN.fullmatch(part)
        if match is None:
            if part and part != "none":
                rest.append(part)
            continue
        steps = int(match.group(1) or 4)
        if inpaint is not None or steps < 1:
            raise ValueError(f"phoenix_smooth takes one inpaint<steps>[soft[<steps>]], steps >= 1, got {value!r}")
        soft = bool(match.group(2))
        inpaint = compose(steps, (int(match.group(3)) if match.group(3) else None) if soft else 0)
    if len(rest) > 1:
        raise ValueError(f"phoenix_smooth takes one smoother besides inpainting, got {value!r}")
    return inpaint, rest[0] if rest else "none"


def join(inpaint: Inpaint | None, smoother_key: str) -> str:
    """The phoenix_smooth value of an inpainting and a smoother: inpaint4+gauss1, inpaint4, gauss1, none."""
    parts = [inpaint.key] if inpaint is not None else []
    if smoother_key and smoother_key != "none":
        parts.append(smoother_key)
    return "+".join(parts) or "none"


def from_usr_args(usr_args: dict) -> Inpaint | None:
    """The inpainting deploy_policy.yml's phoenix_smooth names for usr_args' task (None: off)."""
    from policy.phoenix.chunk_smoother import _per_task  # noqa: PLC0415

    return split(_per_task(usr_args.get("phoenix_smooth"), str(usr_args.get("task_name") or ""), "none"))[0]


class JoinSteps:
    """How far each chunk's first target is from the last executed target of the chunk before (arm joints, rad, the
    model's own targets): what inpainting is for, logged per episode for every setting."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.last = None
        self.steps: list[float] = []

    def chunk(self, rows: np.ndarray, executed: int) -> None:
        rows = np.asarray(rows, dtype=np.float32)
        if self.last is not None:
            self.steps.append(float(np.abs(rows[0, ARM_DIMS] - self.last[ARM_DIMS]).max()))
        self.last = rows[min(int(executed), len(rows)) - 1]

    def summary(self) -> str | None:
        if not self.steps:
            return None
        steps = np.asarray(self.steps)
        return (
            f"{len(steps)} chunk joins, step at the join median {np.median(steps):.3f} max {steps.max():.3f} rad, "
            f"{int((steps > 0.1).sum())} over 0.1"
        )
