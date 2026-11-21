# ----------------------------------------------------------------------------
# In-policy chunk smoothing of the Phoenix adapter (phoenix_smooth).
#
# With one target per control step and no interpolation, the model's own step-to-step reversals (a joint going back
# and forth by a few mrad) become visible stop-and-reverse motions. Each chunk's arm joints are smoothed before they
# are executed (grippers as predicted):
#   gauss    local linear fit with Gaussian weights (phoenix_smooth_sigma steps) through the last command and the
#            chunk; a wide sigma (2-3 steps) cuts the corners of precise approaches, 1 stays close to the path
#   bspline  least-squares cubic B-spline (ABPolicy's representation, used on the outputs) over the last
#            phoenix_smooth_history commands and the chunk, phoenix_smooth_ctrl control points per chunk: the curve
#            leaves the commands already executed smoothly (ABPolicy's continuity refit)
#   lipo     LiPo (arXiv 2506.05165) on the outputs: the first phoenix_smooth_blend steps blended linearly from the
#            previous chunk's unexecuted tail into the new chunk, then the jerk of the commands sent so far plus the
#            chunk minimised with every joint at most lipoN / 100 rad off it (three times that in the blend), so the
#            path can only move that far -- "lipo1": 0.01 rad
# Reads only the policy's own commands and, before the first one, the observed joints.
# ----------------------------------------------------------------------------

from __future__ import annotations

from collections import deque

import numpy as np

ARM_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
METHODS = ("none", "gauss", "bspline", "lipo")
# the choices a UI offers (a method with its parameter, as phoenix_smooth also takes them)
CHOICES = ("none", "gauss1", "gauss2", "bspline8", "bspline6", "lipo1", "lipo2")
LIPO_BLEND = 5  # steps blended from the previous chunk's tail into the new chunk
LIPO_ITERATIONS = 300
DEGREE = 3


def parse(value, sigma: float = 1.0, ctrl: int = 8) -> tuple[str, float, int]:
    """A phoenix_smooth value -> (method, sigma, ctrl): "gauss" / "bspline" take sigma / ctrl from their own
    settings, "gauss2" / "bspline6" carry them; "lipo1" carries its bound in sigma (centiradians)."""
    text = str(value or "none").strip().lower()
    for method in METHODS:
        if text.startswith(method):
            number = text[len(method) :]
            if method == "gauss":
                return method, float(number) if number else float(sigma), int(ctrl)
            if method == "bspline":
                return method, float(sigma), int(number) if number else int(ctrl)
            if method == "lipo":
                return method, float(number) if number else 1.0, int(ctrl)
            if not number:
                return method, float(sigma), int(ctrl)
    raise ValueError(f"phoenix_smooth must be one of {CHOICES} (or gauss / bspline), got {value!r}")


def gauss_fit(chunk: np.ndarray, anchor: np.ndarray, sigma: float) -> np.ndarray:
    """(T, D) chunk low-passed over time: at each step a straight line fitted with Gaussian weights (``sigma`` steps)
    through ``anchor`` (step 0) and the chunk (steps 1..T). A line, not a weighted mean: a steady motion keeps its
    speed, also at both ends."""
    seq = np.vstack([anchor.reshape(1, -1), chunk])
    t = np.arange(len(seq), dtype=np.float64)
    d = t[None, :] - t[:, None]  # (output step, input step)
    w = np.exp(-0.5 * (d / sigma) ** 2)
    s0, s1, s2 = w.sum(1), (w * d).sum(1), (w * d * d).sum(1)
    fit = w * (s2[:, None] - d * s1[:, None]) / (s0 * s2 - s1**2)[:, None]  # local linear regression weights
    return (fit @ seq)[1:]


def bspline_basis(x: np.ndarray, n_ctrl: int, degree: int = DEGREE) -> np.ndarray:
    """(len(x), n_ctrl) clamped open-uniform B-spline basis on [x[0], x[-1]] (Cox-de Boor)."""
    lo, hi = float(x[0]), float(x[-1])
    inner = np.linspace(lo, hi, n_ctrl - degree + 1)[1:-1]
    knots = np.r_[[lo] * (degree + 1), inner, [hi] * (degree + 1)]
    xs = np.clip(x, lo, np.nextafter(hi, lo))  # the last sample falls into the last span
    basis = np.zeros((len(x), len(knots) - 1))
    for i in range(len(knots) - 1):
        basis[:, i] = (knots[i] <= xs) & (xs < knots[i + 1])
    for p in range(1, degree + 1):
        nxt = np.zeros((len(x), len(knots) - p - 1))
        for i in range(len(knots) - p - 1):
            left = knots[i + p] - knots[i]
            right = knots[i + p + 1] - knots[i + 1]
            if left > 0:
                nxt[:, i] += (xs - knots[i]) / left * basis[:, i]
            if right > 0:
                nxt[:, i] += (knots[i + p + 1] - xs) / right * basis[:, i + 1]
        basis = nxt
    return basis


def bspline_fit(chunk: np.ndarray, history: np.ndarray, n_ctrl: int) -> np.ndarray:
    """(T, D) chunk replaced by a least-squares cubic B-spline through the executed ``history`` (H, D) and the chunk;
    ``n_ctrl`` control points per chunk length (more over the history)."""
    seq = np.vstack([history, chunk]) if len(history) else chunk
    total = len(seq)
    n = int(np.clip(round(n_ctrl * total / len(chunk)), DEGREE + 1, total))
    if n >= total:  # as many points as samples: the fit is the samples themselves
        return chunk.copy()
    basis = bspline_basis(np.arange(total, dtype=np.float64), n)
    ctrl, *_ = np.linalg.lstsq(basis, seq, rcond=None)
    return (basis @ ctrl)[len(history) :]


def third_difference(n: int) -> np.ndarray:
    """(n - 3, n) jerk operator: x[i+3] - 3 x[i+2] + 3 x[i+1] - x[i]."""
    d = np.zeros((max(0, n - 3), n))
    for i in range(n - 3):
        d[i, i : i + 4] = (-1.0, 3.0, -3.0, 1.0)
    return d


def lipo_fit(chunk: np.ndarray, history: np.ndarray, tail: np.ndarray | None, bound: float, blend: int) -> np.ndarray:
    """LiPo on (T, D) ``chunk``: linear blend from the previous chunk's ``tail`` over the first ``blend`` steps, then
    the smallest-jerk path through the executed ``history`` (fixed) and the chunk with every value at most ``bound`` off
    the blended reference (3 x ``bound`` in the blend). Box-constrained least squares, accelerated projected
    gradient."""
    ref = chunk.astype(np.float64).copy()
    m = 0
    if tail is not None and len(tail):
        m = min(blend, len(tail), len(ref))
        alpha = (np.arange(1, m + 1) / (m + 1))[:, None]
        ref[:m] = (1.0 - alpha) * np.asarray(tail[:m], dtype=np.float64) + alpha * ref[:m]
    seq = np.vstack([history, ref]) if len(history) else ref
    h, n = len(seq) - len(ref), len(seq)
    if n < 4:
        return ref
    d = third_difference(n)
    q = d.T @ d
    step = 1.0 / max(np.linalg.eigvalsh(q).max(), 1e-9)
    limit = np.full((n, 1), float(bound))
    limit[h : h + m] = 3.0 * bound
    limit[:h] = 0.0  # what was sent stays
    eps = np.zeros_like(seq)
    y, t = eps.copy(), 1.0
    base = q @ seq
    for _ in range(LIPO_ITERATIONS):
        new = np.clip(y - step * (base + q @ y), -limit, limit)
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = new + (t - 1.0) / t_next * (new - eps)
        eps, t = new, t_next
    return (seq + eps)[h:]


class ChunkSmoother:
    """Smooths each chunk's arm joints (env units) before execution; ``record`` the commands actually sent."""

    def __init__(self, method: str = "none", *, sigma: float = 1.0, ctrl: int = 8, history: int = 8) -> None:
        self.method = str(method or "none").lower()
        if self.method not in METHODS:
            raise ValueError(f"phoenix_smooth must be one of {METHODS}, got {method!r}")
        self.sigma = float(sigma)
        self.ctrl = int(ctrl)
        self.history_len = int(history)
        self.totals = {"chunks": 0, "max_change_rad": 0.0}
        self.reset()

    def reset(self) -> None:
        self._sent: deque[np.ndarray] = deque(maxlen=max(1, self.history_len))

    def record(self, action) -> None:
        self._sent.append(np.asarray(action, dtype=np.float64).reshape(-1)[ARM_DIMS])

    def __call__(self, chunk, state, tail=None) -> list[np.ndarray]:
        """``tail``: the previous chunk's planned but unexecuted targets (lipo blends from them)."""
        chunk = np.asarray(chunk, dtype=np.float64)
        if self.method == "none" or len(chunk) < 2:
            return list(chunk)
        arm = chunk[:, ARM_DIMS]
        if self.method == "gauss":
            anchor = self._sent[-1] if self._sent else np.asarray(state, dtype=np.float64).reshape(-1)[ARM_DIMS]
            smooth = gauss_fit(arm, anchor, self.sigma)
        else:
            history = (
                np.array(self._sent)[-self.history_len :]
                if self._sent and self.history_len > 0
                else np.zeros((0, arm.shape[1]))
            )
            if self.method == "bspline":
                smooth = bspline_fit(arm, history, self.ctrl)
            else:
                arm_tail = None if tail is None or not len(tail) else np.asarray(tail, dtype=np.float64)[:, ARM_DIMS]
                smooth = lipo_fit(arm, history, arm_tail, self.sigma / 100.0, LIPO_BLEND)
        out = chunk.copy()
        out[:, ARM_DIMS] = smooth
        self.totals["chunks"] += 1
        self.totals["max_change_rad"] = max(self.totals["max_change_rad"], float(np.abs(smooth - arm).max()))
        return list(out.astype(np.float32))

    @property
    def key(self) -> str:
        """As phoenix_smooth writes it: gauss1, bspline8, lipo1, none."""
        if self.method == "gauss":
            return f"gauss{self.sigma:g}"
        if self.method == "lipo":
            return f"lipo{self.sigma:g}"
        return f"bspline{self.ctrl}" if self.method == "bspline" else "none"

    def describe(self) -> dict:
        if self.method == "gauss":
            return {"method": "gauss", "sigma": self.sigma}
        if self.method == "bspline":
            return {"method": "bspline", "ctrl": self.ctrl, "history": self.history_len}
        if self.method == "lipo":
            return {"method": "lipo", "bound_rad": self.sigma / 100.0, "blend": LIPO_BLEND, "history": self.history_len}
        return {"method": "none"}


def _per_task(value, task: str, default):
    """A deploy_policy.yml setting given for every task, or per task ({task: value, default: value})."""
    if isinstance(value, dict):
        return value.get(task, value.get("default", default))
    return default if value in (None, "") else value


def from_usr_args(usr_args: dict) -> ChunkSmoother | None:
    """The smoother deploy_policy.yml's phoenix_smooth* settings describe for usr_args' task (None: off)."""
    task = str(usr_args.get("task_name") or "")
    return build(
        _per_task(usr_args.get("phoenix_smooth"), task, "none"),
        sigma=float(_per_task(usr_args.get("phoenix_smooth_sigma"), task, 1.0)),
        ctrl=int(_per_task(usr_args.get("phoenix_smooth_ctrl"), task, 8)),
        history=int(_per_task(usr_args.get("phoenix_smooth_history"), task, 8)),
    )


def build(value, *, sigma: float = 1.0, ctrl: int = 8, history: int = 8) -> ChunkSmoother | None:
    """The smoother a phoenix_smooth value names (None: none); its inpainting part is chunk_inpaint's."""
    from policy.phoenix.chunk_inpaint import split  # noqa: PLC0415

    method, sigma, ctrl = parse(split(value)[1], sigma, ctrl)
    return None if method == "none" else ChunkSmoother(method, sigma=sigma, ctrl=ctrl, history=history)
