"""The adapter (deploy_policy.eval_async) with the real worker and model on a stand-in robot at 10 Hz (no hardware):
observations are a recorded episode's frames (the newest one at or before the current step), commands are only
recorded. Prints the adapter's own lines, then per setting: warm-up, takeover delays, late chunks and the step sizes of
the executed commands at the takeovers and elsewhere.

    python tools/fake_robot_loop.py <repo root> <run dir> <step> <frames.npz> <prompt> <phoenix python> <encoders> <wan base> <out.json>
"""
import json
import sys
import time
import types

import numpy as np
import torch

tree, run_dir, step, frames, prompt, python, encoders, wan_base, out = sys.argv[1:10]
sys.path.insert(0, tree)
from policy.phoenix import deploy_policy as adapter  # noqa: E402

data = dict(np.load(frames))  # unpacked once
rows = data["rows"]
ARM = adapter.ARM_DIMS


class Env:
    _current_instruction = ""

    def __init__(self):
        self.unwrapped = types.SimpleNamespace(single_action_space=types.SimpleNamespace(shape=(14,)), device="cpu")
        self.commands, self.times, self.next_tick = [], [], None

    def obs(self):
        i = max(0, int(np.searchsorted(rows, len(self.commands) + int(rows[0]), side="right") - 1))  # step 0 = the first row
        return {"sensor": {c: {"color": data[c][i][None]} for c in adapter.CAMERAS},
                "robot": {"qpos": torch.from_numpy(data["state"][i][None])}}

    def get_wrapper_attr(self, name):
        return lambda: False

    def step(self, action):
        now = time.monotonic()
        self.next_tick = now if self.next_tick is None else self.next_tick
        time.sleep(max(0.0, self.next_tick - now))
        self.next_tick += 0.1
        self.commands.append(action[0].numpy().copy())
        self.times.append(time.monotonic())
        return self.obs(), 0.0, False, False, {}


results = {}
SETTINGS = [
    ("sync s20", {"phoenix_steps": 20}),
    ("async period 20 d6", {"phoenix_steps": 20, "phoenix_async": "real", "phoenix_inference_delay": 6}),
    ("async period 20 d6 inpaint", {"phoenix_steps": 20, "phoenix_async": "real", "phoenix_inference_delay": 6,
                                    "phoenix_rtc_infer": "inpaint"}),
    ("async period 26 d6 inpaint", {"phoenix_steps": 26, "phoenix_async": "real", "phoenix_inference_delay": 6,
                                    "phoenix_rtc_infer": "inpaint"}),
    ("async period 20 d10 guidance", {"phoenix_steps": 20, "phoenix_async": "real", "phoenix_inference_delay": 10,
                                      "phoenix_rtc_infer": "guidance"}),
    ("sync s20 compile + frame cache", {"phoenix_steps": 20, "phoenix_compile": True, "phoenix_frame_cache": True}),
    ("async period 20 d6 inpaint compile + frame cache", {
        "phoenix_steps": 20, "phoenix_async": "real", "phoenix_inference_delay": 6, "phoenix_rtc_infer": "inpaint",
        "phoenix_compile": True, "phoenix_frame_cache": True}),
]
for name, extra in SETTINGS:
    usr = {"checkpoint_dir": run_dir, "phoenix_step": int(step), "phoenix_python": python, "phoenix_prompt": prompt,
           "phoenix_encoders": encoders, "phoenix_wan_base": wan_base, "phoenix_seed": 0, "phoenix_denoise_steps": 5,
           "phoenix_cuda_visible_devices": 0, "task_name": "t", **extra}
    started = time.monotonic()
    model = adapter.get_model(usr)
    ready = time.monotonic() - started
    env = Env()
    obs = env.obs()
    adapter.reset_model(model)
    begin, infer = time.monotonic(), []
    while len(env.commands) < 150:
        obs, _info, _truncated, times = adapter.eval(env, model, obs)
        infer += [float(t) for t in times]
    wall = time.monotonic() - begin
    commands = np.array(env.commands)
    steps = np.abs(np.diff(commands, axis=0))[:, ARM].max(1)
    timeline = getattr(model, "timeline", None)
    record = {"ready_s": ready, "wall_s": wall, "steps": len(commands), "infer_s_median": float(np.median(infer)),
              "infer_s_max": float(max(infer)), "inferences": len(infer),
              "tick_s_max": float(np.diff(env.times).max()), "ticks_over_0.15s": int((np.diff(env.times) > 0.15).sum()),
              "step_rad_median": float(np.median(steps)), "step_rad_p99": float(np.quantile(steps, 0.99)),
              "step_rad_max": float(steps.max()), "largest_steps_at": np.argsort(steps)[-5:][::-1].tolist(),
              "largest_steps": np.sort(steps)[-5:][::-1].round(4).tolist()}
    if timeline is not None:
        record["async"] = model.describe_async()
        record["timeline"] = timeline.describe()
    else:  # synchronous: the joins are every phoenix_steps
        joins = np.arange(usr["phoenix_steps"], len(commands), usr["phoenix_steps"]) - 1
        record["join_steps"] = steps[joins].round(4).tolist()
    results[name] = record
    print("RESULT", name, json.dumps(record), flush=True)
    adapter.close_model(model)
    time.sleep(2)
open(out, "w").write(json.dumps(results, indent=1))
print("ASYNC_LOOP_DONE", flush=True)
