# wam-toolkit

**English** | [中文](README.zh-CN.md)

Post-training, inference and inference acceleration for [Phoenix-MWA](https://github.com/Anyverse-Dynamics/Phoenix-MWA)
(a world action model that denoises a short future video and a 32-step action chunk together) on a new dual-arm
robot. Everything here wraps the upstream code; none of it is modified or copied.

- **Training**: post-train the released model on a robot it has not seen, with a new soft prompt slot.
- **Inference**: a worker process that holds the model, and an adapter that executes its action chunks.
- **Acceleration**: the same computation with fewer operator calls, 21–27 % less time per inference.
- **Asynchronous execution**: the next chunk is computed while the arm keeps moving, and is made to continue the
  running one.

**Status.** The unit tests pass and every number below was measured offline, with a trained checkpoint on recorded
camera frames and a stand-in robot. The acceleration, asynchronous execution and chunk-joining options have **not**
been compared on a robot for task success. They are off by default; compare with them off before relying on them.

## Contents

| Path | What |
|---|---|
| `policy/phoenix/fmc3_phoenix.py`, `train_fmc3.py`, `finetune.sh`, `configs/` | Post-training on a new embodiment |
| `policy/phoenix/relative_stats.py`, `robosyn_transforms.py` | The relative-joint variant |
| `policy/phoenix/build_rtc_set.py`, `rtc_init.py`, `export_rtc.py` | Data, initial weights and export for upstream's Training RTC |
| `policy/phoenix/phoenix_worker.py` | Inference worker (post-train checkpoints); acceleration; inference-time RTC |
| `policy/phoenix/rtc_worker.py` | Inference worker for Training RTC checkpoints |
| `policy/phoenix/deploy_policy.py`, `deploy_policy.yml` | Adapter: chunk execution, asynchronous chunks, joins, smoothing |
| `policy/phoenix/chunk_inpaint.py`, `chunk_smoother.py` | Chunk joins by inpainting; in-policy smoothing |
| `policy/phoenix/setup_env.sh` | The model's Python environment and the public weights |
| `tools/` | Profiling and offline comparison scripts |
| `tests/` | Tests that need no GPU and no weights |

Not included: any weights, any data. The upstream model code is referenced as a submodule, not copied. Dataset assembly (merging, per-task repeats, cutting
still tails and pauses) is in [lerobot-surgery](https://github.com/heliang-pu/lerobot-surgery). A high-level planner
that chains tasks on top of a model like this one is in
[wam-long-horizon-planner](https://github.com/heliang-pu/wam-long-horizon-planner).

## Requirements

- Upstream **Phoenix-MWA, branch `post-train`, commit `64914764a837461f9e12765e7952760cdba0ff47`**. It is a git
  submodule at `policy/phoenix/Phoenix-MWA`, pinned to that commit (`PHOENIX_ROOT` can point at another checkout).
  The code here calls upstream internals by name, so another commit may not work.
- For Training RTC only: upstream's `prometheus` branch at commit `823f009`.
- A CUDA GPU. One inference needs about 14 GB; training was set up for 80 GB cards.
- `bash policy/phoenix/setup_env.sh` builds the environment (Python 3.11, torch 2.7.1 for CUDA 12.8) with
  [uv](https://docs.astral.sh/uv/) and can download the public encoders, the Wan 2.2 VAE and the pretrained weights.

```bash
git clone --recurse-submodules https://github.com/heliang-pu/wam-toolkit.git && cd wam-toolkit
bash policy/phoenix/setup_env.sh --download-inference --download-pretrain
```

## Training

### Data

One LeRobot v2.1 dataset:

- state and action are 14-D: left arm 6 joints, left gripper, right arm 6 joints, right gripper;
- joints are absolute angles in radians, and the action of a frame is the state of the next frame;
- grippers are opening fractions (0 closed, 1 open);
- three cameras at 480×640: `cam_high`, `cam_left_wrist`, `cam_right_wrist`;
- one frame per control step (the configuration here was used at 10 Hz).

### How a new robot is added

Upstream's online-RGB recipe (`task=robotwin_posttrain`) fits this data, but it is pinned to one dataset export and
to one robot's soft prompt. `fmc3_phoenix.install()` swaps four upstream names before upstream's own `scripts/train.py`
runs, and leaves the upstream files as they are:

- the dataset reader accepts any v2.1 root and a text cache keyed by the prompt;
- the robot gets the unused soft prompt slot 18, through a versioned embodiment registry;
- initialization inherits every pretrained weight except that prompt row, which starts from a fresh N(0, 0.02) row;
- a run made this way can itself be the parent of another run, with every weight as saved.

The joints go into upstream's canonical 80-D layout by its `ConcatLeftAlign` mapping.

### Run

```bash
export PHOENIX_WS=/path/to/workspace     # holds Phoenix-MWA/, venvs/phoenix/, models/phoenix/ and the outputs
bash policy/phoenix/finetune.sh <dataset_dir> <run_id> [Hydra overrides...]
```

Stages (`PHOENIX_STAGES`, default `registry text train`):

1. `registry` writes the embodiment registry with slot 18 assigned.
2. `text` caches the T5 features of the dataset's prompts (upstream's `tools/text_cache.py`, one GPU).
3. `train` runs upstream's trainer under accelerate with DeepSpeed ZeRO-1.

Before `train`, `<output>/<run_id>/dataset_stats.json` must exist: the normalization statistics in upstream's
format. **This repository has no statistics tool.**

Multi-node: run `registry text` once, then `train` on every node with the same `run_id` and `NNODES`, `NODE_RANK`,
`MASTER_ADDR`.

Defaults in `configs/task/fmc3_posttrain.yaml`: batch 8 per GPU with 2 accumulation steps (global batch 256 on 16
GPUs), learning rate 5e-5, 2 epochs, and `fmc3.main_camera_mask_prob: 0.1` (that fraction of samples has the top
camera marked missing, so the model has seen an observation without it).

### Relative joints

`PHOENIX_TASK=fmc3_posttrain_relative` trains the 12 arm joints as `a[t+k] - s[t]` (grippers stay absolute); the
worker adds `s[t]` back. Its statistics need `relative_stats.py`: upstream's aggregate of per-episode quantiles is
close to the extreme of the deltas, which wastes the normalized range.

### Notes

- **Cut still frames.** Frames after the last motion of an episode and long pauses inside it hurt post-training and
  deployment; drop them from the dataset first.
- **No full-state resume** in this upstream release: a stopped run restarts from saved weights with a fresh optimizer
  and schedule. Save checkpoints often enough.
- **Keep the parent's normalization statistics** when continuing from one of your own runs. Recomputed statistics on
  a narrower dataset rescale the actions, and upstream clamps normalized values at ±5.
- 16 samples per GPU does not fit an 80 GB card without gradient checkpointing; 8 with 2 accumulation steps does.

### Training RTC (upstream's `prometheus` branch)

Chunks conditioned on an already executed prefix need upstream's other trainer. The pieces here:

1. `build_rtc_set.py` converts a v2.1 set to the native LeRobot v3 layout that trainer reads (`--crop-square` keeps
   the centred square the post-train reader saw).
2. `rtc_init.py` turns post-train or pretrained weights into initial weights that trainer accepts.
3. Train with upstream's own stages.
4. `export_rtc.py` collects the run into a directory `rtc_worker.py` loads.

## Inference

### Worker

`phoenix_worker.py` runs under the model's interpreter and answers one JSON request per line on stdin:

```text
{"cmd": "act", "obs_path": "<npz>", "prompt": "<trained prompt>"}  ->  {"ok": true, "actions": [[14 floats] x steps]}
```

The npz holds the three camera images (uint8 HWC RGB) and the 14-D state. Actions are absolute joint targets with
gripper opening fractions. Optional fields select the whole 32-step chunk, a noise seed, a camera encoded as missing,
steps to hold by inpainting, and RTC guidance; the file's docstring lists them. It does not decode the predicted
video, which nearly halves the time of an inference and leaves the actions unchanged.

### Adapter

`deploy_policy.py` starts the worker and offers `get_model(usr_args)`, `eval(env, model, obs)` and
`reset_model(model)`, the policy interface of the RoboSynChallenge evaluation harness. `eval` executes one chunk's
steps on `env`, which needs `step(action)`, `get_wrapper_attr("is_task_success")`, and observations of the form
`obs["sensor"][camera]["color"]` and `obs["robot"]["qpos"]`.

```python
from policy.phoenix import deploy_policy

model = deploy_policy.get_model({
    "checkpoint_dir": "/path/to/run",          # config.yaml, dataset_stats.json, checkpoints/weights/, text_cache/
    "phoenix_encoders": "/path/to/encoders",   # csvae.pt, dino110k.pt
    "phoenix_wan_base": "/path/to/wan_base",
    "phoenix_prompt": "Click the bell",
    "phoenix_steps": 20,
})
deploy_policy.reset_model(model)
obs, info, truncated, inference_times = deploy_policy.eval(env, model, obs)
```

The options are documented in `deploy_policy.yml`:

| Option | What it does |
|---|---|
| `phoenix_steps` | Steps of each 32-step chunk executed before the next inference |
| `phoenix_denoise_steps` | Denoising steps (video and actions share one schedule) |
| `phoenix_seed` | Noise seed of every chunk; -1 draws fresh noise each time |
| `phoenix_stall_steps`, `phoenix_stall_fix` | Detect chunks requested after no progress and draw them differently |
| `phoenix_compile` | Acceleration (below) |
| `phoenix_async`, `phoenix_inference_delay` | Asynchronous chunks (below) |
| `phoenix_rtc_infer` | Join asynchronous chunks at inference (below) |
| `phoenix_rtc_prefix` | Hand a Training RTC checkpoint the steps it must continue |
| `phoenix_smooth` | Smooth each chunk before execution, or inpaint synchronous chunks |
| `phoenix_sampling` | Draw N chunks and execute the one that best continues the last |

## Acceleration

`phoenix_compile: true` (worker flag `--compile default`). Nothing is quantized and no step is removed.

### Where the time goes

One inference with 5 denoising steps on an RTX PRO 6000 Blackwell, before any change (about 290–330 ms):

| Part | Time | |
|---|---|---|
| MoT trunk | 221–247 ms | 5 steps × 30 layers; 576 video tokens and 32 action tokens |
| Current-frame encoder | 54–63 ms | DINO backbone 39, Wan VAE 14, CSVAE 4 |
| Image resize on the CPU | 7–8 ms | |
| Everything else | 10–15 ms | |

The trunk is not limited by matrix products. Its linear layers alone, timed separately at the same shapes, take
22 ms per forward against 44 ms measured; the profiler counts about 64 000 operator calls per inference. The rest is
small elementwise kernels (float32 RMS norms, modulation, gates, casts) and the cost of launching them.

### What the option does

1. **Trunk.** Each block's two halves (upstream's `_build_expert_attention_io` and `_apply_expert_post_block`) get
   their own `torch.compile` frame. Upstream's functions are traced as they are; the FlexAttention call between them
   is left alone. Upstream's own `mot_compile_mode` applies to training only.
2. **DINO backbone** compiled the same way.
3. **Repeated frames once.** Upstream's online path repeats the current frame nine times before DINO, which then
   encodes nine identical images per camera. They are encoded once and the result is repeated.

### Result

| GPU | Before | After | |
|---|---|---|---|
| RTX 4090 | 423 ms | 333 ms | −21 % |
| RTX PRO 6000 Blackwell | 324 ms | 236 ms | −27 % (trunk only: 268 ms) |

The first inference compiles (13–21 s); the adapter does it at start on a blank observation. Peak memory is unchanged.

### How much the output changes

Without the option, two runs give bit-identical chunks. With it the fused kernels round differently and the
difference is carried through 30 layers and 5 steps. On 20 recorded frames each of two tasks, same noise:

| | Compiled vs not | A new noise seed |
|---|---|---|
| Arm joints, mean difference | 0.0010–0.0015 rad | 0.008–0.018 rad |
| Arm joints, largest difference in a chunk (median over chunks) | 0.008 rad | 0.03–0.09 rad |
| Gripper opening, mean difference | 0.0007–0.0016 | 0.008–0.010 |

So the change is about a tenth of what drawing new noise does. Whether it affects task success has not been
measured.

### Notes

- `mode="reduce-overhead"` (CUDA graphs) does not work with this split: each layer calls one compiled function per
  expert, and the second call invalidates the first one's outputs.
- Not done: the current frame's tokens only attend to themselves, so their computation is the same in all five
  denoising steps and could be cached. That needs a forward path of its own.

## Asynchronous execution and joining chunks

With `phoenix_async: real` the next chunk is computed in the background once `phoenix_steps` (s) of the running one
have executed, and takes over `phoenix_inference_delay` (d) steps later. A chunk therefore runs from its index d to
d + s + d: keep **s + 2d ≤ 32**. On the stand-in robot at 10 Hz a step took 0.12 s on average instead of 0.134 s,
and no control period was missed.

A plain asynchronous chunk does not know what the arm does meanwhile, so it may not start where the old one was
going. `phoenix_rtc_infer` makes it continue the running chunk: the d steps that execute meanwhile at weight 1, the
rest of the overlap by the soft mask of RTC ([Black et al., 2025](https://arxiv.org/abs/2506.07339)).

- `inpaint` replaces the clean estimate of the overlapping steps at every denoising step. No extra inference time.
- `guidance` is RTC's own algorithm: upstream's implementation of the guidance, put around the joint denoising loop
  it was not connected to. One backward pass per denoising step, about twice the inference time.

Largest joint step at the takeover, median / maximum in rad, with s = 14 and d = 6 (18 steps of the running chunk
left, 12 of them soft), on recorded frames of two tasks:

| | Task A (8 launches) | Task B (20 launches) |
|---|---|---|
| No joining | 0.114 / 0.320 | 0.031 / 0.160 |
| `inpaint` | 0.033 / 0.059 | 0.014 / 0.066 |
| `guidance` | 0.107 / 0.233 | 0.031 / 0.149 |
| For scale: neighbouring steps inside a chunk | 0.030 / 0.072 | 0.017 / 0.050 |

Two findings:

- **`inpaint` needs soft steps.** With s = 20 and d = 6 the running chunk has only the 6 hard steps left, and the
  jump simply moves to the takeover (0.092 / 0.280 on task A). To re-plan every 20 steps, use s = 14 and d = 6.
- **`guidance` hardly moves the chunk on these checkpoints.** The guidance term is proportional to how much the
  model's clean-action estimate follows its noisy action input, and here it barely does until the noise level is
  below 0.1: the cosine between the gradient and the error is within ±0.11 at the five steps used. With guidance
  weight 0 the result equals the unguided one exactly, so the wiring is right; the method does not suit a model whose
  actions are this strongly determined by the observation and the jointly denoised video.

## Tools

All take the `policy/phoenix` directory, a run directory, a step, a frames file from `make_frames.py`, a prompt and
the encoder and Wan base directories; each file's docstring has its command line.

| Script | What |
|---|---|
| `make_frames.py` | Frames of one episode of a v2.1 recording |
| `profile_infer.py`, `profile_ops.py` | Time per stage; per-operator CUDA time |
| `gemm_floor.py` | What the GPU reaches on the trunk's matrix products alone |
| `compare_compiled.py` | Time and output difference, compiled against not |
| `seed_vs_compile.py` | That difference next to a new noise seed |
| `rtc_offline.py` | Joins: none, inpaint, guidance |
| `fake_robot_loop.py` | Adapter + real worker on a stand-in robot at 10 Hz |

## Tests

```bash
python -m pytest -q tests
```

Six tests; two import upstream code and are skipped when the submodule is not checked out.

## Limitations

- No robot comparison of task success for `phoenix_compile`, `phoenix_async` or `phoenix_rtc_infer`.
- The normalization statistics step is not included.
- Tied to one upstream commit and to a 14-D dual-arm joint layout with three cameras.
