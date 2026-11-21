# phoenix-wam-realtime-toolkit

**English** | [中文](README.zh-CN.md)

Post-training, inference and inference acceleration for [Phoenix-MWA](https://github.com/Anyverse-Dynamics/Phoenix-MWA)
(a world action model that denoises a short future video and a 32-step action chunk together) on a new dual-arm
robot. Everything here wraps the upstream code; none of it is modified or copied.

- **Training**: post-train the released model on a robot it has not seen, with a new soft prompt slot.
- **Inference**: a worker process that holds the model, and an adapter that executes its action chunks.
- **Acceleration**: the same computation with fewer operator calls and no repeated work, a third less time per
  inference.
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
git clone --recurse-submodules https://github.com/heliang-pu/phoenix-wam-realtime-toolkit.git && cd phoenix-wam-realtime-toolkit
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
| `phoenix_steps` | Steps of each 32-step chunk that are executed: the re-planning period, synchronous or not |
| `phoenix_denoise_steps` | Denoising steps (video and actions share one schedule) |
| `phoenix_seed` | Noise seed of every chunk; -1 draws fresh noise each time |
| `phoenix_stall_steps`, `phoenix_stall_fix` | Detect chunks requested after no progress and draw them differently |
| `phoenix_compile`, `phoenix_frame_cache` | Acceleration (below) |
| `phoenix_async`, `phoenix_inference_delay` | Asynchronous chunks (below) |
| `phoenix_rtc_infer` | Join asynchronous chunks at inference (below) |
| `phoenix_rtc_prefix` | Hand a Training RTC checkpoint the steps it must continue |
| `phoenix_smooth` | Smooth each chunk before execution, or inpaint synchronous chunks |
| `phoenix_sampling` | Draw N chunks and execute the one that best continues the last |

## Acceleration

Two independent options: `phoenix_compile: true` (worker flag `--compile default`) and `phoenix_frame_cache: true`
(`--frame-cache`). Nothing is quantized and no denoising step is removed.

To tell the parts of `phoenix_compile` apart, the worker also takes `--compile trunk` (the MoT blocks only) and
`--compile encoder` (the DINO backbone and the repeated frame only). Together with `--frame-cache` alone, that makes
three single-acceleration settings to compare against none.

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

### What `phoenix_compile` does

1. **Trunk.** Each block's two halves (upstream's `_build_expert_attention_io` and `_apply_expert_post_block`) get
   their own `torch.compile` frame. Upstream's functions are traced as they are; the FlexAttention call between them
   is left alone. Upstream's own `mot_compile_mode` applies to training only.
2. **DINO backbone** compiled the same way.
3. **Repeated frames once.** Upstream's online path repeats the current frame nine times before DINO, which then
   encodes nine identical images per camera. They are encoded once and the result is repeated.

### What `phoenix_frame_cache` does

The video sequence is [current frame, noisy future]. Upstream's attention mask lets the current frame's tokens, a
third of the video tokens, attend only to themselves, and their timestep is 0 at every step. Their hidden states, and
the keys and values the other tokens read from them, are therefore the same in all five denoising steps, yet they are
computed five times.

With the option, the first step of an inference runs in full and keeps each layer's keys and values of those tokens.
The other steps run upstream's own forward on the future and action tokens only, and the mixed attention gets the kept
keys and values in front of theirs, under the same rows of upstream's mask. Upstream's forward is not rewritten: three
hooks, on `mot.forward`, `_build_expert_attention_io` and `_mixed_attention`.

A step uses the cache only when its current-frame inputs (tokens, time modulation, RoPE, text context) and the mask
equal the kept ones bit for bit, checked in one GPU round trip per step. Anything else runs in full: a new frame, a
padded camera, a backward pass (RTC guidance). Upstream has this idea for a whole static video (`kv_cache.py`), not
for part of the video expert.

### Result

One inference with 5 denoising steps, median of 12 recorded frames:

| GPU | Neither | Frame cache | Compile | Both | |
|---|---|---|---|---|---|
| RTX 4090 | 419–423 ms | 368 ms | 330–333 ms | **280 ms** | −34 % |
| RTX PRO 6000 Blackwell | 324 ms | 298 ms | 236 ms | **214 ms** | −34 % |

The first inference compiles; the adapter does it at start on a blank observation (about 14 s for compile alone, 22 s
with the frame cache, whose second token shape is compiled too). Peak memory is unchanged.

### How much the output changes

Without the option, two runs give bit-identical chunks. With it the fused kernels round differently and the
difference is carried through 30 layers and 5 steps. On 20 recorded frames each of two tasks, same noise:

| | Compiled vs not | A new noise seed |
|---|---|---|
| Arm joints, mean difference | 0.0010–0.0015 rad | 0.008–0.018 rad |
| Arm joints, largest difference in a chunk (median over chunks) | 0.008 rad | 0.03–0.09 rad |
| Gripper opening, mean difference | 0.0007–0.0016 | 0.008–0.010 |

With both options on (12 frames per row, against neither):

| | Both options | A new noise seed |
|---|---|---|
| RTX PRO 6000, task A: arm joints, mean difference | 0.0016 rad | 0.019 rad |
| RTX PRO 6000, task A: gripper opening, mean difference | 0.0020 | 0.011 |
| RTX 4090, task B: arm joints, mean difference | 0.0007 rad | 0.0063 rad |
| RTX 4090, task B: gripper opening, mean difference | 0.0005 | 0.0093 |

The frame cache alone changes less than compiling does (0.0003–0.0007 rad), and with it taken off the chunks are bit
for bit the original ones. In a float64 stand-in of the trunk (in the tests) the cached result equals the full
forward to 1e-10: the real model's difference is rounding.

So the change is about a tenth of what drawing new noise does. Whether it affects task success has not been
measured.

### Notes

- `--compile reduce-overhead` (CUDA graph replay) needs one thing with this split: each layer calls one compiled
  function per expert, and by default the second call frees the first one's outputs. The worker therefore marks the
  graph-tree steps itself, one per MoT call (`torch.compiler.cudagraph_mark_step_begin()`). It then gives the same
  chunks, but on the RTX 4090 it was no faster than `--compile default` (the inference is bound by its matrix
  products there, not by kernel launches) and the warm-up took about 100 s. `phoenix_compile` does not select it.
- A first version of the frame cache compared the mask once per layer; those 120 GPU round trips per inference
  cancelled the whole gain. Checks that force a synchronization belong once per step, not per layer.
- The adapter's side is not worth optimizing: encoding the observation, the npz file and the pipe take about 5 ms
  together.
- What is left without changing the computation is small (the text context's keys and values, the image resize on
  the CPU, the VAE encoder: perhaps 10 ms). Beyond that the computation itself has to change: fewer video denoising
  steps, distillation.

## Asynchronous execution and joining chunks

With `phoenix_async: real` the next chunk is computed in the background while the arm keeps executing.
`phoenix_steps` keeps its meaning, the steps of each chunk that are executed, so it is the re-planning period: the
next chunk is launched `phoenix_inference_delay` (d) steps before the running one has executed `phoenix_steps`, and
takes over d steps later. A chunk therefore runs from its index d to d + `phoenix_steps`:
**d < `phoenix_steps` ≤ 32 − d**, and the adapter refuses anything else. When a chunk is launched, the running one
has 32 − `phoenix_steps` steps left. On the stand-in robot at 10 Hz a step took 0.12 s on average instead of 0.134 s,
and no control period was missed.

A plain asynchronous chunk does not know what the arm does meanwhile, so it may not start where the old one was
going. `phoenix_rtc_infer` makes it continue the running chunk: the d steps that execute meanwhile at weight 1, the
rest of the overlap by the soft mask of RTC ([Black et al., 2025](https://arxiv.org/abs/2506.07339)).

- `inpaint` replaces the clean estimate of the overlapping steps at every denoising step. No extra inference time.
- `guidance` is RTC's own algorithm: upstream's implementation of the guidance, put around the joint denoising loop
  it was not connected to. One backward pass per denoising step, about twice the inference time.

Largest joint step at the takeover, median / maximum in rad, with `phoenix_steps` 14 and d = 6 (18 steps of the
running chunk left, 12 of them soft), on recorded frames of two tasks:

| | Task A (8 launches) | Task B (20 launches) |
|---|---|---|
| No joining | 0.114 / 0.320 | 0.031 / 0.160 |
| `inpaint` | 0.033 / 0.059 | 0.014 / 0.066 |
| `guidance` | 0.107 / 0.233 | 0.031 / 0.149 |
| For scale: neighbouring steps inside a chunk | 0.030 / 0.072 | 0.017 / 0.050 |

Two findings:

- **`inpaint` needs soft steps**, 32 − `phoenix_steps` − d of them. With `phoenix_steps` 20 and d = 6 there are 6,
  and it still works (task A 0.038 / 0.086 against 0.090 / 0.283 without joining, task B 0.018 / 0.069 against
  0.035 / 0.252). With `phoenix_steps` 26 and d = 6 the running chunk has only the 6 hard steps left, and the jump
  simply moves to the takeover (0.092 / 0.280 on task A).
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
| `compare_frame_cache.py` | The same for the frame cache, alone and with compile |
| `seed_vs_compile.py` | That difference next to a new noise seed |
| `rtc_offline.py` | Joins: none, inpaint, guidance |
| `fake_robot_loop.py` | Adapter + real worker on a stand-in robot at 10 Hz |

## Tests

```bash
python -m pytest -q tests
```

Seven tests; two import upstream code and are skipped when the submodule is not checked out.

## Limitations

- No robot comparison of task success for `phoenix_compile`, `phoenix_frame_cache`, `phoenix_async` or
  `phoenix_rtc_infer`.
- The normalization statistics step is not included.
- Tied to one upstream commit and to a 14-D dual-arm joint layout with three cameras.
