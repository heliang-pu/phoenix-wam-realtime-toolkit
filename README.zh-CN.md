# phoenix-wam-realtime-toolkit

[English](README.md) | **中文**

[Phoenix-MWA](https://github.com/Anyverse-Dynamics/Phoenix-MWA)（一个世界动作模型：把一小段未来视频和 32 步动作一起去噪）在一台新的双臂机器人上的后训练、推理和推理加速。
这里的代码都是包在上游代码外面的，没有修改或拷贝上游的任何文件。

- **训练**：把发布的模型后训练到一台它没见过的机器人上，用一个新的软提示槽位。
- **推理**：一个装着模型的推理进程，加一个负责执行动作段的适配器。
- **加速**：算的东西不变，算子调用更少、不做重复计算，一次推理的时间少三分之一。
- **异步执行**：机械臂继续动的同时算下一段动作，并让它接着正在执行的这一段。

**现状。** 单元测试全部通过；下面所有数字都是离线量的，用训练好的权重、录下来的相机画面和一个假机器人。
加速、异步执行和动作段衔接这几个选项**还没有**在机器人上比较过任务成功率。它们默认都是关的，用之前请先和关闭时对比。

## 目录

| 路径 | 内容 |
|---|---|
| `policy/phoenix/fmc3_phoenix.py`、`train_fmc3.py`、`finetune.sh`、`configs/` | 在新机器人上做后训练 |
| `policy/phoenix/relative_stats.py`、`robosyn_transforms.py` | 相对关节角的训练变体 |
| `policy/phoenix/build_rtc_set.py`、`rtc_init.py`、`export_rtc.py` | 上游 Training RTC 用的数据、初始权重和导出 |
| `policy/phoenix/phoenix_worker.py` | 推理进程（后训练权重）；加速；推理时 RTC |
| `policy/phoenix/rtc_worker.py` | Training RTC 权重的推理进程 |
| `policy/phoenix/deploy_policy.py`、`deploy_policy.yml` | 适配器：执行动作段、异步、衔接、平滑 |
| `policy/phoenix/chunk_inpaint.py`、`chunk_smoother.py` | 用补全的办法衔接动作段；策略内平滑 |
| `policy/phoenix/setup_env.sh` | 模型的 Python 环境和公开权重 |
| `tools/` | 测速和离线对比脚本 |
| `tests/` | 不需要 GPU 和权重的测试 |

不包含：任何权重、任何数据。上游模型代码以子模块的方式引用，没有拷贝。数据集的整理（合并、按任务重复、切掉静止的尾巴和停顿）
在 [lerobot-surgery](https://github.com/heliang-pu/lerobot-surgery)。在这类模型上面串联多个任务的高层规划器在
[wam-long-horizon-planner](https://github.com/heliang-pu/wam-long-horizon-planner)。

## 依赖

- 上游 **Phoenix-MWA，`post-train` 分支，提交 `64914764a837461f9e12765e7952760cdba0ff47`**。它是 `policy/phoenix/Phoenix-MWA`
  这个 git 子模块，固定在这个提交（也可以用 `PHOENIX_ROOT` 指向别处的检出）。这里的代码按名字调用上游的内部函数，换一个提交不一定能用。
- 只有 Training RTC 需要：上游 `prometheus` 分支的提交 `823f009`。
- 一块 CUDA 显卡。一次推理约占 14 GB；训练是按 80 GB 的卡配置的。
- `bash policy/phoenix/setup_env.sh` 用 [uv](https://docs.astral.sh/uv/) 建环境（Python 3.11，CUDA 12.8 的 torch 2.7.1），
  还可以下载公开的编码器、Wan 2.2 VAE 和预训练权重。

```bash
git clone --recurse-submodules https://github.com/heliang-pu/phoenix-wam-realtime-toolkit.git && cd phoenix-wam-realtime-toolkit
bash policy/phoenix/setup_env.sh --download-inference --download-pretrain
```

## 训练

### 数据

一个 LeRobot v2.1 数据集：

- 状态和动作都是 14 维：左臂 6 个关节、左夹爪、右臂 6 个关节、右夹爪；
- 关节是绝对角度（弧度），每一帧的动作就是下一帧的状态；
- 夹爪是开口比例（0 闭合，1 张开）；
- 三路 480×640 的相机：`cam_high`、`cam_left_wrist`、`cam_right_wrist`；
- 每个控制步一帧（这套配置是按 10 Hz 用的）。

### 新机器人是怎么加进去的

上游的在线 RGB 配方（`task=robotwin_posttrain`）适用于这种数据，但它写死了只认一份数据导出和一台机器人的软提示。
`fmc3_phoenix.install()` 在上游自己的 `scripts/train.py` 运行之前换掉上游的四个名字，上游的文件保持原样：

- 数据读取器可以读任意一个 v2.1 目录，文本缓存按提示词取键；
- 这台机器人拿到没人用的 18 号软提示槽位，通过一份带版本的机体注册表登记；
- 初始化继承全部预训练权重，只有这一行提示从全新的 N(0, 0.02) 开始；
- 这样训出来的一次训练本身也可以当下一次训练的起点，全部权重原样继承。

关节按上游的 `ConcatLeftAlign` 映射填进它的 80 维标准布局。

### 运行

```bash
export PHOENIX_WS=/path/to/workspace     # 里面有 Phoenix-MWA/、venvs/phoenix/、models/phoenix/ 和输出目录
bash policy/phoenix/finetune.sh <数据集目录> <run_id> [Hydra 覆盖项...]
```

阶段（`PHOENIX_STAGES`，默认 `registry text train`）：

1. `registry`：写出登记了 18 号槽位的机体注册表。
2. `text`：缓存数据集里各条提示词的 T5 特征（上游的 `tools/text_cache.py`，一块 GPU）。
3. `train`：用 accelerate 加 DeepSpeed ZeRO-1 跑上游的训练器。

`train` 之前，`<输出>/<run_id>/dataset_stats.json` 必须已经存在：上游格式的归一化统计。**这个仓库里没有统计工具。**

多机训练：`registry text` 只跑一次，然后每台机器用同一个 `run_id` 加 `NNODES`、`NODE_RANK`、`MASTER_ADDR` 跑 `train`。

`configs/task/fmc3_posttrain.yaml` 的默认值：每卡 batch 8、梯度累积 2（16 卡时总 batch 256），学习率 5e-5，2 个 epoch，
`fmc3.main_camera_mask_prob: 0.1`（这个比例的样本把顶部相机标成缺失，让模型见过没有它的观测）。

### 相对关节角

`PHOENIX_TASK=fmc3_posttrain_relative` 把 12 个手臂关节训练成 `a[t+k] - s[t]`（夹爪仍是绝对量），推理进程会把 `s[t]` 加回去。
它的统计要用 `relative_stats.py`：上游按每个 episode 的分位数再取极值，接近增量的极端值，会浪费归一化后的取值范围。

### 注意事项

- **切掉静止帧。** 每个 episode 最后一次运动之后的帧、中间的长停顿，都会伤害后训练和部署，先从数据集里去掉。
- 上游这一版**不支持完整续训**：中途停了只能从存下的权重重新开始，优化器和学习率调度都是新的。权重要存得足够勤。
- 从自己的某次训练接着训时，**沿用它的归一化统计**。在更窄的数据上重新算统计会改变动作的尺度，而上游把归一化后的值截断在 ±5。
- 不开梯度检查点时，80 GB 的卡每卡放不下 16 个样本；每卡 8 个加 2 步累积可以。

### Training RTC（上游的 `prometheus` 分支）

让动作段以"已经执行的前缀"为条件，需要上游的另一套训练器。这里提供的是配套的几步：

1. `build_rtc_set.py` 把 v2.1 数据集转成那套训练器读的原生 LeRobot v3 布局（`--crop-square` 保留后训练读取器看到的居中正方形）。
2. `rtc_init.py` 把后训练权重或预训练权重变成那套训练器接受的初始权重。
3. 用上游自己的各个阶段训练。
4. `export_rtc.py` 把一次训练整理成 `rtc_worker.py` 能加载的目录。

## 推理

### 推理进程

`phoenix_worker.py` 在模型自己的解释器下运行，从标准输入每行读一条 JSON 请求：

```text
{"cmd": "act", "obs_path": "<npz>", "prompt": "<训练时的提示词>"}  ->  {"ok": true, "actions": [[14 个数] x 步数]}
```

npz 里是三路相机图像（uint8，HWC，RGB）和 14 维状态。动作是绝对关节目标，夹爪是开口比例。可选字段可以要整段 32 步、指定噪声种子、
把某路相机当作缺失、用补全保持开头几步、以及 RTC 引导，文件开头的说明里都列了。它不解码预测出来的视频，这让一次推理的时间接近减半，动作不变。

### 适配器

`deploy_policy.py` 启动推理进程，提供 `get_model(usr_args)`、`eval(env, model, obs)` 和 `reset_model(model)`，
也就是 RoboSynChallenge 评测程序的策略接口。`eval` 在 `env` 上执行一段动作里的若干步；`env` 需要有 `step(action)`、
`get_wrapper_attr("is_task_success")`，观测的形式是 `obs["sensor"][相机]["color"]` 和 `obs["robot"]["qpos"]`。

```python
from policy.phoenix import deploy_policy

model = deploy_policy.get_model({
    "checkpoint_dir": "/path/to/run",          # config.yaml、dataset_stats.json、checkpoints/weights/、text_cache/
    "phoenix_encoders": "/path/to/encoders",   # csvae.pt、dino110k.pt
    "phoenix_wan_base": "/path/to/wan_base",
    "phoenix_prompt": "Click the bell",
    "phoenix_steps": 20,
})
deploy_policy.reset_model(model)
obs, info, truncated, inference_times = deploy_policy.eval(env, model, obs)
```

各个选项在 `deploy_policy.yml` 里有说明：

| 选项 | 作用 |
|---|---|
| `phoenix_steps` | 每段 32 步动作里执行多少步，也就是规划间隔（同步、异步都是这个意思） |
| `phoenix_denoise_steps` | 去噪步数（视频和动作共用一个步数） |
| `phoenix_seed` | 每段动作的噪声种子；-1 表示每段换一份新噪声 |
| `phoenix_stall_steps`、`phoenix_stall_fix` | 发现"没有进展之后请求的动作段"，并换一种方式生成它 |
| `phoenix_compile`、`phoenix_frame_cache` | 加速（见下） |
| `phoenix_async`、`phoenix_inference_delay` | 异步执行（见下） |
| `phoenix_rtc_infer` | 推理时衔接异步的动作段（见下） |
| `phoenix_rtc_prefix` | 把必须接着的那几步交给 Training RTC 权重 |
| `phoenix_smooth` | 执行前平滑每段动作，或用补全衔接同步的动作段 |
| `phoenix_sampling` | 生成 N 段，执行最接得上上一段的那一段 |

## 加速

两个互相独立的选项：`phoenix_compile: true`（推理进程的参数是 `--compile default`）和 `phoenix_frame_cache: true`（`--frame-cache`）。
想分清 `phoenix_compile` 里是哪一部分起作用，推理进程还接受 `--compile trunk`（只编译 MoT 主干）和 `--compile encoder`（只编译 DINO、重复帧只算一次）；
加上单独的 `--frame-cache`，一共三种"只开一项加速"的设置，可以分别和不开对比。
没有做量化，也没有减少去噪步数。

### 时间花在哪

RTX PRO 6000 Blackwell 上去噪 5 步的一次推理，改动之前（约 290–330 ms）：

| 环节 | 时间 | |
|---|---|---|
| MoT 主干 | 221–247 ms | 5 步 × 30 层；576 个视频 token 和 32 个动作 token |
| 当前帧编码 | 54–63 ms | DINO 主干 39，Wan VAE 14，CSVAE 4 |
| CPU 上缩放图像 | 7–8 ms | |
| 其余 | 10–15 ms | |

主干并不是被矩阵乘卡住的。把它的线性层按同样的形状单独计时，一次前向 22 ms，而实际量到的是 44 ms；分析器数出每次推理约 64,000 次算子调用。
其余的时间花在零碎的逐元素小算子（float32 的 RMS 归一化、调制、门控、类型转换）和启动它们的开销上。

### `phoenix_compile` 做了什么

1. **主干。** 每个块的前后两段（上游的 `_build_expert_attention_io` 和 `_apply_expert_post_block`）各套一层 `torch.compile`。
   上游的函数原样被追踪，两段之间的 FlexAttention 调用不动。上游自带的 `mot_compile_mode` 只在训练时生效。
2. **DINO 主干**，同样的办法编译。
3. **重复帧只算一次。** 上游的在线路径把当前帧重复 9 份再送进 DINO，于是每路相机要编码 9 张完全相同的图。改成只编码一次，再把结果复制。

### `phoenix_frame_cache` 做了什么

视频序列是 [当前帧, 带噪的未来帧]。上游的注意力掩码让当前帧的 token（占视频 token 的三分之一）只关注自己，它们的时间步在每一步都是 0。
所以它们的隐状态、以及其他 token 从它们那里读到的键和值，在 5 个去噪步里完全相同，却被算了 5 遍。

开了这个选项后，每次推理的第一步照常完整计算，并留下每一层里这些 token 的键和值。其余的步只让未来帧和动作的 token 走上游自己的前向，
在混合注意力那里把留下的键和值接在它们的前面，掩码用上游掩码里对应的那些行。上游的前向没有被重写：只在
`mot.forward`、`_build_expert_attention_io`、`_mixed_attention` 三处挂了钩子。

一个去噪步只有在当前帧的输入（token、时间调制、RoPE、文本条件）和掩码与留下的那份逐位相同时才用缓存，每步核对一次（一次 GPU 往返）。
其他情况一律完整计算：换了新的一帧、有相机缺失、有反向传播（RTC 引导）。上游对"整段静态视频"有这个做法（`kv_cache.py`），
对视频专家的一部分没有。

### 结果

去噪 5 步的一次推理，12 帧录制画面的中位数：

| 显卡 | 都不开 | 当前帧缓存 | 编译 | 两个都开 | |
|---|---|---|---|---|---|
| RTX 4090 | 419–423 ms | 368 ms | 330–333 ms | **280 ms** | −34 % |
| RTX PRO 6000 Blackwell | 324 ms | 298 ms | 236 ms | **214 ms** | −34 % |

第一次推理要编译，适配器在启动时用一张空白观测先做掉（只开编译约 14 秒；加上当前帧缓存约 22 秒，因为它的第二种 token 形状也要编译）。显存峰值不变。

闭环使用时省下的时间相同。七个任务的仿真评测，两种设置各用一个进程、在同一块 RTX PRO 6000 Blackwell 上并排跑，这块显卡当时还有别的任务在用（所以两边都比上表慢）。
下表是适配器记的一次推理的时间（含观测的序列化和管道）：

| | 推理次数 | 中位数 | 平均 | 90 分位 |
|---|---|---|---|---|
| 都不开 | 4474 | 376 ms | 393 ms | 461 ms |
| 两个都开 | 4332 | **263 ms** | 285 ms | 367 ms |

按任务看，中位数从 360–418 ms 降到 248–304 ms。每轮的第一次推理没有算进去：没有预热时它就是编译，在那里是 24–32 秒。

### 输出变了多少

不开这个选项时，跑两遍得到的动作段逐位相同。开了以后，合并后的算子舍入方式不同，差别经过 30 层、5 步被带下来。
两个任务各 20 帧录制画面，同一份噪声：

| | 编译 对 不编译 | 换一个噪声种子 |
|---|---|---|
| 手臂关节，平均差 | 0.0010–0.0015 rad | 0.008–0.018 rad |
| 手臂关节，每段里的最大差（各段的中位数） | 0.008 rad | 0.03–0.09 rad |
| 夹爪开口比例，平均差 | 0.0007–0.0016 | 0.008–0.010 |

两个选项都开（每行 12 帧，和都不开相比）：

| | 两个都开 | 换一个噪声种子 |
|---|---|---|
| RTX PRO 6000，任务 A：手臂关节，平均差 | 0.0016 rad | 0.019 rad |
| RTX PRO 6000，任务 A：夹爪开口比例，平均差 | 0.0020 | 0.011 |
| RTX 4090，任务 B：手臂关节，平均差 | 0.0007 rad | 0.0063 rad |
| RTX 4090，任务 B：夹爪开口比例，平均差 | 0.0005 | 0.0093 |

单开当前帧缓存带来的差别比编译还小（0.0003–0.0007 rad），取下之后动作段和原来逐位相同。测试里用一个 float64 的替身主干验证过：
缓存的结果和完整前向相差不到 1e-10，所以真模型上的差别就是舍入。

也就是说，这个改动带来的差别大约是换一份噪声的十分之一。在仿真里用同一批评测集对比（两个选项都开，对都不开），没有看到任务成功率下降；机器人上还没有比较过。

### 说明

- `--compile reduce-overhead`（CUDA graph 重放）在这种拆法下要多做一件事：每一层先后调用两个专家各自的编译函数，默认情况下后一次调用会释放前一次的输出。
  所以推理进程自己标记 graph 的"步"，每次调用 MoT 标一次（`torch.compiler.cudagraph_mark_step_begin()`）。这样它能跑、动作段也相同，
  但在 RTX 4090 上并不比 `--compile default` 快（那里的推理时间已经由矩阵乘法决定，不是由启动小算子决定），预热还要约 100 秒。`phoenix_compile` 不会选它。
- 当前帧缓存的第一版每一层都核对一次掩码，每次推理多出 120 次 GPU 往返，把省下的时间全部抵消了。会强制同步的检查应该每步做一次，而不是每层一次。
- 适配器这一侧不值得优化：编码观测、写 npz 文件、走管道，合计约 5 ms。
- 不改变计算内容还能省的已经很少（文本条件的键和值、CPU 上的图像缩放、VAE 编码，大约 10 ms）。再往下就要改计算本身：视频少算几个去噪步、蒸馏。

## 异步执行和动作段的衔接

`phoenix_async: real` 时，下一段动作在后台计算，机械臂不停。`phoenix_steps` 的含义不变，仍是每段动作执行多少步，也就是规划间隔：
正在执行的这一段走满 `phoenix_steps` 步之前的 `phoenix_inference_delay`（d）步，下一段开始计算，d 步之后接上。
所以一段动作实际从它的第 d 步执行到第 d + `phoenix_steps` 步：要求 **d < `phoenix_steps` ≤ 32 − d**，不满足时适配器直接报错。
发起下一段时，正在执行的那段还剩 32 − `phoenix_steps` 步。
在 10 Hz 的假机器人上，每步平均 0.12 秒（同步是 0.134 秒），没有错过任何一个控制周期。

单纯的异步动作段不知道这段时间里机械臂在做什么，所以它的起点不一定接得上旧的一段。`phoenix_rtc_infer` 让它接着正在执行的那一段：
推理期间要执行的 d 步权重为 1，其余重叠的步数按 RTC（[Black 等，2025](https://arxiv.org/abs/2506.07339)）的软掩码。

- `inpaint`：在每个去噪步，把重叠那几步的"干净动作估计"替换掉。不增加推理时间。
- `guidance`：RTC 本身的算法。上游实现了这个引导，但没有接到联合去噪的循环上，这里把它套了上去。每个去噪步多一次反向传播，推理时间约为两倍。

接上那一步的最大关节变化（中位数 / 最大值，rad），`phoenix_steps` 14、d = 6（正在执行的那段还剩 18 步，其中 12 步是软掩码），两个任务的录制画面：

| | 任务 A（8 个接点） | 任务 B（20 个接点） |
|---|---|---|
| 不做衔接 | 0.114 / 0.320 | 0.031 / 0.160 |
| `inpaint` | 0.033 / 0.059 | 0.014 / 0.066 |
| `guidance` | 0.107 / 0.233 | 0.031 / 0.149 |
| 参照：一段动作内部相邻两步 | 0.030 / 0.072 | 0.017 / 0.050 |

两个结论：

- **`inpaint` 需要留出软掩码的步数**，一共 32 − `phoenix_steps` − d 步。`phoenix_steps` 20、d = 6 时有 6 步，仍然有效
  （任务 A 0.038 / 0.086，不做衔接是 0.090 / 0.283；任务 B 0.018 / 0.069，不做衔接是 0.035 / 0.252）。
  `phoenix_steps` 26、d = 6 时，正在执行的那段只剩 6 步权重为 1 的，跳变只是挪到了接上的那一步（任务 A 上 0.092 / 0.280）。
- **`guidance` 在这些权重上几乎不改变动作段。** 引导项正比于"模型的干净动作估计跟着带噪动作输入变化多少"，而这里在噪声水平降到 0.1 以下之前它几乎不跟：
  所用的 5 个去噪步上，梯度和误差的夹角余弦都在 ±0.11 以内。引导权重设成 0 时结果和不加引导完全相同，说明接法没有问题；
  这个方法不适合动作被观测和联合去噪的视频决定得这么强的模型。

## 工具

每个脚本都要给 `policy/phoenix` 目录、一次训练的目录、步数、`make_frames.py` 生成的帧文件、提示词，以及编码器和 Wan 基础权重的目录；
具体命令行写在各文件开头。

| 脚本 | 用途 |
|---|---|
| `make_frames.py` | 从一份 v2.1 录制里取一个 episode 的若干帧 |
| `profile_infer.py`、`profile_ops.py` | 各环节的时间；各类算子的 CUDA 时间 |
| `gemm_floor.py` | 这块显卡只算主干那些矩阵乘能有多快 |
| `compare_compiled.py` | 编译和不编译的时间与输出差别 |
| `compare_frame_cache.py` | 当前帧缓存的同样对比，单开和与编译一起开 |
| `seed_vs_compile.py` | 把这个差别和换噪声种子的差别放在一起 |
| `rtc_offline.py` | 接点对比：不衔接、inpaint、guidance |
| `fake_robot_loop.py` | 适配器加真的推理进程，在 10 Hz 的假机器人上跑 |

## 测试

```bash
python -m pytest -q tests
```

共 7 条；其中两条要导入上游代码，子模块没有检出时会跳过。

## 局限

- `phoenix_compile`、`phoenix_frame_cache`、`phoenix_async`、`phoenix_rtc_infer` 都没有在机器人上比较过任务成功率。
- 不包含归一化统计这一步。
- 绑定在上游的一个提交上，也绑定在"14 维双臂关节 + 三路相机"这种布局上。
