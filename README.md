# SanD-Planner

### Sample-Efficient Diffusion Planner in B-Spline Space for Robust Local Navigation

[![arXiv](https://img.shields.io/badge/arXiv-2602.00923-b31b1b.svg)](https://arxiv.org/abs/2602.00923)

SanD-Planner is a learning-based local trajectory planner for point-goal robot
navigation. It predicts smooth, collision-aware trajectories directly from onboard
depth observations by (i) encoding a short sequence of depth frames with a ResNet
backbone and a Transformer, (ii) generating **B-spline control points** with a
conditional 1-D U-Net **DDPM** (denoising diffusion) model, and (iii) selecting the
best candidate trajectory through an **ESDF**-based safety-and-goal evaluator.

By predicting a compact set of B-spline control points rather than dense waypoints,
the planner decouples trajectory smoothness from the number of generated samples and
produces dynamically feasible paths that are inexpensive to evaluate against an
obstacle field.

---

## Contents

1. [**Installation**](#-installation) — set up SanD-Planner
2. [**Evaluation**](#-evaluation-navdp-benchmark) — reproduce results on the NavDP benchmark
3. [Usage](#usage) · [Repository structure](#repository-structure) · [Configuration](#configuration) · [Pretrained weights & data](#pretrained-weights--data) · [Citation](#citation)

---

# 📦 Installation

**1. Clone and install SanD-Planner**

```bash
git clone https://github.com/WangJinCheng1998/sandplanner.git
cd sandplanner
pip install -r requirements.txt
```

Requires **Python ≥ 3.9** and a **CUDA-enabled PyTorch** build.

**2. (Optional) GPU-accelerated ESDF**

```bash
pip install cupy-cuda12x      # match your CUDA version
```

Without CuPy, a CPU Euclidean-distance-transform fallback is used automatically.

**3. Pretrained weights**

Download a checkpoint and place it under `checkpoints/` (see
[Pretrained weights & data](#pretrained-weights--data)).

> The Isaac Sim simulator and the NavDP benchmark are **only** needed to reproduce the
> navigation evaluation — see [Evaluation](#-evaluation-navdp-benchmark) below.

---

# 🚀 Evaluation (NavDP benchmark)

SanD-Planner is evaluated in the
[NavDP](https://github.com/InternRobotics/NavDP) navigation benchmark, running on
**NVIDIA Isaac Sim 4.2.0** and **Isaac Lab 1.2.0**. The planner runs as an HTTP server
that the NavDP evaluation scripts query at every simulation step.

### Step 1 — Install Isaac Sim 4.2.0 + Isaac Lab 1.2.0

```bash
# Isaac Sim 4.2.0 in a dedicated conda environment
conda create -n isaaclab python=3.10
conda activate isaaclab
pip install --upgrade pip
pip install isaacsim==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 \
    isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 \
    --extra-index-url https://pypi.nvidia.com
isaacsim omni.isaac.sim.python.kit

# Isaac Lab 1.2.0
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && git checkout tags/v1.2.0
./isaaclab.sh -i
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py   # smoke test
```

### Step 2 — Install the NavDP benchmark

```bash
git clone https://github.com/InternRobotics/NavDP.git
cd NavDP
pip install -r requirements.txt
```

Then download the benchmark scene assets following the
[NavDP repository](https://github.com/InternRobotics/NavDP).

### Step 3 — Run the evaluation (two terminals)

```bash
# Terminal 1 — serve SanD-Planner (run from THIS repo)
python sand_planner/server/simple_server.py --port 8890 \
    --checkpoint checkpoints/NoMax.pth

# Terminal 2 — run NavDP point-goal evaluation (run from the NavDP repo)
python eval_pointgoal_wheeled.py --port 8890 \
    --scene_dir {ASSET_SCENE} --scene_index {INDEX} --scene_scale {SCALE}
```

> Isaac Sim / Isaac Lab versions and asset setup follow the NavDP benchmark; please
> refer to the upstream repository for the authoritative, up-to-date instructions.

---

## Usage

### Programmatic inference

```python
from sand_planner import InferenceConfig, SandPlannerInference

config = InferenceConfig(checkpoint_path="checkpoints/NoMax.pth")
planner = SandPlannerInference(config)
results = planner.run_inference("depth_frame.png")        # from a depth image file
best = results["sampled_trajectories"][results["best_index"]]   # (N, 3) trajectory
```

`SandPlannerInference` is the core engine: depth → diffusion → B-spline control points
→ arc-length sampling → ESDF evaluation → best trajectory. Use `run_inference(path)`
for files, or `process_depth_arrays(depth_tensor)` for in-memory tensors.

### Online agent

```python
from sand_planner.agent.sand_planner_agent import SandPlannerAgent
```

`SandPlannerAgent` wraps the engine for closed-loop navigation: call
`reset(batch_size, threshold)` once, then `step_pointgoal(goals, images, depths)` each
step to get the trajectory to execute. This is the interface used by the inference
server and the NavDP benchmark.

### Inference server

```bash
python sand_planner/server/simple_server.py --port 8890 --checkpoint checkpoints/NoMax.pth
```

Exposes `/navigator_reset`, `/pointgoal_step` and `/health` endpoints.

### Training

```bash
export DATASET_ROOT=/path/to/dataset
export STATS_JSON=/path/to/trajectory_stats.json
bash run_train.sh
```

Hyper-parameters (sequence length, number of control points, learning-rate schedule,
normalization, CFG, etc.) are passed as command-line arguments in `run_train.sh`.
Training metrics are logged to Weights & Biases (project `sand-planner`).

---

## Repository structure

```
sand_planner/
├── config.py                 # InferenceConfig: paths, model, diffusion, ESDF & eval params
├── core/
│   ├── orchestrator.py       # SandPlannerInference: end-to-end inference pipeline
│   ├── model_manager.py      # checkpoint loading, model & normalizer construction
│   ├── trajectory_inference.py
│   └── trajectory_evaluator.py
├── agent/
│   ├── sand_planner_agent.py # SandPlannerAgent: online navigation agent
│   └── depth_processor.py    # file- and array-based depth processing
├── nn/
│   ├── ddpm.py               # BSplineDDPM diffusion wrapper
│   ├── depth_encoder.py      # depth image encoder
│   ├── condition_encoders.py # depth + goal conditioning
│   └── models/               # UNet1DConditionModel, ResNet, embeddings, transformers
├── trajectory/
│   ├── arc_length_sampling*.py   # arc-length (re)sampling
│   ├── evaluation_vectorized.py  # vectorized ESDF cost evaluation (runtime path)
│   ├── evaluation.py             # reference (non-vectorized) evaluator
│   └── visualization.py
├── utils/                    # bspline, esdf, normalize, image, traj_opt
├── training/
│   ├── train_bspline_ddpm.py # training entry point
│   └── dataloader_bspline.py # SandPlannerBSplineDataset (multi-scene)
└── server/
    └── simple_server.py      # Flask point-goal inference server

run_train.sh                  # training launcher
```

---

## Configuration

`sand_planner/config.py` (`InferenceConfig`) centralizes all configuration:
checkpoint / data paths, model and diffusion settings, ESDF grid and voxel
parameters, and trajectory-evaluation weights. Relative paths are resolved against
`base_dir`, which defaults to the repository root and can be overridden with the
`SAND_PLANNER_BASE_DIR` environment variable.

## Pretrained weights & data

Three pretrained checkpoints are released. Download the desired file, place it under
`checkpoints/`, and set `InferenceConfig.checkpoint_path` accordingly.

| Checkpoint | Arc-length cap | Control points | Description |
| --- | --- | --- | --- |
| **`NoMax.pth`** *(default)* | none (uncapped) | 8 | Trained on the full-length ground-truth trajectory (no arc-length cap); longer planning horizon. Default in `InferenceConfig`. |
| **`NoMax_12.pth`** | none (uncapped) | 12 | Same setup as `NoMax.pth` but with **12** B-spline control points for higher-resolution trajectories. Requires `num_control_points=12` (see note below). |
| **`Max_2.1m.pth`** | `max_arc_length = 2.1 m` | 8 | Supervision targets are truncated to the first **2.1 m** of the path while the true (possibly distant) goal is kept as the conditioning input — a short, fixed-horizon local plan, stable for close-range maneuvers. |

> ⚠️ The checkpoints do **not** embed the control-point count, so it must match
> `InferenceConfig.num_control_points` (default `8`). To use `NoMax_12.pth`, set it to `12`:
>
> ```python
> config = InferenceConfig(checkpoint_path="checkpoints/NoMax_12.pth", num_control_points=12)
> ```

**Download** (from the [`v1.0` release](https://github.com/WangJinCheng1998/sandplanner/releases/tag/v1.0)):

```bash
mkdir -p checkpoints
wget -P checkpoints \
  https://github.com/WangJinCheng1998/sandplanner/releases/download/v1.0/NoMax.pth \
  https://github.com/WangJinCheng1998/sandplanner/releases/download/v1.0/NoMax_12.pth \
  https://github.com/WangJinCheng1998/sandplanner/releases/download/v1.0/Max_2.1m.pth
```

### Training data

The **`dataset_avoid`** subset (obstacle-avoidance runs) used to train the released
checkpoints is available on the Hugging Face Hub:

**https://huggingface.co/datasets/WJCUCL/sandplanner-dataset-avoid**

```bash
# download the dataset into dataset/dataset_avoid/
hf download WJCUCL/sandplanner-dataset-avoid --repo-type dataset \
    --local-dir dataset/dataset_avoid

# train on it (run_train.sh loads every dataset_* subdir under DATASET_ROOT)
DATASET_ROOT=dataset bash run_train.sh
```

Each `run_XXXX/` holds `traj_xyz.npy`, `traj_yaw.npy`, `traj_pitch.npy` and a `depth/`
folder of depth frames. The benchmark / simulation scene assets are separate — see
[Evaluation](#-evaluation-navdp-benchmark).

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{wang2026sand,
  title   = {SanD-Planner: Sample-Efficient Diffusion Planner in B-Spline Space for Robust Local Navigation},
  author  = {Wang, Jincheng and Bao, Lingfan and Yang, Tong and Plasencia, Diego Martinez and Jiao, Jianhao and Kanoulas, Dimitrios},
  journal = {arXiv preprint arXiv:2602.00923},
  year    = {2026}
}
```

## Acknowledgements

- Simulation and benchmarking are built on
  [**NavDP**](https://github.com/InternRobotics/NavDP) (InternRobotics) — we thank
  the authors for releasing the navigation benchmark and Isaac Sim / Isaac Lab
  evaluation environment.
- The diffusion backbone (`UNet1DConditionModel`, ResNet / embedding / transformer
  building blocks) is adapted from
  [HuggingFace `diffusers`](https://github.com/huggingface/diffusers) (Apache-2.0).

## License

This project is released under the [Apache License 2.0](LICENSE), which is
compatible with the adapted `diffusers` components (also Apache-2.0).

> The NavDP benchmark and its simulation assets are distributed by their authors
> under CC BY-NC-SA 4.0 and are **not** included in this repository; obtain them
> from the [upstream project](https://github.com/InternRobotics/NavDP) under their
> original terms.
