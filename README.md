# SanD-Planner

**Structured-and-Decomposed trajectory planning for vision-based robot navigation.**

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

## Highlights

- **Diffusion in control-point space.** A conditional 1-D U-Net DDPM generates
  8–12 B-spline control points, yielding smooth trajectories without dense
  waypoint regression.
- **Depth-conditioned multi-frame encoder.** ResNet-18 spatial features + a
  Transformer over multi-frame tokens, with optional TEA-lite motion cues and a
  2-D sinusoidal positional encoding.
- **Fast, controllable sampling.** Classifier-free guidance (CFG), v-prediction,
  and DPM-Solver multi-step sampling for low-latency inference.
- **ESDF-based trajectory evaluation.** Vectorized clearance / safety-margin /
  goal-cost scoring against a Euclidean Signed Distance Field (CPU EDT or optional
  GPU / NVBlox), with optional warm-start and best-plan backtracking for temporal
  consistency.
- **Ready-to-serve.** A lightweight Flask inference server exposes a point-goal
  navigation endpoint.

---

## Method overview

```
 depth frames (T×H×W)               point goal (x, y)
        │                                  │
        ▼                                  │
 ResNet-18 + Transformer  ──►  depth tokens │
 (multi-frame, +motion)                    ▼
        └──────────►  Condition encoder (depth + goal tokens)
                                   │  encoder_hidden_states
                                   ▼
                 Conditional 1-D U-Net DDPM  (CFG, v-pred, DPM-Solver)
                                   │
                                   ▼
                    B-spline control points  (N = 8 / 12)
                                   │  clamped cubic B-spline + arc-length resampling
                                   ▼
              candidate trajectories ──► ESDF evaluator ──► best trajectory
```

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
├── utils/                    # bspline, esdf, nvblox_esdf, normalize, image, traj_opt
├── training/
│   ├── train_bspline_ddpm.py # training entry point
│   └── dataloader_bspline.py # SandPlannerBSplineDataset (multi-scene)
└── server/
    └── simple_server.py      # Flask point-goal inference server

run_train.sh                  # training launcher
cp_vs_error_plot.py           # control-points vs. reconstruction-error analysis
elbow_test_6m.py              # elbow test for choosing the number of control points
turn_vs_cp.py                 # turning sharpness vs. control-point analysis
plot_arc_length_dist.py       # dataset arc-length distribution
```

---

## Installation

```bash
git clone https://github.com/WangJinCheng1998/sandplanner.git
cd sandplanner
pip install -r requirements.txt
```

Python ≥ 3.9 and a CUDA-enabled PyTorch build are recommended. GPU-accelerated ESDF
computation is optional — install `cupy` and/or `nvblox_torch`; otherwise a CPU
Euclidean-distance-transform fallback is used.

---

## Usage

### Inference (programmatic)

```python
from sand_planner import InferenceConfig, SandPlannerInference

config = InferenceConfig(checkpoint_path="checkpoints/your_model.pth")
planner = SandPlannerInference(config)
# run inference on depth observations to obtain candidate trajectories
```

### Online agent

```python
from sand_planner.agent.sand_planner_agent import SandPlannerAgent
```

`SandPlannerAgent` provides a `step_pointgoal(goal, image, depth)` interface for
closed-loop point-goal navigation.

### Inference server

```bash
python sand_planner/server/simple_server.py --port 8890 --checkpoint checkpoints/your_model.pth
```

The server exposes `/navigator_reset`, `/pointgoal_step` and `/health` endpoints for
point-goal navigation.

### Training

```bash
export DATASET_ROOT=/path/to/dataset
export STATS_JSON=/path/to/trajectory_stats.json
bash run_train.sh
```

Hyper-parameters (sequence length, number of control points, learning-rate schedule,
normalization, CFG, etc.) are passed as command-line arguments in `run_train.sh`.
Training metrics are logged to Weights & Biases (project `sand-planner`).

### Analysis scripts

The standalone scripts reproduce the design-analysis figures (set `DATASET_ROOT`
first):

```bash
export DATASET_ROOT=/path/to/dataset
python elbow_test_6m.py          # how many control points are enough?
python cp_vs_error_plot.py       # control points vs. reconstruction error
python turn_vs_cp.py             # turning sharpness vs. control points
python plot_arc_length_dist.py   # arc-length distribution of training segments
```

---

## Configuration

`sand_planner/config.py` (`InferenceConfig`) centralizes all configuration:
checkpoint / data paths, model and diffusion settings, ESDF grid and voxel
parameters, and trajectory-evaluation weights. Relative paths are resolved against
`base_dir`, which defaults to the repository root and can be overridden with the
`SAND_PLANNER_BASE_DIR` environment variable.

## Pretrained weights & data

Model checkpoints (`.pth`) and datasets are **not** tracked in this repository due to
their size. Place trained weights under `checkpoints/` and point
`InferenceConfig.checkpoint_path` to the desired file. <!-- TODO: add a download link
for the released checkpoints / dataset. -->

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{sandplanner,
  title   = {SanD-Planner: <full paper title>},
  author  = {<authors>},
  journal = {<venue>},
  year    = {<year>},
}
```

<!-- TODO: replace the placeholders above with the published reference. -->

## Acknowledgements

The diffusion backbone (`UNet1DConditionModel`, ResNet/embedding/transformer
building blocks) is adapted from the
[HuggingFace `diffusers`](https://github.com/huggingface/diffusers) library
(Apache-2.0).

## License

<!-- TODO: choose a license (e.g. Apache-2.0, which is compatible with the adapted
diffusers code) and add a LICENSE file. -->
