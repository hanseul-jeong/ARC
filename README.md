# ARC: Leveraging Compositional Representations for Cross-Problem Learning on VRPs

Official implementation of the NeurIPS 2025 DiffCoALG Workshop (Oral) paper **ARC: Leveraging Compositional Representations for Cross-Problem Learning on VRPs** ([paper](https://openreview.net/forum?id=OFY6wTzQZh))

- arXiv: coming soon

---

## 🗂️ Repository
```
ARC/
├─ baseline/routefinder_v2/routefinder/   # RouteFinder core + Hydra configs
│  ├─ configs/                            # experiment, env, model, logger settings
│  ├─ data/                               # train/val/test VRP datasets
│  ├─ routefinder/                        # Lightning modules, policies, envs
│  ├─ scripts/                            # training / evaluation helpers
│  └─ test.py, run.py                     # shared entrypoints
├─ experiments/
│  ├─ arc/                                # ARC-specific encoder/policy/env/model
│  ├─ cada/, mtpomo/, mvmoe/              # comparison baselines
│  └─ ours/                               # generator/env variants
├─ checkpoint/                            # pretrained checkpoints
├─ logs/train/                            # Hydra + Lightning logs
└─ README.md
```

---

## 🔧 Installation
Requirements: Python ≥ 3.10 and a CUDA-enabled GPU are recommended.

```bash
cd baseline/routefinder_v2/routefinder
pip install -e .
# for traditional solvers (OR-Tools, PyVRP, etc.)
# pip install -e '.[dev,solver]'
```

All RouteFinder sub-dependencies (rl4co, tensordict, etc.) are installed automatically.

---

## 🚀 Training
Convenience scripts live under `baseline/routefinder_v2/routefinder/scripts`.

```bash
cd baseline/routefinder_v2/routefinder
./scripts/train_arc_50_id.sh
```
- ARC-specific hyperparameters (e.g., `nce_lambda`) are defined in `experiments/arc/model.py` and the associated Hydra configs.
- Leave-One-Out and zero-shot setups are provided in `configs/experiment/main/ours/leaveout-*.yaml`.

---

## ✅ Evaluation
Run the provided helper scripts (`baseline/routefinder_v2/routefinder/scripts/test_arc_50_id.sh`, `test_arc_100_id.sh`, etc.) or call the shared RouteFinder `test.py` directly.

```bash
cd baseline/routefinder_v2/routefinder
python test.py \
  --checkpoint checkpoint/ARC/id/50/1.ckpt \
  --problem all \
  --size 50 \
  --batch_size 512 \
  --device cuda
```

---

