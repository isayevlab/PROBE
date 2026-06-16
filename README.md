# PROBE: Post-hoc Reliability frOm Backbone Embeddings

Official code for:

> **Knowing when to trust machine-learned interatomic potentials**  
> Shams Mehdi, Ilkwon Cho, Olexandr Isayev  

---

## Overview

PROBE attaches a lightweight binary classifier to the **frozen** per-atom
representations of a pretrained MLIP, learning to answer one question:
*is this prediction reliable?*

It requires no modification to the underlying model, adds <1% inference
overhead, and generalizes across architectures — demonstrated here on
**AIMNet2** and **MACE-OFF23**.

---

## Repository Structure

```
PROBE/
├── probe/
│   ├── model.py            # PROBEModel architecture
│   ├── train.py            # training loop, loss, evaluation
│   ├── metrics.py          # accuracy, MCC, F1, calibration
│   └── backends/
│       ├── aimnet2.py      # AIMNet2 data loading & batch processing
│       └── mace.py         # MACE data loading & batch processing
├── train_aimnet2.py        # runnable training script for AIMNet2
├── train_mace.py           # runnable training script for MACE
├── environment_aimnet2.yml
└── environment_mace.yml
```

---

## Installation
# Installation commands for MLIPs depend on their respective repositories. Please check if the installation fails.
**For MACE:**
```bash
conda create -n environment_mace python=3.11
conda activate environment_mace
pip3 install torch torchvision
pip install mace-torch
```

**For AIMNet2:**
```bash
conda env create -f environment_aimnet2.yml
conda activate probe_aimnet2

# AIMNet2 must be installed from source:
git clone https://github.com/isayevlab/AIMNet2
pip install -e AIMNet2/
```

---

## Training PROBE

### On MACE-OFF23

1. Edit the `CONFIG` block in `train_mace.py`:

```python
CONFIG = {
    'mace_model_path': '/path/to/MACE-OFF23_large.model',
    'train_xyz':       '/path/to/train.xyz',
    'test_xyz':        '/path/to/test.xyz',
    'output_dir':      './probe_mace_outputs',
    ...
}
```

2. Run:

```bash
python train_mace.py
```

### On AIMNet2

1. Edit the `CONFIG` block in `train_aimnet2.py`:

```python
CONFIG = {
    'checkpoint':    '/path/to/aimnet2_checkpoint.pt',
    'arch_yaml':     '/path/to/aimnet2.yaml',
    'inference_cfg': '/path/to/UQ_aimnet2_config.yaml',
    'output_dir':    './probe_aimnet2_outputs',
    ...
}
```

2. Run:

```bash
python train_aimnet2.py
```

Both scripts auto-detect the class boundary from the training-set error
distribution (50th percentile by default) and save the best checkpoint to
`output_dir/best_model_<timestamp>.pt`.

---

## Architecture

```
Frozen MLIP backbone
        │
        ▼  {h_i} ∈ R^d  per-atom embeddings
  Atom Encoder MLP
  (d → 256, LayerNorm, GELU, dropout=0.1)
        │
        ▼  (+ partial charge injection for AIMNet2)
  Multi-Head Self-Attention  (32 heads × 8 dims)
        │
        ▼
  Masked mean-pool ∥ masked max-pool ∥ energy ∥ N_atoms  ∈ R^514
        │
        ▼  linear projection
  Molecular embedding  ∈ R^256
        │
        ▼
  Classifier MLP  [256 → 128 → 32 → 2]
        │
        ▼
  P(reliable),  P(unreliable)
```

Total trainable parameters: ~567K

---

## Extending to a New MLIP

To apply PROBE to a different MLIP:

1. Write a `process_batch_fn(batch, device)` that returns:
   `(atom_feats [B,N,D], atom_mask [B,N], pred_energy [B], true_energy [B], n_atoms [B])`

2. Instantiate `PROBEModel(backbone_dim=D)`.

3. Call `run_training(model, process_batch_fn, ...)`.

No other changes are needed.

---

## Inference and Atom Importance

```python
import torch
from probe.model import PROBEModel

model = PROBEModel(backbone_dim=256)
model.load_state_dict(torch.load('best_model.pt')['model_state_dict'])
model.eval()

# atom_feats: [B, N, 256], atom_mask: [B, N] bool
with torch.no_grad():
    logits = model(atom_feats, atom_mask, energy=pred_energy)
    probs  = torch.softmax(logits, dim=-1)     # P(reliable), P(unreliable)
    importance = model.get_atom_importance(atom_feats, atom_mask)  # [B, N]
```

---

## License

MIT
