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
# Installation commands for MLIPs depend on their respective repositories. Please check if the installation fails. Install PyTorch version appropriate for your GPU.
**For MACE:**
```bash
conda create -n environment_mace python=3.11
conda activate environment_mace
pip3 install torch torchvision
pip install mace-torch
```

**For AIMNet2:**
```bash
conda create -n environment_aimnet2 python=3.11
conda activate environment_aimnet2
pip3 install torch torchvision
pip install aimnet
pip install "aimnet[train]"
pip install tqdm
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

1. Edit the `CONFIG` block in `train_aimnet2.py`. Check examples for additional files that need to be passed to CONFIG (arch_yaml is used to construct the aimnet2 model, inference_cfg is used to adopt aimnet2's built-in dataloader):

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

### Low-level: a single batch

> **Pick the right class for your checkpoint.** AIMNet2 checkpoints are
> `AIMNet2PROBE` (they include the `charge_proj` weights) and must be loaded into
> an `AIMNet2PROBE`; MACE (and any other backbone) uses the base `PROBEModel`.
> Loading an AIMNet2 checkpoint into a plain `PROBEModel` will fail on a
> state-dict key mismatch.

```python
import torch
from probe.model import PROBEModel                  # MACE / generic backbone
from probe.backends.aimnet2 import AIMNet2PROBE     # AIMNet2 checkpoints

# --- MACE / generic ---
model = PROBEModel(backbone_dim=feat_dim)           # feat_dim from the backbone
# --- AIMNet2 (use this instead for AIMNet2 checkpoints) ---
# model = AIMNet2PROBE()                            # backbone_dim is fixed to 256

model.load_state_dict(torch.load('best_model.pt')['model_state_dict'])
model.eval()

# atom_feats: [B, N, D], atom_mask: [B, N] bool
with torch.no_grad():
    # return_embeddings=True is the DEFAULT, so forward returns (logits, embedding).
    # The 256-d embedding is useful for UMAP / clustering / analysis.
    # For AIMNet2 also pass charges=charges (partial charges from the backbone).
    logits, mol_embedding = model(atom_feats, atom_mask, energy=pred_energy)
    probs = torch.softmax(logits, dim=-1)      # P(reliable), P(unreliable)

    # Pass return_embeddings=False if you only want logits:
    logits = model(atom_feats, atom_mask, energy=pred_energy,
                   return_embeddings=False)

    importance = model.get_atom_importance(atom_feats, atom_mask)  # [B, N]
```

For AIMNet2, also pass `charges=charges` (partial charges from the backbone) so the
charge-injection layer is used — the training/inference loops do this for you.

### High-level: full dataset → `.npz` (`probe/inference.py`)

`run_inference` streams a whole dataloader through a trained model, collecting
per-molecule probabilities **and** the 256-d `mol_embedding`. Results are written
to disk in chunks (so large datasets don't OOM) and, if `output_path` is given,
saved to a compressed `.npz` (ragged atomic numbers go to a sibling `.npy`).

**AIMNet2:**
```python
from probe.backends.aimnet2 import load_aimnet2, get_aimnet2_dataloaders, AIMNet2PROBE
from probe.inference import run_inference, make_aimnet2_infer_fn

aimnet2 = load_aimnet2(checkpoint, arch_yaml, device)
model   = AIMNet2PROBE().to(device)
model.load_state_dict(torch.load('best_model.pt')['model_state_dict'])

_, loader = get_aimnet2_dataloaders(inference_cfg)
infer_fn  = make_aimnet2_infer_fn(aimnet2)

results = run_inference(model, infer_fn, loader, device,
                        output_path='inference_results.npz')
```

**MACE:**
```python
from probe.backends.mace import load_mace, get_z_table, load_xyz_dataloader  # or your loader
from probe.model import PROBEModel
from probe.inference import run_inference, make_mace_infer_fn

extractor = load_mace(model_path, device)
model     = PROBEModel(backbone_dim=extractor.feat_dim).to(device)
model.load_state_dict(torch.load('best_model.pt')['model_state_dict'])

z_table  = get_z_table(extractor)
infer_fn = make_mace_infer_fn(extractor, z_table=z_table)  # z_table → save atomic numbers

results = run_inference(model, infer_fn, loader, device,
                        output_path='inference_results.npz')
```

`results` (and the saved `.npz`) contains:

| key | shape | description |
|-----|-------|-------------|
| `probabilities`    | `[N, n_classes]` | softmax — `[:, 0]`=P(reliable), `[:, 1]`=P(unreliable) |
| `predictions`      | `[N]`            | argmax class |
| `predicted_energy` | `[N]`            | MLIP predicted energy (eV) |
| `n_atoms`          | `[N]`            | atoms per molecule |
| `mol_embedding`    | `[N, 256]`       | molecular embedding |
| `net_charge`       | `[N]`            | molecular net charge (AIMNet2; zeros if unavailable) |
| `atomic_numbers`   | list of `[n_i]`  | per-molecule atomic numbers (ragged → saved as `<stem>_atomic_numbers.npy`) |

Reload with:
```python
import numpy as np
d = np.load('inference_results.npz')
results = {k: d[k] for k in d.files}
results['atomic_numbers'] = np.load('inference_results_atomic_numbers.npy',
                                    allow_pickle=True)
```

---

## License

MIT

