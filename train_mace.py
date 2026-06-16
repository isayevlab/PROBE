"""
Train PROBE on MACE-OFF23.

Edit the CONFIG block below, then run:
    python train_mace.py
"""

import numpy as np
import torch
from tqdm.auto import tqdm

from probe.model import PROBEModel
from probe.backends.mace import (
    load_mace, get_z_table,
    train_val_split_loader, process_batch_mace,
)
from probe.train import run_training, compute_error_boundary

# ============================================================
# Configuration — edit these paths before running
# ============================================================
CONFIG = {
    # Paths
    'mace_model_path': '/work/nvme/bbjt/smehdi1/mace_data/MACE-OFF23_large.model',
    'train_xyz': '/work/nvme/bbjt/smehdi1/mace_data/train_large_neut_no_bad_clean.xyz',
    'test_xyz': '/work/nvme/bbjt/smehdi1/mace_data/test_large_neut_all.xyz',
    'output_dir':        './probe_mace_outputs',

    # Device
    'device':            'cuda' if torch.cuda.is_available() else 'cpu',
    'ev_to_kcalmol':     23.06,

    # Data
    'batch_size':        256,
    'valid_fraction':    0.1,

    # Error boundary
    'error_boundary_percentile': 50,

    # Training
    'lr':                5e-5,
    'weight_decay':      1e-4,
    'epochs':            1000,
    'early_stopping_patience': 10,
    'scheduler_patience':      5,
    'scheduler_factor':        0.9,
    'min_lr':            5e-6,
    'gradient_clip_norm':1.0,

    # Architecture (backbone_dim auto-detected from MACE)
    'atom_encoder_hidden':       [256, 128],
    'atom_encoder_output_dim':   256,
    'mol_attention_heads':       32,
    'classifier_hidden':         [256, 128, 32],
    'dropout':                   0.1,

    # Evaluation
    'high_conf_cutoffs': {0: 0.8, 1: 0.8},
}

# ============================================================
# Main
# ============================================================
def main():
    device = CONFIG['device']

    # 1. Load frozen MACE backbone
    extractor = load_mace(CONFIG['mace_model_path'], device)
    z_table   = get_z_table(extractor)
    r_max     = float(extractor.mace_model.r_max)

    # 2. Load data (90/10 train/val split from train_xyz)
    print("Loading data...")
    train_loader, val_loader = train_val_split_loader(
        CONFIG['train_xyz'], z_table, r_max,
        CONFIG['batch_size'], CONFIG['valid_fraction'],
    )

    # 3. Compute error boundary from training set
    print("Computing error distribution on training set...")
    errors_kcal = []
    for batch in tqdm(train_loader, desc='Scanning errors'):
        _, _, pred_e, true_e, _ = process_batch_mace(batch, device, extractor)
        err = torch.abs(true_e - pred_e)
        valid = ~torch.isnan(err)
        errors_kcal.extend((err[valid].cpu().numpy() * CONFIG['ev_to_kcalmol']).tolist())

    boundary_kcal = compute_error_boundary(
        np.array(errors_kcal), CONFIG['error_boundary_percentile'])
    boundary_ev   = boundary_kcal / CONFIG['ev_to_kcalmol']
    error_bins    = torch.tensor([0.0, boundary_ev], device=device)

    # 4. Build PROBE model (backbone_dim auto-detected)
    model = PROBEModel(
        backbone_dim=extractor.feat_dim,
        atom_encoder_hidden=CONFIG['atom_encoder_hidden'],
        atom_encoder_output_dim=CONFIG['atom_encoder_output_dim'],
        mol_attention_heads=CONFIG['mol_attention_heads'],
        classifier_hidden=CONFIG['classifier_hidden'],
        dropout=CONFIG['dropout'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PROBE parameters: {total_params:,}")

    # 5. Define process function
    process_fn = lambda batch, dev: process_batch_mace(batch, dev, extractor)

    # 6. Train
    history = run_training(
        model=model,
        process_batch_fn=process_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        error_bins=error_bins,
        device=device,
        output_dir=CONFIG['output_dir'],
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay'],
        epochs=CONFIG['epochs'],
        early_stopping_patience=CONFIG['early_stopping_patience'],
        scheduler_patience=CONFIG['scheduler_patience'],
        scheduler_factor=CONFIG['scheduler_factor'],
        min_lr=CONFIG['min_lr'],
        gradient_clip_norm=CONFIG['gradient_clip_norm'],
        high_conf_cutoffs=CONFIG['high_conf_cutoffs'],
    )

    print(f"\nTraining complete. Best epoch: {history['best_epoch']}")
    print(f"Checkpoint saved to: {CONFIG['output_dir']}/")


if __name__ == '__main__':
    main()
