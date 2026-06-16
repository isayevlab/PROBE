"""
Train PROBE on AIMNet2.

Edit the CONFIG block below, then run:
    python train_aimnet2.py
"""

import numpy as np
import torch
from tqdm.auto import tqdm

from probe.backends.aimnet2 import (
    load_aimnet2, get_aimnet2_dataloaders,
    AIMNet2PROBE, process_batch_aimnet2,
)
from probe.train import run_training, compute_error_boundary, scalar_to_bin_index

# ============================================================
# Configuration — edit these paths before running
# ============================================================
CONFIG = {
    # Paths
    'checkpoint':        'aimnet2_b973c_3_curate_nodisp.pt',
    'arch_yaml':         'aimnet2.yaml',
    'inference_cfg':     'UQ_aimnet2_20M_b973c_4M_test.yaml',
    'output_dir':        './probe_aimnet2_outputs',

    # Device
    'device':            'cuda' if torch.cuda.is_available() else 'cpu',
    'ev_to_kcalmol':     23.06,

    # Error boundary (50th percentile = balanced classes)
    'error_boundary_percentile': 50,

    # Training
    'lr':                5e-5,
    'weight_decay':      1e-4,
    'epochs':            1000,
    'early_stopping_patience': 25,
    'scheduler_patience':      5,
    'scheduler_factor':        0.9,
    'min_lr':            5e-6,
    'gradient_clip_norm':1.0,

    # Architecture
    'aim_dim':                   256,
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

    # 1. Load frozen AIMNet2 backbone
    aimnet2 = load_aimnet2(CONFIG['checkpoint'], CONFIG['arch_yaml'], device)

    # 2. Load data
    print("Loading data...")
    train_loader, val_loader = get_aimnet2_dataloaders(CONFIG['inference_cfg'])

    # 3. Compute error boundary from training set
    print("Computing error distribution on training set...")
    errors_kcal = []
    for x, y in tqdm(train_loader, desc='Scanning errors'):
        x_dev = {k: v.to(device) for k, v in x.items()}
        y_dev = {k: v.to(device) for k, v in y.items()}
        with torch.no_grad():
            out = aimnet2(x_dev)
        err = torch.abs(y_dev['energy'] - out['energy'])
        valid = ~torch.isnan(err)
        errors_kcal.extend((err[valid].cpu().numpy() * CONFIG['ev_to_kcalmol']).tolist())

    boundary_kcal = compute_error_boundary(
        np.array(errors_kcal), CONFIG['error_boundary_percentile'])
    boundary_ev   = boundary_kcal / CONFIG['ev_to_kcalmol']
    error_bins    = torch.tensor([0.0, boundary_ev], device=device)

    # 4. Build PROBE model
    model = AIMNet2PROBE(
        atom_encoder_output_dim=CONFIG['atom_encoder_output_dim'],
        atom_encoder_hidden=CONFIG['atom_encoder_hidden'],
        mol_attention_heads=CONFIG['mol_attention_heads'],
        classifier_hidden=CONFIG['classifier_hidden'],
        dropout=CONFIG['dropout'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PROBE parameters: {total_params:,}")

    # 5. Define process function (wraps AIMNet2-specific batch extraction).
    #    Returns the 6-tuple *with* charges so the training loop can feed them
    #    into the charge-injection layer (see AIMNet2PROBE.forward).
    def process_fn(batch, dev):
        return process_batch_aimnet2(batch, dev, aimnet2)

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
