"""
Training utilities for PROBE.

  - uncertainty_loss_fn   — size-normalized cross-entropy
  - scalar_to_bin_index   — maps per-molecule errors → class indices
  - train_epoch           — one training epoch
  - evaluate              — evaluation loop
  - run_training          — full training loop with early stopping

These functions are backend-agnostic. They accept a `process_batch_fn`
callable that hides the MLIP-specific forward pass. See probe_aimnet2.py
and probe_mace.py for backend-specific implementations.
"""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from .metrics import confusion_matrix_torch, compute_all_metrics


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def uncertainty_loss_fn(logits: torch.Tensor, targets: torch.Tensor,
                        n_atoms: torch.Tensor,
                        class_weights: Optional[torch.Tensor] = None,
                        label_smoothing: float = 0.0) -> torch.Tensor:
    """Cross-entropy normalized by sqrt(n_atoms).

    Prevents large molecules from dominating the gradient.
    """
    if class_weights is not None:
        class_weights = class_weights.to(dtype=logits.dtype, device=logits.device)
    targets = targets.to(device=logits.device)
    ce = F.cross_entropy(logits, targets, weight=class_weights,
                         reduction='none', label_smoothing=label_smoothing)
    normalized = ce / n_atoms.to(dtype=logits.dtype).sqrt().clamp(min=1.0)
    return normalized.mean()


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def scalar_to_bin_index(errors: torch.Tensor,
                        bin_edges: torch.Tensor) -> torch.Tensor:
    """Map per-molecule absolute errors to class indices.

    bin_edges: 1-D tensor [0, boundary]  →  class 0 = reliable, class 1 = unreliable
    """
    bin_idx = torch.bucketize(errors, bin_edges, right=False) - 1
    bin_idx = torch.clamp(bin_idx, 0, len(bin_edges) - 1)
    return bin_idx


def compute_error_boundary(errors_kcal: np.ndarray,
                           percentile: float = 50) -> float:
    """Return the p-th percentile of the error distribution in kcal/mol."""
    boundary = float(np.percentile(errors_kcal, percentile))
    print(f"Error boundary ({percentile}th percentile): {boundary:.4f} kcal/mol")
    return boundary


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def train_epoch(model, process_batch_fn: Callable, dataloader,
                optimizer, error_bins: torch.Tensor, device,
                class_weights=None, label_smoothing: float = 0.0,
                gradient_clip_norm: float = 1.0) -> float:
    """Run one training epoch.

    Args:
        process_batch_fn: callable(batch, device) →
            (atom_feats [B,N,D], atom_mask [B,N], energy [B], true_energy [B], n_atoms [B])
    Returns:
        mean training loss for the epoch
    """
    model.train()
    total_loss, n_batches = 0.0, 0

    for batch in tqdm(dataloader, desc='Training', leave=False):
        atom_feats, atom_mask, pred_energy, true_energy, n_atoms = \
            process_batch_fn(batch, device)

        abs_errors = torch.abs(true_energy - pred_energy)
        target_classes = scalar_to_bin_index(abs_errors, error_bins)
        valid = ~torch.isnan(pred_energy)
        if not valid.any():
            continue

        atom_feats_v  = atom_feats[valid]
        atom_mask_v   = atom_mask[valid]
        pred_energy_v = pred_energy[valid]
        target_v      = target_classes[valid]
        n_atoms_v     = n_atoms[valid]

        optimizer.zero_grad()
        logits = model(atom_feats_v, atom_mask_v, energy=pred_energy_v)
        loss = uncertainty_loss_fn(logits, target_v, n_atoms_v,
                                   class_weights, label_smoothing)
        loss.backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, process_batch_fn: Callable, dataloader,
             error_bins: torch.Tensor, device,
             n_classes: int = 2,
             high_conf_cutoffs: Optional[Dict] = None) -> dict:
    """Evaluate the model on a dataloader.

    Returns a dict with accuracy, MCC, F1, probabilities, predictions, errors, etc.
    """
    model.eval()
    all_logits, all_targets, all_errors, all_n_atoms = [], [], [], []

    for batch in tqdm(dataloader, desc='Evaluating', leave=False):
        atom_feats, atom_mask, pred_energy, true_energy, n_atoms = \
            process_batch_fn(batch, device)

        abs_errors = torch.abs(true_energy - pred_energy)
        target_classes = scalar_to_bin_index(abs_errors, error_bins)
        valid = ~torch.isnan(pred_energy)
        if not valid.any():
            continue

        atom_feats_v  = atom_feats[valid]
        atom_mask_v   = atom_mask[valid]
        pred_energy_v = pred_energy[valid]
        target_v      = target_classes[valid]
        errors_v      = abs_errors[valid]
        n_atoms_v     = n_atoms[valid]

        logits = model(atom_feats_v, atom_mask_v, energy=pred_energy_v)

        all_logits.append(logits.cpu())
        all_targets.append(target_v.cpu())
        all_errors.append(errors_v.cpu())
        all_n_atoms.append(n_atoms_v.cpu())

    all_logits  = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    all_errors  = torch.cat(all_errors)
    all_n_atoms = torch.cat(all_n_atoms)

    all_probs = F.softmax(all_logits, dim=-1)
    all_preds = all_probs.argmax(dim=-1)

    cm   = confusion_matrix_torch(all_preds, all_targets, n_classes)
    loss = (F.cross_entropy(all_logits, all_targets, reduction='none') /
            all_n_atoms.float().sqrt().clamp(min=1.0)).mean().item()

    results = compute_all_metrics(cm)
    results['loss']          = loss
    results['probabilities'] = all_probs.numpy()
    results['predictions']   = all_preds.numpy()
    results['targets']       = all_targets.numpy()
    results['errors']        = all_errors.numpy()

    if high_conf_cutoffs is not None:
        from .metrics import high_confidence_analysis
        results['high_conf'] = high_confidence_analysis(
            all_probs, all_preds, all_targets, high_conf_cutoffs, n_classes)

    return results


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def run_training(model, process_batch_fn: Callable,
                 train_loader, val_loader,
                 error_bins: torch.Tensor, device,
                 output_dir: str = './probe_outputs',
                 lr: float = 5e-5, weight_decay: float = 1e-4,
                 epochs: int = 1000,
                 early_stopping_patience: int = 25,
                 scheduler_patience: int = 5,
                 scheduler_factor: float = 0.9,
                 min_lr: float = 5e-6,
                 gradient_clip_norm: float = 1.0,
                 class_weights=None,
                 label_smoothing: float = 0.0,
                 high_conf_cutoffs: Optional[Dict] = None) -> dict:
    """
    Full training loop with validation, LR scheduling, and early stopping.

    Saves:
        best_model_<timestamp>.pt  — best model by validation loss

    Returns:
        history dict with per-epoch train/val metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=scheduler_factor,
                                  patience=scheduler_patience,
                                  min_lr=min_lr)

    best_val_loss = float('inf')
    best_state    = None
    best_epoch    = 0
    patience_ctr  = 0
    history: dict = {'train_loss': [], 'val_loss': [],
                     'val_acc': [], 'val_mcc': [], 'val_f1': []}

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, process_batch_fn, train_loader, optimizer,
            error_bins, device, class_weights, label_smoothing, gradient_clip_norm
        )
        val_results = evaluate(
            model, process_batch_fn, val_loader, error_bins, device,
            n_classes=model.n_classes, high_conf_cutoffs=high_conf_cutoffs
        )
        val_loss = val_results['loss']
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_results['accuracy'])
        history['val_mcc'].append(val_results['mcc'])
        history['val_f1'].append(val_results['f1'])

        print(f"Epoch {epoch:4d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | acc={val_results['accuracy']:.4f} | "
              f"mcc={val_results['mcc']:.4f} | f1={val_results['f1']:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            best_state    = copy.deepcopy(model.state_dict())
            patience_ctr  = 0
            ckpt_path = output_dir / f'best_model_{timestamp}.pt'
            torch.save({
                'model_state_dict': best_state,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_metrics': val_results,
                'error_bins': error_bins.cpu().tolist(),
            }, ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(best epoch {best_epoch}, val_loss={best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history['best_epoch'] = best_epoch
    return history
