"""
AIMNet2 backend for PROBE.

Usage:
    from probe.backends.aimnet2 import load_aimnet2, process_batch_aimnet2, AIMNet2PROBE

    aimnet2 = load_aimnet2(checkpoint, arch_yaml, device)
    model   = AIMNet2PROBE(backbone=aimnet2)

    # In training loop:
    process_fn = lambda batch, dev: process_batch_aimnet2(batch, dev, aimnet2)
    run_training(model, process_fn, ...)
"""

import torch
from omegaconf import OmegaConf
from aimnet.config import build_module
from aimnet.train.utils import get_loaders

from ..model import PROBEModel


# ---------------------------------------------------------------------------
# Load backbone
# ---------------------------------------------------------------------------

def load_aimnet2(checkpoint_path: str, arch_yaml: str,
                 device: str = 'cuda') -> torch.nn.Module:
    """Load a frozen AIMNet2 model from a .pt checkpoint."""
    model = build_module(arch_yaml)
    state = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded AIMNet2 from {checkpoint_path}")
    return model


def get_aimnet2_dataloaders(inference_cfg_path: str):
    """Return (train_loader, val_loader) from an AIMNet2 OmegaConf config."""
    cfg = OmegaConf.load(inference_cfg_path)
    return get_loaders(cfg.data)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch_aimnet2(batch, device: str, aimnet2_model):
    """
    Run AIMNet2 on one batch and return PROBE-compatible tensors.

    Returns:
        atom_feats:   [B, N, 256]  AIM vectors
        atom_mask:    [B, N] bool
        pred_energy:  [B]
        true_energy:  [B]
        n_atoms:      [B]
    """
    x, y = batch
    x_dev = {k: v.to(device, non_blocking=True) for k, v in x.items()}
    y_dev = {k: v.to(device, non_blocking=True) for k, v in y.items()}

    with torch.no_grad():
        out = aimnet2_model(x_dev)

    # AIMNet2 runs in float64; cast head inputs to float32 so they match the
    # PROBE head params (otherwise torch.cat promotes the pooled vector to
    # Double and the Linear layers raise a dtype mismatch).
    atom_mask   = x_dev['numbers'] > 0                  # [B, N]
    atom_feats  = out['aim'].float()                    # [B, N, 256]
    charges     = out['charges'].float()                # [B, N]
    pred_energy = out['energy'].float()                 # [B]
    true_energy = y_dev['energy'].float()               # [B]
    n_atoms     = atom_mask.sum(dim=1).float()          # [B]

    return atom_feats, atom_mask, pred_energy, true_energy, n_atoms, charges


# ---------------------------------------------------------------------------
# AIMNet2-specific PROBE subclass (adds charge injection)
# ---------------------------------------------------------------------------

import torch.nn as nn

class AIMNet2PROBE(PROBEModel):
    """
    PROBE for AIMNet2. Adds a learnable charge injection layer on top of
    the base PROBEModel atom encoder.
    """

    def __init__(self, atom_encoder_output_dim: int = 256, **kwargs):
        super().__init__(backbone_dim=256,
                         atom_encoder_output_dim=atom_encoder_output_dim,
                         **kwargs)
        self.charge_proj = nn.Sequential(
            nn.Linear(1, atom_encoder_output_dim),
            nn.LayerNorm(atom_encoder_output_dim),
            nn.GELU(),
        )

    def forward(self, atom_feats, atom_mask, energy=None, charges=None,
                return_attention=False, return_embeddings=True):
        """
        Args:
            atom_feats: [B, N, 256]  AIM vectors
            atom_mask:  [B, N] bool
            energy:     [B]
            charges:    [B, N]  partial charges from AIMNet2
        Returns:
            logits, or (logits, embedding) when return_embeddings=True (default).
            If return_attention=True, attention weights are inserted before the
            embedding: (logits, attn) or (logits, attn, embedding).
        """
        z = self.atom_encoder(atom_feats)
        if charges is not None:
            z = z + self.charge_proj(charges.unsqueeze(-1))

        # Use parent's attention + pooling
        attended, attn_w = self.mol_attention(z, mask=atom_mask,
                                               return_attention=True)
        attended = self.mol_attention_norm(attended + z)
        self._last_attention_weights = attn_w.detach()

        out = self.pool_and_classify(attended, atom_mask, energy,
                                     return_embeddings=return_embeddings)
        if return_attention:
            if return_embeddings:
                logits, emb = out
                return logits, self._last_attention_weights, emb
            return out, self._last_attention_weights
        return out
