"""
Inference for PROBE — runs a trained classifier over a dataset and collects
per-molecule reliability probabilities together with the 256-d molecular
embedding (the raw `proj` output, identical to the notebook's `mol_embedding`).

Results are streamed to disk in chunks so very large datasets do not OOM, then
reloaded and concatenated. If `output_path` is given, the final arrays are saved
to a compressed `.npz` (and atomic numbers, which are ragged, to a sibling
`.npy`), exactly mirroring the notebook's save format.

Usage (AIMNet2):
    from probe.backends.aimnet2 import load_aimnet2, get_aimnet2_dataloaders, AIMNet2PROBE
    from probe.inference import run_inference, make_aimnet2_infer_fn

    aimnet2 = load_aimnet2(checkpoint, arch_yaml, device)
    model   = AIMNet2PROBE(...).to(device)
    model.load_state_dict(torch.load('best_model.pt')['model_state_dict'])

    infer_fn = make_aimnet2_infer_fn(aimnet2)
    results  = run_inference(model, infer_fn, loader, device,
                             output_path='inference_results.npz')

Usage (MACE):
    from probe.inference import run_inference, make_mace_infer_fn
    infer_fn = make_mace_infer_fn(extractor, z_table=z_table)   # z_table optional
    results  = run_inference(model, infer_fn, loader, device,
                             output_path='inference_results.npz')

`results` is a dict with: probabilities [N, n_classes], predictions [N],
predicted_energy [N], n_atoms [N], mol_embedding [N, 256], and (when the
backend provides them) net_charge [N] and atomic_numbers (list of arrays).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Core inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model,
                  infer_batch_fn: Callable,
                  dataloader,
                  device,
                  output_path: Optional[str] = None,
                  chunk_size: int = 50_000,
                  tmp_dir: str = '/tmp/probe_chunks',
                  max_samples: Optional[int] = None) -> dict:
    """Run PROBE over a dataloader and collect probabilities + embeddings.

    Args:
        model:           a trained PROBEModel / AIMNet2PROBE (already on `device`).
        infer_batch_fn:  callable(batch, device) -> dict with keys
                         'atom_feats' [B,N,D], 'atom_mask' [B,N] bool,
                         'energy' [B], 'n_atoms' [B], and optionally
                         'charges' [B,N], 'numbers' [B,N], 'net_charge' [B].
        output_path:     if given, save final arrays to this `.npz`
                         (atomic numbers go to `<stem>_atomic_numbers.npy`).
        chunk_size:      flush to disk roughly every this many molecules.
        max_samples:     stop after this many molecules (None = all).

    Returns:
        dict of concatenated numpy arrays.
    """
    model.eval()
    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob('chunk_*.npz'):
        f.unlink()
    for f in tmp.glob('chunk_*_atnums.npy'):
        f.unlink()

    buf = {k: [] for k in ('probabilities', 'predictions', 'predicted_energy',
                           'n_atoms', 'net_charge', 'mol_embedding')}
    atomic_numbers: list = []
    total_collected = 0
    chunk_idx = 0

    def flush_chunk():
        nonlocal chunk_idx
        if not buf['probabilities']:
            return
        np.savez_compressed(
            tmp / f'chunk_{chunk_idx:05d}.npz',
            probabilities    = torch.cat(buf['probabilities']).numpy(),
            predictions      = torch.cat(buf['predictions']).numpy(),
            predicted_energy = torch.cat(buf['predicted_energy']).numpy(),
            n_atoms          = torch.cat(buf['n_atoms']).numpy(),
            net_charge       = (torch.cat(buf['net_charge']).numpy()
                                if buf['net_charge'] else np.array([])),
            mol_embedding    = torch.cat(buf['mol_embedding']).numpy(),
        )
        np.save(tmp / f'chunk_{chunk_idx:05d}_atnums.npy',
                np.array(atomic_numbers, dtype=object), allow_pickle=True)
        chunk_idx += 1
        for v in buf.values():
            v.clear()
        atomic_numbers.clear()

    for batch in tqdm(dataloader, desc='Inference'):
        d = infer_batch_fn(batch, device)

        atom_feats = d['atom_feats']
        atom_mask  = d['atom_mask']
        energy     = d['energy']
        n_atoms    = d['n_atoms']

        valid = ~torch.isnan(energy)
        if not valid.any():
            continue

        feats_v  = atom_feats[valid]
        mask_v   = atom_mask[valid]
        energy_v = energy[valid]
        extra = {}
        if d.get('charges') is not None:
            extra['charges'] = d['charges'][valid]

        logits, embedding = model(feats_v, mask_v, energy=energy_v,
                                  return_embeddings=True, **extra)
        probs = F.softmax(logits, dim=-1)

        buf['probabilities'].append(probs.cpu())
        buf['predictions'].append(probs.argmax(dim=-1).cpu())
        buf['predicted_energy'].append(energy_v.cpu())
        buf['n_atoms'].append(n_atoms[valid].cpu())
        buf['mol_embedding'].append(embedding.cpu())

        if d.get('net_charge') is not None:
            buf['net_charge'].append(d['net_charge'][valid].cpu())

        if d.get('numbers') is not None:
            numbers_v = d['numbers'][valid]
            for j in range(numbers_v.shape[0]):
                mj = numbers_v[j] > 0
                atomic_numbers.append(numbers_v[j][mj].cpu().numpy())

        bsz = int(valid.shape[0])
        total_collected += int(valid.sum().item())
        if total_collected % chunk_size < bsz:
            flush_chunk()
        if max_samples is not None and total_collected >= max_samples:
            break

    flush_chunk()

    # ---- reload + concatenate ----
    print(f'Reloading {chunk_idx} chunks from {tmp}...')
    chunks = sorted(tmp.glob('chunk_?????.npz'))
    arrs = {k: [] for k in ('probabilities', 'predictions', 'predicted_energy',
                            'n_atoms', 'net_charge', 'mol_embedding')}
    all_atnums: list = []
    for c in tqdm(chunks, desc='Loading chunks'):
        dd = np.load(c)
        for k in arrs:
            arrs[k].append(dd[k])
        an = np.load(tmp / f'{c.stem}_atnums.npy', allow_pickle=True)
        all_atnums.extend(an.tolist())

    results = {k: np.concatenate(arrs[k]) for k in arrs
               if arrs[k] and sum(a.size for a in arrs[k]) > 0}
    if all_atnums:
        results['atomic_numbers'] = all_atnums
    if 'net_charge' not in results and 'predictions' in results:
        results['net_charge'] = np.zeros(len(results['predictions']), dtype=np.int8)

    n = len(results.get('predictions', []))
    print(f'Done. {n:,} molecules.')

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_arrs = {k: v for k, v in results.items() if k != 'atomic_numbers'}
        np.savez_compressed(out, **save_arrs)
        if 'atomic_numbers' in results:
            an_path = out.with_name(out.stem + '_atomic_numbers.npy')
            np.save(an_path, np.array(results['atomic_numbers'], dtype=object),
                    allow_pickle=True)
            print(f'Saved atomic numbers to {an_path}')
        print(f'Saved results to {out}')

    return results


# ---------------------------------------------------------------------------
# Backend adapters
# ---------------------------------------------------------------------------

def make_aimnet2_infer_fn(aimnet2_model):
    """Build an infer_batch_fn for AIMNet2 (charges + atomic numbers + net charge)."""

    @torch.no_grad()
    def fn(batch, device):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_dev = {k: v.to(device, non_blocking=True) for k, v in x.items()}
        out = aimnet2_model(x_dev)
        atom_mask = x_dev['numbers'] > 0
        d = {
            'atom_feats': out['aim'].float(),
            'atom_mask':  atom_mask,
            'energy':     out['energy'].float(),
            'charges':    out['charges'].float(),
            'n_atoms':    atom_mask.sum(dim=1).float(),
            'numbers':    x_dev['numbers'],
        }
        if 'charge' in x_dev:
            d['net_charge'] = x_dev['charge']
        return d

    return fn


def make_mace_infer_fn(extractor, z_table=None):
    """Build an infer_batch_fn for MACE.

    If `z_table` is given, atomic numbers are decoded from the one-hot
    node_attrs so they are saved alongside the embeddings.
    """
    from .backends.mace import process_batch_mace

    @torch.no_grad()
    def fn(batch, device):
        batch = batch.to(device)   # ensure node_attrs/ptr are on `device` below
        atom_feats, atom_mask, pred_energy, _true, n_atoms = \
            process_batch_mace(batch, device, extractor)
        d = {
            'atom_feats': atom_feats,
            'atom_mask':  atom_mask,
            'energy':     pred_energy,
            'n_atoms':    n_atoms,
        }
        if z_table is not None and hasattr(batch, 'node_attrs'):
            zs = torch.as_tensor(z_table.zs, device=device)
            node_z = zs[batch.node_attrs.argmax(dim=1)]   # [n_atoms_total]
            ptr = batch.ptr
            B = ptr.shape[0] - 1
            N_max = atom_mask.shape[1]
            numbers = torch.zeros(B, N_max, dtype=torch.long, device=device)
            for i in range(B):
                s, e = ptr[i].item(), ptr[i + 1].item()
                numbers[i, :e - s] = node_z[s:e]
            d['numbers'] = numbers
        return d

    return fn
