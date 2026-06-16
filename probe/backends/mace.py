"""
MACE backend for PROBE.

Usage:
    from probe.backends.mace import load_mace, process_batch_mace, MACEFeatureExtractor

    extractor = load_mace(model_path, device)
    model     = PROBEModel(backbone_dim=extractor.feat_dim)

    # In training loop:
    process_fn = lambda batch, dev: process_batch_mace(batch, dev, extractor)
    run_training(model, process_fn, ...)
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

# e3nn and MACE must be installed / on sys.path
try:
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    from e3nn import o3
    from mace.tools import AtomicNumberTable, atomic_numbers_to_indices
    from mace.tools import torch_geometric
    from mace.data.atomic_data import AtomicData
    from mace.data.utils import Configuration
    import ase.io
    import ase.data
    _MACE_AVAILABLE = True
except ImportError:
    _MACE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Feature extractor (forward hook)
# ---------------------------------------------------------------------------

class MACEFeatureExtractor(nn.Module):
    """Wraps a MACE model and captures scalar node features from the last
    product block via a forward hook. The backbone is never modified."""

    def __init__(self, mace_model):
        if not _MACE_AVAILABLE:
            raise ImportError("MACE is not installed. "
                              "See https://github.com/ACEsuit/mace")
        super().__init__()
        self.mace_model = mace_model
        self._last_feats = None

        last_product = self.mace_model.products[-1]
        last_product.register_forward_hook(self._hook)

        # Auto-detect scalar feature dimension (L=0 irreps)
        self.feat_dim = last_product.linear.irreps_out.count(o3.Irrep(0, 1))
        print(f"MACEFeatureExtractor: feat_dim={self.feat_dim} "
              f"(from products[-1])")

    def _hook(self, module, input, output):
        self._last_feats = output

    @torch.no_grad()
    def forward(self, data, compute_force: bool = False):
        """Returns (mace_output_dict, node_feats [n_atoms, feat_dim])."""
        self._last_feats = None
        out = self.mace_model(data, compute_force=compute_force)
        assert self._last_feats is not None, "Forward hook did not fire."
        return out, self._last_feats


# ---------------------------------------------------------------------------
# Load backbone
# ---------------------------------------------------------------------------

def load_mace(model_path: str, device: str = 'cuda') -> MACEFeatureExtractor:
    """Load a frozen MACE model and return a MACEFeatureExtractor."""
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    mace_model = torch.load(model_path, map_location=device)
    mace_model.to(device).eval().float()   # cast to float32
    for p in mace_model.parameters():
        p.requires_grad = False
    extractor = MACEFeatureExtractor(mace_model)
    print(f"r_max={mace_model.r_max:.2f}, "
          f"num_interactions={mace_model.num_interactions.item()}, "
          f"feat_dim={extractor.feat_dim}")
    return extractor


def get_z_table(mace_extractor: MACEFeatureExtractor) -> 'AtomicNumberTable':
    zs = mace_extractor.mace_model.atomic_numbers.cpu().tolist()
    return AtomicNumberTable(sorted(zs))


# ---------------------------------------------------------------------------
# Data loading (XYZ → PyG DataLoader)
# ---------------------------------------------------------------------------

def _get_energy(atoms):
    for key in ('energy', 'REF_energy', 'DFT_energy'):
        val = atoms.info.get(key)
        if val is not None:
            return float(val)
    if atoms.calc is not None and 'energy' in getattr(atoms.calc, 'results', {}):
        return float(atoms.calc.results['energy'])
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        return None


def atoms_to_atomic_data(atoms, z_table, r_max, heads=None):
    if heads is None:
        heads = ['Default']
    energy = _get_energy(atoms) or 0.0
    config = Configuration(
        atomic_numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions(),
        properties={
            'energy': energy,
            'forces': atoms.arrays.get('forces', np.zeros((len(atoms), 3))),
        },
        property_weights={'energy': 1.0, 'forces': 1.0},
        pbc=tuple(atoms.get_pbc().tolist()),
        cell=np.array(atoms.get_cell()),
        config_type=atoms.info.get('config_type', 'Default'),
        head='Default',
    )
    return AtomicData.from_config(config, z_table=z_table,
                                  cutoff=r_max, heads=heads)


def load_xyz_dataloader(xyz_path: str, z_table, r_max: float,
                        batch_size: int, shuffle: bool = True,
                        max_structures: int = None):
    """Load an XYZ file into a PyG DataLoader."""
    from tqdm.auto import tqdm
    atoms_list = ase.io.read(xyz_path, index=':')
    if max_structures:
        atoms_list = atoms_list[:max_structures]
    dataset = []
    for atoms in tqdm(atoms_list, desc='Converting', leave=False):
        try:
            dataset.append(atoms_to_atomic_data(atoms, z_table, r_max))
        except Exception:
            pass
    return torch_geometric.DataLoader(dataset, batch_size=batch_size,
                                      shuffle=shuffle, drop_last=False)


def train_val_split_loader(xyz_path: str, z_table, r_max: float,
                           batch_size: int, valid_fraction: float = 0.1,
                           seed: int = 42):
    """Load XYZ and return (train_loader, val_loader)."""
    import numpy as np
    from tqdm.auto import tqdm
    atoms_list = ase.io.read(xyz_path, index=':')
    rng = np.random.default_rng(seed)
    idx = np.arange(len(atoms_list))
    rng.shuffle(idx)
    n_val = max(1, int(len(atoms_list) * valid_fraction))
    val_set = set(idx[:n_val].tolist())
    train_data, val_data = [], []
    for i, atoms in enumerate(tqdm(atoms_list, desc='Converting', leave=False)):
        try:
            d = atoms_to_atomic_data(atoms, z_table, r_max)
            (val_data if i in val_set else train_data).append(d)
        except Exception:
            pass
    train_loader = torch_geometric.DataLoader(train_data, batch_size=batch_size,
                                              shuffle=True,  drop_last=False)
    val_loader   = torch_geometric.DataLoader(val_data,   batch_size=batch_size,
                                              shuffle=False, drop_last=False)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Batch processing  (PyG-style flat atom tensors → [B, N_max, D] padded)
# ---------------------------------------------------------------------------

def process_batch_mace(batch, device: str, extractor: MACEFeatureExtractor):
    """
    Run MACE on a PyG batch and return PROBE-compatible padded tensors.

    Returns:
        atom_feats:   [B, N_max, feat_dim]
        atom_mask:    [B, N_max] bool
        pred_energy:  [B]
        true_energy:  [B]
        n_atoms:      [B]
    """
    # Cast batch floats to float32 to match MACE model
    batch = batch.to(device)
    for key in batch.keys:
        attr = getattr(batch, key, None)
        if isinstance(attr, torch.Tensor) and attr.is_floating_point():
            setattr(batch, key, attr.float())

    mace_out, node_feats_flat = extractor(batch)  # node_feats: [n_atoms_total, D]

    ptr         = batch.ptr                         # [B+1]
    pred_energy = mace_out['energy']                # [B]
    true_energy = batch.energy                      # [B]
    B           = ptr.shape[0] - 1
    D           = node_feats_flat.shape[1]
    sizes       = (ptr[1:] - ptr[:-1]).tolist()
    N_max       = max(sizes)

    atom_feats = torch.zeros(B, N_max, D, device=device)
    atom_mask  = torch.zeros(B, N_max, dtype=torch.bool, device=device)
    for i in range(B):
        s, e = ptr[i].item(), ptr[i + 1].item()
        n = e - s
        atom_feats[i, :n] = node_feats_flat[s:e]
        atom_mask[i, :n]  = True

    n_atoms = atom_mask.sum(dim=1).float()
    return atom_feats, atom_mask, pred_energy, true_energy, n_atoms
