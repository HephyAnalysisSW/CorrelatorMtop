import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import numexpr as ne

class H5Dataset(Dataset):
    def __init__(self, filename, gen_keys, rec_keys, selection: str = None, fraction: float = 1.0, seed: int = 42):
        self.filename = filename
        self.gen_keys = gen_keys
        self.rec_keys = rec_keys
        self.selection = selection
        self.fraction = fraction
        self.seed = seed
        self.file = None

        # Build mask and subsample
        with h5py.File(self.filename, 'r') as f:
            total_len = f[gen_keys[0]].shape[0]

            if selection is None:
                indices = np.arange(total_len)
            else:
                try:
                    context = {key: f[key][:] for key in self._extract_keys(selection)}
                    mask = ne.evaluate(selection, local_dict=context)
                    indices = np.where(mask)[0]
                except Exception as e:
                    raise ValueError(f"Failed to apply selection '{selection}': {e}")

            # Apply fractional subsampling
            if not 0 < fraction <= 1.0:
                raise ValueError(f"fraction must be in (0, 1], got {fraction}")

            if fraction < 1.0:
                np.random.seed(seed)
                sample_size = int(len(indices) * fraction)
                indices = np.random.choice(indices, size=sample_size, replace=False)
                indices = np.sort(indices)

            self.indices = indices

    def _extract_keys(self, selection_str):
        import re
        return list(set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', selection_str)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.filename, 'r')
        real_idx = self.indices[idx]
        gen = np.stack([self.file[k][real_idx] for k in self.gen_keys])
        rec = np.stack([self.file[k][real_idx] for k in self.rec_keys])
        return torch.tensor(gen, dtype=torch.float32), torch.tensor(rec, dtype=torch.float32)
