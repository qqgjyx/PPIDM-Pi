"""MIMIC-IV-ECG dataset stub and data-loading utilities.

This module provides the dataset interface expected by the training
pipeline.  Actual data must be downloaded separately from PhysioNet
(credentialed access required).
"""

import torch
from torch.utils.data import DataLoader, Dataset

from ppidm_pi.physics import ELECTROLYTE_BINS as _PHYS_BINS

MIMIC_IV_ECG_URL = "https://physionet.org/content/mimic-iv-ecg/1.0/"

ELECTROLYTE_BINS = {
    ion: {"edges": _PHYS_BINS[ion]["edges"], "labels": ["low", "normal", "high"]}
    for ion in _PHYS_BINS
}


class ECGElectrolyteDataset(Dataset):
    """MIMIC-IV-ECG electrolyte dataset stub.

    Expected data layout::

        root/
        ├── ecg/          # .npy files, each shape [12, 256] (12-lead, 256 samples)
        └── labels.csv    # columns: filename, K_bin, Ca_bin, Mg_bin  (values 0/1/2)

    Preprocessing (to be done externally):
      1. Download MIMIC-IV-ECG from PhysioNet (credentialed access).
      2. Extract 10-second 12-lead ECGs at 500 Hz → segment into beats → resample to 256.
      3. Match ECGs with lab results from MIMIC-IV Clinical (chartevents).
      4. Bin electrolyte values into low/normal/high using ``ELECTROLYTE_BINS``.
      5. Save each beat as a ``.npy`` file and create ``labels.csv``.

    Args:
        root: Path to preprocessed data directory.
        electrolyte: One of ``"K"``, ``"Ca"``, ``"Mg"``.
        split: One of ``"train"``, ``"val"``, ``"test"``.
    """

    def __init__(self, root: str, electrolyte: str = "K", split: str = "train"):
        self.root = root
        self.electrolyte = electrolyte
        self.split = split
        self.samples: list[tuple[str, int]] = []
        # In a real implementation, parse labels.csv and populate self.samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return ``(ecg_beat [12, 256], bin_label)``."""
        raise NotImplementedError(
            "Download and preprocess MIMIC-IV-ECG data first. "
            f"See {MIMIC_IV_ECG_URL}"
        )


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Standard DataLoader wrapper."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
