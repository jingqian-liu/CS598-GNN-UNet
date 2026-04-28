"""
PyTorch Dataset + Lightning DataModule for the ATOM3D LBA task.
Returns protein-ligand HeteroData graphs ready for UnetHeteroGVPForLBA.
"""
import os
from typing import Optional

import atom3d.datasets.datasets as da
import lightning as L
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data, HeteroData

from proteinworkshop.datasets.components.atom3d_dataset import biopandas_mapping
from proteinworkshop.datasets.components.hetero_graph import build_hetero_graph
from proteinworkshop.features.factory import ProteinFeaturiser
from proteinworkshop.features.sequence_features import amino_acid_one_hot
from proteinworkshop.models.graph_encoders.ligand_gvp import featurize_ligand_from_lba

_DATA_ROOT = "/scratch/ziyiz14/data/GNN_Unet/data"

# Shared featuriser config (CA-level, same as fold classification)
_FEATURISER = ProteinFeaturiser(
    representation="CA",
    scalar_node_features=[
        "amino_acid_one_hot",
        "dihedrals",
        "sidechain_torsions",
        "alpha",
        "kappa",
    ],
    vector_node_features=["orientation"],
    edge_types=["knn_16"],
    scalar_edge_features=[
        "edge_distance",
        "edge_type",
        "node_features",
        "sequence_distance",
    ],
    vector_edge_features=["edge_vectors"],
)


def _featurise_protein(elem: dict) -> Data:
    """Convert raw LBA dict entry → featurised protein CA-level Data."""
    from graphein.protein.tensor.io import protein_to_pyg

    df = elem["atoms_protein"]
    df = df[df["element"] != "H"].reset_index(drop=True)
    df = df.rename(columns=biopandas_mapping)
    df["residue_name"] = df["residue_name"].str.replace("WAT", "HOH")
    df["record_name"] = "ATOM"
    df["atom_number"] = range(1, len(df) + 1)

    protein = protein_to_pyg(df=df)
    protein.x = torch.zeros(protein.coords.shape[0])
    protein.amino_acid_one_hot = amino_acid_one_hot(protein)
    protein.seq_pos = torch.arange(protein.coords.shape[0]).unsqueeze(-1)

    batch = Batch.from_data_list([protein])
    batch = _FEATURISER(batch)

    result = Data()
    result.pos              = batch.pos
    result.x                = batch.x
    result.x_vector_attr    = batch.x_vector_attr
    result.edge_index       = batch.edge_index
    result.edge_attr        = batch.edge_attr
    result.edge_vector_attr = batch.edge_vector_attr
    return result


def _build_sample(elem: dict, cross_cutoff: float) -> HeteroData:
    """Full pipeline: raw LMDB entry → HeteroData graph."""
    protein = _featurise_protein(elem)
    ligand  = featurize_ligand_from_lba(elem["atoms_ligand"], elem["bonds"])
    hg      = build_hetero_graph(protein, ligand, cross_cutoff=cross_cutoff)
    hg["graph_y"] = torch.tensor(
        elem["scores"]["neglog_aff"], dtype=torch.float32
    )
    return hg


class LBAHeteroDataset(Dataset):
    """
    Wraps an ATOM3D LMDB LBA dataset; returns protein-ligand HeteroData graphs.

    Samples that fail featurisation (bad valence, missing atoms, etc.) are
    skipped transparently by trying the next index.
    """

    def __init__(self, lmdb_path: str, cross_cutoff: float = 6.0):
        self.raw     = da.LMDBDataset(lmdb_path)
        self.cutoff  = cross_cutoff
        self._len    = len(self.raw)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Optional[HeteroData]:
        # Try up to 5 consecutive samples to recover from rare featurisation errors
        for attempt in range(idx, min(idx + 5, self._len)):
            try:
                return _build_sample(self.raw[attempt], self.cutoff)
            except Exception:
                continue
        return None   # extremely rare; filtered in collate_fn


def _collate(batch):
    """Filter None entries then batch with PyG Batch.from_data_list."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return Batch.from_data_list(batch)


class LBAHeteroDataModule(L.LightningDataModule):
    """
    Lightning DataModule for the ATOM3D LBA task using HeteroData graphs.

    :param lba_split: 30 or 60 (sequence identity threshold for train/val split)
    :param cross_cutoff: distance cutoff (Å) for protein-ligand cross edges
    :param batch_size: mini-batch size
    :param num_workers: DataLoader worker processes (0 = main process only)
    :param pin_memory: pin DataLoader memory for GPU transfer
    :param data_root: path to the LBA splits directory
    """

    def __init__(
        self,
        lba_split: int = 30,
        cross_cutoff: float = 6.0,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_root: str = _DATA_ROOT,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_ds: Optional[LBAHeteroDataset] = None
        self.val_ds:   Optional[LBAHeteroDataset] = None
        self.test_ds:  Optional[LBAHeteroDataset] = None

    def _split_path(self, phase: str) -> str:
        return os.path.join(
            self.hparams.data_root,
            f"split-by-sequence-identity-{self.hparams.lba_split}",
            "data",
            phase,
        )

    def setup(self, stage: Optional[str] = None):
        if self.train_ds is None:
            self.train_ds = LBAHeteroDataset(self._split_path("train"), self.hparams.cross_cutoff)
            self.val_ds   = LBAHeteroDataset(self._split_path("val"),   self.hparams.cross_cutoff)
            self.test_ds  = LBAHeteroDataset(self._split_path("test"),  self.hparams.cross_cutoff)

    def _loader(self, ds, shuffle=False, drop_last=False):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=_collate,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._loader(self.val_ds)

    def test_dataloader(self):
        return self._loader(self.test_ds)
