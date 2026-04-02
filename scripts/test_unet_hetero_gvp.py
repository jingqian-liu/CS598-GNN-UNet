"""
Test script: forward pass of UnetHeteroGVP on a batch of 8 LBA HeteroData graphs.

Usage:
    python scripts/test_unet_hetero_gvp.py [--split 30|60] [--cutoff 6.0] [--layers 5]
"""
import argparse
import sys

import torch
from torch_geometric.data import Batch, Data, HeteroData

sys.path.insert(0, "/data/server5/jl126/GNN_UNet")

import atom3d.datasets.datasets as da
from proteinworkshop.datasets.components.atom3d_dataset import (
    LBATransform,
    biopandas_mapping,
)
from proteinworkshop.datasets.components.hetero_graph import build_hetero_graph
from proteinworkshop.features.factory import ProteinFeaturiser
from proteinworkshop.features.sequence_features import amino_acid_one_hot
from proteinworkshop.models.graph_encoders.ligand_gvp import featurize_ligand_from_lba
from proteinworkshop.models.graph_encoders.unet_hetero_gvp import (
    UnetHeteroGVP,
    UnetHeteroGVPForLBA,
)

BATCH_SIZE = 8
BASE = "/data/server5/jl126/GNN_UNet/proteinworkshop/data/LBA/splits"

FEATURISER = ProteinFeaturiser(
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


def featurise_protein(elem: dict) -> Data:
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
    batch = FEATURISER(batch)

    result = Data()
    result.pos              = batch.pos
    result.x                = batch.x
    result.x_vector_attr    = batch.x_vector_attr
    result.edge_index       = batch.edge_index
    result.edge_attr        = batch.edge_attr
    result.edge_vector_attr = batch.edge_vector_attr
    return result


def main(lba_split: int = 30, cross_cutoff: float = 6.0, num_layers: int = 5):
    data_path = f"{BASE}/split-by-sequence-identity-{lba_split}/data/train"
    print(f"LBA split-{lba_split} | cross cutoff: {cross_cutoff} Å | layers: {num_layers}\n")

    raw_dataset = da.LMDBDataset(data_path)

    # Build batch of HeteroData graphs
    hetero_list = []
    skipped = 0
    idx = 0
    while len(hetero_list) < BATCH_SIZE and idx < len(raw_dataset):
        try:
            elem    = raw_dataset[idx]
            protein = featurise_protein(elem)
            ligand  = featurize_ligand_from_lba(elem["atoms_ligand"], elem["bonds"])
            hg      = build_hetero_graph(protein, ligand, cross_cutoff=cross_cutoff)
            hg["graph_y"] = torch.tensor(elem["scores"]["neglog_aff"])
            hetero_list.append(hg)
        except Exception as e:
            print(f"  [warn] skipped sample {idx}: {e}")
            skipped += 1
        idx += 1

    print(f"Built {len(hetero_list)} HeteroData graphs (skipped {skipped})\n")

    batch = Batch.from_data_list(hetero_list)

    # ------------------------------------------------------------------ #
    # Encoder-only test                                                   #
    # ------------------------------------------------------------------ #
    encoder = UnetHeteroGVP(
        s_dim=128, v_dim=16, s_dim_edge=32, v_dim_edge=1,
        r_max=10.0, num_bessel=8, num_polynomial_cutoff=5,
        num_layers=num_layers, pool="sum",
        fps_ratio=0.6, cross_cutoff=cross_cutoff, drop_rate=0.1,
    )
    encoder.eval()

    print("Running encoder forward pass ...")
    with torch.no_grad():
        enc_out = encoder(batch)

    enc_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Encoder parameters : {enc_params:,}")
    print(f"  graph_embedding    : {tuple(enc_out['graph_embedding'].shape)}")
    print(f"  node_embedding     : {tuple(enc_out['node_embedding'].shape)}")
    print(f"  lig_node_embedding : {tuple(enc_out['lig_node_embedding'].shape)}")

    # ------------------------------------------------------------------ #
    # Full model (encoder + regression head) test                         #
    # ------------------------------------------------------------------ #
    print()
    model = UnetHeteroGVPForLBA(
        s_dim=128, v_dim=16, s_dim_edge=32, v_dim_edge=1,
        r_max=10.0, num_bessel=8, num_polynomial_cutoff=5,
        num_layers=num_layers, pool="sum",
        fps_ratio=0.6, cross_cutoff=cross_cutoff, enc_drop_rate=0.1,
        head_hidden_dim=256, head_drop_rate=0.1,
    )
    model.eval()

    print("Running full model forward pass (with loss) ...")
    with torch.no_grad():
        out = model(batch)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters   : {total_params:,}")

    targets = batch["graph_y"].float()
    print(f"\n{'Graph':>6}  {'Target':>8}  {'Pred':>8}  {'|Err|':>8}")
    print(f"  {'-'*38}")
    for i in range(BATCH_SIZE):
        t = targets[i].item()
        p = out["pred"][i].item()
        print(f"  {i:>4}  {t:>8.3f}  {p:>8.3f}  {abs(t-p):>8.3f}")

    print(f"\n  MSE loss : {out['loss'].item():.4f}")
    print("\nFull model OK.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",  type=int,   default=30, choices=[30, 60])
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--layers", type=int,   default=5)
    args = parser.parse_args()
    main(args.split, args.cutoff, args.layers)
