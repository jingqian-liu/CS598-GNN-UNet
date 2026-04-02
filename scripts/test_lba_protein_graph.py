"""
Test script: load a batch of 8 protein graphs from the LBA dataset
(full protein chain, CA-level, ligand excluded) and print graph dimensions.

Usage:
    python scripts/test_lba_protein_graph.py [--split 30|60]
"""
import argparse
import sys

import torch
from torch_geometric.data import Batch

sys.path.insert(0, "/data/server5/jl126/GNN_UNet")

import atom3d.datasets.datasets as da
from proteinworkshop.datasets.components.atom3d_dataset import LBATransform
from proteinworkshop.features.factory import ProteinFeaturiser

BATCH_SIZE = 8
BASE = "/data/server5/jl126/GNN_UNet/proteinworkshop/data/LBA/splits"


def main(lba_split: int = 30):
    data_path = f"{BASE}/split-by-sequence-identity-{lba_split}/data/train"
    print(f"Loading LBA split-{lba_split} from: {data_path}")

    dataset = da.LMDBDataset(data_path, transform=LBATransform())
    print(f"Dataset size (train): {len(dataset)}\n")

    # ------------------------------------------------------------------ #
    # Featuriser — same as ca_everything used in fold classification       #
    # ------------------------------------------------------------------ #
    featuriser = ProteinFeaturiser(
        representation="CA",
        scalar_node_features=[
            "amino_acid_one_hot",   # 23-dim
            "dihedrals",            # 6-dim  (phi, psi, omega as sin/cos)
            "sidechain_torsions",   # 8-dim  (chi1-chi4 as sin/cos)
            "alpha",                # 2-dim  (virtual CA angle, sin/cos)
            "kappa",                # 2-dim  (virtual CA angle, sin/cos)
        ],
        vector_node_features=["orientation"],   # 2x3
        edge_types=["knn_16"],
        scalar_edge_features=[
            "edge_distance",        # 1-dim
            "edge_type",            # 1-dim
            "node_features",        # 82-dim (concat of two endpoint node vecs)
            "sequence_distance",    # 1-dim
        ],
        vector_edge_features=["edge_vectors"],  # 1x3
    )

    # ------------------------------------------------------------------ #
    # Build batch of 8                                                     #
    # ------------------------------------------------------------------ #
    graphs = []
    skipped = 0
    idx = 0
    while len(graphs) < BATCH_SIZE and idx < len(dataset):
        try:
            g = dataset[idx]
            # seq_pos required by PositionalEncoding inside featuriser
            g.seq_pos = torch.arange(g.coords.shape[0]).unsqueeze(-1)
            graphs.append(g)
        except Exception as e:
            print(f"  [warn] skipped sample {idx}: {e}")
            skipped += 1
        idx += 1

    print(f"Loaded {len(graphs)} graphs (skipped {skipped} errors)\n")

    batch = Batch.from_data_list(graphs)
    result = featuriser(batch)

    # ------------------------------------------------------------------ #
    # Per-graph stats                                                      #
    # ------------------------------------------------------------------ #
    print("=" * 55)
    print(f"{'Per-graph summary':^55}")
    print("=" * 55)
    node_counts = [(result.batch == i).sum().item() for i in range(BATCH_SIZE)]
    edge_counts = [
        (result.batch[result.edge_index[0]] == i).sum().item()
        for i in range(BATCH_SIZE)
    ]
    print(f"  {'Graph':<8} {'#nodes (residues)':<22} {'#edges (KNN-16)'}")
    print(f"  {'-'*50}")
    for i, (n, e) in enumerate(zip(node_counts, edge_counts)):
        print(f"  {i:<8} {n:<22} {e}")

    # ------------------------------------------------------------------ #
    # Feature dimensions                                                   #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 55)
    print(f"{'Feature dimensions (full batch)':^55}")
    print("=" * 55)
    print(f"  Total nodes               : {result.pos.shape[0]}")
    print(f"  Total edges               : {result.edge_index.shape[1]}")
    print()
    print(f"  Node features (scalar)    : {result.x.shape}  [N x 41]")
    print(f"    - amino_acid_one_hot    : 23")
    print(f"    - dihedrals             : 6  (phi,psi,omega sin/cos)")
    print(f"    - sidechain_torsions    : 8  (chi1-chi4 sin/cos)")
    print(f"    - alpha                 : 2  (sin/cos)")
    print(f"    - kappa                 : 2  (sin/cos)")
    print()
    print(f"  Node features (vector)    : {result.x_vector_attr.shape}  [N x 2 x 3]")
    print(f"    - orientation           : 2 vectors (fwd + bwd along backbone)")
    print()
    print(f"  Edge features (scalar)    : {result.edge_attr.shape}  [E x 85]")
    print(f"    - edge_distance         : 1")
    print(f"    - edge_type             : 1")
    print(f"    - node_features (cat)   : 82  (41+41 endpoint features)")
    print(f"    - sequence_distance     : 1")
    print()
    print(f"  Edge features (vector)    : {result.edge_vector_attr.shape}  [E x 1 x 3]")
    print(f"    - edge_vectors          : normalised direction vector")
    print()
    print(f"  Labels (graph_y)          : {result.graph_y}  (neglog affinity)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, default=30, choices=[30, 60])
    args = parser.parse_args()
    main(args.split)
