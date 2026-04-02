"""
Test script: load a batch of 8 ligand graphs from the LBA dataset and print
graph dimensions (nodes, edges, feature dimensions).

Usage:
    python scripts/test_lba_ligand_graph.py [--split 30|60]
"""
import argparse
import sys

import torch
from torch_geometric.data import Batch

sys.path.insert(0, "/data/server5/jl126/GNN_UNet")

import atom3d.datasets.datasets as da
from proteinworkshop.models.graph_encoders.ligand_gvp import (
    BOND_FEATURE_DIM,
    NODE_FEATURE_DIM,
    featurize_ligand_from_lba,
)

BATCH_SIZE = 8
BASE = "/data/server5/jl126/GNN_UNet/proteinworkshop/data/LBA/splits"


def main(lba_split: int = 30):
    data_path = f"{BASE}/split-by-sequence-identity-{lba_split}/data/train"
    print(f"Loading LBA split-{lba_split} from: {data_path}\n")

    dataset = da.LMDBDataset(data_path)

    # ------------------------------------------------------------------ #
    # Build batch of 8 ligand graphs                                       #
    # ------------------------------------------------------------------ #
    graphs = []
    skipped = 0
    idx = 0
    while len(graphs) < BATCH_SIZE and idx < len(dataset):
        try:
            elem = dataset[idx]
            g = featurize_ligand_from_lba(elem["atoms_ligand"], elem["bonds"])
            g.graph_y = torch.tensor(elem["scores"]["neglog_aff"])
            g.pdb_id = elem["id"]
            graphs.append(g)
        except Exception as e:
            print(f"  [warn] skipped sample {idx}: {e}")
            skipped += 1
        idx += 1

    print(f"Loaded {len(graphs)} ligand graphs (skipped {skipped} errors)\n")

    batch = Batch.from_data_list(graphs)

    # ------------------------------------------------------------------ #
    # Per-graph stats                                                      #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print(f"{'Per-graph summary':^60}")
    print("=" * 60)
    print(f"  {'Graph':<8} {'PDB':<8} {'#nodes (heavy atoms)':<24} {'#edges (bonds x2)'}")
    print(f"  {'-'*55}")
    node_counts = [(batch.batch == i).sum().item() for i in range(BATCH_SIZE)]
    edge_counts = [
        (batch.batch[batch.edge_index[0]] == i).sum().item()
        for i in range(BATCH_SIZE)
    ]
    for i, (g, n, e) in enumerate(zip(graphs, node_counts, edge_counts)):
        print(f"  {i:<8} {g.pdb_id:<8} {n:<24} {e}")

    # ------------------------------------------------------------------ #
    # Feature dimensions                                                   #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print(f"{'Feature dimensions (full batch)':^60}")
    print("=" * 60)
    print(f"  Total nodes               : {batch.x.shape[0]}")
    print(f"  Total edges               : {batch.edge_index.shape[1]}")
    print()
    print(f"  Node features (scalar)    : {tuple(batch.x.shape)}  [N x {NODE_FEATURE_DIM}]")
    print(f"    - atom_type one-hot     : 11  (C,N,O,S,F,Cl,Br,I,P,B,other)")
    print(f"    - hybridization one-hot : 7   (S,SP,SP2,SP3,SP3D,SP3D2,other)")
    print(f"    - degree one-hot        : 8   (0-6, other)")
    print(f"    - num H one-hot         : 6   (0-4, other)")
    print(f"    - formal charge one-hot : 6   (-2,-1,0,1,2, other)")
    print(f"    - is aromatic           : 1")
    print(f"    - is in ring            : 1")
    print()
    print(f"  Node positions (3D)       : {tuple(batch.pos.shape)}  [N x 3]")
    print()
    print(f"  Edge features (scalar)    : {tuple(batch.edge_attr.shape)}  [E x {BOND_FEATURE_DIM}]")
    print(f"    - bond type one-hot     : 4   (SINGLE,DOUBLE,TRIPLE,AROMATIC)")
    print(f"    - is in ring            : 1")
    print(f"    - is conjugated         : 1")
    print()
    print(f"  Labels (graph_y)          : {batch.graph_y}  (neglog affinity)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, default=30, choices=[30, 60])
    args = parser.parse_args()
    main(args.split)
