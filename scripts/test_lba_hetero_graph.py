"""
Test script: build a batch of 8 protein-ligand HeteroData graphs from LBA
and print the dimensions of all node/edge features per type.

Usage:
    python scripts/test_lba_hetero_graph.py [--split 30|60] [--cutoff 6.0]
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
from proteinworkshop.models.graph_encoders.ligand_gvp import (
    featurize_ligand_from_lba,
)

BATCH_SIZE = 8
BASE = "/data/server5/jl126/GNN_UNet/proteinworkshop/data/LBA/splits"

# ------------------------------------------------------------------ #
# Shared protein featuriser (ca_everything)                            #
# ------------------------------------------------------------------ #
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
    """Convert LBA atoms_protein dict entry → featurised protein Data."""
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

    # Featuriser requires Batch; manually copy non-standard attrs back to Data
    batch = Batch.from_data_list([protein])
    batch = FEATURISER(batch)

    result = Data()
    result.pos             = batch.pos
    result.x               = batch.x
    result.x_vector_attr   = batch.x_vector_attr
    result.edge_index      = batch.edge_index
    result.edge_attr       = batch.edge_attr
    result.edge_vector_attr = batch.edge_vector_attr
    return result


def main(lba_split: int = 30, cross_cutoff: float = 6.0):
    data_path = f"{BASE}/split-by-sequence-identity-{lba_split}/data/train"
    print(f"Loading LBA split-{lba_split} | cross cutoff: {cross_cutoff} Å\n")

    raw_dataset = da.LMDBDataset(data_path)

    # ------------------------------------------------------------------ #
    # Build batch of 8 HeteroData graphs                                  #
    # ------------------------------------------------------------------ #
    hetero_list = []
    skipped = 0
    idx = 0
    while len(hetero_list) < BATCH_SIZE and idx < len(raw_dataset):
        try:
            elem = raw_dataset[idx]
            protein = featurise_protein(elem)
            ligand  = featurize_ligand_from_lba(
                elem["atoms_ligand"], elem["bonds"]
            )
            hg = build_hetero_graph(protein, ligand, cross_cutoff=cross_cutoff)
            hg["graph_y"] = torch.tensor(elem["scores"]["neglog_aff"])
            hg["pdb_id"]  = elem["id"]
            hetero_list.append(hg)
        except Exception as e:
            print(f"  [warn] skipped sample {idx}: {e}")
            skipped += 1
        idx += 1

    print(f"Built {len(hetero_list)} HeteroData graphs (skipped {skipped})\n")

    # ------------------------------------------------------------------ #
    # Batch                                                                #
    # ------------------------------------------------------------------ #
    batch = Batch.from_data_list(hetero_list)

    # ------------------------------------------------------------------ #
    # Per-graph summary                                                    #
    # ------------------------------------------------------------------ #
    prot_batch = batch["protein"].batch
    lig_batch  = batch["ligand"].batch

    print("=" * 70)
    print(f"{'Per-graph summary':^70}")
    print("=" * 70)
    print(f"  {'#':<4} {'PDB':<8} {'Prot nodes':>12} {'Lig nodes':>10} "
          f"{'PP edges':>10} {'LL edges':>10} {'Cross→':>10} {'←Cross':>10}")
    print(f"  {'-'*68}")

    pp_batch  = batch["protein", "contact",   "protein"].edge_index
    ll_batch  = batch["ligand",  "bond",       "ligand"].edge_index
    pl_batch  = batch["protein", "interacts",  "ligand"].edge_index
    lp_batch  = batch["ligand",  "interacts",  "protein"].edge_index

    for i in range(BATCH_SIZE):
        n_p  = (prot_batch == i).sum().item()
        n_l  = (lig_batch  == i).sum().item()
        e_pp = (prot_batch[pp_batch[0]] == i).sum().item()
        e_ll = (lig_batch [ll_batch[0]] == i).sum().item()
        e_pl = (prot_batch[pl_batch[0]] == i).sum().item()
        e_lp = (lig_batch [lp_batch[0]] == i).sum().item()
        pdb  = hetero_list[i]["pdb_id"]
        print(f"  {i:<4} {pdb:<8} {n_p:>12} {n_l:>10} "
              f"{e_pp:>10} {e_ll:>10} {e_pl:>10} {e_lp:>10}")

    # ------------------------------------------------------------------ #
    # Feature dimensions                                                   #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 70)
    print(f"{'Feature dimensions (full batch)':^70}")
    print("=" * 70)

    print()
    print("  NODE TYPES")
    print(f"  {'protein'.upper()}")
    print(f"    x (scalar)        : {tuple(batch['protein'].x.shape)}"
          f"  — 41 dims")
    print(f"      amino_acid_one_hot : 23")
    print(f"      dihedrals          : 6  (phi,psi,omega sin/cos)")
    print(f"      sidechain_torsions : 8  (chi1-chi4 sin/cos)")
    print(f"      alpha              : 2  (sin/cos)")
    print(f"      kappa              : 2  (sin/cos)")
    print(f"    x_vector_attr     : {tuple(batch['protein'].x_vector_attr.shape)}"
          f"  — orientation [2 x 3]")
    print(f"    pos               : {tuple(batch['protein'].pos.shape)}"
          f"  — Cα coordinates")
    print()
    print(f"  {'ligand'.upper()}")
    print(f"    x (scalar)        : {tuple(batch['ligand'].x.shape)}"
          f"  — 40 dims")
    print(f"      atom_type          : 11  one-hot")
    print(f"      hybridization      : 7   one-hot")
    print(f"      degree             : 8   one-hot")
    print(f"      num_H              : 6   one-hot")
    print(f"      formal_charge      : 6   one-hot")
    print(f"      is_aromatic        : 1   binary")
    print(f"      is_in_ring         : 1   binary")
    print(f"    pos               : {tuple(batch['ligand'].pos.shape)}"
          f"  — heavy atom coordinates")

    print()
    print("  EDGE TYPES")
    pp_ea  = batch["protein", "contact",   "protein"].edge_attr
    pp_ev  = batch["protein", "contact",   "protein"].edge_vector_attr
    ll_ea  = batch["ligand",  "bond",       "ligand"].edge_attr
    pl_ea  = batch["protein", "interacts",  "ligand"].edge_attr
    pl_ev  = batch["protein", "interacts",  "ligand"].edge_vector_attr

    print(f"  ('protein', 'contact', 'protein')  — KNN-16")
    print(f"    edge_index        : {tuple(pp_batch.shape)}")
    print(f"    edge_attr         : {tuple(pp_ea.shape)}  — 85 dims")
    print(f"      edge_distance      : 1   Euclidean dist between Cα")
    print(f"      edge_type          : 1   KNN type index")
    print(f"      node_features_cat  : 82  concat src+dst node features")
    print(f"      sequence_distance  : 1   j-i sequence position")
    print(f"    edge_vector_attr  : {tuple(pp_ev.shape)}  — norm. direction Cα→Cα")

    ll_ev  = batch["ligand",  "bond",       "ligand"].edge_vector_attr
    print()
    print(f"  ('ligand', 'bond', 'ligand')  — covalent bonds")
    print(f"    edge_index        : {tuple(ll_batch.shape)}")
    print(f"    edge_attr         : {tuple(ll_ea.shape)}  — 6 dims")
    print(f"      bond_type          : 4   one-hot (SINGLE/DOUBLE/TRIPLE/AROM)")
    print(f"      is_in_ring         : 1   binary")
    print(f"      is_conjugated      : 1   binary")
    print(f"    edge_vector_attr  : {tuple(ll_ev.shape)}  — norm. direction atom→atom")

    print()
    print(f"  ('protein', 'interacts', 'ligand')  — eps_{cross_cutoff:.0f} Å")
    print(f"    edge_index        : {tuple(pl_batch.shape)}")
    print(f"    edge_attr         : {tuple(pl_ea.shape)}  — 1 dim (distance)")
    print(f"    edge_vector_attr  : {tuple(pl_ev.shape)}  — norm. direction Cα→atom")

    lp_ea  = batch["ligand",  "interacts",  "protein"].edge_attr
    lp_ev  = batch["ligand",  "interacts",  "protein"].edge_vector_attr
    print()
    print(f"  ('ligand', 'interacts', 'protein')  — eps_{cross_cutoff:.0f} Å (reverse)")
    print(f"    edge_index        : {tuple(lp_batch.shape)}")
    print(f"    edge_attr         : {tuple(lp_ea.shape)}  — 1 dim (distance)")
    print(f"    edge_vector_attr  : {tuple(lp_ev.shape)}  — norm. direction atom→Cα")

    print()
    print(f"  Labels (graph_y): {batch['graph_y']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",  type=int,   default=30, choices=[30, 60])
    parser.add_argument("--cutoff", type=float, default=6.0)
    args = parser.parse_args()
    main(args.split, args.cutoff)
