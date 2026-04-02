"""Build a protein-ligand heterogeneous graph (PyG HeteroData).

Node types
----------
    'protein' : one node per residue (Cα-level)
    'ligand'  : one node per heavy atom

Edge types
----------
    ('protein', 'contact',   'protein') — KNN-16 on Cα, existing PP edges
    ('ligand',  'bond',      'ligand')  — covalent bonds, existing LL edges
    ('protein', 'interacts', 'ligand')  — eps_6 Å distance cutoff, protein→ligand
    ('ligand',  'interacts', 'protein') — eps_6 Å distance cutoff, ligand→protein
"""

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import radius as pyg_radius


def build_hetero_graph(
    protein: Data,
    ligand: Data,
    cross_cutoff: float = 6.0,
) -> HeteroData:
    """Combine a featurised protein graph and a ligand graph into a HeteroData.

    :param protein: Featurised protein graph produced by ``ProteinFeaturiser``.
        Expected attributes: ``pos`` [N_p, 3], ``x`` [N_p, 41],
        ``x_vector_attr`` [N_p, 2, 3], ``edge_index`` [2, E_pp],
        ``edge_attr`` [E_pp, 85], ``edge_vector_attr`` [E_pp, 1, 3].
    :param ligand: Ligand graph produced by ``featurize_ligand_from_lba``.
        Expected attributes: ``pos`` [N_l, 3], ``x`` [N_l, 40],
        ``edge_index`` [2, E_ll], ``edge_attr`` [E_ll, 6].
    :param cross_cutoff: Distance cutoff in Å for protein-ligand cross edges.
        Default 6.0 Å (direct binding-pocket contacts).
    :return: ``HeteroData`` with four edge types.
    """
    data = HeteroData()

    # ------------------------------------------------------------------ #
    # Protein nodes                                                        #
    # ------------------------------------------------------------------ #
    data["protein"].x = protein.x                          # [N_p, 41]
    data["protein"].x_vector_attr = protein.x_vector_attr  # [N_p, 2, 3]
    data["protein"].pos = protein.pos                      # [N_p, 3]

    # ------------------------------------------------------------------ #
    # Ligand nodes                                                         #
    # ------------------------------------------------------------------ #
    data["ligand"].x = ligand.x    # [N_l, 40]
    data["ligand"].pos = ligand.pos  # [N_l, 3]

    # ------------------------------------------------------------------ #
    # Protein-Protein edges  ('protein', 'contact', 'protein')            #
    # KNN-16 on Cα coordinates — carried over directly from featuriser    #
    # ------------------------------------------------------------------ #
    data["protein", "contact", "protein"].edge_index = protein.edge_index
    data["protein", "contact", "protein"].edge_attr = protein.edge_attr
    data["protein", "contact", "protein"].edge_vector_attr = (
        protein.edge_vector_attr
    )

    # ------------------------------------------------------------------ #
    # Ligand-Ligand edges  ('ligand', 'bond', 'ligand')                   #
    # Covalent bonds (bidirectional) — carried over from featuriser       #
    # ------------------------------------------------------------------ #
    ll_src, ll_dst = ligand.edge_index[0], ligand.edge_index[1]
    ll_vecs = ligand.pos[ll_src] - ligand.pos[ll_dst]              # [E_ll, 3]
    ll_lens = torch.linalg.norm(ll_vecs, dim=-1, keepdim=True)     # [E_ll, 1]
    ll_unit = torch.nan_to_num(ll_vecs / ll_lens).unsqueeze(-2)    # [E_ll, 1, 3]

    data["ligand", "bond", "ligand"].edge_index = ligand.edge_index
    data["ligand", "bond", "ligand"].edge_attr = ligand.edge_attr
    data["ligand", "bond", "ligand"].edge_vector_attr = ll_unit

    # ------------------------------------------------------------------ #
    # Cross edges  ('protein'/'ligand', 'interacts', 'ligand'/'protein')  #
    # eps_6 Å cutoff between Cα positions and ligand heavy-atom positions #
    # radius(x=ligand, y=protein, r) → row0=protein idx, row1=ligand idx  #
    # ------------------------------------------------------------------ #
    cross = pyg_radius(ligand.pos, protein.pos, r=cross_cutoff)
    prot_idx = cross[0]  # [E_cross]
    lig_idx  = cross[1]  # [E_cross]

    # Geometric features shared by both directions
    vecs = protein.pos[prot_idx] - ligand.pos[lig_idx]        # [E, 3]
    dists = torch.linalg.norm(vecs, dim=-1, keepdim=True)     # [E, 1]
    unit_vecs = torch.nan_to_num(vecs / dists).unsqueeze(-2)  # [E, 1, 3]

    # protein → ligand
    data["protein", "interacts", "ligand"].edge_index = torch.stack(
        [prot_idx, lig_idx], dim=0
    )
    data["protein", "interacts", "ligand"].edge_attr = dists          # [E, 1]
    data["protein", "interacts", "ligand"].edge_vector_attr = unit_vecs  # [E, 1, 3]

    # ligand → protein  (reverse direction, negate vector)
    data["ligand", "interacts", "protein"].edge_index = torch.stack(
        [lig_idx, prot_idx], dim=0
    )
    data["ligand", "interacts", "protein"].edge_attr = dists
    data["ligand", "interacts", "protein"].edge_vector_attr = -unit_vecs

    return data
