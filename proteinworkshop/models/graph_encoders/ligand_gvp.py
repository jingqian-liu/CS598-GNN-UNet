"""Ligand GVP encoder for small molecules.

Encodes a 3D ligand graph using Geometric Vector Perceptrons (GVP),
producing per-atom node embeddings and a global graph embedding.
The output format matches EncoderOutput, making it compatible with the
rest of the ProteinWorkshop pipeline.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import radius_graph

import proteinworkshop.models.graph_encoders.layers.gvp as gvp
from proteinworkshop.models.graph_encoders.components import blocks
from proteinworkshop.models.utils import get_aggregation
from proteinworkshop.types import EncoderOutput

# fmt: off
# Atom type vocabulary (heavy atoms only)
ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "other"]
# Hybridization vocabulary
HYBRIDIZATIONS = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "other"]
# Degree vocabulary (number of explicit bonds)
DEGREES = [0, 1, 2, 3, 4, 5, 6, "other"]
# Hydrogen count vocabulary
H_COUNTS = [0, 1, 2, 3, 4, "other"]
# Formal charge vocabulary
FORMAL_CHARGES = [-2, -1, 0, 1, 2, "other"]
# Bond type vocabulary
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
# fmt: on

# Total scalar node feature dimension (sum of all one-hots + binary flags)
# ATOM_TYPES(11) + HYBRIDIZATIONS(7) + DEGREES(8) + H_COUNTS(6) +
# FORMAL_CHARGES(6) + aromaticity(1) + in_ring(1) = 40
NODE_FEATURE_DIM = (
    len(ATOM_TYPES)
    + len(HYBRIDIZATIONS)
    + len(DEGREES)
    + len(H_COUNTS)
    + len(FORMAL_CHARGES)
    + 2  # aromaticity, in_ring
)

# Total scalar edge feature dimension from featurizer (bond type + flags)
# BOND_TYPES(4) + in_ring(1) + conjugated(1) = 6
BOND_FEATURE_DIM = len(BOND_TYPES) + 2


def _one_hot(value, vocab: list) -> list:
    """One-hot encode value against vocab, using last bin for unknowns."""
    if value not in vocab:
        value = vocab[-1]  # "other" bucket
    return [int(value == v) for v in vocab]


def featurize_ligand(mol) -> Data:
    """Convert an RDKit molecule with a 3D conformer into a PyG Data object.

    The molecule must already have a 3D conformer (e.g. generated via
    ``AllChem.EmbedMolecule`` or loaded from a PDB/SDF).

    Node features (``data.x``):
        - Atom type one-hot          (11 dims)
        - Hybridization one-hot      ( 7 dims)
        - Degree one-hot             ( 8 dims)
        - Num Hs one-hot             ( 6 dims)
        - Formal charge one-hot      ( 6 dims)
        - Is aromatic                ( 1 dim )
        - Is in ring                 ( 1 dim )
        Total: 40 dims

    Edge features (``data.edge_attr``):
        - Bond type one-hot          ( 4 dims)
        - Is in ring                 ( 1 dim )
        - Is conjugated              ( 1 dim )
        Total: 6 dims

    Edges are bidirectional bonds (no self-loops). Distance-based
    non-bonded edges can optionally be added inside LigandGVPEncoder.

    :param mol: RDKit ``Mol`` object with at least one conformer.
    :return: PyG ``Data`` with fields ``x``, ``pos``, ``edge_index``,
             ``edge_attr``.
    """
    from rdkit.Chem import rdMolDescriptors  # noqa: F401 (lazy import)

    conf = mol.GetConformer()

    # ---- Node features ----
    node_feats = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        hyb = str(atom.GetHybridization()).split(".")[-1]
        feat = (
            _one_hot(symbol, ATOM_TYPES)
            + _one_hot(hyb, HYBRIDIZATIONS)
            + _one_hot(atom.GetDegree(), DEGREES)
            + _one_hot(atom.GetTotalNumHs(), H_COUNTS)
            + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
            + [int(atom.GetIsAromatic()), int(atom.IsInRing())]
        )
        node_feats.append(feat)

    x = torch.tensor(node_feats, dtype=torch.float)  # [N, NODE_FEATURE_DIM]

    # ---- 3D positions ----
    pos = torch.tensor(
        conf.GetPositions(), dtype=torch.float
    )  # [N, 3]

    # ---- Edges (bidirectional bonds) ----
    src, dst, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        btype = str(bond.GetBondTypeAsDouble())
        # Map float bond order to string key used in BOND_TYPES
        _btype_map = {"1.0": "SINGLE", "2.0": "DOUBLE", "3.0": "TRIPLE", "1.5": "AROMATIC"}
        btype_key = _btype_map.get(btype, "SINGLE")
        feat = (
            _one_hot(btype_key, BOND_TYPES)
            + [int(bond.IsInRing()), int(bond.GetIsConjugated())]
        )
        # Add both directions
        src += [i, j]
        dst += [j, i]
        edge_feats += [feat, feat]

    edge_index = torch.tensor([src, dst], dtype=torch.long)  # [2, E]
    edge_attr = torch.tensor(edge_feats, dtype=torch.float)  # [E, BOND_FEATURE_DIM]

    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)


def featurize_ligand_from_lba(atoms_df, bonds_df) -> Data:
    """Convert LBA ``atoms_ligand`` + ``bonds`` DataFrames into a PyG Data object.

    The LBA dataset stores ligands as heavy-atom-only DataFrames with explicit
    bond tables (Kekulized: bond type 1.0/2.0/3.0). This function builds an
    RDKit mol from that data, sanitizes it so that hybridization, aromaticity,
    ring membership, and conjugation flags are all computed correctly, then
    delegates to :func:`featurize_ligand`.

    Two quirks of the LBA dataset are handled here:
    - Element symbols are stored in uppercase (e.g. ``'CL'``, ``'BR'``);
      RDKit requires title-case (``'Cl'``, ``'Br'``).
    - Some molecules fail full sanitization (e.g. unusual valence from the
      way the PDB records bonds). In that case we fall back to computing only
      the subset of sanitization flags needed for featurization.

    :param atoms_df: ``atoms_ligand`` DataFrame from an LBA LMDB entry.
        Must contain columns ``element``, ``x``, ``y``, ``z``.
    :param bonds_df: ``bonds`` DataFrame from an LBA LMDB entry.
        Must contain columns ``atom1`` (int), ``atom2`` (int), ``type`` (float:
        1.0 = single, 2.0 = double, 3.0 = triple, 1.5 = aromatic).
    :return: PyG ``Data`` with fields ``x``, ``pos``, ``edge_index``,
        ``edge_attr`` — identical format to :func:`featurize_ligand`.
    """
    from rdkit import Chem

    bond_type_map = {
        1.0: Chem.BondType.SINGLE,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        1.5: Chem.BondType.AROMATIC,
    }

    atoms = atoms_df.reset_index(drop=True)

    rw = Chem.RWMol()
    for _, row in atoms.iterrows():
        # LBA stores elements in ALL-CAPS (e.g. 'CL', 'BR'); RDKit needs
        # title-case ('Cl', 'Br'). str.capitalize() handles all cases.
        element = str(row["element"]).capitalize()
        rw.AddAtom(Chem.Atom(element))

    for _, row in bonds_df.iterrows():
        i, j = int(row["atom1"]), int(row["atom2"])
        btype = bond_type_map.get(float(row["type"]), Chem.BondType.SINGLE)
        if rw.GetBondBetweenAtoms(i, j) is None:
            rw.AddBond(i, j, btype)

    mol = rw.GetMol()

    # Full sanitization computes hybridization, aromaticity, ring info, and
    # conjugation — all needed by featurize_ligand. If it fails (e.g. unusual
    # valence from PDB bond records), fall back to the subset that doesn't
    # enforce valence rules but still computes the other properties.
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_FINDRADICALS
            | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
            | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
        )

    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, row in atoms.iterrows():
        conf.SetAtomPosition(idx, (float(row["x"]), float(row["y"]), float(row["z"])))
    mol.AddConformer(conf, assignId=True)

    return featurize_ligand(mol)


class LigandGVPEncoder(torch.nn.Module):
    """GVP-GNN encoder for 3D ligand graphs.

    Architecture mirrors ``GVPGNNModel`` (protein encoder) so that the
    output embedding dimension is directly compatible for downstream
    protein-ligand interaction layers.

    Node and edge embeddings are initialised from ligand atom / bond
    features.  Distances are encoded with the same Bessel radial basis
    used in the protein encoder.  Optionally, non-bonded edges up to
    ``cutoff`` Å are added on top of covalent bonds.

    :param s_dim: Scalar channel width for node embeddings. (default: 128)
    :param v_dim: Vector channel width for node embeddings. (default: 16)
    :param s_dim_edge: Scalar channel width for edge embeddings. (default: 32)
    :param v_dim_edge: Vector channel width for edge embeddings. (default: 1)
    :param r_max: Maximum distance for radial basis / non-bonded edges.
        (default: 10.0 Å)
    :param num_bessel: Number of Bessel basis functions. (default: 8)
    :param num_polynomial_cutoff: Polynomial cutoff order. (default: 5)
    :param num_layers: Number of GVPConvLayers. (default: 4)
    :param pool: Pooling method for graph-level embedding. (default: ``"sum"``)
    :param residual: Use residual connections. (default: ``True``)
    :param add_distance_edges: If ``True``, augment bond edges with all
        atom pairs within ``r_max`` Å (non-bonded contacts). (default: ``False``)
    """

    def __init__(
        self,
        s_dim: int = 128,
        v_dim: int = 16,
        s_dim_edge: int = 32,
        v_dim_edge: int = 1,
        r_max: float = 10.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        num_layers: int = 4,
        pool: str = "sum",
        residual: bool = True,
        add_distance_edges: bool = False,
    ):
        super().__init__()
        _NODE_DIM = (s_dim, v_dim)
        _EDGE_DIM = (s_dim_edge, v_dim_edge)
        self.r_max = r_max
        self.num_layers = num_layers
        self.add_distance_edges = add_distance_edges
        activations = (F.relu, None)

        # ---- Node input embedding ----
        # LazyLinear infers NODE_FEATURE_DIM at first forward pass
        self.emb_in = torch.nn.LazyLinear(s_dim)
        self.W_v = torch.nn.Sequential(
            gvp.LayerNorm((s_dim, 0)),
            gvp.GVP(
                (s_dim, 0),
                _NODE_DIM,
                activations=(None, None),
                vector_gate=True,
            ),
        )

        # ---- Edge input embedding ----
        # Radial basis encodes distance; bond features are concatenated
        self.radial_embedding = blocks.RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        # Bond scalar features are projected to match radial dim before
        # passing into the GVP edge embedder via a small linear layer
        radial_dim = self.radial_embedding.out_dim  # = num_bessel
        self.bond_proj = torch.nn.Linear(BOND_FEATURE_DIM, radial_dim)
        self.W_e = torch.nn.Sequential(
            gvp.LayerNorm((radial_dim, 1)),
            gvp.GVP(
                (radial_dim, 1),
                _EDGE_DIM,
                activations=(None, None),
                vector_gate=True,
            ),
        )

        # ---- GVP message-passing layers ----
        self.layers = torch.nn.ModuleList(
            gvp.GVPConvLayer(
                _NODE_DIM,
                _EDGE_DIM,
                activations=activations,
                vector_gate=True,
                residual=residual,
            )
            for _ in range(num_layers)
        )

        # ---- Output projection (vector → scalar) ----
        self.W_out = torch.nn.Sequential(
            gvp.LayerNorm(_NODE_DIM),
            gvp.GVP(
                _NODE_DIM,
                (s_dim, 0),
                activations=activations,
                vector_gate=True,
            ),
        )

        # ---- Global pooling ----
        self.readout = get_aggregation(pool)

    def forward(self, batch: Batch) -> EncoderOutput:
        """Encode a batch of ligand graphs.

        :param batch: PyG ``Batch`` with fields:
            - ``x``          : node features  ``[N, NODE_FEATURE_DIM]``
            - ``pos``        : 3-D coordinates ``[N, 3]``
            - ``edge_index`` : bond edges      ``[2, E]``
            - ``edge_attr``  : bond features   ``[E, BOND_FEATURE_DIM]``
                               (set to zeros for distance-only edges)
            - ``batch``      : batch assignment ``[N]``
        :return: ``EncoderOutput`` dict with:
            - ``node_embedding``  ``[N, s_dim]``
            - ``graph_embedding`` ``[B, s_dim]``
        """
        edge_index = batch.edge_index
        pos = batch.pos

        # Optionally add non-bonded distance edges within r_max
        if self.add_distance_edges:
            dist_edge_index = radius_graph(
                pos, r=self.r_max, batch=batch.batch, loop=False
            )
            # Pad bond attrs with zeros for new non-bonded edges
            n_dist = dist_edge_index.size(1)
            dist_edge_attr = torch.zeros(
                n_dist, BOND_FEATURE_DIM, device=pos.device
            )
            edge_index = torch.cat([edge_index, dist_edge_index], dim=1)
            edge_attr = torch.cat([batch.edge_attr, dist_edge_attr], dim=0)
        else:
            edge_attr = batch.edge_attr

        # ---- Edge geometric features ----
        vectors = pos[edge_index[0]] - pos[edge_index[1]]  # [E, 3]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [E, 1]
        unit_vectors = torch.nan_to_num(
            torch.div(vectors, lengths)
        ).unsqueeze(-2)  # [E, 1, 3]

        radial = self.radial_embedding(lengths)  # [E, num_bessel]

        # Combine radial basis with bond features (additive after projection)
        bond_scalar = self.bond_proj(edge_attr)  # [E, num_bessel]
        edge_scalar = radial + bond_scalar       # [E, num_bessel]

        h_E = (edge_scalar, unit_vectors)

        # ---- Node embedding ----
        h_V = self.emb_in(batch.x)   # [N, s_dim]
        h_V = self.W_v(h_V)          # (scalar [N, s_dim], vector [N, v_dim, 3])
        h_E = self.W_e(h_E)          # (scalar [E, s_dim_edge], vector [E, v_dim_edge, 3])

        # ---- Message passing ----
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)

        # ---- Readout ----
        out = self.W_out(h_V)  # [N, s_dim]

        return EncoderOutput(
            {
                "node_embedding": out,
                "graph_embedding": self.readout(out, batch.batch),
            }
        )
