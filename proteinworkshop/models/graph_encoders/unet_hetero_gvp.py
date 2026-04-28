"""
Heterogeneous UNet-GVP encoder for protein-ligand binding affinity (LBA).

Architecture
------------
- Protein residue nodes (Cα-level) are downsampled via FPS every other layer.
  Ligand heavy-atom nodes are kept at full atom resolution throughout.
- Four typed GVP message-passing streams per layer:
    PP  protein→protein  (KNN-16, rebuilt after each FPS step)
    LL  ligand→ligand    (covalent bonds, fixed throughout)
    PL  protein→ligand   (distance cutoff, rebuilt after each FPS step)
    LP  ligand→protein   (distance cutoff, rebuilt after each FPS step)
- All four message streams use messages computed from the PRE-update state
  (simultaneous message passing semantics).
- Skip connections are additive at FPS-selected positions (UNet style).
  Only protein nodes have skip connections; ligand embeddings evolve freely.
- Output: global pool (protein) ++ global pool (ligand) → [B, 2*s_dim].
"""

import functools

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import ModuleList
from torch_geometric.nn import fps, MLP, GINConv
from torch_geometric.nn import radius as pyg_radius
from torch_scatter import scatter_add

import proteinworkshop.models.graph_encoders.layers.gvp as gvp
from proteinworkshop.models.graph_encoders.components import blocks
from proteinworkshop.models.utils import get_aggregation
from proteinworkshop.models.graph_encoders.unet_gvp_enc_dec_add import (
    compute_new_edges,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _edge_feats(pos_src, pos_dst, edge_index, radial_emb):
    """
    Compute (radial_scalars, unit_direction) edge feature tuple.

    :param pos_src: [N_src, 3] source node positions
    :param pos_dst: [N_dst, 3] destination node positions
    :param edge_index: [2, E] row 0 = src idx, row 1 = dst idx
    :param radial_emb: RadialEmbeddingBlock
    :return: tuple ((s [E, R], v [E, 1, 3]))

    Note: lengths are clamped at 1e-6 Å to prevent NaN in the Bessel basis
    (sin(w*r)/r → 0/0 singularity).  Self-loops or duplicate coordinates
    in the input can otherwise produce NaN that propagates through training.
    """
    vecs    = pos_src[edge_index[0]] - pos_dst[edge_index[1]]         # [E, 3]
    lengths = torch.linalg.norm(vecs, dim=-1, keepdim=True)           # [E, 1]
    lengths_safe = lengths.clamp(min=1e-6)                             # prevents 0/0 in Bessel
    unit    = torch.nan_to_num(vecs / lengths_safe).unsqueeze(-2)     # [E, 1, 3]
    return radial_emb(lengths_safe), unit


# ------------------------------------------------------------------ #
# BipartiteGVPConv                                                     #
# ------------------------------------------------------------------ #

class BipartiteGVPConv(gvp.GVPConv):
    """
    GVPConv extended for bipartite graphs where source and destination
    node sets have the same embedding dimensions but are different objects
    (e.g. protein nodes → ligand nodes).

    Inherits all weight modules from GVPConv; only overrides forward().
    """

    def forward(self, x_src, x_dst, edge_index, edge_attr, N_src: int, N_dst: int):
        """
        :param x_src: (s [N_src, si], v [N_src, vi, 3])
        :param x_dst: (s [N_dst, si], v [N_dst, vi, 3])
        :param edge_index: [2, E] row 0 = src indices, row 1 = dst indices
        :param edge_attr: (s_e [E, se], v_e [E, ve, 3])
        :param N_src: number of source nodes
        :param N_dst: number of destination nodes
        :return: updated destination embeddings (s [N_dst, so], v [N_dst, vo, 3])
        """
        s_src, v_src = x_src
        s_dst, v_dst = x_dst
        v_src_flat = v_src.contiguous().view(N_src, self.vi * 3)
        v_dst_flat = v_dst.contiguous().view(N_dst, self.vi * 3)
        out = self.propagate(
            edge_index,
            s=(s_src, s_dst),
            v=(v_src_flat, v_dst_flat),
            edge_attr=edge_attr,
            size=(N_src, N_dst),
        )
        return gvp._split(out, self.vo)


# ------------------------------------------------------------------ #
# HeteroGVPLayer                                                       #
# ------------------------------------------------------------------ #

class HeteroGVPLayer(nn.Module):
    """
    One complete heterogeneous GVP message-passing layer.

    All four message streams (PP, LL, PL, LP) are computed from the
    PRE-update state, then combined to update protein and ligand nodes
    separately (each with its own norm, dropout, and feedforward).
    """

    def __init__(
        self,
        node_dims,
        edge_dims,
        n_message: int = 3,
        n_feedforward: int = 2,
        drop_rate: float = 0.1,
        activations=(F.relu, torch.sigmoid),
        vector_gate: bool = True,
    ):
        super().__init__()

        def _conv():
            return gvp.GVPConv(
                node_dims, node_dims, edge_dims,
                n_message, activations=activations, vector_gate=vector_gate,
            )

        def _biconv():
            return BipartiteGVPConv(
                node_dims, node_dims, edge_dims,
                n_message, activations=activations, vector_gate=vector_gate,
            )

        self.conv_pp = _conv()       # protein → protein
        self.conv_ll = _conv()       # ligand  → ligand
        self.conv_lp = _biconv()     # ligand  → protein
        self.conv_pl = _biconv()     # protein → ligand

        GVP_ = functools.partial(gvp.GVP, activations=activations, vector_gate=vector_gate)

        def _ff():
            if n_feedforward == 1:
                return nn.Sequential(GVP_(node_dims, node_dims, activations=(None, None)))
            hid = (4 * node_dims[0], 2 * node_dims[1])
            layers = [GVP_(node_dims, hid)]
            for _ in range(n_feedforward - 2):
                layers.append(GVP_(hid, hid))
            layers.append(GVP_(hid, node_dims, activations=(None, None)))
            return nn.Sequential(*layers)

        # Separate norm / dropout / FF for protein and ligand
        self.norm_p = nn.ModuleList([gvp.LayerNorm(node_dims) for _ in range(2)])
        self.norm_l = nn.ModuleList([gvp.LayerNorm(node_dims) for _ in range(2)])
        self.drop_p = nn.ModuleList([gvp.Dropout(drop_rate)   for _ in range(2)])
        self.drop_l = nn.ModuleList([gvp.Dropout(drop_rate)   for _ in range(2)])
        self.ff_p   = _ff()
        self.ff_l   = _ff()

    def forward(
        self,
        h_prot, h_lig,
        edge_pp, h_E_pp,
        edge_ll, h_E_ll,
        edge_pl, h_E_pl,
        edge_lp, h_E_lp,
    ):
        """
        :param h_prot: (s [N_p, s], v [N_p, v, 3])  protein node embeddings
        :param h_lig:  (s [N_l, s], v [N_l, v, 3])  ligand  node embeddings
        :param edge_*: [2, E] edge index tensors
        :param h_E_*:  (s_e [E, se], v_e [E, ve, 3]) edge embedding tuples
        :return: updated (h_prot, h_lig)
        """
        N_p = h_prot[0].shape[0]
        N_l = h_lig[0].shape[0]

        # --- Compute all messages from pre-update state ---
        dh_pp = self.conv_pp(h_prot, edge_pp, h_E_pp)                       # PP
        dh_ll = self.conv_ll(h_lig,  edge_ll, h_E_ll)                       # LL
        dh_lp = self.conv_lp(h_lig, h_prot, edge_lp, h_E_lp, N_l, N_p)    # LP → protein
        dh_pl = self.conv_pl(h_prot, h_lig, edge_pl, h_E_pl, N_p, N_l)     # PL → ligand

        # --- Update protein: PP + LP messages ---
        dh_p   = gvp.tuple_sum(dh_pp, dh_lp)
        h_prot = self.norm_p[0](gvp.tuple_sum(h_prot, self.drop_p[0](dh_p)))
        h_prot = self.norm_p[1](gvp.tuple_sum(h_prot, self.drop_p[1](self.ff_p(h_prot))))

        # --- Update ligand: LL + PL messages ---
        dh_l  = gvp.tuple_sum(dh_ll, dh_pl)
        h_lig = self.norm_l[0](gvp.tuple_sum(h_lig, self.drop_l[0](dh_l)))
        h_lig = self.norm_l[1](gvp.tuple_sum(h_lig, self.drop_l[1](self.ff_l(h_lig))))

        return h_prot, h_lig


# ------------------------------------------------------------------ #
# UnetHeteroGVP                                                        #
# ------------------------------------------------------------------ #

class UnetHeteroGVP(nn.Module):
    """
    Heterogeneous UNet-GVP encoder for protein-ligand graphs.

    Protein nodes are down-sampled via FPS (every other layer);
    ligand nodes remain at full atom resolution throughout.
    After each FPS step, PP and cross (PL/LP) edges are rebuilt from
    the new positions; LL edges are fixed.
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
        num_layers: int = 5,
        pool: str = "sum",
        fps_ratio: float = 0.6,
        cross_cutoff: float = 6.0,
        drop_rate: float = 0.1,
    ):
        """
        :param s_dim: scalar node embedding dimension
        :param v_dim: vector node embedding dimension (number of 3-D vectors)
        :param s_dim_edge: scalar edge embedding dimension
        :param v_dim_edge: vector edge embedding dimension
        :param r_max: maximum radius for Bessel basis
        :param num_bessel: number of Bessel radial basis functions
        :param num_polynomial_cutoff: polynomial envelope order
        :param num_layers: total number of down-path GNN layers
        :param pool: global pooling method (``"sum"`` / ``"mean"`` / ``"max"``)
        :param fps_ratio: FPS sampling ratio for protein nodes (0 < ratio < 1)
        :param cross_cutoff: distance cutoff (Å) for protein-ligand cross edges
        :param drop_rate: dropout probability
        """
        super().__init__()
        node_dims     = (s_dim, v_dim)
        edge_dims     = (s_dim_edge, v_dim_edge)
        self.s_dim        = s_dim
        self.fps_ratio    = fps_ratio
        self.cross_cutoff = cross_cutoff
        num_up_downs      = num_layers // 2
        self.num_up_downs = num_up_downs
        activations       = (F.relu, None)

        # ---------------------------------------------------------- #
        # Node input embeddings                                        #
        # ---------------------------------------------------------- #
        # Scalar projection (LazyLinear handles arbitrary input dim)
        # followed by GVP (s,0) → (s, v) to introduce vector channels.
        self.emb_in_prot = nn.LazyLinear(s_dim)
        self.emb_in_lig  = nn.LazyLinear(s_dim)
        self.W_v_prot = nn.Sequential(
            gvp.LayerNorm((s_dim, 0)),
            gvp.GVP((s_dim, 0), node_dims, activations=(None, None), vector_gate=True),
        )
        self.W_v_lig = nn.Sequential(
            gvp.LayerNorm((s_dim, 0)),
            gvp.GVP((s_dim, 0), node_dims, activations=(None, None), vector_gate=True),
        )

        # ---------------------------------------------------------- #
        # Edge embeddings                                               #
        # ---------------------------------------------------------- #
        self.radial_emb = blocks.RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        radial_dim = self.radial_emb.out_dim

        def _radial_embedder():
            return nn.Sequential(
                gvp.LayerNorm((radial_dim, 1)),
                gvp.GVP((radial_dim, 1), edge_dims, activations=(None, None), vector_gate=True),
            )

        def _bond_embedder():
            # LL bond features: 6 scalar (bond type 4-hot, ring, conjugated), 1 vector
            return nn.Sequential(
                gvp.LayerNorm((6, 1)),
                gvp.GVP((6, 1), edge_dims, activations=(None, None), vector_gate=True),
            )

        # PP edges: one embedder per FPS level (down + up)
        # W_e_pp_d[0] is shared between level-0 initial embedding and first FPS level
        self.W_e_pp_d  = ModuleList([_radial_embedder() for _ in range(num_up_downs)])
        self.W_e_pp_up = ModuleList([_radial_embedder() for _ in range(num_up_downs)])

        # LL edges: single fixed embedder (ligand topology never changes)
        self.W_e_ll = _bond_embedder()

        # Cross PL/LP edges: same embedder for both directions (distance = same;
        # direction vectors are negated for LP, which GVP handles equivariantly)
        self.W_e_cross_d  = ModuleList([_radial_embedder() for _ in range(num_up_downs)])
        self.W_e_cross_up = ModuleList([_radial_embedder() for _ in range(num_up_downs)])

        # ---------------------------------------------------------- #
        # GNN layers                                                   #
        # ---------------------------------------------------------- #
        def _layer():
            return HeteroGVPLayer(
                node_dims, edge_dims,
                drop_rate=drop_rate,
                activations=activations,
                vector_gate=True,
            )

        self.layers_d  = nn.ModuleList([_layer() for _ in range(num_layers)])
        self.layers_up = nn.ModuleList([_layer() for _ in range(num_up_downs)])

        # ---------------------------------------------------------- #
        # FPS aggregation (protein scalar via GINConv, vector via scatter) #
        # ---------------------------------------------------------- #
        self.reds_prot = ModuleList()
        for _ in range(num_up_downs):
            mlp = MLP([s_dim, s_dim, s_dim], act="relu", norm=None)
            self.reds_prot.append(GINConv(nn=mlp, train_eps=False))

        # ---------------------------------------------------------- #
        # Output projection                                            #
        # ---------------------------------------------------------- #
        self.W_out_prot = nn.Sequential(
            gvp.LayerNorm(node_dims),
            gvp.GVP(node_dims, (s_dim, 0), activations=activations, vector_gate=True),
        )
        self.W_out_lig = nn.Sequential(
            gvp.LayerNorm(node_dims),
            gvp.GVP(node_dims, (s_dim, 0), activations=activations, vector_gate=True),
        )

        self.readout = get_aggregation(pool)

    # -------------------------------------------------------------- #
    # Forward                                                          #
    # -------------------------------------------------------------- #

    def forward(self, batch):
        """
        :param batch: PyG HeteroData batch (from ``Batch.from_data_list``).
            Expected attributes:

            * ``batch["protein"].x``           [N_p, 41]
            * ``batch["protein"].pos``          [N_p, 3]
            * ``batch["protein"].batch``        [N_p]
            * ``batch["ligand"].x``             [N_l, 40]
            * ``batch["ligand"].pos``           [N_l, 3]
            * ``batch["ligand"].batch``         [N_l]
            * ``batch["protein","contact","protein"].edge_index``       [2, E_pp]
            * ``batch["ligand","bond","ligand"].edge_index``            [2, E_ll]
            * ``batch["ligand","bond","ligand"].edge_attr``             [E_ll, 6]
            * ``batch["ligand","bond","ligand"].edge_vector_attr``      [E_ll, 1, 3]
            * ``batch["protein","interacts","ligand"].edge_index``      [2, E_pl]
            * ``batch["ligand","interacts","protein"].edge_index``      [2, E_lp]

        :return: dict with keys:
            * ``"graph_embedding"``      [B, 2*s_dim]  protein+ligand pooled
            * ``"node_embedding"``       [N_p, s_dim]  protein node embeddings
            * ``"lig_node_embedding"``   [N_l, s_dim]  ligand  node embeddings
        """
        # ---------------------------------------------------------- #
        # Unpack                                                        #
        # ---------------------------------------------------------- #
        pos_prot   = batch["protein"].pos
        batch_prot = batch["protein"].batch
        pos_lig    = batch["ligand"].pos
        batch_lig  = batch["ligand"].batch

        edge_pp = batch["protein", "contact",   "protein"].edge_index
        edge_ll = batch["ligand",  "bond",       "ligand"].edge_index
        edge_pl = batch["protein", "interacts",  "ligand"].edge_index
        edge_lp = batch["ligand",  "interacts",  "protein"].edge_index

        ll_raw_s = batch["ligand", "bond", "ligand"].edge_attr         # [E_ll, 6]
        ll_raw_v = batch["ligand", "bond", "ligand"].edge_vector_attr  # [E_ll, 1, 3]

        # ---------------------------------------------------------- #
        # Initial node embeddings                                       #
        # ---------------------------------------------------------- #
        h_prot = self.W_v_prot(self.emb_in_prot(batch["protein"].x))
        h_lig  = self.W_v_lig(self.emb_in_lig(batch["ligand"].x))

        # ---------------------------------------------------------- #
        # Initial edge embeddings                                       #
        # PP: use W_e_pp_d[0] (shared with first FPS level, like orig) #
        # LL: computed once, reused throughout                          #
        # Cross: use W_e_cross_d[0]                                    #
        # ---------------------------------------------------------- #
        pp_s, pp_v = _edge_feats(pos_prot, pos_prot, edge_pp, self.radial_emb)
        h_E_pp = self.W_e_pp_d[0]((pp_s, pp_v))

        h_E_ll = self.W_e_ll((ll_raw_s, ll_raw_v))

        pl_s, pl_v = _edge_feats(pos_prot, pos_lig, edge_pl, self.radial_emb)
        h_E_pl = self.W_e_cross_d[0]((pl_s,  pl_v))
        h_E_lp = self.W_e_cross_d[0]((pl_s, -pl_v))

        # ---------------------------------------------------------- #
        # Down path                                                     #
        # ---------------------------------------------------------- #
        # skip_stack: saves (h_prot, pos_prot, batch_prot, edge_pp)
        #             at the pre-FPS resolution for each FPS step.
        # fps_stack:  saves the fps_idx that maps into the saved level.
        skip_stack = []
        fps_stack  = []

        for i, layer in enumerate(self.layers_d):
            if i % 2 == 1:
                # Save state at current (pre-FPS) resolution
                skip_stack.append((h_prot, pos_prot, batch_prot, edge_pp))

                # FPS on protein nodes
                fps_idx = fps(pos_prot, batch_prot, self.fps_ratio)
                fps_stack.append(fps_idx)

                # Aggregate protein features into selected nodes
                level = i // 2 - 1   # 0 for i=1, 1 for i=3, …
                row, col = edge_pp
                h_s = self.reds_prot[level](h_prot[0], edge_pp)[fps_idx]
                h_v = scatter_add(
                    h_prot[1][row], col, dim=0, dim_size=h_prot[1].shape[0]
                )[fps_idx]
                h_prot = (h_s, h_v)

                pos_prot   = pos_prot[fps_idx]
                batch_prot = batch_prot[fps_idx]

                # Rebuild PP edges (KNN-16 on down-sampled Cα)
                edge_pp, _ = compute_new_edges(pos_prot, batch_prot, "knn_16")
                pp_s, pp_v = _edge_feats(pos_prot, pos_prot, edge_pp, self.radial_emb)
                h_E_pp = self.W_e_pp_d[level]((pp_s, pp_v))

                # Rebuild cross edges (distance cutoff between new Cα and ligand atoms)
                cross  = pyg_radius(
                    pos_lig, pos_prot, r=self.cross_cutoff,
                    batch_x=batch_lig, batch_y=batch_prot,
                )
                edge_pl = cross              # row0 = prot, row1 = lig
                edge_lp = cross[[1, 0]]      # row0 = lig,  row1 = prot

                pl_s, pl_v = _edge_feats(pos_prot, pos_lig, edge_pl, self.radial_emb)
                h_E_pl = self.W_e_cross_d[level]((pl_s,  pl_v))
                h_E_lp = self.W_e_cross_d[level]((pl_s, -pl_v))

            h_prot, h_lig = layer(
                h_prot, h_lig,
                edge_pp, h_E_pp,
                edge_ll, h_E_ll,
                edge_pl, h_E_pl,
                edge_lp, h_E_lp,
            )

        # ---------------------------------------------------------- #
        # Up path                                                       #
        # ---------------------------------------------------------- #
        for i, layer in enumerate(self.layers_up):
            fps_idx = fps_stack.pop()
            h_prot_saved, pos_prot, batch_prot, edge_pp = skip_stack.pop()

            # Additive skip connection at FPS-selected positions
            h_s_saved, h_v_saved = h_prot_saved
            h_s_saved[fps_idx] = h_s_saved[fps_idx] + h_prot[0]
            h_v_saved[fps_idx] = h_v_saved[fps_idx] + h_prot[1]
            h_prot = (h_s_saved, h_v_saved)

            # Recompute PP edge features at restored resolution
            pp_s, pp_v = _edge_feats(pos_prot, pos_prot, edge_pp, self.radial_emb)
            h_E_pp = self.W_e_pp_up[i]((pp_s, pp_v))

            # Rebuild cross edges for restored protein resolution
            cross  = pyg_radius(
                pos_lig, pos_prot, r=self.cross_cutoff,
                batch_x=batch_lig, batch_y=batch_prot,
            )
            edge_pl = cross
            edge_lp = cross[[1, 0]]

            pl_s, pl_v = _edge_feats(pos_prot, pos_lig, edge_pl, self.radial_emb)
            h_E_pl = self.W_e_cross_up[i]((pl_s,  pl_v))
            h_E_lp = self.W_e_cross_up[i]((pl_s, -pl_v))

            h_prot, h_lig = layer(
                h_prot, h_lig,
                edge_pp, h_E_pp,
                edge_ll, h_E_ll,
                edge_pl, h_E_pl,
                edge_lp, h_E_lp,
            )

        # ---------------------------------------------------------- #
        # Output projection + global pooling                           #
        # ---------------------------------------------------------- #
        out_prot = self.W_out_prot(h_prot)   # [N_p, s_dim]
        out_lig  = self.W_out_lig(h_lig)     # [N_l, s_dim]

        prot_graph_emb = self.readout(out_prot, batch_prot)   # [B, s_dim]
        lig_graph_emb  = self.readout(out_lig,  batch_lig)    # [B, s_dim]

        graph_emb = torch.cat([prot_graph_emb, lig_graph_emb], dim=-1)  # [B, 2*s_dim]

        return {
            "graph_embedding":     graph_emb,   # [B, 2*s_dim]
            "node_embedding":      out_prot,     # [N_p, s_dim]  (full-resolution protein)
            "lig_node_embedding":  out_lig,      # [N_l, s_dim]
        }

class UnetHeteroGVPEncoderOnly(nn.Module):
    """
    Heterogeneous UNet-GVP encoder for protein-ligand graphs.
    Only use the down path for prediction.

    Protein nodes are down-sampled via FPS (every other layer);
    ligand nodes remain at full atom resolution throughout.
    After each FPS step, PP and cross (PL/LP) edges are rebuilt from
    the new positions; LL edges are fixed.
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
        num_layers: int = 5,
        pool: str = "sum",
        fps_ratio: float = 0.6,
        cross_cutoff: float = 6.0,
        drop_rate: float = 0.1,
    ):
        """
        :param s_dim: scalar node embedding dimension
        :param v_dim: vector node embedding dimension (number of 3-D vectors)
        :param s_dim_edge: scalar edge embedding dimension
        :param v_dim_edge: vector edge embedding dimension
        :param r_max: maximum radius for Bessel basis
        :param num_bessel: number of Bessel radial basis functions
        :param num_polynomial_cutoff: polynomial envelope order
        :param num_layers: total number of down-path GNN layers
        :param pool: global pooling method (``"sum"`` / ``"mean"`` / ``"max"``)
        :param fps_ratio: FPS sampling ratio for protein nodes (0 < ratio < 1)
        :param cross_cutoff: distance cutoff (Å) for protein-ligand cross edges
        :param drop_rate: dropout probability
        """
        super().__init__()
        node_dims     = (s_dim, v_dim)
        edge_dims     = (s_dim_edge, v_dim_edge)
        self.s_dim        = s_dim
        self.fps_ratio    = fps_ratio
        self.cross_cutoff = cross_cutoff
        num_up_downs      = num_layers // 2
        self.num_up_downs = num_up_downs
        activations       = (F.relu, None)

        # ---------------------------------------------------------- #
        # Node input embeddings                                        #
        # ---------------------------------------------------------- #
        # Scalar projection (LazyLinear handles arbitrary input dim)
        # followed by GVP (s,0) → (s, v) to introduce vector channels.
        self.emb_in_prot = nn.LazyLinear(s_dim)
        self.emb_in_lig  = nn.LazyLinear(s_dim)
        self.W_v_prot = nn.Sequential(
            gvp.LayerNorm((s_dim, 0)),
            gvp.GVP((s_dim, 0), node_dims, activations=(None, None), vector_gate=True),
        )
        self.W_v_lig = nn.Sequential(
            gvp.LayerNorm((s_dim, 0)),
            gvp.GVP((s_dim, 0), node_dims, activations=(None, None), vector_gate=True),
        )

        # ---------------------------------------------------------- #
        # Edge embeddings                                               #
        # ---------------------------------------------------------- #
        self.radial_emb = blocks.RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        radial_dim = self.radial_emb.out_dim

        def _radial_embedder():
            return nn.Sequential(
                gvp.LayerNorm((radial_dim, 1)),
                gvp.GVP((radial_dim, 1), edge_dims, activations=(None, None), vector_gate=True),
            )

        def _bond_embedder():
            # LL bond features: 6 scalar (bond type 4-hot, ring, conjugated), 1 vector
            return nn.Sequential(
                gvp.LayerNorm((6, 1)),
                gvp.GVP((6, 1), edge_dims, activations=(None, None), vector_gate=True),
            )

        # PP edges: one embedder per FPS level (down + up)
        # W_e_pp_d[0] is shared between level-0 initial embedding and first FPS level
        self.W_e_pp_d  = ModuleList([_radial_embedder() for _ in range(num_up_downs)])
        self.W_e_pp_up = ModuleList([_radial_embedder() for _ in range(num_up_downs)])

        # LL edges: single fixed embedder (ligand topology never changes)
        self.W_e_ll = _bond_embedder()

        # Cross PL/LP edges: same embedder for both directions (distance = same;
        # direction vectors are negated for LP, which GVP handles equivariantly)
        self.W_e_cross_d  = ModuleList([_radial_embedder() for _ in range(num_up_downs)])
        self.W_e_cross_up = ModuleList([_radial_embedder() for _ in range(num_up_downs)])

        # ---------------------------------------------------------- #
        # GNN layers                                                   #
        # ---------------------------------------------------------- #
        def _layer():
            return HeteroGVPLayer(
                node_dims, edge_dims,
                drop_rate=drop_rate,
                activations=activations,
                vector_gate=True,
            )

        self.layers_d  = nn.ModuleList([_layer() for _ in range(num_layers)])
        self.layers_up = nn.ModuleList([_layer() for _ in range(num_up_downs)])

        # ---------------------------------------------------------- #
        # FPS aggregation (protein scalar via GINConv, vector via scatter) #
        # ---------------------------------------------------------- #
        self.reds_prot = ModuleList()
        for _ in range(num_up_downs):
            mlp = MLP([s_dim, s_dim, s_dim], act="relu", norm=None)
            self.reds_prot.append(GINConv(nn=mlp, train_eps=False))

        # ---------------------------------------------------------- #
        # Output projection                                            #
        # ---------------------------------------------------------- #
        self.W_out_prot = nn.Sequential(
            gvp.LayerNorm(node_dims),
            gvp.GVP(node_dims, (s_dim, 0), activations=activations, vector_gate=True),
        )
        self.W_out_lig = nn.Sequential(
            gvp.LayerNorm(node_dims),
            gvp.GVP(node_dims, (s_dim, 0), activations=activations, vector_gate=True),
        )

        self.readout = get_aggregation(pool)

    # -------------------------------------------------------------- #
    # Forward                                                          #
    # -------------------------------------------------------------- #

    def forward(self, batch):
        """
        :param batch: PyG HeteroData batch (from ``Batch.from_data_list``).
            Expected attributes:

            * ``batch["protein"].x``           [N_p, 41]
            * ``batch["protein"].pos``          [N_p, 3]
            * ``batch["protein"].batch``        [N_p]
            * ``batch["ligand"].x``             [N_l, 40]
            * ``batch["ligand"].pos``           [N_l, 3]
            * ``batch["ligand"].batch``         [N_l]
            * ``batch["protein","contact","protein"].edge_index``       [2, E_pp]
            * ``batch["ligand","bond","ligand"].edge_index``            [2, E_ll]
            * ``batch["ligand","bond","ligand"].edge_attr``             [E_ll, 6]
            * ``batch["ligand","bond","ligand"].edge_vector_attr``      [E_ll, 1, 3]
            * ``batch["protein","interacts","ligand"].edge_index``      [2, E_pl]
            * ``batch["ligand","interacts","protein"].edge_index``      [2, E_lp]

        :return: dict with keys:
            * ``"graph_embedding"``      [B, 2*s_dim]  protein+ligand pooled
            * ``"node_embedding"``       [N_p, s_dim]  protein node embeddings
            * ``"lig_node_embedding"``   [N_l, s_dim]  ligand  node embeddings
        """
        # ---------------------------------------------------------- #
        # Unpack                                                        #
        # ---------------------------------------------------------- #
        pos_prot   = batch["protein"].pos
        batch_prot = batch["protein"].batch
        pos_lig    = batch["ligand"].pos
        batch_lig  = batch["ligand"].batch

        edge_pp = batch["protein", "contact",   "protein"].edge_index
        edge_ll = batch["ligand",  "bond",       "ligand"].edge_index
        edge_pl = batch["protein", "interacts",  "ligand"].edge_index
        edge_lp = batch["ligand",  "interacts",  "protein"].edge_index

        ll_raw_s = batch["ligand", "bond", "ligand"].edge_attr         # [E_ll, 6]
        ll_raw_v = batch["ligand", "bond", "ligand"].edge_vector_attr  # [E_ll, 1, 3]

        # ---------------------------------------------------------- #
        # Initial node embeddings                                       #
        # ---------------------------------------------------------- #
        h_prot = self.W_v_prot(self.emb_in_prot(batch["protein"].x))
        h_lig  = self.W_v_lig(self.emb_in_lig(batch["ligand"].x))

        # ---------------------------------------------------------- #
        # Initial edge embeddings                                       #
        # PP: use W_e_pp_d[0] (shared with first FPS level, like orig) #
        # LL: computed once, reused throughout                          #
        # Cross: use W_e_cross_d[0]                                    #
        # ---------------------------------------------------------- #
        pp_s, pp_v = _edge_feats(pos_prot, pos_prot, edge_pp, self.radial_emb)
        h_E_pp = self.W_e_pp_d[0]((pp_s, pp_v))

        h_E_ll = self.W_e_ll((ll_raw_s, ll_raw_v))

        pl_s, pl_v = _edge_feats(pos_prot, pos_lig, edge_pl, self.radial_emb)
        h_E_pl = self.W_e_cross_d[0]((pl_s,  pl_v))
        h_E_lp = self.W_e_cross_d[0]((pl_s, -pl_v))

        # ---------------------------------------------------------- #
        # Down path                                                     #
        # ---------------------------------------------------------- #
        # skip_stack: saves (h_prot, pos_prot, batch_prot, edge_pp)
        #             at the pre-FPS resolution for each FPS step.
        # fps_stack:  saves the fps_idx that maps into the saved level.
        skip_stack = []
        fps_stack  = []

        for i, layer in enumerate(self.layers_d):
            if i % 2 == 1:
                # Save state at current (pre-FPS) resolution
                skip_stack.append((h_prot, pos_prot, batch_prot, edge_pp))

                # FPS on protein nodes
                fps_idx = fps(pos_prot, batch_prot, self.fps_ratio)
                fps_stack.append(fps_idx)

                # Aggregate protein features into selected nodes
                level = i // 2 - 1   # 0 for i=1, 1 for i=3, …
                row, col = edge_pp
                h_s = self.reds_prot[level](h_prot[0], edge_pp)[fps_idx]
                h_v = scatter_add(
                    h_prot[1][row], col, dim=0, dim_size=h_prot[1].shape[0]
                )[fps_idx]
                h_prot = (h_s, h_v)

                pos_prot   = pos_prot[fps_idx]
                batch_prot = batch_prot[fps_idx]

                # Rebuild PP edges (KNN-16 on down-sampled Cα)
                edge_pp, _ = compute_new_edges(pos_prot, batch_prot, "knn_16")
                pp_s, pp_v = _edge_feats(pos_prot, pos_prot, edge_pp, self.radial_emb)
                h_E_pp = self.W_e_pp_d[level]((pp_s, pp_v))

                # Rebuild cross edges (distance cutoff between new Cα and ligand atoms)
                cross  = pyg_radius(
                    pos_lig, pos_prot, r=self.cross_cutoff,
                    batch_x=batch_lig, batch_y=batch_prot,
                )
                edge_pl = cross              # row0 = prot, row1 = lig
                edge_lp = cross[[1, 0]]      # row0 = lig,  row1 = prot

                pl_s, pl_v = _edge_feats(pos_prot, pos_lig, edge_pl, self.radial_emb)
                h_E_pl = self.W_e_cross_d[level]((pl_s,  pl_v))
                h_E_lp = self.W_e_cross_d[level]((pl_s, -pl_v))

            h_prot, h_lig = layer(
                h_prot, h_lig,
                edge_pp, h_E_pp,
                edge_ll, h_E_ll,
                edge_pl, h_E_pl,
                edge_lp, h_E_lp,
            )

        # ---------------------------------------------------------- #
        # Output projection + global pooling                           #
        # ---------------------------------------------------------- #
        out_prot = self.W_out_prot(h_prot)   # [N_p, s_dim]
        out_lig  = self.W_out_lig(h_lig)     # [N_l, s_dim]

        prot_graph_emb = self.readout(out_prot, batch_prot)   # [B, s_dim]
        lig_graph_emb  = self.readout(out_lig,  batch_lig)    # [B, s_dim]

        graph_emb = torch.cat([prot_graph_emb, lig_graph_emb], dim=-1)  # [B, 2*s_dim]

        return {
            "graph_embedding":     graph_emb,   # [B, 2*s_dim]
            "node_embedding":      out_prot,     # [N_p, s_dim]  (full-resolution protein)
            "lig_node_embedding":  out_lig,      # [N_l, s_dim]
        }


# ------------------------------------------------------------------ #
# LBA Regression Head                                                  #
# ------------------------------------------------------------------ #

class LBARegressionHead(nn.Module):
    """
    MLP regression head that maps a graph-level embedding to a single
    binding affinity scalar (neglog Kd/Ki/IC50).

    Architecture: LayerNorm → Linear(in, h) → GELU → Dropout
                           → Linear(h, h//2) → GELU → Dropout
                           → Linear(h//2, 1)

    Using GELU (smoother than ReLU) and LayerNorm on the input are
    common choices for regression on pooled GNN embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        drop_rate: float = 0.1,
    ):
        """
        :param in_dim: Input dimension (= 2*s_dim from UnetHeteroGVP).
        :param hidden_dim: First hidden layer width.
        :param drop_rate: Dropout probability applied after each hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim,         hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim,     hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, graph_emb: torch.Tensor) -> torch.Tensor:
        """
        :param graph_emb: [B, in_dim]
        :return: [B] predicted binding affinity (neglog units)
        """
        return self.net(graph_emb).squeeze(-1)   # [B]


# ------------------------------------------------------------------ #
# Full model: encoder + regression head                                #
# ------------------------------------------------------------------ #

class UnetHeteroGVPForLBA(nn.Module):
    """
    End-to-end protein-ligand binding affinity predictor.

    Encoder : UnetHeteroGVP  →  graph_embedding [B, 2*s_dim]
    Head    : LBARegressionHead  →  predicted_affinity [B]

    Loss is MSE between predicted and target neglog affinity.
    """

    def __init__(
        self,
        # encoder hyper-parameters
        s_dim: int = 128,
        v_dim: int = 16,
        s_dim_edge: int = 32,
        v_dim_edge: int = 1,
        r_max: float = 10.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        num_layers: int = 5,
        pool: str = "sum",
        fps_ratio: float = 0.6,
        cross_cutoff: float = 6.0,
        enc_drop_rate: float = 0.1,
        # head hyper-parameters
        head_hidden_dim: int = 256,
        head_drop_rate: float = 0.1,
    ):
        super().__init__()
        self.encoder = UnetHeteroGVP(
            s_dim=s_dim,
            v_dim=v_dim,
            s_dim_edge=s_dim_edge,
            v_dim_edge=v_dim_edge,
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            num_layers=num_layers,
            pool=pool,
            fps_ratio=fps_ratio,
            cross_cutoff=cross_cutoff,
            drop_rate=enc_drop_rate,
        )
        self.head = LBARegressionHead(
            in_dim=2 * s_dim,
            hidden_dim=head_hidden_dim,
            drop_rate=head_drop_rate,
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        """
        :param batch: HeteroData batch (see UnetHeteroGVP.forward).
        :return: dict with keys:
            * ``"pred"``   [B]  predicted affinities
            * ``"loss"``   scalar MSE loss (only if ``batch["graph_y"]`` exists)
        """
        enc_out = self.encoder(batch)
        pred    = self.head(enc_out["graph_embedding"])   # [B]

        out = {"pred": pred}
        if hasattr(batch, "graph_y") or "graph_y" in batch:
            target = batch["graph_y"].float()
            out["loss"] = self.loss_fn(pred, target)
        return out

class UnetHeteroGVPEncoderOnlyForLBA(nn.Module):
    """
    End-to-end protein-ligand binding affinity predictor.
    Only use the down path for prediction.

    Encoder : UnetHeteroGVP  →  graph_embedding [B, 2*s_dim]
    Head    : LBARegressionHead  →  predicted_affinity [B]

    Loss is MSE between predicted and target neglog affinity.
    """

    def __init__(
        self,
        # encoder hyper-parameters
        s_dim: int = 128,
        v_dim: int = 16,
        s_dim_edge: int = 32,
        v_dim_edge: int = 1,
        r_max: float = 10.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        num_layers: int = 5,
        pool: str = "sum",
        fps_ratio: float = 0.6,
        cross_cutoff: float = 6.0,
        enc_drop_rate: float = 0.1,
        # head hyper-parameters
        head_hidden_dim: int = 256,
        head_drop_rate: float = 0.1,
    ):
        super().__init__()
        self.encoder = UnetHeteroGVPEncoderOnly(
            s_dim=s_dim,
            v_dim=v_dim,
            s_dim_edge=s_dim_edge,
            v_dim_edge=v_dim_edge,
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            num_layers=num_layers,
            pool=pool,
            fps_ratio=fps_ratio,
            cross_cutoff=cross_cutoff,
            drop_rate=enc_drop_rate,
        )
        self.head = LBARegressionHead(
            in_dim=2 * s_dim,
            hidden_dim=head_hidden_dim,
            drop_rate=head_drop_rate,
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        """
        :param batch: HeteroData batch (see UnetHeteroGVP.forward).
        :return: dict with keys:
            * ``"pred"``   [B]  predicted affinities
            * ``"loss"``   scalar MSE loss (only if ``batch["graph_y"]`` exists)
        """
        enc_out = self.encoder(batch)
        pred    = self.head(enc_out["graph_embedding"])   # [B]

        out = {"pred": pred}
        if hasattr(batch, "graph_y") or "graph_y" in batch:
            target = batch["graph_y"].float()
            out["loss"] = self.loss_fn(pred, target)
        return out