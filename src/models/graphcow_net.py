"""
GraphCoW-Net — Top-level wrapper

Glues together:
  * GraphCoWEncoder         (frozen DINOv3 + tri-planar + pyramid)
  * GraphCoWVoxelDecoder    (Stream A)
  * GraphCoWGraphDecoder    (Stream B + bidirectional coupling hooks)

Forward (training) expects:
    x: [B, 1, D, H, W]  isotropic-voxel patch.

Forward returns a dict with:
    'logits'          : voxel logits [B, num_classes, D, H, W]
    'aux'             : dict of deep-supervision logits (u1/u2/u3)
    'features'        : dict of encoder pyramid + decoder feature maps
    'graph'           : dict from the graph decoder, carrying
                        node_presence, node_bbox, node_poly, node_radius,
                        edge_presence, edge_poly, edge_radius,
                        tube_mask, tube_per_class, prelim (aux graph heads)

Phase control:
  * `mode="voxel_only"`    -- run encoder + Stream A only; skip Stream B.
  * `mode="graph_only"`    -- run encoder + Stream B only; Stream A is not
                              executed. Used in Phase 2 bootstrap. Voxel
                              logits will still be produced for logging
                              if `force_voxel_logits=True`, but otherwise
                              we skip that compute to speed up.
  * `mode="joint"`         -- Stream A and Stream B both run. Stream A is
                              run ONCE without graph conditioning to
                              produce u3/u4, then Stream B uses them for
                              refinement and emits cond maps. We do NOT
                              currently re-run Stream A with the cond
                              maps: the graph->voxel coupling is applied
                              via the consistency loss rather than by
                              re-entering Stream A with its own output.
                              This avoids a second forward pass through
                              the largest decoder stages.

Memory footprint at 192^3, bf16, batch=1 (A100 40GB):
    Encoder           :  ~9 GB
    Stream A          :  ~7-8 GB
    Stream B (+tube)  :  ~4-5 GB
    Optimizer state   :  ~2 GB (bf16 params + fp32 master for AdamW)
    CUDA workspace    :  ~2 GB
    -------
    Total             :  ~25 GB

Note: the tube renderer is slow in Python -- ~0.5 s/step at 192^3. For
production runs we render only periodically during Phase 3 (every N steps)
while the consistency loss is computed on those rendered batches.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .graphcow_encoder import GraphCoWEncoder
from .graphcow_decoder import GraphCoWVoxelDecoder
from .graphcow_graph_decoder import (
    GraphCoWGraphDecoder, COW_NODE_NAMES, COW_EDGES, N_NODES, N_EDGES,
)


class GraphCoWNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 14,
        variant: str = "vits16",
        pyramid_channels: Tuple[int, int, int, int, int] = (32, 64, 128, 256, 384),
        dec_channels: Tuple[int, int, int, int, int] = (16, 32, 64, 128, 256),
        d_model_graph: int = 256,
        K_node: int = 8,
        K_edge: int = 6,
        cond_channels: Tuple[int, int] = (16, 16),        # (u4, u3)
        use_graph: bool = True,
        graph_refinement: bool = True,
        cache_dir: Optional[str] = None,
        unfreeze_top_k: int = 0,
    ):
        super().__init__()
        self.use_graph = use_graph
        self.num_classes = num_classes

        self.encoder = GraphCoWEncoder(
            variant=variant,
            pyramid_channels=pyramid_channels,
            cache_dir=cache_dir,
            unfreeze_top_k=unfreeze_top_k,
        )

        self.decoder = GraphCoWVoxelDecoder(
            num_classes=num_classes,
            enc_channels=pyramid_channels,
            dec_channels=dec_channels,
            cond_channels=cond_channels,
            grad_ckpt=True,
        )

        if use_graph:
            self.graph = GraphCoWGraphDecoder(
                feat_channels=(pyramid_channels[1], pyramid_channels[3], pyramid_channels[4]),
                d_model=d_model_graph,
                K_node=K_node,
                K_edge=K_edge,
                cond_channels=cond_channels,
                use_refinement=graph_refinement,
            )
        else:
            self.graph = None

    # --------------------------------------------------------------- forward

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "joint",
        render_tube: bool = True,
        force_voxel_logits: bool = False,
        attn_supervision_mask: Optional[torch.Tensor] = None,
        attn_target_classes: tuple = (8, 9, 10, 13),
        full_vol: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        x:        [B, 1, D, H, W] training patch (192³).
        full_vol: [B, 1, D_full, H_full, W_full] full brain volume (optional).
            When provided, DINOv3 processes full-brain slices for global CoW
            topology context. The stem still processes the patch for fine detail.
            Snake Conv decoder delineates precise vessel boundaries.
            This is the key architectural improvement for Pcom/Acom/3rd-A2.
        attn_supervision_mask: [B, D, H, W] long GT label for attention loss.
        """
        assert mode in ("voxel_only", "graph_only", "joint"), mode

        if attn_supervision_mask is not None and self.encoder.triplanar.vit_trainable:
            per_layer, attn_loss = self.encoder.triplanar.forward_with_attn_supervision(
                x, attn_supervision_mask, target_classes=attn_target_classes,
            )
            feats = self.encoder._build_pyramid(per_layer, x)
        else:
            feats = self.encoder(x, full_vol=full_vol)
            attn_loss = None

        # --- Stream A (always run in voxel_only/joint) ---------------------
        dec_out: Optional[Dict[str, torch.Tensor]] = None
        if mode in ("voxel_only", "joint") or force_voxel_logits:
            dec_out = self.decoder(feats, cond_q4=None, cond_q3=None)

        # --- Stream B ------------------------------------------------------
        graph_out: Optional[Dict[str, torch.Tensor]] = None
        if self.graph is not None and mode in ("graph_only", "joint"):
            # In joint mode, feed Stream A's u3/u4 to Stream B for refinement.
            decoder_feats = None
            if mode == "joint" and dec_out is not None:
                decoder_feats = {
                    "u3": dec_out["features"]["u3"],
                    "u4": dec_out["features"]["u4"],
                }
            graph_out = self.graph(feats, decoder_features=decoder_feats,
                                   render_tube=render_tube)

        out: Dict[str, torch.Tensor] = {}
        if dec_out is not None:
            out["logits"] = dec_out["logits"]
            out["aux"] = dec_out["aux"]
            out["features"] = {
                **feats,
                **{k: v for k, v in dec_out["features"].items()},
            }
        else:
            out["features"] = feats
        if graph_out is not None:
            out["graph"] = graph_out
        if attn_loss is not None:
            out["attn_loss"] = attn_loss
        return out

    # ----------------------------------------------------- optimizer groups

    def trainable_parameter_groups(
        self,
        lr_encoder_adapter: float = 5e-5,
        lr_decoder: float = 1e-4,
        lr_graph: float = 2e-4,
        lr_vit: float = 1e-5,
        weight_decay: float = 1e-4,
    ):
        """Parameter groups with their own LRs.

        Groups:
          * vit        -- unfrozen top-K DINOv3 blocks (lowest LR, stays near
                          pretrained optimum).
          * enc_adapter-- stem + tri-planar fusion + pyramid refinement.
          * dec        -- voxel decoder (Stream A).
          * graph      -- graph decoder + tube renderer (Stream B).
        Plus a no-WD group per base LR for norm/bias tensors.
        """
        vit_param_ids = {
            id(p) for p in self.encoder.triplanar.vit.parameters()
            if p.requires_grad
        }
        vit_params = []
        enc_adapter = []
        dec_params = []
        graph_params = []
        no_wd = []

        def is_no_wd(name: str, p: torch.nn.Parameter) -> bool:
            # Disable WD on biases and norm weights (standard practice).
            return name.endswith(".bias") or "norm" in name.lower() or p.ndim == 1

        for n, p in self.encoder.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in vit_param_ids:
                if is_no_wd(n, p):
                    no_wd.append((p, lr_vit))
                else:
                    vit_params.append(p)
                continue
            if is_no_wd(n, p):
                no_wd.append((p, lr_encoder_adapter))
            else:
                enc_adapter.append(p)

        for n, p in self.decoder.named_parameters():
            if not p.requires_grad:
                continue
            if is_no_wd(n, p):
                no_wd.append((p, lr_decoder))
            else:
                dec_params.append(p)

        if self.graph is not None:
            for n, p in self.graph.named_parameters():
                if not p.requires_grad:
                    continue
                if is_no_wd(n, p):
                    no_wd.append((p, lr_graph))
                else:
                    graph_params.append(p)

        groups = [
            {"params": enc_adapter, "lr": lr_encoder_adapter, "weight_decay": weight_decay},
            {"params": dec_params, "lr": lr_decoder, "weight_decay": weight_decay},
        ]
        if vit_params:
            groups.append({"params": vit_params, "lr": lr_vit,
                           "weight_decay": weight_decay})
        if graph_params:
            groups.append({"params": graph_params, "lr": lr_graph,
                           "weight_decay": weight_decay})
        # Bundle no-weight-decay params into one group per base LR.
        from collections import defaultdict
        by_lr = defaultdict(list)
        for p, lr in no_wd:
            by_lr[lr].append(p)
        for lr, params in by_lr.items():
            groups.append({"params": params, "lr": lr, "weight_decay": 0.0})
        return groups


# ---------------------------------------------------------------------------
# Atlas projection (test-time correction; skeleton only for now)
# ---------------------------------------------------------------------------

class CoWAtlasProjection:
    """Project a predicted CoW graph onto the nearest canonical variant.

    The 21 CoW variants (Bosc et al. 2025) are encoded as presence masks
    over the 13 node classes + 20 edge classes. At test time, given the
    graph decoder's presence vector, we round each presence to the closest
    atlas entry under Hamming distance (nodes) + Jaccard (edges).

    This class is test-time-only and deliberately light: the actual
    variant table is provided at construction (see configs/atlas.json).
    """

    def __init__(
        self,
        node_presence_table: torch.Tensor,    # [V, N_nodes]  bool
        edge_presence_table: torch.Tensor,    # [V, N_edges]  bool
        node_weight: float = 1.0,
        edge_weight: float = 0.5,
    ):
        assert node_presence_table.dtype == torch.bool
        assert edge_presence_table.dtype == torch.bool
        self.node_tab = node_presence_table
        self.edge_tab = edge_presence_table
        self.node_weight = node_weight
        self.edge_weight = edge_weight

    def project(
        self,
        node_presence: torch.Tensor,   # [B, N_nodes] sigmoid probs in (0,1)
        edge_presence: torch.Tensor,   # [B, N_edges]
        node_thresh: float = 0.5,
        edge_thresh: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (node_on, edge_on, variant_idx) all on cpu."""
        node_on = (node_presence >= node_thresh).cpu()
        edge_on = (edge_presence >= edge_thresh).cpu()

        B = node_on.shape[0]
        V = self.node_tab.shape[0]
        # Distance: weighted Hamming on nodes + Hamming on edges.
        node_d = (node_on.unsqueeze(1) != self.node_tab.unsqueeze(0)).float().sum(-1)
        edge_d = (edge_on.unsqueeze(1) != self.edge_tab.unsqueeze(0)).float().sum(-1)
        d = self.node_weight * node_d + self.edge_weight * edge_d   # [B, V]
        variant = d.argmin(dim=1)
        proj_nodes = self.node_tab[variant]
        proj_edges = self.edge_tab[variant]
        return proj_nodes, proj_edges, variant


__all__ = [
    "GraphCoWNet",
    "CoWAtlasProjection",
    "COW_NODE_NAMES",
    "COW_EDGES",
    "N_NODES",
    "N_EDGES",
]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="vits16")
    ap.add_argument("--patch", type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--mode", default="joint",
                    choices=("voxel_only", "graph_only", "joint"))
    ap.add_argument("--no-tube", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 and device == "cuda" else torch.float32

    net = GraphCoWNet(variant=args.variant).to(device)
    if dtype == torch.bfloat16:
        net = net.to(dtype)
    net.train()

    D, H, W = args.patch
    x = torch.randn(1, 1, D, H, W, device=device, dtype=dtype)
    with torch.cuda.amp.autocast(enabled=dtype == torch.bfloat16, dtype=torch.bfloat16):
        out = net(x, mode=args.mode, render_tube=not args.no_tube)

    def describe(prefix: str, obj):
        if isinstance(obj, torch.Tensor):
            print(f"{prefix:40s}: {tuple(obj.shape)}  {obj.dtype}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                describe(f"{prefix}.{k}", v)

    for k, v in out.items():
        describe(k, v)

    print(f"Params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M total, "
          f"{sum(p.numel() for p in net.parameters() if p.requires_grad)/1e6:.2f}M trainable")
    if device == "cuda":
        print(f"GPU peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
