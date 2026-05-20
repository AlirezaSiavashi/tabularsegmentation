"""
GraphCoW-Net — Stream B (Graph Decoder) + Bidirectional Coupling

The graph decoder predicts the Circle of Willis as a first-class geometric
object: 13 anatomy-anchored **node queries** (one per CoW vessel segment) and
20 **edge queries** (canonical CoW adjacency). Each node emits a centerline
polyline + per-vertex radius + a bbox; each edge emits a connection logit +
path polyline + radii. A differentiable tube rasterizer then renders those
polylines back into a voxel-space occupancy map which (a) serves as the
graph->voxel coupling signal for the voxel decoder and (b) provides the
voxel/graph consistency loss.

Design pillars (TMI rationale):

  1. Class-anchored queries (no DETR matching). Each of the 13 node queries
     carries the identity of a specific vessel segment. This converts the
     thousand-voxel segmentation of, e.g., Pcom into a ~30-parameter
     polyline regression -- class imbalance evaporates. Edge queries are
     likewise class-anchored to canonical CoW edges (BA-PCAs, ICA-MCA, ...).
     A learnable per-class query embedding + class-identity positional
     embedding is what gives the decoder its built-in anatomical prior.

  2. Two-scale cross-attention. F5 (stride 32, 6^3) gives global context
     -- necessary for "does this segment exist in this patient?" F2
     (stride 4, 48^3) gives the fine spatial resolution needed to localize
     the centerline at voxel accuracy. Deeper (F4) is queried once as a
     mid-scale bridge. Standard DETR/Mask2Former cascaded design.

  3. Bbox-masked attention at the last block. Once the queries have a
     rough bbox estimate, we restrict the key-value set to features within
     each query's bbox (soft mask, not a hard crop). This is our voxel->
     graph direction of the bidirectional coupling.

  4. Differentiable tube rendering. Polylines are rasterized to a voxel
     grid using a signed-distance formulation: for each voxel v and each
     segment (p_i, p_{i+1}) of the polyline, we compute the unsigned
     distance to the capsule (segment + radius) and combine via soft-min.
     A sigmoid converts the SDF to a (0, 1) occupancy. This gives us:
       * a gating mask for graph->voxel coupling (modulates u3);
       * a rendered-tube tensor that is L2-compared to the voxel logits
         for the coupling loss;
       * gradients through every polyline vertex and every radius.
     All math is pure torch with no custom kernels; per-batch cost is
     modest because only local windows around each node bbox are rendered.

  5. Cheap enough at 192^3 on A100 40GB. Queries: 13 + 20 = 33 vectors,
     d_model = 256. Cross-attention over F5 is 33 -> 216 tokens (negligible).
     Over F2 is 33 -> 110k tokens -- handled with 2 blocks and chunked
     attention. Total Stream B footprint: ~3 GB fwd + ~2 GB grad.

Inputs (forward):
  features: Dict with keys "F2", "F4", "F5" (all [B, C_l, d_l, h_l, w_l]).
  feat_channels: Tuple (c_f2, c_f4, c_f5). Must match the encoder.
  decoder_features (optional): Dict "u3", "u4" from Stream A for the
      voxel->graph refinement step. Not required; if omitted, the graph
      decoder runs in forward-only mode (Phase 2 bootstrap).

Outputs:
  A dict with:
    node_presence:  [B, N_nodes]          -- sigmoid later, BCE target
    node_bbox:      [B, N_nodes, 6]       -- (z0,y0,x0,z1,y1,x1) in [0,1]
    node_poly:      [B, N_nodes, K, 3]    -- centerline vertices in [0,1]
    node_radius:    [B, N_nodes, K]       -- per-vertex radii in voxel units
    edge_presence:  [B, N_edges]
    edge_poly:      [B, N_edges, K, 3]
    edge_radius:    [B, N_edges, K]
    cond_q4:        [B, c_cond_u4, d4, h4, w4]   -- FiLM conditioning for u4
    cond_q3:        [B, c_cond_u3, d3, h3, w3]   -- FiLM conditioning for u3
    tube_mask:      [B, 1, D, H, W]              -- differentiable rendered tube
                                                   (passes through soft SDF)
    tube_per_class: [B, N_nodes, D, H, W]        -- one channel per node

Notes on coordinate convention:
  * All polyline vertices live in the [0, 1]^3 normalized patch frame with
    order (z, y, x). At render time we scale by (D-1, H-1, W-1).
  * Bboxes are (z0, y0, x0, z1, y1, x1) with 0 <= low < high <= 1.
  * Radii are in voxel units of the input volume (so 1.0 == one full-res
    voxel). Predicted via softplus on a raw scalar to keep positive.

Memory (A100 40 GB, bf16, 192^3, batch=1):
  Queries + cross-attn F5:       ~40 MB
  Cross-attn F2 (two blocks):    ~1.3 GB  (chunked, 32x32x32 key windows)
  Bbox-masked attn at u3:        ~0.8 GB
  Polyline heads:                ~10 MB
  Tube rendering (chunked):      ~0.6 GB
  FiLM condition maps (u3, u4):  ~80 MB
  -> total Stream B: ~3 GB fwd; + ~2 GB grad = ~5 GB.
  Combined with encoder (~9 GB) and decoder (~7-8 GB): ~21-22 GB total.
  Leaves ~18 GB for optimizer state, workspace, label tensors, losses.

References:
  DETR: https://arxiv.org/abs/2005.12872
  Mask2Former cross-attn in decoder: https://arxiv.org/abs/2112.01527
  SDF-TopoNet (tube SDF): https://arxiv.org/abs/2403.14042
  Betti Matching: https://arxiv.org/abs/2407.04683
  TopCoW variant atlas: Bosc et al. 2025 (21 canonical CoW topologies)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graphcow_encoder import ConvBlock3D, _norm3d


# ---------------------------------------------------------------------------
# CoW anatomical ontology
# ---------------------------------------------------------------------------

# 13 CoW vessel segments (background is class 0 in the voxel decoder; here
# they are indices 0..12 over the N_nodes=13 query slots).
COW_NODE_NAMES: Tuple[str, ...] = (
    "BA",       # basilar
    "R-PCA", "L-PCA",
    "R-ICA", "L-ICA",
    "R-MCA", "L-MCA",
    "R-Pcom", "L-Pcom",
    "Acom",
    "R-ACA", "L-ACA",
    "3rd-A2",
)

# 20 canonical CoW edges. Each edge connects two node indices; directional
# semantics are irrelevant (we treat them as undirected pairs). Chosen from
# TopCoW 2024 anatomy doc; a few are redundant for robustness (both Pcom ends).
# Index pairs refer to COW_NODE_NAMES above.
COW_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),   # BA - R-PCA
    (0, 2),   # BA - L-PCA
    (1, 7),   # R-PCA - R-Pcom
    (2, 8),   # L-PCA - L-Pcom
    (3, 7),   # R-ICA - R-Pcom
    (4, 8),   # L-ICA - L-Pcom
    (3, 5),   # R-ICA - R-MCA
    (4, 6),   # L-ICA - L-MCA
    (3, 10),  # R-ICA - R-ACA
    (4, 11),  # L-ICA - L-ACA
    (10, 9),  # R-ACA - Acom
    (11, 9),  # L-ACA - Acom
    (10, 12), # R-ACA - 3rd-A2
    (11, 12), # L-ACA - 3rd-A2
    (1, 2),   # R-PCA - L-PCA (BA bifurcation neighbors; not always present
              # but the edge presence head decides)
    (5, 6),   # R-MCA - L-MCA (symmetry edge; head decides)
    (7, 8),   # R-Pcom - L-Pcom (symmetry, head decides)
    (10, 11), # R-ACA - L-ACA (symmetry)
    (3, 4),   # R-ICA - L-ICA (symmetry; presence useful for variant typing)
    (0, 12),  # BA - 3rd-A2 (rare fetal variant routing; head decides)
)

N_NODES: int = len(COW_NODE_NAMES)
N_EDGES: int = len(COW_EDGES)


# ---------------------------------------------------------------------------
# Multi-head cross-attention (queries attend over a 3D feature map)
# ---------------------------------------------------------------------------

class CrossAttn3D(nn.Module):
    """Standard multi-head cross-attention: queries [B, N, d] attend over
    a 3D feature map [B, C, D, H, W].

    * A 1x1x1 conv projects C -> d for keys and values.
    * 3D learned positional embedding is added to the keys/values. This
      tells the queries where each voxel sits. We use a factored (z, y, x)
      positional encoding: three small learned tables interpolated to the
      current feature-map size. Standard trick for variable-size inputs.
    * The attention itself is vanilla scaled-dot-product with dropout.
    * `chunk_kv` > 0 splits the KV set along its spatial axis for memory
      control. We never need chunked queries because N is tiny (33).
    """

    def __init__(
        self,
        d_model: int,
        feat_ch: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        pos_table_size: int = 64,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.k_proj = nn.Conv3d(feat_ch, d_model, kernel_size=1, bias=False)
        self.v_proj = nn.Conv3d(feat_ch, d_model, kernel_size=1, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Factored positional embedding for keys: separate learnable tables
        # per axis, interpolated to the current feature-map size then summed.
        self.pos_z = nn.Parameter(torch.zeros(1, d_model, pos_table_size, 1, 1))
        self.pos_y = nn.Parameter(torch.zeros(1, d_model, 1, pos_table_size, 1))
        self.pos_x = nn.Parameter(torch.zeros(1, d_model, 1, 1, pos_table_size))
        for p in (self.pos_z, self.pos_y, self.pos_x):
            nn.init.trunc_normal_(p, std=0.02)

    def _build_pos(self, D: int, H: int, W: int, device, dtype) -> torch.Tensor:
        pz = F.interpolate(self.pos_z.float(), size=(D, 1, 1), mode="trilinear",
                           align_corners=True)
        py = F.interpolate(self.pos_y.float(), size=(1, H, 1), mode="trilinear",
                           align_corners=True)
        px = F.interpolate(self.pos_x.float(), size=(1, 1, W), mode="trilinear",
                           align_corners=True)
        return (pz + py + px).to(device=device, dtype=dtype)  # [1, d, D, H, W]

    def forward(
        self,
        queries: torch.Tensor,           # [B, N, d]
        feat: torch.Tensor,              # [B, C, D, H, W]
        kv_mask: Optional[torch.Tensor] = None,  # [B, N, D*H*W] bool mask; True=keep
        chunk_kv: int = 0,               # 0 disables chunking
    ) -> torch.Tensor:
        B, N, d = queries.shape
        _, C, D, H, W = feat.shape
        k_full = self.k_proj(feat) + self._build_pos(D, H, W, feat.device, feat.dtype)
        v_full = self.v_proj(feat)

        # Flatten KV to [B, d, M]
        M = D * H * W
        k_flat = k_full.reshape(B, d, M)
        v_flat = v_full.reshape(B, d, M)

        # Project Q and split heads: [B, H, N, hd]
        q = self.q_proj(queries).reshape(B, N, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)          # [B, H, N, hd]
        scale = 1.0 / math.sqrt(self.head_dim)

        # K, V heads: [B, H, hd, M]  (transpose applied in matmul below)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # t: [B, d, M] -> [B, H, hd, M]
            return t.reshape(B, self.n_heads, self.head_dim, -1)

        k_h = split_heads(k_flat)
        v_h = split_heads(v_flat)

        if chunk_kv and M > chunk_kv:
            # Streaming softmax over KV chunks. We accumulate numerator +
            # running max + denominator in one pass, then normalize. This
            # keeps peak memory at O(N * chunk_kv) instead of O(N * M).
            # Numerically stable softmax using max-subtraction per chunk.
            out = torch.zeros(B, self.n_heads, N, self.head_dim,
                              device=queries.device, dtype=queries.dtype)
            running_max = torch.full((B, self.n_heads, N, 1), -float("inf"),
                                     device=queries.device, dtype=queries.dtype)
            running_denom = torch.zeros(B, self.n_heads, N, 1,
                                        device=queries.device, dtype=queries.dtype)
            for start in range(0, M, chunk_kv):
                end = min(M, start + chunk_kv)
                k_c = k_h[..., start:end]         # [B, H, hd, m]
                v_c = v_h[..., start:end]         # [B, H, hd, m]
                # [B, H, N, m]
                s = torch.einsum("bhnd,bhdm->bhnm", q, k_c) * scale
                if kv_mask is not None:
                    m_c = kv_mask[..., start:end].unsqueeze(1)   # [B, 1, N, m]
                    s = s.masked_fill(~m_c, -1e4)
                chunk_max = s.max(dim=-1, keepdim=True).values   # [B, H, N, 1]
                new_max = torch.maximum(running_max, chunk_max)
                renorm = torch.exp(running_max - new_max)
                exp_s = torch.exp(s - new_max)
                running_denom = running_denom * renorm + exp_s.sum(dim=-1, keepdim=True)
                # [B, H, N, hd]  += attn_c @ v_c  with the renorm factor
                out = out * renorm + torch.einsum("bhnm,bhdm->bhnd", exp_s, v_c)
                running_max = new_max
            out = out / running_denom.clamp_min(1e-6)
        else:
            s = torch.einsum("bhnd,bhdm->bhnm", q, k_h) * scale  # [B, H, N, M]
            if kv_mask is not None:
                s = s.masked_fill(~kv_mask.unsqueeze(1), -1e4)
            a = F.softmax(s, dim=-1)
            a = self.dropout(a)
            out = torch.einsum("bhnm,bhdm->bhnd", a, v_h)         # [B, H, N, hd]

        out = out.permute(0, 2, 1, 3).reshape(B, N, d)            # [B, N, d]
        return self.o_proj(out)


class QueryBlock(nn.Module):
    """Standard transformer decoder block: self-attn on queries, then cross-
    attn over features, then FFN. Pre-norm variant for training stability.
    """

    def __init__(
        self,
        d_model: int,
        feat_ch: int,
        n_heads: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_sa = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.norm_ca = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttn3D(d_model, feat_ch, n_heads=n_heads,
                                      dropout=dropout)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        feat: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
        chunk_kv: int = 0,
    ) -> torch.Tensor:
        q = queries
        # Self-attn (queries talk among themselves; shares info across sibling
        # segments, e.g. R-Pcom <-> R-ICA).
        qn = self.norm_sa(q)
        q = q + self.self_attn(qn, qn, qn, need_weights=False)[0]
        # Cross-attn (queries pull from feature map).
        qn = self.norm_ca(q)
        q = q + self.cross_attn(qn, feat, kv_mask=kv_mask, chunk_kv=chunk_kv)
        # FFN.
        qn = self.norm_ff(q)
        q = q + self.ffn(qn)
        return q


# ---------------------------------------------------------------------------
# Polyline & radius heads
# ---------------------------------------------------------------------------

class NodeHead(nn.Module):
    """From a [B, N, d] query tensor, produce:
      * presence logit              [B, N]
      * bbox in [0,1]^6             [B, N, 6]  (z0,y0,x0,z1,y1,x1)
      * centerline polyline         [B, N, K, 3]  in [0,1]
      * per-vertex radius (voxels)  [B, N, K]
    """

    def __init__(self, d_model: int, K: int, radius_scale: float = 4.0):
        super().__init__()
        self.K = K
        self.radius_scale = radius_scale

        self.norm = nn.LayerNorm(d_model)
        self.presence = nn.Linear(d_model, 1)
        # Bbox: 6 raw -> sigmoid -> enforce low < high via softplus offset.
        self.bbox = nn.Linear(d_model, 6)
        # Polyline: K*3 raw -> sigmoid. The normalized positions are then
        # projected so the polyline lies inside the predicted bbox (+ margin)
        # at render time -- gives the model an easy initial layout.
        self.poly = nn.Linear(d_model, K * 3)
        # Radius: K raw -> softplus * radius_scale. Stays positive.
        self.radius = nn.Linear(d_model, K)

    def forward(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.norm(q)
        B, N, d = h.shape

        presence = self.presence(h).squeeze(-1)            # [B, N]

        bbox_raw = self.bbox(h)                            # [B, N, 6]
        bbox_s = torch.sigmoid(bbox_raw)                   # all in (0,1)
        lo, hi_raw = bbox_s[..., :3], bbox_s[..., 3:]
        # ensure hi > lo by at least eps via softplus offset
        hi = lo + F.softplus(hi_raw - lo) + 1e-3
        hi = hi.clamp_max(1.0)
        bbox = torch.cat([lo, hi], dim=-1)                 # [B, N, 6]

        poly_local = torch.sigmoid(self.poly(h)).view(B, N, self.K, 3)   # [0,1]^3
        # Project polyline from its own [0,1] box into the predicted bbox.
        # This makes the polyline initialization already roughly inside the
        # bbox, which speeds up training noticeably.
        lo_e = lo.unsqueeze(2)                             # [B, N, 1, 3]
        hi_e = hi.unsqueeze(2)
        poly = lo_e + (hi_e - lo_e) * poly_local           # [B, N, K, 3]

        radius = F.softplus(self.radius(h)) * self.radius_scale / math.log(2.0)
        # Scale-factor so that an all-zero raw input yields radius ~= radius_scale.
        # (softplus(0)/ln(2) == 1.)

        return {
            "presence": presence,
            "bbox": bbox,
            "poly": poly,
            "radius": radius,
        }


class EdgeHead(nn.Module):
    """From a [B, N_edges, d] query tensor + the node query tensor, produce:
      * connection logit   [B, N_edges]
      * path polyline      [B, N_edges, K, 3]  in [0,1]
      * per-vertex radius  [B, N_edges, K]     in voxels
    The path endpoints are softly anchored to the connected nodes' polyline
    endpoints so that edges share junctions with their endpoint segments.
    """

    def __init__(self, d_model: int, K: int, edges: Sequence[Tuple[int, int]],
                 radius_scale: float = 2.0):
        super().__init__()
        self.K = K
        self.register_buffer(
            "edge_idx",
            torch.tensor(edges, dtype=torch.long),   # [N_edges, 2]
            persistent=False,
        )
        self.radius_scale = radius_scale

        self.norm = nn.LayerNorm(d_model)
        self.presence = nn.Linear(d_model, 1)
        # Interior K-2 vertices only; endpoints come from nodes.
        self.interior = nn.Linear(d_model, max(0, K - 2) * 3)
        self.radius = nn.Linear(d_model, K)

    def forward(
        self,
        q_edges: torch.Tensor,           # [B, N_edges, d]
        node_poly: torch.Tensor,         # [B, N_nodes, K_n, 3]
    ) -> Dict[str, torch.Tensor]:
        B, Ne, d = q_edges.shape
        h = self.norm(q_edges)
        presence = self.presence(h).squeeze(-1)

        # Endpoints from the two connected nodes' polyline endpoints. We pick
        # node.poly[:, 0] for one side and node.poly[:, -1] for the other; the
        # head learns to pair them correctly via the interior.
        a = self.edge_idx[:, 0]          # [Ne]
        b = self.edge_idx[:, 1]
        p_a = node_poly[:, a, 0, :]       # [B, Ne, 3]
        p_b = node_poly[:, b, -1, :]      # [B, Ne, 3]

        if self.K > 2:
            interior = torch.sigmoid(self.interior(h)).view(B, Ne, self.K - 2, 3)
            poly = torch.cat([p_a.unsqueeze(2), interior, p_b.unsqueeze(2)], dim=2)
        else:
            poly = torch.stack([p_a, p_b], dim=2)

        radius = F.softplus(self.radius(h)) * self.radius_scale / math.log(2.0)
        return {"presence": presence, "poly": poly, "radius": radius}


# ---------------------------------------------------------------------------
# Differentiable tube renderer
# ---------------------------------------------------------------------------

def _capsule_distance(
    voxel_xyz: torch.Tensor,       # [..., 3]  (z, y, x) in voxel units
    p0: torch.Tensor,              # [..., 3]
    p1: torch.Tensor,              # [..., 3]
    r0: torch.Tensor,              # [...]
    r1: torch.Tensor,              # [...]
) -> torch.Tensor:
    """Unsigned signed-distance-like score for a capsule (line segment with
    per-endpoint radius). Returns (dist_to_axis - interp_radius); negative
    inside, positive outside. All tensors broadcast over the leading dims.
    """
    ab = p1 - p0                                   # [...,3]
    ap = voxel_xyz - p0                            # [...,3]
    # Parameter t along the segment, clamped to [0,1].
    ab_sq = (ab * ab).sum(dim=-1, keepdim=True).clamp_min(1e-6)
    t = (ap * ab).sum(dim=-1, keepdim=True) / ab_sq
    t = t.clamp(0.0, 1.0)
    closest = p0 + t * ab
    d = (voxel_xyz - closest).norm(dim=-1)         # [...]
    r = r0 + t.squeeze(-1) * (r1 - r0)             # [...]
    return d - r


class TubeRenderer(nn.Module):
    """Render polylines (with per-vertex radii) to a differentiable voxel
    occupancy map via soft-min of capsule signed-distances.

    For each polyline with vertices p_0..p_{K-1} and radii r_0..r_{K-1},
    each segment (p_i, p_{i+1}) is a capsule; the polyline is their union.
    The voxel value equals sigmoid(-min_i sdf_i / tau). tau controls
    sharpness; smaller -> sharper, higher gradient noise.

    Rendering cost is O(N * K * window_vox) where window_vox is the number of
    voxels within (max_radius + pad) around the polyline's bounding box.
    Outside the window we return 0. This keeps rendering cheap even when the
    volume is 192^3.

    All math is done in fp32 for numerical stability regardless of input
    dtype; output is cast back.
    """

    def __init__(self, tau: float = 0.5, pad_voxels: float = 4.0):
        super().__init__()
        self.tau = tau
        self.pad_voxels = pad_voxels

    def forward(
        self,
        poly: torch.Tensor,            # [B, N, K, 3] in [0,1]
        radius: torch.Tensor,          # [B, N, K] in voxel units
        presence: Optional[torch.Tensor],   # [B, N] sigmoid-scaled weight or None
        shape: Tuple[int, int, int],   # (D, H, W) of the target volume
    ) -> torch.Tensor:
        """Returns per-class rendered tubes [B, N, D, H, W] in (0, 1).
        Outside each polyline's bbox-window, values are ~0.
        """
        D, H, W = shape
        B, N, K, _ = poly.shape
        device = poly.device
        orig_dtype = poly.dtype
        poly = poly.float()
        radius = radius.float()
        pres = None if presence is None else presence.float()

        # Convert polyline from normalized [0,1] to voxel index space.
        scale = torch.tensor([D - 1, H - 1, W - 1], device=device, dtype=torch.float32)
        poly_v = poly * scale                                  # [B, N, K, 3]

        out = torch.zeros(B, N, D, H, W, device=device, dtype=torch.float32)

        # Iterate polylines (B*N is 13-33 at most; fine to Python-loop).
        for b in range(B):
            for n in range(N):
                pv = poly_v[b, n]                              # [K, 3]
                rv = radius[b, n]                              # [K]
                p0s = pv[:-1]                                  # [K-1, 3]
                p1s = pv[1:]                                   # [K-1, 3]
                r0s = rv[:-1]
                r1s = rv[1:]

                # Window = polyline bbox enlarged by max radius + pad.
                r_max = float(rv.max().item())
                pad = r_max + self.pad_voxels
                mn = pv.min(dim=0).values - pad
                mx = pv.max(dim=0).values + pad
                z0 = max(0, int(mn[0].floor().item()))
                y0 = max(0, int(mn[1].floor().item()))
                x0 = max(0, int(mn[2].floor().item()))
                z1 = min(D, int(mx[0].ceil().item()) + 1)
                y1 = min(H, int(mx[1].ceil().item()) + 1)
                x1 = min(W, int(mx[2].ceil().item()) + 1)
                if z1 <= z0 or y1 <= y0 or x1 <= x0:
                    continue

                dz = z1 - z0
                dy = y1 - y0
                dx = x1 - x0
                zs = torch.arange(z0, z1, device=device, dtype=torch.float32)
                ys = torch.arange(y0, y1, device=device, dtype=torch.float32)
                xs = torch.arange(x0, x1, device=device, dtype=torch.float32)
                zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
                grid = torch.stack([zz, yy, xx], dim=-1)       # [dz, dy, dx, 3]

                # For each segment compute capsule distance and take the
                # minimum (soft-min over segments).
                # Expand for broadcasting: [K-1, dz, dy, dx, 3] / [K-1, 1,...].
                gs = grid.unsqueeze(0)                          # [1, dz, dy, dx, 3]
                p0s_e = p0s.view(-1, 1, 1, 1, 3)
                p1s_e = p1s.view(-1, 1, 1, 1, 3)
                r0s_e = r0s.view(-1, 1, 1, 1)
                r1s_e = r1s.view(-1, 1, 1, 1)
                # Broadcasting happens inside _capsule_distance.
                sdf_each = _capsule_distance(gs, p0s_e, p1s_e, r0s_e, r1s_e)
                # Soft-min across segments for a smoother gradient near
                # vertex joints, log-sum-exp form: -log(sum exp(-x/tau))*tau.
                sdf = -torch.logsumexp(-sdf_each / self.tau, dim=0) * self.tau

                occ = torch.sigmoid(-sdf / self.tau)            # (0,1)
                if pres is not None:
                    occ = occ * pres[b, n]                     # fade absent classes

                out[b, n, z0:z1, y0:y1, x0:x1] = occ

        return out.to(orig_dtype)


# ---------------------------------------------------------------------------
# Bidirectional-coupling condition projector
# ---------------------------------------------------------------------------

class GraphToVoxelCond(nn.Module):
    """Project per-class tube renderings into conditioning feature maps at
    the strides needed by Stream A (u3: stride 8, u4: stride 16).

    Input:  per_class  [B, N_nodes, D, H, W]
            presence_w [B, N_nodes]   -- optional additional gating (e.g. sigmoid
                                         of the node-presence logit); applied
                                         multiplicatively so absent segments
                                         contribute nothing.
    Output: (cond_u4, cond_u3) tensors at their respective strides and
            channel widths.

    A single 3x3x3 conv maps the per-class stack to each conditioning
    width; downsampling is done with a strided pool of the tube occupancy
    (NOT of the conv output, so that the spatial structure of the tube is
    preserved regardless of channel mixing).
    """

    def __init__(
        self,
        n_nodes: int,
        out_ch_u4: int,
        out_ch_u3: int,
    ):
        super().__init__()
        self.pool_u4 = nn.AvgPool3d(kernel_size=16, stride=16)
        self.pool_u3 = nn.AvgPool3d(kernel_size=8, stride=8)
        # 3x3x3 conv + GN + GELU to a small channel width.
        def make(out_ch: int) -> nn.Module:
            mid = max(16, out_ch)
            return nn.Sequential(
                nn.Conv3d(n_nodes, mid, kernel_size=3, padding=1, bias=False),
                _norm3d(mid), nn.GELU(),
                nn.Conv3d(mid, out_ch, kernel_size=1, bias=True),
            )
        self.proj_u4 = make(out_ch_u4)
        self.proj_u3 = make(out_ch_u3)

    def forward(
        self,
        per_class: torch.Tensor,
        presence_w: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if presence_w is not None:
            per_class = per_class * presence_w.view(*presence_w.shape, 1, 1, 1)
        # Downsample the per-class occupancy first, then mix channels.
        tu4 = self.pool_u4(per_class)
        tu3 = self.pool_u3(per_class)
        return self.proj_u4(tu4), self.proj_u3(tu3)


# ---------------------------------------------------------------------------
# Bbox-masked attention helpers (voxel -> graph refinement)
# ---------------------------------------------------------------------------

def _bbox_kv_mask(
    bbox: torch.Tensor,            # [B, N, 6] in [0,1]
    shape: Tuple[int, int, int],
    pad_voxels: float = 2.0,
    device=None,
) -> torch.Tensor:
    """Build a boolean mask [B, N, D*H*W] that is True for voxels lying inside
    each query's padded bbox. Used to restrict the KV set for the refinement
    cross-attn block.
    """
    B, N, _ = bbox.shape
    D, H, W = shape
    if device is None:
        device = bbox.device
    scale = torch.tensor([D - 1, H - 1, W - 1], device=device,
                         dtype=torch.float32).view(1, 1, 3)
    lo = (bbox[..., :3].float() * scale - pad_voxels).clamp_min(0.0)
    hi = (bbox[..., 3:].float() * scale + pad_voxels).clamp(max=torch.tensor(
        [D - 1, H - 1, W - 1], device=device, dtype=torch.float32))
    # Voxel index grid.
    zs = torch.arange(D, device=device).view(D, 1, 1)
    ys = torch.arange(H, device=device).view(1, H, 1)
    xs = torch.arange(W, device=device).view(1, 1, W)
    # [B, N, D, H, W] True inside bbox.
    m = (
        (zs.float() >= lo[..., 0:1].unsqueeze(-1).unsqueeze(-1)) &
        (zs.float() <= hi[..., 0:1].unsqueeze(-1).unsqueeze(-1)) &
        (ys.float() >= lo[..., 1:2].unsqueeze(-1).unsqueeze(-1)) &
        (ys.float() <= hi[..., 1:2].unsqueeze(-1).unsqueeze(-1)) &
        (xs.float() >= lo[..., 2:3].unsqueeze(-1).unsqueeze(-1)) &
        (xs.float() <= hi[..., 2:3].unsqueeze(-1).unsqueeze(-1))
    )
    return m.view(B, N, D * H * W)


# ---------------------------------------------------------------------------
# Full Stream-B graph decoder
# ---------------------------------------------------------------------------

class GraphCoWGraphDecoder(nn.Module):
    """The full Stream B graph decoder.

    Queries:
      * N_nodes = 13 class-anchored node queries
      * N_edges = 20 class-anchored edge queries
      * d_model = 256 (default)

    Pipeline:
      1. Initialize queries = learned class embeddings (fixed ordering).
      2. Block 1: cross-attn over F5 (coarse, global context).
      3. Block 2: cross-attn over F4 (mid scale).
      4. Block 3: cross-attn over F2 (fine) with chunked KV.
      5. Emit preliminary node bbox/polyline/radius + edge polyline/radius.
      6. Refinement block: voxel->graph bbox-masked attn over u3 (or F2 if
         u3 absent). This is step "Voxel->Graph" of the bidirectional coupling.
      7. Re-emit refined heads.
      8. Tube renderer -> per-class occupancy volumes.
      9. Graph->Voxel conditioning maps for u3 and u4 (FiLM into Stream A).

    Output keys listed in module docstring.
    """

    def __init__(
        self,
        feat_channels: Tuple[int, int, int] = (64, 256, 384),   # (c_f2, c_f4, c_f5)
        d_model: int = 256,
        n_heads: int = 4,
        K_node: int = 8,
        K_edge: int = 6,
        node_names: Sequence[str] = COW_NODE_NAMES,
        edges: Sequence[Tuple[int, int]] = COW_EDGES,
        cond_channels: Tuple[int, int] = (16, 16),              # (u4, u3)
        f2_chunk: int = 32 * 32 * 32,
        radius_scale_node: float = 4.0,
        radius_scale_edge: float = 2.0,
        tube_tau: float = 0.5,
        use_refinement: bool = True,
    ):
        super().__init__()
        c_f2, c_f4, c_f5 = feat_channels
        self.d_model = d_model
        self.n_nodes = len(node_names)
        self.n_edges = len(edges)
        self.K_node = K_node
        self.K_edge = K_edge
        self.f2_chunk = f2_chunk
        self.use_refinement = use_refinement

        # Learned content embedding per class (nodes + edges).
        self.node_emb = nn.Parameter(torch.zeros(1, self.n_nodes, d_model))
        self.edge_emb = nn.Parameter(torch.zeros(1, self.n_edges, d_model))
        nn.init.trunc_normal_(self.node_emb, std=0.02)
        nn.init.trunc_normal_(self.edge_emb, std=0.02)

        # Positional / identity embeddings (separate from the content embedding
        # to match standard DETR design; content is updated across blocks,
        # positional stays fixed and is re-added per block).
        self.node_pos = nn.Parameter(torch.zeros(1, self.n_nodes, d_model))
        self.edge_pos = nn.Parameter(torch.zeros(1, self.n_edges, d_model))
        nn.init.trunc_normal_(self.node_pos, std=0.02)
        nn.init.trunc_normal_(self.edge_pos, std=0.02)

        # Cross-attn blocks. Nodes and edges share blocks (they attend jointly
        # so edge queries can condition on node context via self-attn).
        self.block_f5 = QueryBlock(d_model, c_f5, n_heads=n_heads)
        self.block_f4 = QueryBlock(d_model, c_f4, n_heads=n_heads)
        self.block_f2 = QueryBlock(d_model, c_f2, n_heads=n_heads)

        # Preliminary heads.
        self.node_head0 = NodeHead(d_model, K_node, radius_scale=radius_scale_node)
        self.edge_head0 = EdgeHead(d_model, K_edge, edges=edges,
                                   radius_scale=radius_scale_edge)

        # Refinement: bbox-masked cross-attn over F2 (or u3 when provided).
        self.block_refine = QueryBlock(d_model, c_f2, n_heads=n_heads)
        self.node_head1 = NodeHead(d_model, K_node, radius_scale=radius_scale_node)
        self.edge_head1 = EdgeHead(d_model, K_edge, edges=edges,
                                   radius_scale=radius_scale_edge)

        # Tube renderer (parameter-free; pure geometry).
        self.tube = TubeRenderer(tau=tube_tau)

        # Graph->voxel conditioning projector.
        self.cond_proj = GraphToVoxelCond(
            n_nodes=self.n_nodes,
            out_ch_u4=cond_channels[0],
            out_ch_u3=cond_channels[1],
        )

        # Cache the edge index buffer so it lives on the same device.
        self.register_buffer(
            "_edge_idx",
            torch.tensor(edges, dtype=torch.long),
            persistent=False,
        )

    # ------------------------------------------------------------------ helpers

    def _init_queries(self, B: int, device, dtype):
        q_nodes = (self.node_emb + self.node_pos).expand(B, -1, -1).to(
            device=device, dtype=dtype)
        q_edges = (self.edge_emb + self.edge_pos).expand(B, -1, -1).to(
            device=device, dtype=dtype)
        return q_nodes, q_edges

    def _emit_nodes_edges(
        self, q_nodes: torch.Tensor, q_edges: torch.Tensor,
        head_node: NodeHead, head_edge: EdgeHead,
    ) -> Dict[str, torch.Tensor]:
        nh = head_node(q_nodes)
        eh = head_edge(q_edges, node_poly=nh["poly"])
        return {
            "node_presence": nh["presence"],
            "node_bbox": nh["bbox"],
            "node_poly": nh["poly"],
            "node_radius": nh["radius"],
            "edge_presence": eh["presence"],
            "edge_poly": eh["poly"],
            "edge_radius": eh["radius"],
        }

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        decoder_features: Optional[Dict[str, torch.Tensor]] = None,
        render_tube: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        features: dict with keys F2, F4, F5 from the encoder.
        decoder_features: optional dict with keys u3, u4 from Stream A used
            to refine queries via voxel->graph attention. In Phase 2
            (graph bootstrap) pass None; in Phase 3 (joint) pass them.
        render_tube: if False, skip the tube rasterization step (for speed
            during the first few forward passes of Phase 2 while the
            polyline predictions are still noise -- rendering them is wasted
            work because the coupling loss is off).
        """
        f2 = features["F2"]
        f4 = features["F4"]
        f5 = features["F5"]
        B, _, D2, H2, W2 = f2.shape
        device = f2.device
        dtype = f2.dtype

        # ------- 1. Initial queries ----------------------------------------
        q_nodes, q_edges = self._init_queries(B, device, dtype)

        # Concatenate nodes + edges into a single query stream so that
        # self-attn across both lets edges condition on nodes (and vice
        # versa). We split back at emit time.
        def run_block(block: QueryBlock, feat: torch.Tensor,
                      q_nodes: torch.Tensor, q_edges: torch.Tensor,
                      kv_mask: Optional[torch.Tensor] = None,
                      chunk_kv: int = 0):
            q = torch.cat([q_nodes, q_edges], dim=1)
            q = block(q, feat, kv_mask=kv_mask, chunk_kv=chunk_kv)
            return q[:, :self.n_nodes], q[:, self.n_nodes:]

        # ------- 2. Cross-attn F5 (global) ---------------------------------
        q_nodes, q_edges = run_block(self.block_f5, f5, q_nodes, q_edges)

        # ------- 3. Cross-attn F4 (mid) ------------------------------------
        q_nodes, q_edges = run_block(self.block_f4, f4, q_nodes, q_edges)

        # ------- 4. Cross-attn F2 (fine, chunked) --------------------------
        q_nodes, q_edges = run_block(
            self.block_f2, f2, q_nodes, q_edges, chunk_kv=self.f2_chunk,
        )

        # ------- 5. Preliminary heads --------------------------------------
        prelim = self._emit_nodes_edges(
            q_nodes, q_edges, self.node_head0, self.edge_head0,
        )

        # ------- 6. Refinement (voxel -> graph) ----------------------------
        # Attend to u3 features (if provided from Stream A) or back to F2.
        # Restrict KV to voxels inside each query's preliminary bbox.
        if self.use_refinement:
            if decoder_features is not None and "u3" in decoder_features:
                refine_feat = decoder_features["u3"]
            else:
                refine_feat = f2
            _, _, Dr, Hr, Wr = refine_feat.shape

            # Build bbox mask for nodes; edges use the union of their two
            # endpoint nodes' bboxes (adequate approximation). Stack into
            # [B, N_nodes+N_edges, Dr*Hr*Wr].
            node_bbox = prelim["node_bbox"]
            edge_a = self._edge_idx[:, 0]
            edge_b = self._edge_idx[:, 1]
            # Union of two bboxes: elementwise min of lows and max of highs.
            lo_a = node_bbox[:, edge_a, :3]
            hi_a = node_bbox[:, edge_a, 3:]
            lo_b = node_bbox[:, edge_b, :3]
            hi_b = node_bbox[:, edge_b, 3:]
            lo_e = torch.minimum(lo_a, lo_b)
            hi_e = torch.maximum(hi_a, hi_b)
            edge_bbox = torch.cat([lo_e, hi_e], dim=-1)

            all_bbox = torch.cat([node_bbox, edge_bbox], dim=1)
            kv_mask = _bbox_kv_mask(
                all_bbox, shape=(Dr, Hr, Wr), pad_voxels=2.0, device=device,
            )
            # Chunking matters if refine_feat is large (F2 case).
            M = Dr * Hr * Wr
            chunk = self.f2_chunk if M > self.f2_chunk else 0
            q_nodes, q_edges = run_block(
                self.block_refine, refine_feat,
                q_nodes, q_edges, kv_mask=kv_mask, chunk_kv=chunk,
            )

            # ------- 7. Refined heads --------------------------------------
            final = self._emit_nodes_edges(
                q_nodes, q_edges, self.node_head1, self.edge_head1,
            )
        else:
            final = prelim

        # ------- 8. Tube rendering -----------------------------------------
        # We need the target full-res shape; infer from F2 (stride 4).
        D, H, W = D2 * 4, H2 * 4, W2 * 4
        if render_tube:
            node_pres_w = torch.sigmoid(final["node_presence"].detach())  # detached so
                                                                           # gradient to
                                                                           # presence is
                                                                           # only via the
                                                                           # supervised
                                                                           # loss
            per_class = self.tube(
                final["node_poly"], final["node_radius"],
                presence=node_pres_w,
                shape=(D, H, W),
            )                                                   # [B, N_nodes, D, H, W]
            tube_mask = per_class.sum(dim=1, keepdim=True).clamp(0.0, 1.0)  # union
        else:
            per_class = f2.new_zeros(B, self.n_nodes, D, H, W)
            tube_mask = f2.new_zeros(B, 1, D, H, W)

        # ------- 9. Graph -> voxel conditioning ----------------------------
        presence_soft = torch.sigmoid(final["node_presence"])       # [B, N_nodes]
        cond_q4, cond_q3 = self.cond_proj(per_class, presence_w=presence_soft)

        out = {
            **final,
            "prelim": prelim,          # kept for auxiliary supervision
            "cond_q4": cond_q4,
            "cond_q3": cond_q3,
            "tube_mask": tube_mask,
            "tube_per_class": per_class,
        }
        return out

    # -------------------------------------------------------------- helpers

    @property
    def node_names(self) -> Tuple[str, ...]:
        return COW_NODE_NAMES

    @property
    def edge_pairs(self) -> Tuple[Tuple[int, int], ...]:
        return COW_EDGES


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--patch", type=int, nargs=3, default=[192, 192, 192],
                   metavar=("D", "H", "W"))
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--with-u3", action="store_true",
                   help="simulate Stream A output for refinement")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 and device == "cuda" else torch.float32

    D, H, W = args.patch
    enc_ch = (32, 64, 128, 256, 384)
    feats = {
        "F2": torch.randn(1, enc_ch[1], D // 4,  H // 4,  W // 4,  device=device, dtype=dtype),
        "F4": torch.randn(1, enc_ch[3], D // 16, H // 16, W // 16, device=device, dtype=dtype),
        "F5": torch.randn(1, enc_ch[4], D // 32, H // 32, W // 32, device=device, dtype=dtype),
    }
    dec_feats = None
    if args.with_u3:
        dec_feats = {
            "u3": torch.randn(1, 64, D // 8, H // 8, W // 8, device=device, dtype=dtype),
            "u4": torch.randn(1, 128, D // 16, H // 16, W // 16, device=device, dtype=dtype),
        }

    gd = GraphCoWGraphDecoder(feat_channels=(enc_ch[1], enc_ch[3], enc_ch[4])).to(device)
    if dtype == torch.bfloat16:
        gd = gd.to(dtype)
    gd.train()

    with torch.cuda.amp.autocast(enabled=dtype == torch.bfloat16, dtype=torch.bfloat16):
        out = gd(feats, decoder_features=dec_feats, render_tube=True)

    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:20s}: {tuple(v.shape)}  dtype={v.dtype}")
        elif isinstance(v, dict):
            for kk, vv in v.items():
                print(f"{k+'.'+kk:20s}: {tuple(vv.shape)}  dtype={vv.dtype}")

    n_all = sum(p.numel() for p in gd.parameters())
    n_train = sum(p.numel() for p in gd.parameters() if p.requires_grad)
    print(f"Total params: {n_all/1e6:.2f}M, trainable: {n_train/1e6:.2f}M")
    if device == "cuda":
        print(f"GPU peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
