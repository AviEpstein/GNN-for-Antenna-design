"""
Antenna Mesh → Far‑Field (Eθ, Eφ) predictor
PyTorch + PyTorch Geometric (global‑latent, fixed θ–φ grid, optional freq cond)

Inputs per sample (single graph; batching can be added with PyG Batch later):
    inputs = {
        'pos':           FloatTensor [N, 3],        # vertex positions (m)
        'node_normals':  FloatTensor [N, 3],        # per‑vertex normals (unit)
        'node_type':     FloatTensor [N, C],        # one‑hot node types (C classes)
        'node_probs':    FloatTensor [N, 1],        # probability node exists
        'edge_index':    LongTensor  [2, E],        # mesh edges (both directions OK)
        # Optional conditioning:
        'frequency':     FloatTensor [1] or [B,1],  # GHz (optional)
        # Supervision (for training):
        'target_ff':     FloatTensor [M, 2],        # Far‑field on fixed θ–φ grid: [Eθ, Eφ] (linear)
    }

Model uses a fixed direction grid (θ, φ) shared across the dataset:
    model.set_direction_grid(theta_phi)  # theta_phi: [M, 2] with radians

It builds a global latent from the mesh via Encode–Process–Decode GNN, conditions on
frequency (if provided), crosses with an embedding of each grid direction, and predicts
normalized [Eθ, Eφ] at each direction; de‑normalization is handled by an OnlineNormalizer.
"""
from __future__ import annotations
from dataclasses import dataclass

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch


# -----------------------------
# Helpers
# -----------------------------
class OnlineNormalizer(nn.Module):
    def __init__(self, size: int, eps: float = 1e-8, max_accumulations: int = 10**6):
        super().__init__()
        self.register_buffer("acc_count", torch.zeros((), dtype=torch.float32))
        self.register_buffer("num_accumulations", torch.zeros((), dtype=torch.float32))
        self.register_buffer("acc_sum", torch.zeros(size, dtype=torch.float32))
        self.register_buffer("acc_sum_sq", torch.zeros(size, dtype=torch.float32))
        self.eps = float(eps)
        self.max_accumulations = float(max_accumulations)

    def forward(self, x: torch.Tensor, accumulate: bool = True) -> torch.Tensor:
        if accumulate and (self.num_accumulations.item() < self.max_accumulations):
            self._accumulate(x)
        mean = self._mean()
        std = self._std(mean)
        return (x - mean) / std

    @torch.no_grad()
    def _accumulate(self, x: torch.Tensor) -> None:
        self.acc_sum += x.sum(dim=0)
        self.acc_sum_sq += (x * x).sum(dim=0)
        self.acc_count += float(x.shape[0])
        self.num_accumulations += 1.0

    def _mean(self) -> torch.Tensor:
        c = torch.clamp(self.acc_count, min=1.0)
        return self.acc_sum / c

    def _std(self, mean: torch.Tensor) -> torch.Tensor:
        c = torch.clamp(self.acc_count, min=1.0)
        var = self.acc_sum_sq / c - mean * mean
        return torch.clamp(var, min=0.0).sqrt().clamp_min(self.eps)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        mean = self._mean()
        std = self._std(mean)
        return y * std + mean

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, layers: int, layer_norm: bool = True):
        super().__init__()
        mods = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        mods += [nn.Linear(d, out_dim)]
        if layer_norm:
            mods += [nn.LayerNorm(out_dim)]
        self.net = nn.Sequential(*mods)
    def forward(self, x):
        return self.net(x)

class GraphNetBlock(nn.Module):
    def __init__(self, node_latent: int, edge_latent: int, hidden: int, layers: int):
        super().__init__()
        self.edge_fn = MLP(node_latent * 2 + edge_latent, edge_latent, hidden, layers, layer_norm=True)
        self.node_fn = MLP(node_latent + edge_latent, node_latent, hidden, layers, layer_norm=True)
    def forward(self, x, edge_index, edge_attr):
        s, r = edge_index
        e_in = torch.cat([x.index_select(0, s), x.index_select(0, r), edge_attr], dim=-1)
        e_out = self.edge_fn(e_in)
        agg = scatter_add(e_out, r, dim=0, dim_size=x.size(0))
        n_out = self.node_fn(torch.cat([x, agg], dim=-1))
        return x + n_out, edge_attr + e_out


class GraphNetBlockMultiEdge(nn.Module):
    def __init__(self, node_latent: int, edge_latent: int, hidden: int, layers: int):
        super().__init__()
        # Two edge MLPs: mesh/world
        self.edge_fn_mesh  = MLP(node_latent * 2 + edge_latent, edge_latent, hidden, layers, layer_norm=True)
        self.edge_fn_world = MLP(node_latent * 2 + edge_latent, edge_latent, hidden, layers, layer_norm=True)
        # One node MLP that takes node + sum(messages_mesh + messages_world)
        self.node_fn = MLP(node_latent + edge_latent, node_latent, hidden, layers, layer_norm=True)

    def forward(self, x, eM_idx, eM_attr, eW_idx, eW_attr):
        # Mesh edges
        sM, rM = eM_idx
        eM_in  = torch.cat([x.index_select(0, sM), x.index_select(0, rM), eM_attr], dim=-1)
        eM_out = self.edge_fn_mesh(eM_in)
        aggM   = scatter_add(eM_out, rM, dim=0, dim_size=x.size(0))

        # World edges
        sW, rW = eW_idx
        eW_in  = torch.cat([x.index_select(0, sW), x.index_select(0, rW), eW_attr], dim=-1)
        eW_out = self.edge_fn_world(eW_in)
        aggW   = scatter_add(eW_out, rW, dim=0, dim_size=x.size(0))

        # Combine messages
        agg = aggM + aggW
        n_out = self.node_fn(torch.cat([x, agg], dim=-1))
        return x + n_out, eM_attr + eM_out, eW_attr + eW_out


# -----------------------------
# Model
# -----------------------------
@dataclass
class AntennaFFConfig:
    num_node_types: int = 5      # one‑hot length C
    node_latent: int = 128
    edge_latent: int = 128
    hidden: int = 128
    layers: int = 2
    mp_steps: int = 15
    dir_latent: int = 64         # embedding of (θ,φ) (via unit direction)
    global_latent: int = 128
    world_radius: float = 3 #0.05    # m 0.05 is about 2x average vertex spacing; adjust based on mesh density


# =============================
# Option B: Direction→Nodes Cross‑Attention Head
# =============================
class AntennaFarFieldAttnModel(nn.Module):
    """Mesh encoder (same as Option A) + cross‑attention where each direction
    queries *all* node embeddings (keys/values).

    Steps
    1) Encode mesh nodes with Encode–Process–Decode → node embeddings X ∈ ℝ^{N×D}.
    2) (Optional) Build a global latent g from mean(X) and add frequency conditioning.
    3) Embed each direction to query vectors Q ∈ ℝ^{M×Dh}.
    4) Scaled dot‑product attention: Attn = softmax((Q Kᵀ)/√d + bias_from_node_probs),
       with K,V from node embeddings (linear projections).
    5) Context = Attn·V, then fuse with g and pass to head → [Eθ,Eφ].

    Notes
    - Node existence `node_probs ∈ (0,1]` is used as an *additive logit bias* so that
      low‑probability nodes contribute less to attention.
    - Supports multi‑head attention (H heads). Uses standard linear projections per head.
    """
    def __init__(self, cfg: AntennaFFConfig, num_heads: int = 4):
        super().__init__()
        self.cfg = cfg
        self.num_heads = num_heads
        self.register_buffer("theta_phi", torch.empty(0, 2), persistent=False)
        self.register_buffer("dir_cart", torch.empty(0, 3), persistent=False)

        # ---- normalizers (reuse shapes from Option A) ----
        node_in = 3 + 3 + cfg.num_node_types + 1
        self.node_norm = OnlineNormalizer(size=node_in)
        self.edge_norm = OnlineNormalizer(size=7)
        self.freq_norm = OnlineNormalizer(size=1)
        self.dir_norm = OnlineNormalizer(size=3)
        self.out_norm = OnlineNormalizer(size=2)

        # ---- encoders / processor ----
        self.node_enc = MLP(node_in, cfg.node_latent, cfg.hidden, cfg.layers, layer_norm=True)

        # Two separate encoders: mesh-edge features vs world-edge features
        self.edge_enc_mesh  = MLP(7, cfg.edge_latent, cfg.hidden, cfg.layers, layer_norm=True)
        self.edge_enc_world = MLP(7, cfg.edge_latent, cfg.hidden, cfg.layers, layer_norm=True)

        # Multi-edge message passing blocks
        self.blocks = nn.ModuleList([
            GraphNetBlockMultiEdge(
                node_latent=cfg.node_latent,
                edge_latent=cfg.edge_latent,
                hidden=cfg.hidden,
                layers=cfg.layers
            )
            for _ in range(cfg.mp_steps)
        ])

        # ---- projections for global/freq ----
        self.global_proj = MLP(cfg.node_latent, cfg.global_latent, cfg.hidden, cfg.layers, layer_norm=True)
        self.freq_proj = MLP(1, cfg.global_latent, cfg.hidden, cfg.layers, layer_norm=True)

        # ---- direction/query projections ----
        self.dir_proj = nn.Linear(3, cfg.dir_latent)

        # ---- multi‑head attention projections ----
        d_model = cfg.node_latent
        assert d_model % num_heads == 0, "node_latent must be divisible by num_heads"
        self.dk = d_model // num_heads
        self.q_proj = nn.Linear(cfg.dir_latent, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, cfg.hidden)

        # ---- head ----
        # Use (context + global) → hidden → [Eθ,Eφ]
        self.cross = MLP(cfg.hidden + cfg.global_latent, cfg.hidden, cfg.hidden, cfg.layers, layer_norm=True)
        self.head = MLP(cfg.hidden, 2, cfg.hidden, cfg.layers, layer_norm=False)

        if cfg.predict_surface_current:
            current_dim = 6
            self.current_decoder = torch.nn.Sequential(
            torch.nn.Linear(cfg.hidden, cfg.hidden * 2),
            torch.nn.LayerNorm(cfg.hidden * 2),
            torch.nn.GELU(),
            torch.nn.Linear(cfg.hidden * 2, cfg.hidden),
            torch.nn.LayerNorm(cfg.hidden),
            torch.nn.GELU(),
            torch.nn.Linear(cfg.hidden, current_dim) # e.g., 2 or 3 for complex vector components
            )

    # ---------- utilities ----------
    @staticmethod
    def _sph_to_cart(theta_phi: torch.Tensor) -> torch.Tensor:
        theta = theta_phi[:, 0]; phi = theta_phi[:, 1]
        st = torch.sin(theta); ct = torch.cos(theta)
        cp = torch.cos(phi);   sp = torch.sin(phi)
        return torch.stack([st*cp, st*sp, ct], dim=-1)

    @torch.no_grad()
    def set_direction_grid(self, theta_phi: torch.Tensor) -> None:
        assert theta_phi.dim() == 2 and theta_phi.size(-1) == 2
        self.theta_phi = theta_phi.detach().to(dtype=torch.float32, device=self.theta_phi.device if self.theta_phi.numel()>0 else theta_phi.device)
        self.dir_cart = F.normalize(self._sph_to_cart(self.theta_phi), dim=-1)

    # ---------- graph build & encode ----------
    def _build_graph(self, inputs: dict, is_training: bool):
        V = inputs['pos']  # [N,3] world coords
        if V.dim() == 3:
            # (unchanged) densified → flattened path
            mask = inputs.get('mask', None)
            if mask is None:
                raise ValueError('Dense batched `pos` provided but `mask` missing')
            B, Nmax, _ = V.shape
            mask_bool = mask.bool()
            V = V[mask_bool]
            batch = torch.arange(B, device=V.device).unsqueeze(1).expand(B, Nmax)[mask_bool]
            normals = inputs['node_normals'][mask_bool]
            onehot  = inputs['node_type'][mask_bool]
            probs   = inputs['node_probs'][mask_bool]
        else:
            normals = inputs['node_normals']
            onehot  = inputs['node_type']
            probs   = inputs['node_probs']
            batch   = inputs.get('batch', None)

        # Node features (same as before)
        node_feat = torch.cat([V, normals, onehot, probs], dim=-1)
        node_feat = self.node_norm(node_feat, accumulate=is_training)

        # -------------------------
        # Mesh edges: use given edge_index
        # -------------------------
        eM = inputs['edge_index']  # [2, E_M]
        sM, rM = eM[0], eM[1]
        relM   = V.index_select(0, sM) - V.index_select(0, rM)
        lenM   = relM.norm(dim=-1, keepdim=True)
        dnormM = (normals.index_select(0, sM) - normals.index_select(0, rM)).abs()
        edge_feat_mesh = torch.cat([relM, lenM, dnormM], dim=-1)
        edge_feat_mesh = self.edge_norm(edge_feat_mesh, accumulate=is_training)

        # -------------------------
        # World edges: build radius graph in world space, per graph in batch (if any)
        # -------------------------

        rw_idx = torch_geometric.nn.radius_graph(
            x=V,
            r=self.cfg.world_radius,
            batch=batch,
            loop=False,
            flow='source_to_target',
            num_workers=1,
                )

        # Remove pairs that are already mesh edges (treat edges as directed to match message passing)
        if eM.numel() > 0 and rw_idx.numel() > 0:
            # Create a hash for fast set-diff
            def _hash_edges(e):
                return e[0] * (V.size(0)) + e[1]
            mesh_keys  = _hash_edges(eM)
            world_keys = _hash_edges(rw_idx)
            mask_keep  = ~torch.isin(world_keys, mesh_keys)
            rw_idx     = rw_idx[:, mask_keep]

        sW, rW = rw_idx[0], rw_idx[1]
        relW   = V.index_select(0, sW) - V.index_select(0, rW)
        lenW   = relW.norm(dim=-1, keepdim=True)
        dnormW = (normals.index_select(0, sW) - normals.index_select(0, rW)).abs()
        edge_feat_world = torch.cat([relW, lenW, dnormW], dim=-1)
        # You can either reuse the same normalizer or keep a second one; reuse for simplicity:
        edge_feat_world = self.edge_norm(edge_feat_world, accumulate=is_training)

        return node_feat, (eM, edge_feat_mesh), (rw_idx, edge_feat_world), probs, batch

    def _encode_nodes(self, node_feat):
        return self.node_enc(node_feat)  # [N, D]


    # ---------- attention core ----------
    def _multihead_attention(self, Q, K, V, node_probs):
        """
        Q: [M, D] (queries from directions)
        K,V: [N, D] (from node embeddings)
        node_probs: [N,1] in (0,1]
        Returns: context [M, D]
        """
        Bq = 1  # single-graph API; for batching, group by graph id
        M, D = Q.shape; N = K.shape[0]
        H = self.num_heads; Dh = self.dk
        # project
        Qh = self.q_proj(Q).view(M, H, Dh).transpose(0,1)   # [H, M, Dh]
        Kh = self.k_proj(K).view(N, H, Dh).transpose(0,1)   # [H, N, Dh]
        Vh = self.v_proj(V).view(N, H, Dh).transpose(0,1)   # [H, N, Dh]
        # scaled dot‑product
        logits = torch.matmul(Qh, Kh.transpose(1,2)) / math.sqrt(Dh)  # [H, M, N]
        # additive bias from node_probs
        bias = torch.log(torch.clamp(node_probs.squeeze(-1), min=1e-6)).unsqueeze(0).unsqueeze(0)  # [1,1,N]
        logits = logits + bias
        attn = torch.softmax(logits, dim=-1)  # [H, M, N]
        ctx = torch.matmul(attn, Vh)          # [H, M, Dh]
        ctx = ctx.transpose(0,1).contiguous().view(M, H*Dh)  # [M, D]
        return self.out_proj(ctx)             # [M, hidden]

    # ---------- forward ----------
    def forward(self, inputs: dict, is_training: bool) -> torch.Tensor:
        x_in, (eM, eM_attr), (eW, eW_attr), probs, batch = self._build_graph(inputs, is_training)
        X = self._encode_nodes(x_in)  # [N, D]
        X = X*probs # scale node embeddings by node existence prob
        # Encode edges
        eM_lat = self.edge_enc_mesh(eM_attr)    # [E_M, L]
        eW_lat = self.edge_enc_world(eW_attr)   # [E_W, L]

        # Multi-edge message passing
        for blk in self.blocks:
            X, eM_lat, eW_lat = blk(X, eM, eM_lat, eW, eW_lat)

        # (unchanged) direction queries + heads...
        assert self.dir_cart.numel() > 0, "Call set_direction_grid() first."
        d = self.dir_norm(self.dir_cart, accumulate=is_training)
        Q = self.dir_proj(d)

        if batch is not None:
            B = int(batch.max().item()) + 1
            outs = []
            for i in range(B):
                m = (batch == i)
                Xi    = X[m]
                probi = probs[m]
                gi = self.global_proj(Xi.mean(dim=0, keepdim=True)) if Xi.size(0) else \
                    self.global_proj(torch.zeros(1, self.cfg.node_latent, device=X.device))
                if 'frequency' in inputs and inputs['frequency'] is not None:
                    f = self.freq_norm(inputs['frequency'][i:i+1].reshape(-1,1).to(gi.dtype), accumulate=is_training)
                    gi = gi + self.freq_proj(f)
                ctx = self._multihead_attention(Q, Xi, Xi, probi)
                h   = self.cross(torch.cat([ctx, gi.expand(ctx.size(0), -1)], dim=-1))
                y   = self.head(h)
                outs.append(y.unsqueeze(0))
            if self.cfg.predict_surface_current:
                # Also predict surface current at each vertex
                current_out = self.current_decoder(X)  # [N, current_dim]
                ff_out = torch.cat(outs, dim=0)
                output = {
                    'radiation_image': ff_out,
                    'surface_current': current_out
                    }
                return   output
            else:
                return torch.cat(outs, dim=0)

        g  = self.global_proj(X.mean(dim=0, keepdim=True))
        if 'frequency' in inputs and inputs['frequency'] is not None:
            f = self.freq_norm(inputs['frequency'].reshape(-1,1).to(g.dtype), accumulate=is_training)
            g = g + self.freq_proj(f)
        ctx = self._multihead_attention(Q, X, X, probs)
        h   = self.cross(torch.cat([ctx, g.expand(ctx.size(0), -1)], dim=-1))
        return self.head(h)

    def loss(self, inputs: dict, loss_type: str = 'mse') -> torch.Tensor:
        y_norm = self.forward(inputs, is_training=True)
        target = inputs['target_ff']
        t_norm = self.out_norm(target, accumulate=True)
        if loss_type == 'mae':
            err = (t_norm - y_norm).abs()
        elif loss_type == 'huber':
            err = F.huber_loss(y_norm, t_norm, reduction='none')
        else:
            err = (t_norm - y_norm) ** 2
        return err.mean()

    @torch.no_grad()
    def infer(self, inputs: dict) -> torch.Tensor:
        y_norm = self.forward(inputs, is_training=False)
        return self.out_norm.inverse(y_norm)


class MeshGraphNetWrapper(nn.Module):
    def __init__(self, config, device):

        super(MeshGraphNetWrapper, self).__init__()
        self.config = config
        self.setup_config()
        C=10
        gcfg = AntennaFFConfig(num_node_types=C)
        gcfg.predict_surface_current = config.get('predict_surface_current', False)
        cfg = gcfg
        attn_model = AntennaFarFieldAttnModel(gcfg, num_heads=4).to(device)

        S = self.config.get('radiation_image_shape')[0]
        theta = torch.linspace(0.0, math.pi, S)
        phi = torch.linspace(-math.pi, math.pi, S)
        TH, PH = torch.meshgrid(theta, phi, indexing='ij')
        theta_phi = torch.stack([TH.reshape(-1), PH.reshape(-1)], dim=-1).to(device)
        attn_model.set_direction_grid(theta_phi)
        self.model = attn_model

    def forward(self, data, radiation_image_shape=(64,64), is_training=True):
        # Convert PyG Batch -> dense tensors then flatten with explicit batch mapping.
        pos_dense, mask = to_dense_batch(data.pos, data.batch)              # [B, Nmax, 3], [B, Nmax]
        node_normals_dense, _ = to_dense_batch(data.node_normals, data.batch)
        node_type_dense, _ = to_dense_batch(data.node_type, data.batch)
        node_probs_dense, _ = to_dense_batch(data.node_probs, data.batch)

        B, Nmax, _ = pos_dense.shape
        mask_bool = mask.bool()
        # Flatten valid nodes in batch order to match the model's expected sparse input
        pos_flat = pos_dense[mask_bool]
        node_normals_flat = node_normals_dense[mask_bool]
        node_type_flat = node_type_dense[mask_bool]

        node_probs_flat = node_probs_dense[mask_bool]
        # build explicit batch vector mapping each flattened node to its graph index
        batch_idx = torch.arange(B, device=pos_dense.device).unsqueeze(1).expand(B, Nmax)
        batch_flat = batch_idx[mask_bool]

        edge_index = data.edge_index
        inputs = {
            'pos': pos_flat,
            'node_normals': node_normals_flat,
            'node_type': node_type_flat,
            'node_probs': node_probs_flat,
            'edge_index': edge_index,
            'batch': batch_flat,
            'frequency': None,
            'target_ff': None,
        }
        x = self.model(inputs, is_training=is_training)  # expect [B, M, 2]
        if isinstance(x, dict):
            ff = x['radiation_image']
            ff = torch.sum(ff, dim=-1)
            ff = ff.view(-1, self.config.get('radiation_image_shape')[0], self.config.get('radiation_image_shape')[1]).unsqueeze(-1)
            x['radiation_image'] = ff
            return  x
        else:
            x = torch.sum(x, dim=-1)
            x = x.view(-1, self.config.get('radiation_image_shape')[0], self.config.get('radiation_image_shape')[1])
            return x.unsqueeze(-1)  # [B, 1, H, W]

    def setup_config(self):
        self.config['k'] = 20
        self.config['emb_dims'] = 1024
        self.config['dropout'] = 0.5
