"""Frequency conditioning with explicit bandwidth control.

Drop-in replacement for the frequency-embedding parts of
utilities/frequency_conditioning.py.

Key change vs. the old version: instead of a geometric ladder of Fourier
bands (2^0 ... 2^15 cycles, which turns 4 training frequencies into 4
quasi-orthogonal codes), the band frequencies are chosen so the FASTEST
band completes only `max_cycles` periods across the normalized k range.
This forces the embedding to be a smooth, low-resolution function of k,
which is the only thing 4 training frequencies can support -- and it is
what makes zero-shot interpolation to a held-out frequency work.

Target behavior (verify with `diagnose_embedding` below): the cosine
similarity between embeddings should decay smoothly and monotonically
with |delta k| -- adjacent training frequencies ~0.7-0.9 similar,
opposite ends of the band ~0.1-0.3, no sign flips.
"""

import math

import torch
import torch.nn as nn

C0_M_PER_S = 299_792_458.0

# meters per unit of data.pos
_UNIT_TO_METERS = {
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
}


def wavenumber_from_freq(freq_hz: torch.Tensor, length_unit: str = "m") -> torch.Tensor:
    """k = 2*pi*f/c, expressed in [1/length_unit] so k * data.pos is unitless."""
    if length_unit not in _UNIT_TO_METERS:
        raise ValueError(f"unknown length_unit '{length_unit}', expected one of {list(_UNIT_TO_METERS)}")
    scale = _UNIT_TO_METERS[length_unit]  # k[1/unit] = k[1/m] * (m per unit)
    return 2.0 * math.pi * freq_hz / C0_M_PER_S * scale


def build_freq_embedding_factory(freqs_hz, length_unit: str = "m", margin: float = 0.25):
    """Return {'k_min', 'k_max'} spanning the training k range plus a margin.

    margin is a fraction of the k span added on each side (0.25 -> +/-25%).
    The margin (a) leaves headroom for mild extrapolation and (b) compresses
    the training frequencies into the interior of [0, 1], further lowering
    the effective resolution of the embedding.
    """
    freqs = torch.as_tensor([float(f) for f in freqs_hz], dtype=torch.float64)
    k = wavenumber_from_freq(freqs, length_unit)
    k_lo, k_hi = float(k.min()), float(k.max())
    span = k_hi - k_lo
    if span <= 0:
        raise ValueError("need at least two distinct frequencies to build k bounds")
    return {
        "k_min": k_lo - margin * span,
        "k_max": k_hi + margin * span,
    }


class FrequencyEmbedding(nn.Module):
    """Low-bandwidth Fourier embedding of the wavenumber k.

    k is normalized to t = (k - k_min) / (k_max - k_min) in [0, 1].
    Features are [2t-1, sin(w_j t), cos(w_j t)] where the band cycle counts
    c_j are spaced linearly from `min_cycles` to `max_cycles` across the
    normalized range (w_j = 2*pi*c_j). A small MLP projects to `out_dim`.

    With num_bands=4 and max_cycles=2 the fastest feature varies over
    ~1/6 of the range per half-period -- smooth enough to interpolate
    between training frequencies, too smooth to memorize them.
    """

    def __init__(
        self,
        out_dim: int,
        num_bands: int = 4,
        k_min: float = 0.0,
        k_max: float = 1.0,
        min_cycles: float = 0.5,
        max_cycles: float = 2.0,
    ):
        super().__init__()
        if k_max <= k_min:
            raise ValueError(f"k_max ({k_max}) must be > k_min ({k_min})")
        self.k_min = float(k_min)
        self.k_max = float(k_max)

        cycles = torch.linspace(float(min_cycles), float(max_cycles), int(num_bands))
        self.register_buffer("omegas", 2.0 * math.pi * cycles)  # [num_bands]

        feat_dim = 1 + 2 * int(num_bands)  # linear term + sin/cos per band
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def features(self, k: torch.Tensor) -> torch.Tensor:
        """Raw (deterministic) Fourier features, before the learned projection."""
        k = k.view(-1).float()
        t = (k - self.k_min) / (self.k_max - self.k_min)

        # Never silently collapse out-of-range frequencies onto the boundary:
        # the linear term and the slow bands still extrapolate, so just warn.
        if bool((t < -0.05).any() or (t > 1.05).any()):
            print(
                f"[FrequencyEmbedding] warning: k in [{k.min():.3g}, {k.max():.3g}] "
                f"outside embedding bounds [{self.k_min:.3g}, {self.k_max:.3g}]"
            )

        phase = t.unsqueeze(-1) * self.omegas  # [B, num_bands]
        return torch.cat(
            [
                (2.0 * t - 1.0).unsqueeze(-1),  # [B, 1]
                torch.sin(phase),               # [B, num_bands]
                torch.cos(phase),               # [B, num_bands]
            ],
            dim=-1,
        )

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """k: [B] wavenumbers -> [B, out_dim] embedding."""
        return self.proj(self.features(k))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def diagnose_embedding(freq_embed: FrequencyEmbedding, freqs_hz, length_unit: str = "m"):
    """Print k values and the pairwise cosine-similarity matrix of embeddings.

    The RAW-FEATURE matrix is the one that matters for judging embedding
    bandwidth: it is deterministic and should decay smoothly and monotonically
    with |delta k| (adjacent training freqs ~0.7-0.9, opposite band ends near 0).
    The PROJECTED matrix will look noisy at random init and only becomes
    meaningful after training; check it on a trained checkpoint to see what
    resolution the model actually learned to use.

    Red flags (raw matrix): near-zero/negative similarity between adjacent
    frequencies (memorization regime), or an all-ones matrix (collapsed).
    """
    freqs = torch.as_tensor(list(freqs_hz), dtype=torch.float32)
    k = wavenumber_from_freq(freqs, length_unit)
    print("k:", k)

    feats = freq_embed.features(k)
    sim_raw = torch.nn.functional.cosine_similarity(feats.unsqueeze(1), feats.unsqueeze(0), dim=-1)
    print("raw-feature cosine similarity (bandwidth check):")
    print(sim_raw)

    emb = freq_embed(k)
    sim_proj = torch.nn.functional.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=-1)
    print("projected-embedding cosine similarity (meaningful only after training):")
    print(sim_proj)
    return sim_raw, sim_proj
