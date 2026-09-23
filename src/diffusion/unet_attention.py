import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Small helpers (swappable)
# -------------------------
class Conv(nn.Module):
    """Conv → BN → GELU"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
    def forward(self, x): return self.net(x)

class DownBlock(nn.Module):
    """Downsample by 2, then refine."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = Conv(in_ch, out_ch, k=3, s=2, p=1)    # /2
        self.refine = Conv(out_ch, out_ch, k=3, s=1, p=1)
    def forward(self, x):
        x = self.down(x)
        x = self.refine(x)
        return x

class UpBlock(nn.Module):
    """Nearest upsample by 2, then fuse with conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch, k=3, s=1, p=1)
        self.conv2 = Conv(out_ch, out_ch, k=3, s=1, p=1)
    def forward(self, x):
        # x is concatenated features before calling this
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Flatten(nn.Module):
    """(B,C,H,W) -> (B,C,1,1) via GAP."""
    def __init__(self): super().__init__()
    def forward(self, x): return F.adaptive_avg_pool2d(x, output_size=1)

# tiny MLP for time embeddings
class FC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU()
        )
    def forward(self, x): return self.net(x)

# ---------------------------------
# Cross-attention utility (inlined)
# ---------------------------------
class CrossAttnBlock(nn.Module):
    """Cross-attend a feature map (queries) to external KV tokens."""
    def __init__(self, c_q: int, c_kv: int, n_heads: int = 4, head_dim: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner = n_heads * head_dim

        self.to_q = nn.Conv2d(c_q, inner, 1, bias=False)
        self.to_k = nn.Linear(c_kv, inner, bias=False)
        self.to_v = nn.Linear(c_kv, inner, bias=False)
        self.proj = nn.Conv2d(inner, c_q, 1, bias=False)

    def forward(self, feat: torch.Tensor, kv_tokens: torch.Tensor):
        """
        feat: (B, Cq, H, W)
        kv_tokens: (B, T, Ckv)
        """
        B, Cq, H, W = feat.shape
        T = kv_tokens.shape[1]

        q = self.to_q(feat)                                 # (B, inner, H, W)
        q = q.view(B, self.n_heads, self.head_dim, H*W)     # (B,h,d,L)
        q = q.permute(0, 1, 3, 2).reshape(B*self.n_heads, H*W, self.head_dim)  # (B*h,L,d)

        k = self.to_k(kv_tokens)                             # (B,T,inner)
        v = self.to_v(kv_tokens)                             # (B,T,inner)
        k = k.view(B, T, self.n_heads, self.head_dim).permute(0,2,1,3).reshape(B*self.n_heads, T, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim).permute(0,2,1,3).reshape(B*self.n_heads, T, self.head_dim)

        # Fused SDPA (uses FlashAttention backend when available)
        attn = F.scaled_dot_product_attention(q, k, v)       # (B*h, L, d)
        attn = attn.reshape(B, self.n_heads, H*W, self.head_dim).permute(0,1,3,2)  # (B,h,d,L)
        attn = attn.reshape(B, self.n_heads*self.head_dim, H, W)
        return self.proj(attn) + feat                        # residual

# ---------------------------------------
# Far-field -> multi-scale token encoder
# ---------------------------------------
class FarfieldTokenEncoder(nn.Module):
    """
    Produce KV tokens aligned to UNet scales:
      - tokens8: from 8x8 features -> T8 = 64 (or reduced)
      - tokens4: from 4x4 features -> T4 = 16
    """
    def __init__(self, ff_in_ch: int, H: int, reduce8: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(ff_in_ch, H, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((16,16))  # normalize far-field spatial size
        )
        self.to8 = nn.Sequential(
            nn.Conv2d(H, H, 3, stride=2, padding=1), nn.GELU()   # 16 -> 8
        )
        self.to4 = nn.Sequential(
            nn.Conv2d(H, 2*H, 3, stride=2, padding=1), nn.GELU() # 8 -> 4
        )
        self.reduce8 = reduce8
        if reduce8 > 1:
            self.pool8 = nn.AvgPool2d(kernel_size=reduce8, stride=reduce8)  # e.g., 8->4

    def forward(self, c):
        """
        c: far-field tensor (B, ff_in_ch, Hf, Wf), e.g. channels=[Eθ_mag, Eθ_phase, Eφ_mag, Eφ_phase]
        """
        x16 = self.stem(c)                # (B, H, 16, 16)
        f8  = self.to8(x16)               # (B, H, 8, 8)
        f4  = self.to4(f8)                # (B, 2H, 4, 4)
        if hasattr(self, "pool8"):
            f8 = self.pool8(f8)           # optionally (B, H, 4, 4) to cut tokens by 4×

        # flatten to tokens
        B, Ch8, H8, W8 = f8.shape
        B, Ch4, H4, W4 = f4.shape
        tokens8 = f8.permute(0,2,3,1).reshape(B, H8*W8, Ch8)   # (B, T8,  H)
        tokens4 = f4.permute(0,2,3,1).reshape(B, H4*W4, Ch4)   # (B, 16,  2H)
        return tokens8, tokens4

# ----------------------------------------------
# The full ConditionalDenoisingUNetSmall (16x16)
# ----------------------------------------------
class ConditionalDenoisingUNetSmallAttention(nn.Module):
    """
    16x16 denoising U-Net with time embedding and far-field cross-attention conditioning.
    Path: 16 -> 8 -> 4 -> 8 -> 16
    """
    def __init__(
        self,
        in_channels: int,     # channels of x (noisy image)
        ff_in_ch: int,        # channels of far-field condition c
        num_hiddens: int,     # base width H
        n_heads: int = 4,
        head_dim: int = 64,
        tokens8_reduce: int = 1,  # 1=64 tokens; 2=16 tokens from 8x8 stage
    ):
        super().__init__()
        H = num_hiddens

        # Encoder
        self.conv_in = Conv(in_channels, H)          # 16x16
        self.down1   = DownBlock(H, H)               # 16->8
        self.down2   = DownBlock(H, 2*H)             # 8->4

        # Latent squeeze/expand
        self.flatten   = Flatten()                   # (B,2H,4,4)->(B,2H,1,1)
        self.unflatten = nn.Upsample(size=(4,4), mode="nearest")

        # Decoder
        self.up1     = UpBlock(4*H, H)              # 4->8 (input is cat([enc3, unflattened]))
        self.up2     = UpBlock(2*H, H)              # 8->16 (cat with enc2)
        self.conv_out = Conv(2*H, H)
        self.conv2D   = nn.Conv2d(H, in_channels, kernel_size=1)

        # Time embeddings
        self.fc1_t = FC(1, H)       # for 8x8
        self.fc2_t = FC(1, 2*H)     # for 4x4

        # Far-field -> tokens
        self.ff_tokens = FarfieldTokenEncoder(ff_in_ch, H, reduce8=tokens8_reduce)

        # Cross-attention blocks
        self.ca4 = CrossAttnBlock(c_q=2*H, c_kv=2*H, n_heads=n_heads, head_dim=head_dim)  # 4x4
        self.ca8 = CrossAttnBlock(c_q=H,   c_kv=H,   n_heads=n_heads, head_dim=head_dim)  # 8x8

    def forward(
        self,
        x: torch.Tensor,  # (B, Cx, 16, 16)
        c: torch.Tensor,  # (B, Cff, Hf, Wf) far-field
        t: torch.Tensor,  # (B, 1) normalized time
        mask: torch.Tensor | None = None,  # optional (B,1,1,1) to gate conditioning
    ) -> torch.Tensor:
        assert x.shape[-2:] == (16, 16), "Expect input shape to be (16, 16)."

        # Prepare far-field tokens once
        tokens8, tokens4 = self.ff_tokens(c.unsqueeze(1))  # (B, T8, H), (B, 16, 2H)
        if mask is not None:
            # broadcast mask over tokens
            tokens8 = tokens8 * mask.unsqueeze(-1)  # Broadcast mask to match tokens8 shape
            tokens4 = tokens4 * mask.unsqueeze(-1)  # Broadcast mask to match tokens4 shape

        # Time embeddings
        enc_t_1 = self.fc1_t(t).unsqueeze(-1).unsqueeze(-1)   # (B,H,1,1)
        enc_t_2 = self.fc2_t(t).unsqueeze(-1).unsqueeze(-1)   # (B,2H,1,1)

        # ----- Encoder -----
        enc1 = self.conv_in(x)          # (B,H,16,16)
        enc2 = self.down1(enc1)         # (B,H,8,8)
        enc3 = self.down2(enc2)         # (B,2H,4,4)

        # ----- Bottleneck / Time -----
        z    = self.flatten(enc3)       # (B,2H,1,1)
        unfl = self.unflatten(z) + enc_t_2  # (B,2H,4,4)

        # Cross-attend at 4x4
        enc3 = self.ca4(enc3, tokens4)  # (B,2H,4,4)

        # ----- Decoder -----
        # 4 -> 8
        dec1 = self.up1(torch.cat([enc3, unfl], dim=1))  # (B,H,8,8)

        # add time then cross-attend at 8x8
        dec1 = dec1 + enc_t_1
        dec1 = self.ca8(dec1, tokens8)                   # (B,H,8,8)

        # 8 -> 16
        dec2 = self.up2(torch.cat([enc2, dec1], dim=1))  # (B,H,16,16)

        out  = self.conv_out(torch.cat([enc1, dec2], dim=1))  # (B,H,16,16)
        return self.conv2D(out)                               # (B,Cx,16,16)

# ----------------------------------------------
# The full ConditionalDenoisingUNetSmall (16x16)
# ----------------------------------------------
class ConditionalDenoisingUNetSmallAttentionWithConcat(nn.Module):
    """
    16x16 denoising U-Net with time embedding and far-field cross-attention conditioning.
    Path: 16 -> 8 -> 4 -> 8 -> 16
    """
    def __init__(
        self,
        config,
        in_channels: int,     # channels of x (noisy image)
        ff_in_ch: int,        # channels of far-field condition c
        num_hiddens: int,     # base width H
        n_heads: int = 4,
        head_dim: int = 64,
        tokens8_reduce: int = 1,  # 1=64 tokens; 2=16 tokens from 8x8 stage
    ):
        super().__init__()
        H = num_hiddens

        # Encoder
        self.conv_in = Conv(in_channels + ff_in_ch, H)  # 16x16
        self.down1   = DownBlock(H, H)               # 16->8
        self.down2   = DownBlock(H, 2*H)             # 8->4

        # Latent squeeze/expand
        self.flatten   = Flatten()                   # (B,2H,4,4)->(B,2H,1,1)
        self.unflatten = nn.Upsample(size=(4,4), mode="nearest")

        # Decoder
        self.up1     = UpBlock(4*H, H)              # 4->8 (input is cat([enc3, unflattened]))
        self.up2     = UpBlock(2*H, H)              # 8->16 (cat with enc2)
        self.conv_out = Conv(2*H, H)
        self.conv2D   = nn.Conv2d(H, in_channels, kernel_size=1)

        # Time embeddings
        self.fc1_t = FC(1, H)       # for 8x8
        self.fc2_t = FC(1, 2*H)     # for 4x4

        # Far-field -> tokens
        self.ff_tokens = FarfieldTokenEncoder(ff_in_ch, H, reduce8=tokens8_reduce)

        # Cross-attention blocks
        self.ca4 = CrossAttnBlock(c_q=2*H, c_kv=2*H, n_heads=n_heads, head_dim=head_dim)  # 4x4
        self.ca8 = CrossAttnBlock(c_q=H,   c_kv=H,   n_heads=n_heads, head_dim=head_dim)  # 8x8


    def forward(
        self,
        x: torch.Tensor,  # (B, Cx, 16, 16)
        c: torch.Tensor,  # (B, Cff, Hf, Wf) far-field
        t: torch.Tensor,  # (B, 1) normalized time
        mask: torch.Tensor | None = None,  # optional (B,1,1,1) to gate conditioning
    ) -> torch.Tensor:
        assert x.shape[-2:] == (16, 16), "Expect input shape to be (16, 16)."

        # Resize condition to match input x (16x16)
        c_resized = F.interpolate(c, size=(16, 16), mode='bilinear', align_corners=False)

        # Prepare far-field tokens once
        tokens8, tokens4 = self.ff_tokens(c)  # (B, T8, H), (B, 16, 2H)
        if mask is not None:
            # broadcast mask over tokens
            tokens8 = tokens8 * mask.unsqueeze(-1)  # Broadcast mask to match tokens8 shape
            tokens4 = tokens4 * mask.unsqueeze(-1)  # Broadcast mask to match tokens4 shape
            c_resized = c_resized * mask.unsqueeze(-1).unsqueeze(-1)  # Broadcast mask to match c_resized shape

        # Time embeddings
        enc_t_1 = self.fc1_t(t).unsqueeze(-1).unsqueeze(-1)   # (B,H,1,1)
        enc_t_2 = self.fc2_t(t).unsqueeze(-1).unsqueeze(-1)   # (B,2H,1,1)

        # ----- Encoder -----
        # Concatenate noisy image + condition
        x = torch.cat([x, c_resized], dim=1)
        enc1 = self.conv_in(x)          # (B,H,16,16)
        enc2 = self.down1(enc1)         # (B,H,8,8)
        enc3 = self.down2(enc2)         # (B,2H,4,4)

        # ----- Bottleneck / Time -----
        z    = self.flatten(enc3)       # (B,2H,1,1)
        unfl = self.unflatten(z) + enc_t_2  # (B,2H,4,4)

        # Cross-attend at 4x4
        enc3 = self.ca4(enc3, tokens4)  # (B,2H,4,4)

        # ----- Decoder -----
        # 4 -> 8
        dec1 = self.up1(torch.cat([enc3, unfl], dim=1))  # (B,H,8,8)

        # add time then cross-attend at 8x8
        dec1 = dec1 + enc_t_1
        dec1 = self.ca8(dec1, tokens8)                   # (B,H,8,8)

        # 8 -> 16
        dec2 = self.up2(torch.cat([enc2, dec1], dim=1))  # (B,H,16,16)

        out  = self.conv_out(torch.cat([enc1, dec2], dim=1))  # (B,H,16,16)
        return self.conv2D(out)


# -------------------------
# Quick sanity check (CPU)
# -------------------------
if __name__ == "__main__":
    B = 2
    Cx = 1
    Cff = 4          # e.g., [Eθ_mag, Eθ_phase, Eφ_mag, Eφ_phase]
    H = 128

    model = ConditionalDenoisingUNetSmallAttention(
        in_channels=Cx,
        ff_in_ch=Cff,
        num_hiddens=H,
        n_heads=4,
        head_dim=64,
        tokens8_reduce=1,  # set to 2 to reduce 8x8 tokens from 64 -> 16
    )

    x  = torch.randn(B, Cx, 16, 16)
    ff = torch.randn(B, Cff, 34, 34)   # arbitrary far-field size; encoder normalizes to 16x16
    t  = torch.rand(B, 1)              # normalized diffusion time
    y  = model(x, ff, t)               # (B,Cx,16,16)
    print("out:", y.shape)
