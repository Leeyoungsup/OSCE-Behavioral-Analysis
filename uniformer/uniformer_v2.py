import torch
from torch import nn
from einops import rearrange

# Patch Embedding using ViT style (Conv2d)
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, T, H, W] => apply on each frame
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w -> b t (h w) c', b=B, t=T)
        return x  # [B, T, N, C]

# MHSA-T: Temporal attention block
class MHSA_T(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, N, C]
        B, T, N, C = x.shape
        x = rearrange(x, 'b t n c -> b n t c')

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n t (h d) -> b n h t d', h=self.heads), qkv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b n h t d -> b n t (h d)')
        out = self.to_out(out)
        out = rearrange(out, 'b n t c -> b t n c')
        return out

# MHSA-S: Spatial attention block
class MHSA_S(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, N, C]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b t n (h d) -> b t h n d', h=self.heads), qkv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b t h n d -> b t n (h d)')
        return self.to_out(out)

# FeedForward
class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# UniFormerV2 Block
class UniFormerV2Block(nn.Module):
    def __init__(self, dim, attn_type='T', heads=8, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHSA_T(dim, heads, dropout) if attn_type == 'T' else MHSA_S(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# UniFormerV2 Model (Single Input)
class UniFormerV2(nn.Module):
    def __init__(self, num_classes=400, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=[2,2], heads=8, dropout=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        blocks = []
        for i, blk_type in enumerate(['T', 'S'] * (len(depth)//2)):
            blocks.append(UniFormerV2Block(embed_dim, blk_type, heads, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)   # [B, T, N, C]
        x = self.blocks(x)        # [B, T, N, C]
        x = self.norm(x)          # [B, T, N, C]
        x = x.mean(dim=2).mean(dim=1)  # Global average pool: [B, C]
        return self.head(x)       # [B, num_classes]
    
    

class MultiVideoUniFormerV2(nn.Module):
    def __init__(self, num_classes=10, embed_dim=768, depth=[2, 2], heads=8, dropout=0.):
        super().__init__()
        # 세 개의 UniFormerV2 인스턴스 (각각 개별 비디오 입력용)
        self.model1 = UniFormerV2(num_classes=embed_dim, embed_dim=embed_dim, depth=depth, heads=heads, dropout=dropout)
        self.model2 = UniFormerV2(num_classes=embed_dim, embed_dim=embed_dim, depth=depth, heads=heads, dropout=dropout)
        self.model3 = UniFormerV2(num_classes=embed_dim, embed_dim=embed_dim, depth=depth, heads=heads, dropout=dropout)

        # 세 개의 임베딩을 결합 후 최종 분류
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, video1, video2, video3):
        # 각 비디오별로 특징 추출
        feat1 = self.model1(video1)  # [B, embed_dim]
        feat2 = self.model2(video2)
        feat3 = self.model3(video3)

        # 특징 결합
        fused = torch.cat([feat1, feat2, feat3], dim=1)  # [B, embed_dim * 3]

        # 최종 분류
        out = self.classifier(fused)  # [B, num_classes]
        return out