import torch
import torch.nn as nn
import torch.nn.functional as F


class MyTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_blocks):
        super().__init__()
        self.patchify = nn.Conv2d(3, dim, 8, stride=8)
        self.emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim) for _ in range(num_blocks)
        ])
        self.output_layer = nn.ConvTranspose2d(dim, 3, 8, stride=8)

    def forward(self, x):
        # x: B3HW
        x = self.patchify(x)
        # x: B, D, H//p, W//p
        B, D, Hp, Wp = x.shape
        x = x.view(B, D, -1)
        # B D, Hp*Wp
        x = x.permute(0, 2, 1)
        # B, Hp*Wp, D
        # x: [B, L]
        x += self.pos_emb...
        x = self.emb(x)
        # emb: [B, L, D]
        for block in self.blocks:
            x = block(x)
            # emb: [B, L, D]
        # emb: [B, L, D]
        x = x.permute(0, 2, 1)
        # emb: [B, D, Hp * Wp]
        x = x.view(B, D, Hp, Wp)
        return self.output_layer(x)


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # block avec la SA
        self.sa_norm = nn.RMSNorm(dim)
        self.sa_block = SABlock(dim)
        # block de MLP
        self.mlp_norm = nn.RMSNorm(dim)
        self.mlp_block = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = self.sa_block(self.sa_norm(x)) + x
        x = self.mlp_block(self.mlp_norm(x)) + x
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.k = self.Linear(dim, dim)
        self.q = self.Linear(dim, dim)
        self.v = self.Linear(dim, dim)
        self.o = self.Linear(dim, dim)
        self.num_heads = num_heads

    def forward(self, x):
        #x: BLD
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        #k, q, v: BLD
        B, L, D = k.shape
        k = k.view(B, L, self.num_heads, D // self.num_heads)
        q = q.view(B, L, self.num_heads, D // self.num_heads)
        v = v.view(B, L, self.num_heads, D // self.num_heads)

        # k: BLHk
        k = k.permute(0, 2, 1, 3)  # B H L k
        q = q.permute(0, 2, 1, 3)  # B H L k
        v = v.permute(0, 2, 1, 3)  # B H L k

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # out: B H L k
        out = out.permute(0, 2, 1, 3).contiguous()
        # out: B L H k
        out = out.view(B, L, D)
        # out: B L (H k)

        o = self.o(out)

        return o

