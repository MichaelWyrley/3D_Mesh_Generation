# MODIFIED FROM https://github.com/facebookresearch/DiT/blob/main/models.py
import sys
sys.path.append('')
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class PointFF(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0):
        super(PointFF, self).__init__()

        self.activation = nn.GELU(approximate='tanh')

        self.w_1 = nn.Linear(in_features, hidden_features)
        self.w_2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        z = self.w_1(x)
        z = self.activation(z)
        z = self.dropout(z)
        z = self.w_2(z)

        return z


# adaLN-Zero from transformer paper
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = PointFF(in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        z = modulate(self.norm1(x), shift_msa, scale_msa)
        attention_output, _ = self.attn(z,z,z)
        x = x + gate_msa.unsqueeze(1) * attention_output
        z = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(z)
        return x
    

class PatchEmbedding(nn.Module):
    def __init__(self, in_size, emb_dimention):
        super(PatchEmbedding, self).__init__()
        self.emb_dimention = emb_dimention
        
        self.fc = nn.Linear(in_size, emb_dimention, bias=False)
        self.ln = nn.LayerNorm([emb_dimention])

    def position_encoding(self, x):
        b, n, _ = x.shape
        pos = torch.arange(n, device=x.device).float()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.emb_dimention, 2, device=x.device).float() / self.emb_dimention))
        pos = pos[:, None] * inv_freq[None, :]
        pos_enc = torch.cat([pos.sin(), pos.cos()], dim=-1)[None, :, :].expand(b, -1, -1)
        return pos_enc

    def forward(self, x):
        # b, n, _ = x.shape
        # x = x.reshape((b*n, -1))
        x = self.fc(x)
        
        x = self.ln(x)
        # x.reshape((b,n,-1))
        x += self.position_encoding(x)
        return x

# DIRECTLY TAKEN FROM https://github.com/facebookresearch/DiT/blob/main/models.py
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        self.hidden_size = hidden_size
    
    def time_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc


    def forward(self, t):
        t = t + self.time_encoding(t, self.hidden_size)
        t_emb = self.mlp(t)
        return t_emb


class DiT_adaLN_zero(nn.Module):
    def __init__(self, in_dim=3, depth=12, emb_dimention=768, num_heads=12, num_classes=55, device='cuda', condition_prob=1.0):
        super(DiT_adaLN_zero, self).__init__()
        self.condition_prob = condition_prob
        self.num_heads = num_heads
        self.emb_dimention = emb_dimention
        self.device = device

        self.blocks = nn.ModuleList([DiTBlock(emb_dimention, num_heads) for _ in range(depth)])

        self.condition_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dimention, 2 * emb_dimention),
        )
        self.layernorm = nn.LayerNorm(emb_dimention, elementwise_affine=False, eps=1e-6)
        self.fc = nn.Linear(emb_dimention, in_dim )


        self.label_emb = nn.Embedding(num_classes, emb_dimention)
        self.patch_embedding = PatchEmbedding(in_dim, emb_dimention)
        self.time_embedding = TimestepEmbedder(emb_dimention)

        self.initialise_weights()

        self.to(device)
    
    def initialise_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedding.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedding.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.condition_mlp[-1].weight, 0)
        nn.init.constant_(self.condition_mlp[-1].bias, 0)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, t, c=None):

        # Encode time and condition
        t = self.time_embedding(t)
        if c is not None and random.random() < self.condition_prob:
            c = self.label_emb(c)
            t = t + c
        # Patchify and encode image
        x = self.patch_embedding(x) 

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, t)

        
        # Use adaptive Layer Normalization
        shift, scale = self.condition_mlp(t).chunk(2, dim=1)
        # Reshape x to the original image shape
        x = self.layernorm(x)
        x = modulate(x, shift, scale)

        x = self.fc(x)
        
        return x

        
        

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # net = UNet(device="cpu")
    net = DiT_adaLN_zero(num_classes=10, device=device).to(device)

    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 21, 6).to(device)
    t = x.new_tensor([0.02] * x.shape[0]).unsqueeze(-1).to(device)
    # y = x.new_tensor([1] * x.shape[0]).long().to(device)
    print(net(x, t).shape)
