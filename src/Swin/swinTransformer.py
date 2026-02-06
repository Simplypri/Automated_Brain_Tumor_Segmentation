import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Utils: window partition / reverse ---------

def window_partition(x, window_size):
    """
    x: (B, H, W, C)
    return: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()   # (B, nWh, nWw, Ws, Ws, C)
    x = x.view(-1, window_size, window_size, C)    # (B*nWh*nWw, Ws, Ws, C)
    return x

def window_reverse(windows, window_size, H, W):
    """
    windows: (num_windows*B, window_size, window_size, C)
    return: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x

# --------- MLP ---------

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# --------- Window Multi-Head Self Attention ---------

class WindowAttention(nn.Module):
    """W-MSA with relative position bias."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size-1)*(2*window_size-1), num_heads)
        )  # (2Ws-1)^2, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, Ws, Ws)
        coords_flatten = torch.flatten(coords, 1)  # (2, Ws*Ws)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Ws*Ws, Ws*Ws)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Ws*Ws, Ws*Ws, 2)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2*window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (Ws*Ws, Ws*Ws)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        """
        x: (B*nW, Ws*Ws, C)
        mask: (nW, Ws*Ws, Ws*Ws) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: (B_, nH, N, head_dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, nH, N, N)

        # add relative position bias
        rpbt = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1
        )  # (N, N, nH)
        rpbt = rpbt.permute(2, 0, 1).unsqueeze(0)  # (1, nH, N, N)
        attn = attn + rpbt

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# --------- Swin Transformer Block (W-MSA / SW-MSA) ---------

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        H, W = input_resolution
        if min(H, W) <= window_size:
            self.window_size = min(H, W)
            self.shift_size = 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = nn.Identity() if drop_path == 0. else DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

        # attention mask for SW-MSA
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # (1, H, W, 1)
            h_slices = (slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None))
            w_slices = (slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # (nW, Ws, Ws, 1)
            mask_windows = mask_windows.view(-1, self.window_size*self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        """
        x: (B, H*W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (B*nW, Ws, Ws, C)
        x_windows = x_windows.view(-1, self.window_size*self.window_size, C)  # (B*nW, Ws*Ws, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # (B*nW, Ws*Ws, C)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # (B, H, W, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H*W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# --------- Patch Merging (downsample, double channels) ---------

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        """
        x: (B, H*W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Wrong token length"

        x = x.view(B, H, W, C)
        # 2x downsample by concatenating 2x2 neighbors
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)                # (B, H/2*W/2, 4C)
        x = self.norm(x)
        x = self.reduction(x)                   # (B, H/2*W/2, 2C)
        return x

# --------- Stochastic depth (DropPath) ---------

class DropPath(nn.Module):
    """Drop paths per sample (Stochastic Depth)."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep) * random_tensor

# --------- Patch Embedding ---------

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)            # (B, embed_dim, H/ps, W/ps)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        x = self.norm(x)
        return x, (H, W)

# --------- Basic Swin Stage (multiple blocks + optional merging) ---------

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path if isinstance(drop_path, float) else drop_path[i],
                )
            )
        self.downsample = PatchMerging(input_resolution, dim) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

# --------- Swin Transformer (backbone) ---------

class SwinTransformer(nn.Module):
    """
    depths: number of blocks in each stage, e.g., [2,2,6,2]
    num_heads: attention heads per stage, e.g., [3,6,12,24]
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24),
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.2, include_head=True):
        super().__init__()

        self.include_head = include_head
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        layers = []
        dims = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
        res = None

        x_dummy = torch.zeros(1, in_chans, img_size, img_size)
        _, res = self.patch_embed(x_dummy)
        H, W = res

        dp_ptr = 0
        for i in range(len(depths)):
            layer = BasicLayer(
                dim=dims[i],
                input_resolution=(H, W),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[dp_ptr:dp_ptr+depths[i]],
                downsample=(i < len(depths)-1),
            )
            layers.append(layer)
            dp_ptr += depths[i]
            if i < len(depths)-1:
                H, W = H // 2, W // 2  # after PatchMerging

        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(dims[-1])
        if include_head:
            self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x):
        # x: (B, C, H, W)
        x, res = self.patch_embed(x)    # (B, HW, C0), (H0, W0)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)                # (B, HW/16, C4)
        x = x.mean(dim=1)               # global average pool (token-wise)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.include_head:
            x = self.head(x)
        return x
