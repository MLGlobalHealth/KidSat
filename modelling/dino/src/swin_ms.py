import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=1024, patch_size=14, in_chans=13, embed_dim=96):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        x = self.norm(x)
        return x

class MLP(nn.Module):
    """Multilayer Perceptron."""
    def __init__(self, in_features, hidden_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    """Window-based multi-head self attention."""
    def __init__(self, dim, num_heads, window_size, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        # Note: Implement window-based attention with proper masking and relative position bias
        # For brevity, the detailed implementation is omitted
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, num_heads=num_heads, window_size=window_size
        )
        self.drop_path = nn.Identity()  # Can use DropPath for stochastic depth
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x):
        # Implement shifted window mechanism here
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer."""

    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"

        x = x.view(B, H, W, C)

        # Pad if the size is not even
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = nn.functional.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # Downsample and concatenate features
        x0 = x[:, 0::2, 0::2, :]   # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]   # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]   # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]   # [B, H/2, W/2, C]

        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]

        x = x.view(B, -1, 4 * C)  # [B, (H/2)*(W/2), 4*C]
        x = self.norm(x)
        x = self.reduction(x)     # [B, (H/2)*(W/2), 2*C]

        return x

class SwinTransformer(nn.Module):
    """Swin Transformer Model."""

    def __init__(self, img_size=1024, patch_size=16, in_chans=13, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        patches_resolution = (img_size // patch_size, img_size // patch_size)
        self.patches_resolution = patches_resolution

        # Build layers
        self.layers = nn.ModuleList()
        dim = embed_dim
        input_resolution = (patches_resolution[0], patches_resolution[1])
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList()
            for i_block in range(depths[i_layer]):
                shift_size = 0 if (i_block % 2 == 0) else window_size // 2
                block = SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    drop=0.0
                )
                layer.append(block)
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                # Add Patch Merging layer
                merge_layer = PatchMerging(input_resolution, dim)
                self.layers.append(merge_layer)
                dim *= 2  # Double the embedding dimension
                # Update input_resolution after merging
                input_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Identity()  # Output the feature vector directly

    def forward(self, x):
        x = self.patch_embed(x)  # [B, N, C]
        for layer in self.layers:
            if isinstance(layer, nn.ModuleList):
                for blk in layer:
                    x = blk(x)
            elif isinstance(layer, PatchMerging):
                x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling over patches
        x = self.head(x)    # Output feature vector
        return x

