import argparse

import numpy as np
import torch
from SoftPool import SoftPool2d
from timm.layers import to_2tuple, trunc_normal_, DropPath
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int,
                    default=6, help='output channel of network')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--vit_patches_size', type=int,
                    default=8, help='vit_patches_size, default is 16')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=2, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')

args = parser.parse_args()


class BasicConv(nn.Module):
    """
    Custom neural network module, including: convolution, batch normalization, and activation functions
    To be reused as a basic module subsequently
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def window_partition(x, window_size: int):
    """
    Divide the feature map into non-overlapping windows of window_size
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Restore the individual windows into a single feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 512.
        patch_size (int): Patch token size. Default: 8.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = [patch_size[0] // 2, patch_size[1] // 2]       # stride:[8,8]
        patches_resolution = [img_size[0] // stride[0], img_size[1] // stride[1]]       # patches_resolution:[64,64]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=2)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class WindowAttention(nn.Module):
    """
    Window attention
    Used to capture local relationship information in images for better feature extraction and model training
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    Multi-layer Perceptron (MLP)
    Consists of two fully connected layers, two dropout layers, and an activation function,
    used for non-linear transformation of input data.
    Extracts high-level features from the input data.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer basic module
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=16, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            # Determine whether to perform window shifting; if needed, proceed to SW-MSA, otherwise, go to W-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"
        shortcut = x  # ([12, 4096, 96])
        x = self.norm1(x)

        x = x.view(B, H, W, C)

        # Window shifting
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # Inverse window shifting
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class NewPatchMerging(nn.Module):
    """
    Feature compression module NPM, which compresses and fuses the input features
    to achieve effective fusion and information extraction of features at different scales
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dconv = nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, dilation=2, padding=2)
        self.dconv1 = nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, dilation=1, padding=1)
        self.dconv2 = nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, dilation=2, padding=2)
        self.dconv3 = nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, dilation=4, padding=4)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim * 6, dim * 2, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Conv2d(dim, 2 * dim, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(dim, 2 * dim, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(2 * dim, 2 * dim, kernel_size=1, stride=2)
        self.bn2 = nn.BatchNorm2d(2 * dim)
        self.gn1 = nn.GroupNorm(dim // 2, dim)
        self.pool = SoftPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.gelu = nn.GELU()

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape  # ([12, 4096, 96])

        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)  # ([12, 64, 64, 384])
        x = x.view(B, C, H, W)

        short = x
        x = self.gelu(self.bn2(self.conv1(x)))

        x1 = self.dconv1(x)
        x2 = self.dconv2(x)
        x3 = self.dconv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv_cat(x)
        # x = self.bn2(self.dconv(x))

        x = self.gelu(self.bn2(self.conv3(x)))
        short = self.gn1(self.pool(short))

        short = self.gelu(self.bn2(self.conv2(short)))

        x = x + short
        x = x.view(B, -1, 2 * C)  # B H/2*W/2 4*C

        return x


class BasicLayer(nn.Module):
    """
    Swin Transformer basic layer, a layer can contain multiple Swin Transformer blocks (2, 2, 6, 2)
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, no_down=0, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.no_down = no_down
        self.bn = nn.BatchNorm2d(dim // 2)
        self.relu = nn.ReLU(inplace=True)
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.gelu = nn.GELU()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        self.conv1 = nn.Conv2d(dim, dim // 2, kernel_size=3, dilation=2, stride=1, padding=2)
        self.conv2 = nn.Conv2d(dim // 2, dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim // 2)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        # Trans_feature
        i = 0
        for blk in self.blocks:
            if i % 2 == 0:

                s = x
                H, W = self.input_resolution
                B, L, C = s.shape  # ([12, 4096, 96])

                assert L == H * W, "input feature has wrong size"
                assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
                s = s.view(B, H, W, C)  # ([12, 64, 64, 384])
                s = s.view(B, C, H, W)

                s = self.gelu(self.bn1(self.conv1(s)))
                s_h, s_w = self.pool_h(s), self.pool_w(s)  # .permute(0, 1, 3, 2)
                s = torch.matmul(s_h, s_w)

                s = self.gelu(self.bn2(self.conv2(s)))
            x = blk(x)
            i=i+1
            if i % 2 == 0:
                H, W = self.input_resolution
                B, L, C = x.shape  # ([12, 4096, 96])
                assert L == H * W, "input feature has wrong size"
                assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
                x = x.view(B, H, W, C)  # ([12, 64, 64, 384])
                x = x.view(B, C, H, W)
                x=x+s
                x = x.view(B, -1, C)

        return x


class SwinTransformer(nn.Module):
    """
    Construct the overall architecture of Swin-Transformer:
    Composed of multiple BasicLayers;
    Each BasicLayer contains multiple SwinTransformerBlock blocks and other feature processing modules.
    """

    # block_units 3,4,9
    def __init__(self, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width  # 64
        # self.num_classes = 6  # 6
        depths = [2, 2, 6, 2]
        self.num_layers = len(depths)
        self.patch_norm = True
        embed_dim = 96  # 96
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.embeddings = PatchEmbed(img_size=256, patch_size=8, in_chans=3,
                                     embed_dim=embed_dim,
                                     norm_layer=nn.LayerNorm if self.patch_norm else None)
        # num_patches = self.embeddings.num_patches
        patches_resolution = self.embeddings.patches_resolution
        self.patches_resolution = patches_resolution

        self.mlp_ratio = 4.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        use_checkpoint = False
        # stochastic depth

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # build layers
        '''
        trans.num_heads = [3, 6, 12, 24]
        trans.depths = [2, 2, 6, 2] 
        '''
        # w=32
        outchannel = [width * 4, width * 8, width * 16, width * 32]
        self.layers = nn.ModuleList()
        self.Down_features = nn.ModuleList()
        self.Conv3 = nn.ModuleList()

        for i_layer in range(self.num_layers):  # i_layer 0,1,2,3
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=[3, 6, 12, 24][i_layer],
                               window_size=8,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               no_down=i_layer,
                               use_checkpoint=use_checkpoint)
            downsample = NewPatchMerging(input_resolution=layer.input_resolution, dim=layer.dim, norm_layer=norm_layer)

            Conv3x3 = BasicConv(in_planes=layer.dim, out_planes=outchannel[i_layer], kernel_size=1, stride=1,
                                groups=1, relu=True, bn=True, bias=False)
            self.Conv3.append(Conv3x3)
            self.Down_features.append(downsample)
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        Trans_features = []
        b, c, in_size, _ = x.shape  # torch.Size([12, 3, 512, 512])
        #print('www',x.shape)
        trans_x = x
        trans_x = self.embeddings(trans_x)  # [12, 4096, 96]
        i_layer = 0
        self.num_layers = len(self.layers)
        for layer in self.layers:
            trans_x = layer(trans_x)
            B, n_patch, hidden = trans_x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
            h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
            Trans_x = trans_x.permute(0, 2, 1)
            Trans_x = Trans_x.contiguous().view(B, hidden, h, w)
            Trans_features.append(Trans_x)

            if (i_layer < self.num_layers - 1):
                trans_x = self.Down_features[i_layer](trans_x)
            i_layer = i_layer + 1

        for i in range(4):
            s1=self.Conv3[i](Trans_features[i])
        return s1


class swin_transformer(nn.Module):
    def __init__(self, num_classes=6, zero_head=False):
        super(swin_transformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = 'seg'
        self.swintransformer = SwinTransformer(width_factor=0.5)

    def forward(self, x):
        x = self.swintransformer(x)  # (B, n_patch, hidden)

        return x

if __name__ == '__main__':

    img = torch.rand((2, 3, 256, 256))
    img = img.to(device)
    net = swin_transformer()
    net = net.to(device)
    output = net(img)
    print(output.shape)