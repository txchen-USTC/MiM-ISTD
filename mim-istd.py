import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass







def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Mlp(nn.Module):
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


class SE(nn.Module):
    def __init__(self, dim, hidden_ratio=None):
        super().__init__()
        hidden_ratio = hidden_ratio or 1
        self.dim = dim
        hidden_dim = int(dim * hidden_ratio)
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()
        )

    def forward(self, x):
        a = x.mean(dim=1, keepdim=True) # B, 1, C
        a = self.fc(a)
        x = a * x
        return x

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x, H, W, relative_pos=None):
        B, N, C = x.shape
        #print('x input',x.shape)
        x = x.permute(0, 2, 1).reshape(B, H, W, C)

        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out=out.reshape(B,N,C)
        #print('x output',out.shape)
        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.pool = nn.AvgPool2d(sr_ratio, stride=sr_ratio)
            self.linear = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W, relative_pos=None):
        B, N, C = x.shape
        print('x input',x.shape)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            #print('outer_tokens before pool',x_.shape)
            x_ = self.pool(x_).reshape(B, C, -1).permute(0, 2, 1)
            #print('outer_tokens after pool',x_.shape)
            x_ = self.norm(self.linear(x_))
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if relative_pos is not None:
            attn += relative_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        print('x output',x.shape)
        return x


class Block(nn.Module):
    """ MiM-ISTD Block
    """
    def __init__(self, outer_dim, inner_dim, outer_head, inner_head, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0, sr_ratio=1):
        super().__init__()
        self.has_inner = inner_dim > 0
        if self.has_inner:
            # Inner
            self.inner_norm1 = norm_layer(num_words * inner_dim)

            self.inner_attn = SS2D(d_model=inner_dim, dropout=0, d_state=16)


            self.proj_norm1 = norm_layer(num_words * inner_dim)
            self.proj = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
            self.proj_norm2 = norm_layer(outer_dim)
        # Outer
        self.outer_norm1 = norm_layer(outer_dim)

        self.outer_attn = SS2D(d_model=outer_dim, dropout=0, d_state=16)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()



    def forward(self, x, outer_tokens, H_out, W_out, H_in, W_in, relative_pos):
        B, N, C = outer_tokens.size()
        #print('outer_tokens input',outer_tokens.shape)
        if self.has_inner:
            '''b,c,n=x.shape
            # 计算新的 H 和 W，即 sqrt(b/2)
            H = W = int(math.sqrt(b // 2))

            # 重塑张量 x 为 (2n, c, H, W)
            #print('x shape',x.shape)
            x_reshaped = x.reshape(2 * n, c, H, W)
            #print('x reshaped',x.shape)
            x_reshaped = self.inner_convs(x_reshaped)
            x = x + x_reshaped.reshape(b, c, n)'''

            x = x + self.drop_path(self.inner_attn(self.inner_norm1(x.reshape(B, N, -1)).reshape(B*N, H_in*W_in, -1), H_in, W_in)) # B*N, k*k, c
            outer_tokens = outer_tokens + self.proj_norm2(self.proj(self.proj_norm1(x.reshape(B, N, -1)))) # B, N, C
        outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens), H_out, W_out, relative_pos))
        return x, outer_tokens


class SentenceAggregation(nn.Module):
    """ Sentence Aggregation
    """
    def __init__(self, dim_in, dim_out, stride=2, act_layer=nn.GELU):
        super().__init__()
        self.stride = stride
        self.norm = nn.LayerNorm(dim_in)
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=2*stride-1, padding=stride-1, stride=stride),
        )
        
    def forward(self, x, H, W):#b,hw,c->b,hw/4,2c
        B, N, C = x.shape # B, N, C
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.conv(x)
        H, W = math.ceil(H / self.stride), math.ceil(W / self.stride)
        x = x.reshape(B, -1, H * W).transpose(1, 2)
        return x, H, W


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):#(b,h,w,c)->(b,h/2,w/2,2c)
        B, N, C = x.shape
        x=x.reshape(B,int(math.sqrt(N)),int(math.sqrt(N)),C)
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        b, h, w, c = x.shape
        x=x.reshape(b,h*w,c)

        return x,h,w


class PatchMerging2D_word(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim_in, dim_out, stride=2, act_layer=nn.GELU):
        super().__init__()
        self.stride = stride
        self.dim_out = dim_out
        self.norm = nn.LayerNorm(dim_in)
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=2*stride-1, padding=stride-1, stride=stride),
        )

    def forward(self, x, H_out, W_out, H_in, W_in):#(b,h,w,c)->(b,h/2,w/2,2c)
        B_N, M, C = x.shape # B*N, M, C
        x = self.norm(x)
        x = x.reshape(-1, H_out, W_out, H_in, W_in, C)
        # padding to fit (1333, 800) in detection.
        pad_input = (H_out % 2 == 1) or (W_out % 2 == 1)
        if pad_input:
            x = F.pad(x.permute(0, 3, 4, 5, 1, 2), (0, W_out % 2, 0, H_out % 2))
            x = x.permute(0, 4, 5, 1, 2, 3)

        H,W=x.shape[1],x.shape[2]      
        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        # patch merge
        x1 = x[:, 0::2, 0::2, :, :, :]  # B, H/2, W/2, H_in, W_in, C
        x2 = x[:, 1::2, 0::2, :, :, :]
        x3 = x[:, 0::2, 1::2, :, :, :]
        x4 = x[:, 1::2, 1::2, :, :, :]

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([torch.cat([x1, x2], 3), torch.cat([x3, x4], 3)], 4) # B, H/2, W/2, 2*H_in, 2*W_in, C
        x = x.reshape(-1, 2*H_in, 2*W_in, C).permute(0, 3, 1, 2) # B_N/4, C, 2*H_in, 2*W_in
        x = self.conv(x)  # B_N/4, C, H_in, W_in
        x = x.reshape(-1, self.dim_out, M).transpose(1, 2)

        return x
    
class WordAggregation(nn.Module):
    """ Word Aggregation
    """
    def __init__(self, dim_in, dim_out, stride=2, act_layer=nn.GELU):
        super().__init__()
        self.stride = stride
        self.dim_out = dim_out
        self.norm = nn.LayerNorm(dim_in)
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=2*stride-1, padding=stride-1, stride=stride),
        )

    def forward(self, x, H_out, W_out, H_in, W_in):
        B_N, M, C = x.shape # B*N, M, C
        x = self.norm(x)
        x = x.reshape(-1, H_out, W_out, H_in, W_in, C)
        
        # padding to fit (1333, 800) in detection.
        pad_input = (H_out % 2 == 1) or (W_out % 2 == 1)
        if pad_input:
            x = F.pad(x.permute(0, 3, 4, 5, 1, 2), (0, W_out % 2, 0, H_out % 2))
            x = x.permute(0, 4, 5, 1, 2, 3)            
        # patch merge
        x1 = x[:, 0::2, 0::2, :, :, :]  # B, H/2, W/2, H_in, W_in, C
        x2 = x[:, 1::2, 0::2, :, :, :]
        x3 = x[:, 0::2, 1::2, :, :, :]
        x4 = x[:, 1::2, 1::2, :, :, :]
        x = torch.cat([torch.cat([x1, x2], 3), torch.cat([x3, x4], 3)], 4) # B, H/2, W/2, 2*H_in, 2*W_in, C
        x = x.reshape(-1, 2*H_in, 2*W_in, C).permute(0, 3, 1, 2) # B_N/4, C, 2*H_in, 2*W_in
        x = self.conv(x)  # B_N/4, C, H_in, W_in
        x = x.reshape(-1, self.dim_out, M).transpose(1, 2)
        return x


class Stem(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_chans=3, outer_dim=768, inner_dim=24):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.inner_dim = inner_dim
        self.num_patches = img_size[0] // 8 * img_size[1] // 8
        self.num_words = 4#16    #outer h/4,w/4
        
        self.common_conv = nn.Sequential(
            nn.Conv2d(in_chans, inner_dim*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim*2),
            nn.ReLU(inplace=True),
        )
        self.inner_convs = nn.Sequential(
            nn.Conv2d(inner_dim*2, inner_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(inplace=False),
        )
        self.outer_convs = nn.Sequential(
            nn.Conv2d(inner_dim*2, inner_dim*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim*4, inner_dim*8, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim*8, outer_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(outer_dim),
            nn.ReLU(inplace=False),
        )
        '''self.outer_convs = nn.Sequential(
            nn.Conv2d(inner_dim*2, inner_dim*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim*4, outer_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(outer_dim),
            nn.ReLU(inplace=False),
        )#outer h/4,w/4'''
        
        self.unfold = nn.Unfold(kernel_size=4, padding=0, stride=4)#每个视觉词对应于原图像中的 4x4 像素区域
        #self.unfold = nn.Unfold(kernel_size=2, padding=0, stride=2)#每个视觉词对应于原图像中的 2x2 像素区域

    def forward(self, x):
        B, C, H, W = x.shape
        H_out, W_out = H // 8, W // 8
        H_in, W_in = 4, 4#这里 H_in 和 W_in 被设置为 4，所以每个视觉句子由 4x4 个视觉词组成。
        #H_out, W_out = H // 4, W // 4#outer h/4,w/4
        #H_in, W_in = 2, 2#outer h/4,w/4
        x = self.common_conv(x)
        # inner_tokens
        inner_tokens = self.inner_convs(x) # B, C, H, W
        inner_tokens = self.unfold(inner_tokens).transpose(1, 2) # B, N, Ck2
        inner_tokens = inner_tokens.reshape(B * H_out * W_out, self.inner_dim, H_in*W_in).transpose(1, 2) # B*N, C, 4*4
        # outer_tokens
        outer_tokens = self.outer_convs(x) # B, C, H_out, W_out
        outer_tokens = outer_tokens.permute(0, 2, 3, 1).reshape(B, H_out * W_out, -1)
        return inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in)


class Stage(nn.Module):
    """ PyramidTNT stage
    """
    def __init__(self, num_blocks, outer_dim, inner_dim, outer_head, inner_head, num_patches, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0, sr_ratio=1):
        super().__init__()
        blocks = []
        drop_path = drop_path if isinstance(drop_path, list) else [drop_path] * num_blocks
        
        for j in range(num_blocks):
            if j == 0:
                _inner_dim = inner_dim
            elif j == 1 and num_blocks > 6:
                _inner_dim = inner_dim
            else:
                _inner_dim = -1
            blocks.append(Block(
                outer_dim, _inner_dim, outer_head=outer_head, inner_head=inner_head,
                num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                attn_drop=attn_drop, drop_path=drop_path[j], act_layer=act_layer, norm_layer=norm_layer,
                se=se, sr_ratio=sr_ratio))

        self.blocks = nn.ModuleList(blocks)
        self.relative_pos = nn.Parameter(torch.randn(
                        1, outer_head, num_patches, num_patches // sr_ratio // sr_ratio))

    def forward(self, inner_tokens, outer_tokens, H_out, W_out, H_in, W_in):
        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens, H_out, W_out, H_in, W_in, self.relative_pos)
        return inner_tokens, outer_tokens


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        # 步长为2的2x2转置卷积
        self.transposed_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        )
        # 批量归一化
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        # GeLU 激活函数
        self.gelu1 = nn.GELU()
        # 步长为1的3x3卷积
        self.conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        # 另一个批量归一化
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        # 另一个 GeLU 激活函数
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = self.transposed_conv(x)
        x = self.batch_norm1(x)
        x = self.gelu1(x)
        x = self.conv(x)
        x = self.batch_norm2(x)
        x = self.gelu2(x)
        return x





class PyramidMiM(nn.Module):
    """ Pyramid MiM-ISTD encoder including conv stem for computer vision
    """
    def __init__(self, configs=None, img_size=512, in_chans=3, num_classes=1, mlp_ratio=4., qkv_bias=False,
                qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, se=0):
        super().__init__()
        self.num_classes = num_classes
        #depths = [2, 8, 4, 2]#configs['depths']
        depths = [2, 4, 9, 2]
        #depths = [2, 10, 6, 2]
        #outer_dims = [128, 128*2, 128*4, 128*8]#configs['outer_dims']
        #inner_dims = [8, 8*2, 8*4, 8*8]#configs['inner_dims']
        #outer_dims = [32, 32*2, 32*4, 32*8]#configs['outer_dims']
        outer_dims = [16, 16*2, 16*4, 16*8]#configs['outer_dims']
        inner_dims = [4, 4*2, 4*4, 4*8]#configs['inner_dims']   original mim-istd
        #inner_dims = [16, 16*2, 16*4, 16*8]#configs['inner_dims']
        outer_heads = [2, 2*2, 2*4, 2*8]#configs['outer_heads']
        inner_heads = [1, 1*2, 1*4, 1*8]#configs['inner_heads']
        sr_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule 
        self.num_features = outer_dims[-1]  # num_features for consistency with other models       

        self.patch_embed = Stem(
            img_size=img_size, in_chans=in_chans, outer_dim=outer_dims[0], inner_dim=inner_dims[0])
        num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words
        
        self.outer_pos = nn.Parameter(torch.zeros(1, num_patches, outer_dims[0]))
        self.inner_pos = nn.Parameter(torch.zeros(1, num_words, inner_dims[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        depth = 0
        self.word_merges = nn.ModuleList([])
        self.sentence_merges = nn.ModuleList([])
        self.stages = nn.ModuleList([])
        for i in range(4):
            if i > 0:
                #self.word_merges.append(WordAggregation(inner_dims[i-1], inner_dims[i], stride=2))
                self.word_merges.append(PatchMerging2D_word(inner_dims[i-1], inner_dims[i], stride=2))
                #self.sentence_merges.append(SentenceAggregation(outer_dims[i-1], outer_dims[i], stride=2))
                self.sentence_merges.append(PatchMerging2D(outer_dims[i-1]))
            self.stages.append(Stage(depths[i], outer_dim=outer_dims[i], inner_dim=inner_dims[i],
                        outer_head=outer_heads[i], inner_head=inner_heads[i],
                        num_patches=num_patches // (2 ** i) // (2 ** i), num_words=num_words, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[depth:depth+depths[i]], norm_layer=norm_layer, se=se, sr_ratio=sr_ratios[i])
            )
            depth += depths[i]
        
        self.norm = norm_layer(outer_dims[-1])

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(outer_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        #self.head = nn.Linear(outer_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        '''self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )'''
        self.up_blocks = nn.ModuleList([])
        for i in range(4):
            self.up_blocks.append(UpsampleBlock(outer_dims[i],outer_dims[i]))
           

        trunc_normal_(self.outer_pos, std=.02)
        trunc_normal_(self.inner_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'outer_pos', 'inner_pos'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        size = x.size()[2:]
        inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in) = self.patch_embed(x)
        inner_tokens = inner_tokens# + self.inner_pos # B*N, 8*8, C
        #print('elf.outer_pos',self.outer_pos.shape)
        #print('outer_tokens',outer_tokens.shape)
        outer_tokens = outer_tokens# + self.pos_drop(self.outer_pos)  # B, N, D
        outputs=[]
        
        for i in range(4):
            if i > 0:
                #print('before word_merges inner_tokens',inner_tokens.shape)
                inner_tokens = self.word_merges[i-1](inner_tokens, H_out, W_out, H_in, W_in)
                outer_tokens, H_out, W_out = self.sentence_merges[i-1](outer_tokens)#self.sentence_merges[i-1](outer_tokens, H_out, W_out)
                #print('after word_merges inner_tokens',inner_tokens.shape)
            inner_tokens, outer_tokens = self.stages[i](inner_tokens, outer_tokens, H_out, W_out, H_in, W_in)
            b,l,m=outer_tokens.shape
            #print('inner_tokens',inner_tokens.shape)
            #print('outer_tokens',outer_tokens.shape)
            mid_out=outer_tokens.reshape(b,int(math.sqrt(l)),int(math.sqrt(l)),m).permute(0,3,1,2)
            mid_out=self.up_blocks[i](mid_out)

            #print('outputs',mid_out.shape)#[32,32,64,64],[32,64,32,32],[32,128,16,16],[32,256,8,8]
            outputs.append(mid_out)
        #out = self.decode_head(outputs)
        #out = F.interpolate(out, size, **up_kwargs)
        #print('out',out.shape)


        #outer_tokens = self.norm(outer_tokens)
        #print('forward outer_tokens',outer_tokens.shape)
        #return outer_tokens.mean(dim=1)
        return outputs

    def forward(self, x):
        #print('input',x.shape)
        x = self.forward_features(x)
        #print('forward',x.shape)
        #x = self.head(x)
        #print('output',x.shape)
        return x








