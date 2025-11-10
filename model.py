import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]


class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class MST(nn.Module):
    def __init__(self, in_dim=121, out_dim=121, dim=121, stage=2, num_blocks=[2,4,4]):
        super(MST, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = MSAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for (MSAB, FeaDownSample) in self.encoder_layers:
            fea = MSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = LeWinBlcok(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out


class MST_Plus_Plus(nn.Module):
    def __init__(self, in_channels=8, out_channels=121, n_feat=121, stage=3):
        super(MST_Plus_Plus, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        modules_body = [MST(dim=121, stage=2, num_blocks=[1,1,1]) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x = self.conv_in(x)
        h = self.body(x)
        h = self.conv_out(h)
        h += x
        return h[:, :, :h_inp, :w_inp]


def _softplus_inv(y, beta=1.0):
    return torch.log(torch.expm1(beta * y)) / beta


class Filter(nn.Module):
    """
    Input:
      X : [B, C, H, W]
    Output:
      Y : [B, 4, H, W]
    """
    def __init__(self, spectral_sens_csv: str,
                 sum_to_one: bool=False,
                 eps: float=1e-6,
                 dtype=torch.float32,
                 device="cpu"):
        super().__init__()
        self.csv_path = spectral_sens_csv
        self.sum_to_one = sum_to_one
        self.eps = eps
        self._dtype = dtype
        self._device = torch.device(device)

        self._wl_sens_np, self._S_all_np = self._load_sens_csv(self.csv_path)   # (Ls,), (Ls,4)

        # cache dipendente da C
        self.register_buffer("S", None, persistent=False)  # <-- FIX: buffer che segue .to(...)
        self._S_C = None
        self.weight_param = None  # nn.Parameter [C]

    # ---------- forward ----------
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        assert X.dim() == 4, "X deve essere [B, C, H, W]"
        B, C, H, W = X.shape
        device, dtype = X.device, X.dtype                 # <-- FIX: prendi device/dtype da X

        self._ensure_S(C, device, dtype)                 # <-- FIX: pass device/dtype
        self._ensure_params(C, device, dtype)            # <-- FIX: pass device/dtype

        f = self._current_f().to(device=device, dtype=dtype)     # [C]  <-- FIX
        S = self.S.to(device=device, dtype=dtype)                 # [4,C] <-- FIX

        Xf = X * f.view(1, C, 1, 1)                      # [B,C,H,W]
        Y = torch.einsum("bchw,oc->bohw", Xf, S)         # [B,4,H,W]
        return Y

    # ---------- utilities ----------
    def smoothness_penalty(self) -> torch.Tensor:
        assert self.weight_param is not None, "Chiama forward prima."
        f = F.softplus(self.weight_param) + self.eps
        # seconda differenza (Laplaciano 1D)
        if f.numel() >= 3:
            d2 = f[2:] - 2 * f[1:-1] + f[:-2]
            return (d2 ** 2).mean()
        else:
            d1 = f[1:] - f[:-1]
            return (d1 ** 2).mean()

    @torch.no_grad()
    def current_filter(self) -> torch.Tensor:
        return self._current_f().detach().cpu()

    @torch.no_grad()
    def effective_sensor(self) -> torch.Tensor:
        assert self.S is not None and self.weight_param is not None
        f = self._current_f().view(1, -1)
        return (self.S * f).detach().cpu()

    # ---------- internals ----------
    def _ensure_params(self, C: int, device: torch.device, dtype: torch.dtype):
        if self.weight_param is None:
            p = torch.zeros(C, device=device, dtype=dtype)
            with torch.no_grad():
                p.copy_(_softplus_inv(torch.ones_like(p)))  # init f≈1
            self.weight_param = nn.Parameter(p)
        else:
            if self.weight_param.numel() != C:
                raise ValueError(f"C è cambiato ({self.weight_param.numel()} -> {C}). "
                                 "Mantieni C fisso o gestisci la ricreazione del parametro.")
            # assicurati che segua il device/dtype correnti
            if self.weight_param.device != device or self.weight_param.dtype != dtype:
                self.weight_param.data = self.weight_param.data.to(device=device, dtype=dtype)

    def _current_f(self) -> torch.Tensor:
        f = F.softplus(self.weight_param) + self.eps
        if self.sum_to_one:
            f = f / f.sum().clamp_min(self.eps)
        return f

    def _ensure_S(self, C: int, device: torch.device, dtype: torch.dtype):
        """Costruisce/aggiorna S ∈ [4,C] interpolando il CSV sulla griglia 400..1000 con C punti."""
        if (self.S is not None) and (self._S_C == C):
            # già costruita per questo C: porta comunque a device/dtype correnti
            self.S = self.S.to(device=device, dtype=dtype)
            return

        wl_target = np.linspace(400.0, 1000.0, C, dtype=np.float64)
        S_interp = np.zeros((4, C), dtype=np.float64)
        for i in range(4):
            S_interp[i] = np.interp(wl_target, self._wl_sens_np, self._S_all_np[:, i],
                                    left=0.0, right=0.0)
        S_t = torch.from_numpy(S_interp).to(device=device, dtype=dtype)   # <-- FIX
        if self.sum_to_one:
            S_t = S_t / (S_t.sum(dim=1, keepdim=True).clamp_min(self.eps))
        # registra come buffer (sovrascrive quello precedente)
        self.register_buffer("S", S_t, persistent=False)
        self._S_C = C

    @staticmethod
    def _load_sens_csv(csv_path: str):
        """Ritorna (wl_sens ∈ [Ls], S_all ∈ [Ls,4]) in ordine R,G,B,IR."""
        with open(csv_path, "r") as f:
            header = f.readline().strip().split(",")
        hmap = {name.strip().lower(): i for i, name in enumerate(header)}
        if "wavelength" not in hmap:
            raise ValueError("Nel CSV manca 'wavelength'.")

        data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        wl_sens = data[:, hmap["wavelength"]].astype(np.float64)

        def col(name):
            idx = hmap.get(name, None)
            if idx is None:
                raise ValueError(f"Manca la colonna '{name}' nel CSV.")
            return idx

        cols = [col("red"), col("green"), col("blue")]
        ir_idx = hmap.get("ir850", hmap.get("ir", None))
        if ir_idx is None:
            raise ValueError("Manca la colonna 'ir850'/'ir' nel CSV.")
        cols.append(ir_idx)

        S_all = data[:, cols].astype(np.float64)  # (Ls, 4) R,G,B,IR
        return wl_sens, S_all


class JointDualFilterMST(nn.Module):
    """
    Ramo A: filtro unico f_A(C) su radianza [B,C,H,W] + proiezione S[4,C]
    Ramo B: filtro unico f_B(C) su radianza [B,C,H,W] + proiezione S[4,C]
    Fusione: concat 8 canali → MST++
    """
    def __init__(self, spectral_sens_csv: str, device="cpu", dtype=torch.float32):
        super().__init__()
        self.filterA = Filter(spectral_sens_csv, device=device, dtype=dtype)
        self.filterB = Filter(spectral_sens_csv, device=device, dtype=dtype)
        self.mst = MST_Plus_Plus()  # la tua rete già definita

    def smoothness_penalty(self):
        return self.filterA.smoothness_penalty() + self.filterB.smoothness_penalty()

    @torch.no_grad()
    def current_filters(self):
        """Restituisce f_A(λ) e f_B(λ) ∈ [C]."""
        return self.filterA.current_filter(), self.filterB.current_filter()

    @torch.no_grad()
    def effective_sensors(self):
        """Restituisce S_eff_A e S_eff_B ∈ [4,C]."""
        return self.filterA.effective_sensor(), self.filterB.effective_sensor()

    def forward(self, X):
        """
        X: radianza [B, C, H, W]  (C=121 se 400..1000 ogni 5nm, ma può essere qualsiasi)
        """
        xa = self.filterA(X)                 # [B,4,H,W]
        xb = self.filterB(X)                 # [B,4,H,W]
        x8 = torch.cat([xa, xb], dim=1)     # [B,8,H,W]
        return self.mst(x8)                 # [B,121,H,W] (o il tuo output)


class DualFilterVector(nn.Module):
    """
    Usa due Filter esistenti ma con input vettoriali (B, C) invece di (B, C, H, W).
    Output: y ∈ (B, 8) con ordine interleaved: [y1R,y2R,y1G,y2G,y1B,y2B,y1IR,y2IR].
    """
    def __init__(self, spectral_sens_csv: str, device="cpu", dtype=torch.float32):
        super().__init__()
        self.filterA = Filter(spectral_sens_csv, device=device, dtype=dtype)
        self.filterB = Filter(spectral_sens_csv, device=device, dtype=dtype)

    def forward(self, Xvec):  # Xvec: (B, C)
        B, C = Xvec.shape
        X4d = Xvec.view(B, C, 1, 1)     # (B,C,H=1,W=1)

        xa = self.filterA(X4d).view(B, 4)   # (B,4) = [R,G,B,IR] per filtro 1
        xb = self.filterB(X4d).view(B, 4)   # (B,4) = [R,G,B,IR] per filtro 2

        # Interleaving per canale: [y1R,y2R, y1G,y2G, y1B,y2B, y1IR,y2IR]
        y = torch.stack([xa[:,0], xb[:,0], xa[:,1], xb[:,1],
                         xa[:,2], xb[:,2], xa[:,3], xb[:,3]], dim=1)  # (B,8)
        return y

    def filters_smoothness(self) -> torch.Tensor:
        # versione che contribuisce al gradiente
        return self.filterA.smoothness_penalty() + self.filterB.smoothness_penalty()


class ReconMLP(nn.Module):
    def __init__(self, in_dim=8, out_len=121, hidden=(256, 384), nonneg=True, norm_in=True):
        super().__init__()
        self.norm_in = norm_in
        if norm_in:
            self.mu = nn.Parameter(torch.zeros(in_dim))   # learnable
            self.sig = nn.Parameter(torch.ones(in_dim))    # learnable

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]), nn.ReLU(inplace=True),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(inplace=True),
            nn.Linear(hidden[1], out_len)
        )
        self.nonneg = nonneg
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

    def forward(self, y):
        if self.norm_in:
            y = (y - self.mu) / (self.sig.abs() + 1e-6)
        s = self.net(y)
        if self.nonneg:
            s = self.softplus(s)
        return s.clamp(0.0, 1.0)



class PixelReconModel(nn.Module):
    def __init__(self, spectral_sens_csv: str, out_len=121, device="cuda", dtype=torch.float32):
        super().__init__()
        self.meas = DualFilterVector(spectral_sens_csv, device=device, dtype=dtype)
        self.dec = ReconMLP(in_dim=8, out_len=out_len)

    def forward(self, s_true):           # s_true: (B, L=121)  (radiance o riflettanza)
        y = self.meas(s_true)            # (B,8)
        s_pred = self.dec(y)             # (B,121)
        return s_pred, y


class SpectralMLP(nn.Module):
    """
    Input:  (B, 8, 16, 16)
    Output: (B, 121, 16, 16)
    MLP per-pixel: R^8 -> R^121 applicato su ogni (h,w).
    """
    def __init__(self, hidden_dim=256, num_layers=3, activation="gelu", out_activation=None):
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True)

        layers = []
        in_dim = 8
        for _ in range(max(0, num_layers - 1)):
            layers += [nn.Linear(in_dim, hidden_dim), act]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 121))
        if out_activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif out_activation == "softplus":
            layers.append(nn.Softplus(beta=1.0, threshold=20.0))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 8, 16, 16)
        B, C, H, W = x.shape
        assert (C, H, W) == (8, 16, 16), f"atteso (8,16,16), ricevuto {(C,H,W)}"
        x = x.permute(0, 2, 3, 1).contiguous()   # (B,16,16,8)
        x = x.view(B * H * W, 8)                 # (BHW, 8)
        y = self.mlp(x)                          # (BHW, 121)
        y = y.view(B, H, W, 121).permute(0, 3, 1, 2).contiguous()  # (B,121,16,16)
        return y


class ResMLP8to121(nn.Module):
    def __init__(self, width=256, depth=4, nonneg=True, norm_in=True):
        super().__init__()
        self.nonneg = nonneg
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

        self.norm_in = norm_in
        if norm_in:
            self.mu = nn.Parameter(torch.zeros(8))
            self.sig = nn.Parameter(torch.ones(8))

        self.inp = nn.Linear(8, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width, width),
                nn.ReLU(inplace=True),
                nn.Linear(width, width),
            ) for _ in range(depth)
        ])
        self.out = nn.Linear(width, 121)

        # init xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):          # x: (B,8)
        if self.norm_in:
            x = (x - self.mu) / (self.sig.abs() + 1e-6)
        h = F.relu(self.inp(x), inplace=True)
        for blk in self.blocks:
            h = h + blk(h)         # residual
            h = F.relu(h, inplace=True)
        y = self.out(h)            # (B,121)
        return self.softplus(y) if self.nonneg else y


class TinySpecFormer(nn.Module):
    def __init__(self, d_model=192, nhead=6, num_layers=2, nonneg=True):
        super().__init__()
        self.nonneg = nonneg
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

        self.obs_proj = nn.Linear(8, d_model)                   # 8 -> d
        self.q_emb = nn.Parameter(torch.randn(121, d_model))    # 121 query (una per λ)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            batch_first=True, activation="relu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=1)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            batch_first=True, activation="relu"
        )
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.head = nn.Linear(d_model, 1)

    def forward(self, x):                  # x: (B,8)
        mem = self.obs_proj(x).unsqueeze(1)   # (B,1,d)
        mem = self.enc(mem)                   # (B,1,d)
        Q = self.q_emb.unsqueeze(0).expand(x.size(0), -1, -1)  # (B,121,d)
        H = self.dec(Q, mem)                  # (B,121,d)
        y = self.head(H).squeeze(-1)          # (B,121)
        return self.softplus(y) if self.nonneg else y

