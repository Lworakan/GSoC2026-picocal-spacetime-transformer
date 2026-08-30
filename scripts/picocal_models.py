import numpy as np
import torch
import torch.nn as nn

CFG = dict(d=128, nhead=4, layers=3, dropout=0.1, lr=3e-4, wd=1e-4, batch=96, huber_delta=0.1)
NG = 6
NC = 9
QUANTILES = [0.25, 0.5, 0.75]


NPAIR = 5


def pair_feats(x):
    di, dj, tf, tb = x[:, :, 3], x[:, :, 4], x[:, :, 7], x[:, :, 8]
    ddi = di[:, None, :] - di[:, :, None]
    ddj = dj[:, None, :] - dj[:, :, None]
    return torch.stack([ddi, ddj, torch.sqrt(ddi ** 2 + ddj ** 2 + 1e-6),
                        tf[:, None, :] - tf[:, :, None],
                        tb[:, None, :] - tb[:, :, None]], -1)


class GeoEncoderLayer(nn.Module):
    def __init__(self, d, nhead, ff, dropout):
        super().__init__()
        self.nh = nhead
        self.dh = d // nhead
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.bias = nn.Sequential(nn.Linear(NPAIR, 16), nn.ReLU(), nn.Linear(16, nhead))
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, pair, m):
        B, L, d = h.shape
        q, k, v = self.qkv(h).chunk(3, -1)
        q, k, v = (t.view(B, L, self.nh, self.dh).transpose(1, 2) for t in (q, k, v))
        a = q @ k.transpose(-2, -1) / self.dh ** 0.5 + self.bias(pair).permute(0, 3, 1, 2)
        a = a.masked_fill(~m[:, None, None, :], float('-inf'))
        h = self.n1(h + self.drop(self.proj((torch.softmax(a, -1) @ v)
                                            .transpose(1, 2).reshape(B, L, d))))
        return self.n2(h + self.drop(self.ff(h)))


class SpacetimeLayer(nn.Module):
    """Divided space-time attention over calorimeter cells, in the sense of TimeSformer
    (arXiv:2102.05095): attend spatially WITHIN a time slice, then across slices.

    TimeSformer cannot be transplanted directly -- our second axis is two longitudinal samples, not
    T frames, so factorising it saves nothing. What makes the mechanism apply here is turning the
    per-cell TIME into the axis: cells are bucketed by their time pull against the seed-estimated
    reference, and attention is restricted to run inside a bucket before the buckets talk to each
    other through pooled summaries.

    The reason is measured, not aesthetic. Pileup energy has a time rms of 13.2 ns at 15mm against
    1.6 ns for the photon, and a pull cut keeps 94.5% of photon while removing 59% of pileup, so a
    time bucket physically separates the two sources. Four independent experiments (H7, --gatesup,
    --prior-feat, --prior-teach) showed a single scalar gate cannot do that separation; this asks the
    architecture to do it structurally instead. It is also cheaper: sum of within-bucket attention is
    far below one full n^2, which matters now that the window has grown to 17x17.
    """

    def __init__(self, d, nhead, ff, dropout, nslice):
        super().__init__()
        self.spatial = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=ff, dropout=dropout,
                                                  batch_first=True)
        self.cross = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.nslice = nslice

    def forward(self, h, m, sl):
        B, N, D = h.shape
        # spatial pass: a cell may only attend to cells in its own time slice
        same = (sl.unsqueeze(2) == sl.unsqueeze(1))
        keep = same & m.unsqueeze(1) & m.unsqueeze(2)
        # a fully masked row would produce NaN, so let isolated cells see themselves
        eye = torch.eye(N, dtype=torch.bool, device=h.device).unsqueeze(0)
        bias = torch.zeros(B, N, N, device=h.device).masked_fill(~(keep | eye), float('-inf'))
        bias = bias.unsqueeze(1).expand(B, self.spatial.self_attn.num_heads, N, N)
        h = self.spatial(h, src_mask=bias.reshape(-1, N, N))
        # cross-slice pass: each slice is pooled to one token, then every cell reads all slices
        pooled = []
        for k in range(self.nslice):
            w = ((sl == k) & m).float().unsqueeze(-1)
            pooled.append((h * w).sum(1) / w.sum(1).clamp(min=1e-6))
        p = torch.stack(pooled, 1)
        a, _ = self.cross(h, p, p, need_weights=False)
        return self.norm(h + a) * m.unsqueeze(-1)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block (2201.03545): depthwise 7x7, LayerNorm, inverted bottleneck, residual.

    Chosen over plain 3x3 stacks because the window is now wide -- W10 is a 21x21 grid -- and a
    single depthwise 7x7 covers the radial scale that matters at a fraction of the parameters an
    equivalent dense stack would need. The point of using convolution at all is that this detector
    window IS a regular grid whose empty cells are genuinely below the 2.49 MeV threshold rather
    than missing, so nothing is being approximated; and it is O(n) where attention is O(n^2), which
    is what makes the wide window deployable.
    """

    def __init__(self, c, mult=4):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c)
        self.norm = nn.GroupNorm(1, c)
        self.pw1 = nn.Conv2d(c, mult * c, 1)
        self.pw2 = nn.Conv2d(mult * c, c, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))


class CNNSub(nn.Module):
    def __init__(self, in_dim, la0, lb0, ng=NG, cfg=CFG, side=9, modern=False):
        super().__init__()
        self.side = side
        self.gate_mode = 'learned'
        d = cfg['d']
        if modern:
            self.conv = nn.Sequential(nn.Conv2d(in_dim + 1, d, 1), nn.GroupNorm(1, d),
                                      *[ConvNeXtBlock(d) for _ in range(cfg['layers'])])
        else:
            ch = [in_dim + 1, 32, 64, d]
            blocks = []
            for a, b in zip(ch[:-1], ch[1:]):
                blocks += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU()]
            self.conv = nn.Sequential(*blocks)
        self.fconv = nn.Conv2d(d, 1, 1)
        self.head = nn.Sequential(nn.Linear(d + ng, d), nn.ReLU(),
                                  nn.Dropout(cfg['dropout']), nn.Linear(d, 3))
        self.la = nn.Parameter(torch.tensor(float(la0)))
        self.lb = nn.Parameter(torch.tensor(float(lb0)))

    def grid(self, v, pos, L):
        B, _, C = v.shape
        out = v.new_zeros(B, L + 1, C)
        out.scatter_(1, pos.unsqueeze(-1).expand(-1, -1, C), v)
        return out[:, :L]

    def forward(self, x, m, g, ecell, pos=None, sl=None):
        L = self.side ** 2
        mf = m.unsqueeze(-1).float()
        feats = torch.cat([x * mf, mf], -1)
        img = self.grid(feats, pos, L).transpose(1, 2).reshape(-1, feats.shape[-1],
                                                               self.side, self.side)
        eg = self.grid((ecell * m.float()).unsqueeze(-1), pos, L).squeeze(-1)
        h = self.conv(img)
        w = torch.sigmoid(self.fconv(h)).flatten(1)
        base = self.la * torch.log1p((w * eg).sum(1, keepdim=True)) + self.lb
        p = h.mean((2, 3))
        return base + self.head(torch.cat([p, g], 1))


def masked_knn(coord, m, k):
    """Indices of the k nearest neighbours of each token, with padded tokens pushed to infinity
    so they are never selected and never select."""
    d2 = ((coord.unsqueeze(2) - coord.unsqueeze(1)) ** 2).sum(-1)
    bad = ~m
    d2 = d2.masked_fill(bad.unsqueeze(1), float('inf'))
    d2 = d2 + torch.eye(d2.size(1), device=d2.device).unsqueeze(0) * 1e6
    k = min(k, d2.size(1) - 1)
    return d2.topk(k, dim=-1, largest=False).indices


class EdgeConv(nn.Module):
    """ParticleNet's building block (1902.08570): h_i <- max_j MLP([h_i, h_j - h_i]) over the k
    nearest neighbours, with the graph rebuilt in the current feature space each block."""

    def __init__(self, din, dout, k=16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(nn.Linear(2 * din, dout), nn.BatchNorm1d(dout), nn.ReLU(),
                                 nn.Linear(dout, dout), nn.BatchNorm1d(dout), nn.ReLU())
        self.short = nn.Linear(din, dout) if din != dout else nn.Identity()

    def forward(self, h, m, coord):
        B, N, C = h.shape
        idx = masked_knn(coord, m, self.k)
        kk = idx.size(-1)
        nb = torch.gather(h.unsqueeze(1).expand(B, N, N, C), 2,
                          idx.unsqueeze(-1).expand(B, N, kk, C))
        cen = h.unsqueeze(2).expand(B, N, kk, C)
        e = torch.cat([cen, nb - cen], -1).reshape(-1, 2 * C)
        e = self.mlp(e).reshape(B, N, kk, -1)
        out = e.max(dim=2).values
        return (out + self.short(h)) * m.unsqueeze(-1)


class PNetEncoder(nn.Module):
    """ParticleNet-style encoder, used as a like-for-like baseline: same inputs, same physics
    readout, same loss and same splits as the transformer, so the only thing that differs is the
    encoder. Comparing against numbers published on other datasets would prove nothing."""

    def __init__(self, d, k=16, blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([EdgeConv(d, d, k) for _ in range(blocks)])
        self.dyn = True

    def forward(self, h, m, x):
        coord = x[:, :, 3:5]                      # start from the real cell geometry
        for b in self.blocks:
            h = b(h, m, coord)
            if self.dyn:
                coord = h                          # dynamic graph, as in DGCNN/ParticleNet
        return h


class GravNetEncoder(nn.Module):
    """GravNet (1902.07987): project to a low-dimensional learned space, gather k neighbours there
    with a distance-weighted kernel, and concatenate the aggregate back."""

    def __init__(self, d, k=16, ds=4):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(d, ds)
        self.tr = nn.Linear(d, d)
        self.out = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, h, m, x):
        s = self.proj(h)
        idx = masked_knn(s, m, self.k)
        B, N, C = h.shape
        kk = idx.size(-1)
        f = self.tr(h)
        nb = torch.gather(f.unsqueeze(1).expand(B, N, N, C), 2,
                          idx.unsqueeze(-1).expand(B, N, kk, C))
        sn = torch.gather(s.unsqueeze(1).expand(B, N, N, s.size(-1)), 2,
                          idx.unsqueeze(-1).expand(B, N, kk, s.size(-1)))
        w = torch.exp(-10.0 * ((sn - s.unsqueeze(2)) ** 2).sum(-1)).unsqueeze(-1)
        agg = torch.cat([(nb * w).max(dim=2).values, (nb * w).mean(dim=2)], -1)
        return self.out(torch.cat([h, agg], -1)) * m.unsqueeze(-1)


class MixerBlock(nn.Module):
    def __init__(self, d, lmax, drop):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.tok = nn.Sequential(nn.Linear(lmax, 2 * lmax), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(2 * lmax, lmax))
        self.n2 = nn.LayerNorm(d)
        self.ch = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                nn.Dropout(drop), nn.Linear(4 * d, d))

    def forward(self, h, m):
        w = m.unsqueeze(-1).float()
        t = (self.n1(h) * w).transpose(1, 2)
        h = h + self.tok(t).transpose(1, 2) * w
        return h + self.ch(self.n2(h)) * w


class SubNetFQ(nn.Module):
    def __init__(self, in_dim, la0, lb0, ng=NG, cfg=CFG, gate='learned', arch='std', qpool=False,
                 gx=False, aux=False, film=False, nfour=0, tfour=0, slots=0, cellreg=False, widthfb=False):
        super().__init__()
        self.gate_mode = gate
        self.signed = gate == 'signed'
        self.gx = gx
        self.aux = aux
        self.film = film
        self.nfour = nfour
        self.tfour = tfour
        self.arch = arch
        self.qpool = qpool
        self.k_slots = slots
        d = cfg['d']
        # Fourier features on the two in-cell offset channels. Our own measurement says the
        # sub-cell impact position is the hidden variable that fixed estimators cannot use
        # (the seed's energy share swings 40-90% with it), and a linear embedding of a raw
        # coordinate represents high-frequency dependence on it poorly.
        extra_in = (4 * nfour if nfour else 0) + (4 * tfour if tfour else 0)
        self.embed = nn.Linear(in_dim * (2 if gx else 1) + extra_in, d)
        if nfour:
            self.register_buffer('freqs', 2.0 ** torch.arange(nfour).float() * torch.pi)
        if tfour:
            self.register_buffer('tfreqs', 2.0 ** torch.arange(tfour).float() * torch.pi)
        if arch == 'pnet':
            self.pnet = PNetEncoder(d, k=cfg.get('knn') or 16, blocks=cfg['layers'])
        elif arch == 'gravnet':
            self.grav = nn.ModuleList([GravNetEncoder(d, k=cfg.get('knn') or 16) for _ in range(cfg['layers'])])
        elif arch == 'spacetime':
            self.nslice = 6
            self.st = nn.ModuleList([SpacetimeLayer(d, cfg['nhead'], 4 * d, cfg['dropout'],
                                                    self.nslice) for _ in range(cfg['layers'])])
            self.sl_emb = nn.Embedding(self.nslice, d)
        elif arch == 'geo':
            self.geo = nn.ModuleList([GeoEncoderLayer(d, cfg['nhead'], 4 * d, cfg['dropout'])
                                      for _ in range(cfg['layers'])])
        elif arch == 'mixer':
            self.lmax = 289
            self.mix = nn.ModuleList([MixerBlock(d, self.lmax, cfg['dropout'])
                                      for _ in range(cfg['layers'])])
        elif film:
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d, cfg['nhead'], dim_feedforward=4 * d,
                                           dropout=cfg['dropout'], batch_first=True,
                                           norm_first=cfg.get('preln', False))
                for _ in range(cfg['layers'])])
        else:
            layer = nn.TransformerEncoderLayer(d, cfg['nhead'], dim_feedforward=4 * d,
                                               dropout=cfg['dropout'], batch_first=True,
                                               norm_first=cfg.get('preln', False))
            self.enc = nn.TransformerEncoder(layer, cfg['layers'], enable_nested_tensor=False)
        if film:
            # The measured physics context (pileup density, window/cluster ratio, occupancy,
            # region) modulates EVERY block, not only the readout: scale and shift per layer.
            self.mod = nn.ModuleList([nn.Linear(ng, 2 * d) for _ in range(cfg['layers'])])
            for lin in self.mod:
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        if qpool:
            self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.pool_attn = nn.MultiheadAttention(d, cfg['nhead'], dropout=cfg['dropout'],
                                                   batch_first=True)
        if slots:
            # The event is physically a SUM of showers (the photon plus pileup vertices), but no
            # encoder in the ledger carries that superposition as structure: every readout pools
            # all cells into one vector. K slot queries cross-attend to the cells, each cell's
            # energy is then split by a softmax RESPONSIBILITY over the slots, and only slot 0
            # feeds the energy sum and the head. Unlike --gatesup/--prior-teach, nothing forces
            # slot 0 to be the photon: the energy loss alone decides what the slots specialise to,
            # which is the same freedom that let the unsupervised gate beat every supervised one.
            self.slot_q = nn.Parameter(torch.randn(1, slots, d) * 0.02)
            self.slot_attn = nn.MultiheadAttention(d, cfg['nhead'], dropout=cfg['dropout'],
                                                   batch_first=True)
            self.slot_score = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d + ng, d), nn.ReLU(), nn.Dropout(cfg['dropout']),
                                  nn.Linear(d, 3 + (3 if aux else 0)))
        self.fhead = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))
        # The gated sum can only ever REMOVE energy: w in (0,1) multiplies the observed cell
        # energy, so a cell that sampled low, or that fell under the 2.49 MeV threshold, is
        # unrecoverable. Our own error budget says the dominant term in the highest-leverage bin
        # (60 mm low-E, 11.5% of events) is an energy-INDEPENDENT 0.29 GeV -- an additive error,
        # which a multiplicative estimator cannot express whatever it is trained on. cellreg
        # replaces the weight with a per-cell energy regression that may exceed E_i. The residual
        # is initialised at zero so the model starts exactly at the gated sum and has to earn any
        # departure from it.
        self.widthfb = widthfb
        if widthfb:
            nout = 3 + (3 if aux else 0)
            self.whead = nn.Sequential(nn.Linear(d + ng + 2, d), nn.ReLU(),
                                       nn.Dropout(cfg['dropout']), nn.Linear(d, nout))
            nn.init.zeros_(self.whead[-1].weight)
            nn.init.zeros_(self.whead[-1].bias)
        self.cellreg = cellreg
        if cellreg:
            self.rhead = nn.Sequential(nn.Linear(d + 1, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))
            nn.init.zeros_(self.rhead[-1].weight)
            nn.init.zeros_(self.rhead[-1].bias)
        if gate == 'time':
            self.thead = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
            nn.init.constant_(self.thead[-1].bias, 2.0)
        self.la = nn.Parameter(torch.tensor(float(la0)))
        self.lb = nn.Parameter(torch.tensor(float(lb0)))

    def encode(self, x, m, g=None, sl=None):
        if self.gx:
            wm = m.unsqueeze(-1).float()
            mu = (x * wm).sum(1, keepdim=True) / wm.sum(1, keepdim=True).clamp(min=1)
            x = torch.cat([x, mu.expand_as(x)], -1)
        if self.nfour:
            u = x[:, :, 3:5]
            ang = u.unsqueeze(-1) * self.freqs
            x = torch.cat([x, torch.sin(ang).flatten(2), torch.cos(ang).flatten(2)], -1)
        if self.tfour:
            u = x[:, :, 7:9]
            ang = u.unsqueeze(-1) * self.tfreqs
            x = torch.cat([x, torch.sin(ang).flatten(2), torch.cos(ang).flatten(2)], -1)
        h = self.embed(x)
        if self.arch == 'spacetime':
            if sl is None:
                raise RuntimeError('arch spacetime needs the per-cell time slice; pass sl=D["SL"]')
            sl = sl.clamp(0, self.nslice - 1)
            h = h + self.sl_emb(sl)          # which slice a cell is in is itself information
            for i, lyr in enumerate(self.st):
                h = lyr(h, m, sl)
                h = self.apply_film(h, g, i)
            return h
        if self.arch == 'mixer':
            B, L, d = h.shape
            if L > self.lmax:
                raise RuntimeError(f'arch mixer is built for <= {self.lmax} tokens, got {L}')
            if L < self.lmax:
                h = torch.cat([h, h.new_zeros(B, self.lmax - L, d)], 1)
                mf = torch.cat([m, m.new_zeros(B, self.lmax - L)], 1)
            else:
                mf = m
            for blk in self.mix:
                h = blk(h, mf)
            return h[:, :L]
        if self.arch == 'pnet':
            return self.pnet(h, m, x)
        if self.arch == 'gravnet':
            for i, lyr in enumerate(self.grav):
                h = lyr(h, m, x)
                h = self.apply_film(h, g, i)
            return h
        if self.arch == 'geo':
            pair = pair_feats(x)
            for i, lyr in enumerate(self.geo):
                h = lyr(h, pair, m)
                h = self.apply_film(h, g, i)
            return h
        if self.film:
            for i, lyr in enumerate(self.layers):
                h = lyr(h, src_key_padding_mask=~m)
                h = self.apply_film(h, g, i)
            return h
        return self.enc(h, src_key_padding_mask=~m)

    def apply_film(self, h, g, i):
        if not self.film or g is None:
            return h
        sc, sh = self.mod[i](g).unsqueeze(1).chunk(2, -1)
        return h * (1.0 + sc) + sh

    def forward(self, x, m, g, ecell, pos=None, sl=None):
        h = self.encode(x, m, g, sl)
        if self.k_slots:
            s, _ = self.slot_attn(self.slot_q.expand(h.shape[0], -1, -1), h, h,
                                  key_padding_mask=~m, need_weights=False)
            sc = torch.einsum('bnd,bkd->bnk', self.slot_score(h), s) / h.shape[-1] ** 0.5
            w = torch.softmax(sc, -1)[:, :, 0] * m.float()
            self.last_w = w
            tot = (w * ecell).sum(1, keepdim=True)
            base = self.la * torch.log1p(tot.clamp(min=1e-3)) + self.lb
            out = self.head(torch.cat([self.norm(s[:, 0]), g], 1))
            q = base + out[:, :3]
            return torch.cat([q, out[:, 3:]], 1) if self.aux else q
        if self.gate_mode == 'off':
            w = m.float()
        elif self.signed:
            # GLS (w = C^-1 f) puts NEGATIVE weight on the outer rings: their energy is
            # correlated with the pileup contaminating the core, so it must be subtracted.
            # A sigmoid gate cannot express that, which caps the achievable variance.
            w = (1.5 * torch.sigmoid(self.fhead(h).squeeze(-1)) - 0.5) * m.float()
        else:
            w = torch.sigmoid(self.fhead(h).squeeze(-1)) * m.float()
            if self.gate_mode == 'time':
                w = w * torch.sigmoid(self.thead(x[:, :, 7:11]).squeeze(-1))
        self.last_w = w
        if self.cellreg:
            # relu, not softplus: w*E is already non-negative, so relu(w*E + r*E) reduces to w*E
            # exactly when the zero-initialised residual is zero, and the arm therefore starts
            # from the gated sum rather than 0.69 above it. The residual is scaled by the cell's
            # own energy so one correction is meaningful for a 10 MeV cell and a 10 GeV one.
            r = self.rhead(torch.cat([h, torch.log1p(ecell.clamp(min=0)).unsqueeze(-1)], -1)).squeeze(-1)
            cell_e = torch.relu(w * ecell + r * ecell.clamp(min=1e-3)) * m.float()
            tot = cell_e.sum(1, keepdim=True)
        else:
            tot = (w * ecell).sum(1, keepdim=True)
        base = self.la * torch.log1p(tot.clamp(min=1e-3)) + self.lb
        if self.qpool:
            p = self.norm(self.pool_attn(self.q.expand(h.shape[0], -1, -1), h, h,
                                         key_padding_mask=~m, need_weights=False)[0].squeeze(1))
        else:
            wm = m.unsqueeze(-1).float()
            p = self.norm((h * wm).sum(1) / wm.sum(1).clamp(min=1))
        out = self.head(torch.cat([p, g], 1))
        if self.widthfb:
            # The predicted interval width separates the 8.2% tail that carries 21% of the total
            # error at AUC 0.93, and the pipeline spends that signal on nothing but choosing which
            # of three calibration terciles an event lands in. Here the correction head reads it
            # directly: a second pass sees how uncertain the first pass was, so the median can be
            # pulled differently for an event the model already knows it is guessing at. The
            # second head is zero-initialised, so this starts exactly at the one-pass readout.
            wd = (out[:, 2] - out[:, 0]).detach().unsqueeze(-1)
            out = out + self.whead(torch.cat([p, g, wd, wd.abs().log1p()], 1))
        q = base + out[:, :3]
        return torch.cat([q, out[:, 3:]], 1) if self.aux else q

    def gate(self, x, m, ecell, g=None):
        h = self.encode(x, m, g)
        raw = torch.sigmoid(self.fhead(h).squeeze(-1))
        w = ((1.5 * raw - 0.5) if self.signed else raw) * m.float()
        if self.gate_mode == 'time':
            w = w * torch.sigmoid(self.thead(x[:, :, 7:11]).squeeze(-1))
        return w


def pinball_loss(q, yb, qs, w=None):
    d = yb - q
    pin = torch.maximum(qs * d, (qs - 1) * d)
    if w is None:
        return pin.mean()
    return (pin.mean(1) * w).sum() / w.sum().clamp(min=1e-6)


def qd_pinball_loss(q, yb, qs, lam=0.5, tau=0.02, w=None):
    pin = pinball_loss(q, yb, qs, w)
    width = (q[:, 2] - q[:, 1]).abs() + (q[:, 1] - q[:, 0]).abs()
    inside = torch.sigmoid((yb.squeeze(1) - q[:, 0]) / tau) * torch.sigmoid((q[:, 2] - yb.squeeze(1)) / tau)
    return pin + lam * (width.mean() + 10.0 * torch.relu(0.5 - inside.mean()) ** 2)


def width_binned_calibration(qv, qt, yva, n_groups=3, min_n=10):
    wv = qv[:, 2] - qv[:, 0]
    wt = qt[:, 2] - qt[:, 0]
    cuts = np.quantile(wv, np.linspace(0, 1, n_groups + 1)[1:-1])
    gv = np.digitize(wv, cuts)
    gt = np.digitize(wt, cuts)
    pe = np.empty(len(qt))
    for g in range(n_groups):
        if (gv == g).sum() < min_n or (gt == g).sum() == 0:
            a, b = np.polyfit(qv[:, 1], yva, 1)
        else:
            a, b = np.polyfit(qv[gv == g, 1], yva[gv == g], 1)
        pe[gt == g] = np.exp(a * qt[gt == g, 1] + b)
    return pe


def linear_calibration(qv, qt, yva):
    a, b = np.polyfit(qv[:, 1], yva, 1)
    return np.exp(a * qt[:, 1] + b)


def save_model(path, model, extras):
    torch.save(dict(state_dict=model.state_dict(), **extras), path)


def load_model(path, device='cpu'):
    st = torch.load(path, map_location=device, weights_only=False)
    model = SubNetFQ(st['in_dim'], st['la0'], st['lb0'], ng=st.get('ng', NG),
                     gate=st.get('gate', 'learned'), arch=st.get('arch', 'std'),
                     qpool=st.get('qpool', False), gx=st.get('gx', False),
                     aux=st.get('aux', False), film=st.get('film', False),
                     nfour=st.get('nfour', 0), tfour=st.get('tfour', 0),
                     slots=st.get('slots', 0)).to(device)
    model.load_state_dict(st['state_dict'])
    model.eval()
    return model, st
