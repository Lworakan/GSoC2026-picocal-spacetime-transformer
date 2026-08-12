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


class CNNSub(nn.Module):
    def __init__(self, in_dim, la0, lb0, ng=NG, cfg=CFG, side=9):
        super().__init__()
        self.side = side
        self.gate_mode = 'learned'
        d = cfg['d']
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

    def forward(self, x, m, g, ecell, pos=None):
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


class SubNetFQ(nn.Module):
    def __init__(self, in_dim, la0, lb0, ng=NG, cfg=CFG, gate='learned', arch='std', qpool=False,
                 gx=False, aux=False, film=False, nfour=0):
        super().__init__()
        self.gate_mode = gate
        self.signed = gate == 'signed'
        self.gx = gx
        self.aux = aux
        self.film = film
        self.nfour = nfour
        self.arch = arch
        self.qpool = qpool
        d = cfg['d']
        # Fourier features on the two in-cell offset channels. Our own measurement says the
        # sub-cell impact position is the hidden variable that fixed estimators cannot use
        # (the seed's energy share swings 40-90% with it), and a linear embedding of a raw
        # coordinate represents high-frequency dependence on it poorly.
        extra_in = 4 * nfour if nfour else 0
        self.embed = nn.Linear(in_dim * (2 if gx else 1) + extra_in, d)
        if nfour:
            self.register_buffer('freqs', 2.0 ** torch.arange(nfour).float() * torch.pi)
        if arch == 'geo':
            self.geo = nn.ModuleList([GeoEncoderLayer(d, cfg['nhead'], 4 * d, cfg['dropout'])
                                      for _ in range(cfg['layers'])])
        elif film:
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d, cfg['nhead'], dim_feedforward=4 * d,
                                           dropout=cfg['dropout'], batch_first=True)
                for _ in range(cfg['layers'])])
        else:
            layer = nn.TransformerEncoderLayer(d, cfg['nhead'], dim_feedforward=4 * d,
                                               dropout=cfg['dropout'], batch_first=True)
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
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d + ng, d), nn.ReLU(), nn.Dropout(cfg['dropout']),
                                  nn.Linear(d, 3 + (3 if aux else 0)))
        self.fhead = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))
        if gate == 'time':
            self.thead = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
            nn.init.constant_(self.thead[-1].bias, 2.0)
        self.la = nn.Parameter(torch.tensor(float(la0)))
        self.lb = nn.Parameter(torch.tensor(float(lb0)))

    def encode(self, x, m, g=None):
        if self.gx:
            wm = m.unsqueeze(-1).float()
            mu = (x * wm).sum(1, keepdim=True) / wm.sum(1, keepdim=True).clamp(min=1)
            x = torch.cat([x, mu.expand_as(x)], -1)
        if self.nfour:
            u = x[:, :, 3:5]
            ang = u.unsqueeze(-1) * self.freqs
            x = torch.cat([x, torch.sin(ang).flatten(2), torch.cos(ang).flatten(2)], -1)
        h = self.embed(x)
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

    def forward(self, x, m, g, ecell, pos=None):
        h = self.encode(x, m, g)
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
        tot = (w * ecell).sum(1, keepdim=True)
        base = self.la * torch.log1p(tot.clamp(min=1e-3)) + self.lb
        if self.qpool:
            p = self.norm(self.pool_attn(self.q.expand(h.shape[0], -1, -1), h, h,
                                         key_padding_mask=~m, need_weights=False)[0].squeeze(1))
        else:
            wm = m.unsqueeze(-1).float()
            p = self.norm((h * wm).sum(1) / wm.sum(1).clamp(min=1))
        out = self.head(torch.cat([p, g], 1))
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
                     nfour=st.get('nfour', 0)).to(device)
    model.load_state_dict(st['state_dict'])
    model.eval()
    return model, st
