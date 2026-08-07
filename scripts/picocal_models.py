import numpy as np
import torch
import torch.nn as nn

CFG = dict(d=128, nhead=4, layers=3, dropout=0.1, lr=3e-4, wd=1e-4, batch=96, huber_delta=0.1)
NG = 6
NC = 9
QUANTILES = [0.25, 0.5, 0.75]


class SubNetFQ(nn.Module):
    def __init__(self, in_dim, la0, lb0, ng=NG, cfg=CFG, gate='learned'):
        super().__init__()
        self.gate_mode = gate
        d = cfg['d']
        self.embed = nn.Linear(in_dim, d)
        layer = nn.TransformerEncoderLayer(d, cfg['nhead'], dim_feedforward=4 * d,
                                           dropout=cfg['dropout'], batch_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg['layers'], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d + ng, d), nn.ReLU(), nn.Dropout(cfg['dropout']), nn.Linear(d, 3))
        self.fhead = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))
        self.la = nn.Parameter(torch.tensor(float(la0)))
        self.lb = nn.Parameter(torch.tensor(float(lb0)))

    def forward(self, x, m, g, ecell):
        h = self.enc(self.embed(x), src_key_padding_mask=~m)
        if self.gate_mode == 'off':
            w = m.float()
        else:
            w = torch.sigmoid(self.fhead(h).squeeze(-1)) * m.float()
        base = self.la * torch.log1p((w * ecell).sum(1, keepdim=True)) + self.lb
        wm = m.unsqueeze(-1).float()
        p = self.norm((h * wm).sum(1) / wm.sum(1).clamp(min=1))
        return base + self.head(torch.cat([p, g], 1))

    def gate(self, x, m, ecell):
        h = self.enc(self.embed(x), src_key_padding_mask=~m)
        return torch.sigmoid(self.fhead(h).squeeze(-1)) * m.float()


def pinball_loss(q, yb, qs):
    d = yb - q
    return torch.maximum(qs * d, (qs - 1) * d).mean()


def qd_pinball_loss(q, yb, qs, lam=0.5, tau=0.02):
    pin = pinball_loss(q, yb, qs)
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
    model = SubNetFQ(st['in_dim'], st['la0'], st['lb0'], ng=st.get('ng', NG)).to(device)
    model.load_state_dict(st['state_dict'])
    model.eval()
    return model, st
