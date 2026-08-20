import sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, 'scripts')
from picocal_models import load_model
torch.set_num_threads(24)
model, st = load_model('models/SubNetW4CleanAuxExDnRcQdEma_smoke_s0.pt')
D = st["in_dim"]
G = st["ng"]
print('in_dim', D, 'glob', G, 'params', sum(p.numel() for p in model.parameters()))
def rate(L, B=64, N=3072, reps=5):
    x = torch.randn(N, L, D); m = torch.ones(N, L, dtype=torch.bool)
    g = torch.randn(N, G); ec = torch.rand(N, L)
    with torch.no_grad():
        for _ in range(2):
            for i in range(0, N, B):
                model(x[i:i+B], m[i:i+B], g[i:i+B], ec[i:i+B])
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for i in range(0, N, B):
                model(x[i:i+B], m[i:i+B], g[i:i+B], ec[i:i+B])
            ts.append(time.perf_counter() - t0)
    return N / np.median(ts)
r81 = rate(81); r289 = rate(289)
print(f'L=81: {r81:.0f} clusters/s   L=289: {r289:.0f} clusters/s   ratio {r81/r289:.2f}')
