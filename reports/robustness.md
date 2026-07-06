# Robustness — multi-seed and controlled E4

config: {'files': 100, 'epochs': 30, 'seeds': 5, 'window': 3, 'vertex': 100.0, 'region': 3, 'device': 'cuda', 'batch': 64}
R3 test set = 1604 clusters

All over 5 seeds. E1 baseline mean = 0.0687.

| experiment | mean sigma_eff | std | delta vs E1 | significant? |
|---|---|---|---|---|
| E1_transformer_seed | 0.0687 | 0.0117 | (baseline) | - |
| E11_deepsets_seed | 0.1032 | 0.0086 | +0.0345 | within noise |
| E12a_drop_frontback | 0.0568 | 0.0101 | -0.0119 | within noise |
| E12b_raw_energy | 0.0636 | 0.0124 | -0.0051 | within noise |
| E12c_drop_region_onehot | 0.0713 | 0.0163 | +0.0026 | within noise |
| E12d_abs_coords | 0.0736 | 0.0100 | +0.0049 | within noise |

## Controlled E4 — same R3 test set

| model | mean sigma_eff | std |
|---|---|---|
| R3only_on_rte | 0.0638 | 0.0087 |
| allregions_on_rte | 0.0504 | 0.0050 |

all-region vs R3-only on the same test set: delta = -0.0134 (within noise).

