# Experiment results

config: {'files': 3, 'epochs': 2, 'window': 3, 'vertex': 100.0, 'region': 3, 'device': 'cuda', 'batch': 64, 'tag': 'e12smoke'}
kept 1033 clusters; R3 = 293

| experiment | sigma_eff | IQR | bias |
|---|---|---|---|
| E0_B0_sum_calib | 0.0288 | 0.0301 | -0.0099 |
| E0_B1_seed_calib | 0.1898 | 0.2195 | -0.0607 |
| E0_B2_total_energy_calib | 0.0294 | 0.0307 | -0.011 |
| E0b_BDT | 0.045 | 0.0497 | -0.0031 |
| E1_transformer_seed | 0.057 | 0.0534 | -0.8499 |
| E2_transformer_cluster | 0.0317 | 0.0338 | -0.9176 |
| E11_deepsets_seed | 0.0229 | 0.0241 | -0.9418 |
| E4_all_regions_overall | 0.1127 | 0.1405 | -0.83 |
| E4_all_regions_on_R3 | 0.1413 | 0.2117 | -0.7342 |
| E12a_drop_frontback | 0.0511 | 0.0503 | -0.8768 |
| E12b_raw_energy | 0.0368 | 0.0384 | -0.9144 |
| E12c_drop_region_onehot | 0.095 | 0.1045 | -0.8242 |
| E12d_abs_coords | 0.035 | 0.0408 | -0.9201 |
