# Experiment results

config: {'files': 100, 'epochs': 30, 'window': 3, 'vertex': 100.0, 'region': 3, 'device': 'cuda', 'batch': 64, 'tag': 'gpu'}
kept 36852 clusters; R3 = 10692

| experiment | sigma_eff | IQR | bias |
|---|---|---|---|
| E0_B0_sum_calib | 0.0649 | 0.0675 | 0.0208 |
| E0_B1_seed_calib | 0.2251 | 0.2202 | 0.0527 |
| E0_B2_total_energy_calib | 0.0579 | 0.0593 | 0.0184 |
| E0b_BDT | 0.0563 | 0.0518 | 0.0099 |
| E1_transformer_seed | 0.0621 | 0.0617 | 0.0561 |
| E2_transformer_cluster | 0.0911 | 0.0899 | 0.078 |
| E11_deepsets_seed | 0.1136 | 0.1124 | 0.0202 |
| E4_all_regions_overall | 0.0888 | 0.0883 | 0.0777 |
| E4_all_regions_on_R3 | 0.0405 | 0.0399 | 0.0425 |
| E12a_drop_frontback | 0.102 | 0.1084 | 0.0381 |
| E12b_raw_energy | 0.0517 | 0.0462 | 0.0602 |
| E12c_drop_region_onehot | 0.0674 | 0.0671 | 0.0855 |
| E12d_abs_coords | 0.0783 | 0.0807 | 0.0304 |
