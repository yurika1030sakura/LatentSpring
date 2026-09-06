| metric | energy_s2 | control_s2 | energy_s3 | control_s3 |
|---|---|---|---|---|
| systems | 12 | 12 | 12 | 12 |
| within-basin r (old metric) | 0.3812 | 0.3613 | 0.4616 | 0.3187 |
| between-basin r (basin mass) | 0.07696 | 0.1507 | 0.2171 | 0.4293 |
| between-basin slope (ideal 1) | 0.08083 | 0.1994 | 0.8558 | 1.12 |
| NRV basin mass (ideal 0) | 28.08 | 7.295 | 8.415 | 6.487 |
| NRV within basins (ideal 0) | 6.483 | 1.246 | 1.671 | 1.224 |
| kT_eff between basins (eV) | 0.8221 | 0.725 | 0.5483 | 1.387 |
| basin dF error, mass (eV) | 7.833 | 4.584 | 4.592 | 3.78 |
| true basin dF spread, mass (eV) | 2.061 | 2.061 | 2.061 | 2.061 |
| dF err / true (1 = flat, 0 = ideal) | 4.507 | 2.492 | 2.579 | 2.11 |
| basin dF error, minima (eV) | 6.474 | 1.644 | 1.751 | 4.334 |
| true basin dF spread, minima (eV) | 0.05771 | 0.05771 | 0.05771 | 0.05771 |
| pointwise r at minima | -0.1019 | 0.1251 | 0.4226 | -0.1029 |
| pointwise r | Rg (confound ctrl) | 0.01352 | 0.09197 | 0.1997 | 0.07027 |
| KL(P_ref || P_model) | 7.118 | 3.006 | 2.835 | 2.179 |
| total variation | 0.5484 | 0.5843 | 0.5206 | 0.4214 |
| W1 on basin energy (eV) | 0.03332 | 0.04751 | 0.0385 | 0.03331 |
| basin-weight ESS frac | 0.2219 | 0.2616 | 0.25 | 0.2887 |
| IS ESS frac, ref (est. quality) | 0.1948 | 0.1948 | 0.1948 | 0.1948 |
| IS ESS frac, model (est. quality) | 0.1336 | 0.1654 | 0.1539 | 0.1702 |
| top-1 basin match | 0.5 | 0.4167 | 0.5 | 0.6667 |

**Paired per-system comparison (energy_s2 - control_s2, n=12 shared systems)**

| metric | energy | control | Δ | t | p |
|---|---|---|---|---|---|
| r_basin_mass | 0.07696 | 0.1507 | -0.0737 | -0.71 | 0.494 |
| slope_basin_mass | 0.08083 | 0.1994 | -0.1186 | -0.28 | 0.782 |
| nrv_basin_mass | 28.08 | 7.295 | +20.78 | 2.17 | 0.053 |
| dF_err_mass_eV | 7.833 | 4.584 | +3.249 | 2.59 | 0.0253 |
| dF_err_mass_ratio | 4.507 | 2.492 | +2.015 | 2.82 | 0.0167 |
| dF_err_min_eV | 6.474 | 1.644 | +4.83 | 3.29 | 0.00716 |
| kl_ref_model | 7.118 | 3.006 | +4.111 | 1.52 | 0.156 |
| tv | 0.5484 | 0.5843 | -0.03591 | -0.28 | 0.785 |
| mean_within_basin_r | 0.3812 | 0.3613 | +0.01984 | 0.53 | 0.608 |
| nrv_within | 6.483 | 1.246 | +5.236 | 4.42 | 0.00102 |
| r_pointwise | -0.1019 | 0.1251 | -0.2269 | -0.75 | 0.469 |
| ess_frac_basins | 0.2219 | 0.2616 | -0.03979 | -1.98 | 0.0727 |

