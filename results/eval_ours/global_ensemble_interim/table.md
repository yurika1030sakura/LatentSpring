| metric | energy_s2 | control_s2 | energy_s3 | control_s3 |
|---|---|---|---|---|
| systems | 4 | 4 | 4 | 4 |
| within-basin r (old metric) | 0.3043 | 0.3114 | 0.4128 | 0.2485 |
| between-basin r (basin mass) | 0.5204 | 0.4016 | 0.3627 | 0.3611 |
| between-basin slope (ideal 1) | 1.615 | 0.5692 | 0.6558 | 0.4531 |
| NRV basin mass (ideal 0) | 5.873 | 2.236 | 3.147 | 2.522 |
| NRV within basins (ideal 0) | 4.361 | 1.16 | 1.399 | 1.197 |
| kT_eff between basins (eV) | 0.8221 | 2.232 | 0.8158 | 1.524 |
| basin dF error, mass (eV) | 6.457 | 4.111 | 4.478 | 4.325 |
| true basin dF spread, mass (eV) | 2.768 | 2.768 | 2.768 | 2.768 |
| dF err / true (1 = flat, 0 = ideal) | 2.359 | 1.435 | 1.675 | 1.505 |
| basin dF error, minima (eV) | 4.539 | 1.878 | 2.395 | 4.886 |
| true basin dF spread, minima (eV) | 0.05529 | 0.05529 | 0.05529 | 0.05529 |
| pointwise r at minima | -0.07641 | -0.3242 | 0.1551 | -0.2132 |
| pointwise r | Rg (confound ctrl) | -0.05077 | 0.102 | 0.04701 | -0.06284 |
| KL(P_ref || P_model) | 1.583 | 1.343 | 2.129 | 2.545 |
| total variation | 0.2783 | 0.4865 | 0.4848 | 0.4489 |
| W1 on basin energy (eV) | 0.01956 | 0.02067 | 0.01999 | 0.01785 |
| basin-weight ESS frac | 0.1931 | 0.2943 | 0.2088 | 0.1823 |
| IS ESS frac, ref (est. quality) | 0.1948 | 0.1948 | 0.1948 | 0.1948 |
| IS ESS frac, model (est. quality) | 0.1388 | 0.1685 | 0.1476 | 0.1704 |
| top-1 basin match | 0.75 | 0.5 | 0.5 | 0.5 |

**Paired per-system comparison (energy_s2 - control_s2, n=4 shared systems)**

| metric | energy | control | Δ | t | p |
|---|---|---|---|---|---|
| r_basin_mass | 0.5204 | 0.4016 | +0.1188 | 0.60 | 0.588 |
| slope_basin_mass | 1.615 | 0.5692 | +1.046 | 1.51 | 0.228 |
| nrv_basin_mass | 5.873 | 2.236 | +3.637 | 1.70 | 0.187 |
| dF_err_mass_eV | 6.457 | 4.111 | +2.346 | 1.37 | 0.266 |
| dF_err_mass_ratio | 2.359 | 1.435 | +0.9235 | 1.79 | 0.171 |
| dF_err_min_eV | 4.539 | 1.878 | +2.661 | 1.72 | 0.183 |
| kl_ref_model | 1.583 | 1.343 | +0.2403 | 0.75 | 0.506 |
| tv | 0.2783 | 0.4865 | -0.2083 | -2.64 | 0.0774 |
| mean_within_basin_r | 0.3043 | 0.3114 | -0.007081 | -0.27 | 0.804 |
| nrv_within | 4.361 | 1.16 | +3.201 | 2.18 | 0.117 |
| r_pointwise | -0.07641 | -0.3242 | +0.2478 | 1.58 | 0.213 |
| ess_frac_basins | 0.1931 | 0.2943 | -0.1012 | -2.13 | 0.123 |

