# Internal geometry comparison

| Seed | Original | Geometry SC | Fixed geometry | Time-scaled geometry |
|---|---:|---:|---:|---:|
| 1 | 384/768 | 410/768 | 167/768 | 328/768 |
| 2 | 327/768 | 369/768 | 156/768 | 315/768 |

Graph-rate differences in percentage points:

- time_geometry minus plain: -4.43; paired95 [-6.84,-2.02], composition95 [-6.64,-2.41].
- time_geometry minus fixed_geometry: +20.83; paired95 [+18.29,+23.44], composition95 [+17.12,+24.81].
- time_geometry minus geometry: -8.85; paired95 [-11.46,-6.25], composition95 [-11.33,-6.38].
- fixed_geometry minus plain: -25.26; paired95 [-27.93,-22.59], composition95 [-30.27,-20.12].

Prespecified internal-geometry gate passed: **False**.
All6144 source/output records replay. Original/fixed/time variants have identical trainable parameter counts and state shapes; their training data, source, warm start and update recipe match.
The geometry-SC reference used higher training compute, which is reported separately. All inference arms use128 primitive denoiser calls. This reused12-composition development panel is not an untouched confirmation. No final-density or Boltzmann-law result is established.
