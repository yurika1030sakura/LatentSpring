# Tree-coordinate flow: completed negative result

| Seed | Cartesian | Cartesian projected | Product-space |
|---|---:|---:|---:|
| 1 | 311/768 | 202/768 | 165/768 |
| 2 | 295/768 | 193/768 | 156/768 |

Product-space flow eliminates disconnected outputs, but graph validity is substantially WORSE.

- vs cartesian: -18.55 pp; paired95 [-20.83,-16.28], composition95 [-21.88,-15.56].
- vs cartesian_projected: -4.82 pp; paired95 [-6.90,-2.73], composition95 [-7.88,-1.95].

The gate fails. All6256 selected training/reference rows and scaffolds,3072 neural outputs and1536 additional projected readouts replay. Original coordinates/metadata and composition exclusions check; checkpoints and sources restore. No quantum calls.
This shows that contact connectivity is insufficient for chemical validity and that enforcing an unreliable fixed auxiliary tree can harm generation. It does not prove every internal-coordinate or adaptive-chart method must fail. Do not reinterpret zero fragmentation as a learned-method success.
The same clean training rows can be reused as FIT data; all outputs and evaluated compositions stay outside fitting. Next is the separately frozen geometry/latent-feedback test, which retains free coordinate motion and tests pair-localized learned information.
