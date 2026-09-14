# Geometry/relation feedback results

| Seed | One pass | Geometry SC | Pair codes | Pooled codes |
|---|---:|---:|---:|---:|
| 1 | 363/768 | 417/768 | 422/768 | 432/768 |
| 2 | 312/768 | 342/768 | 334/768 | 333/768 |

Graph-rate differences in percentage points; both uncertainty summaries retained:

- latent minus geometry: -0.20; paired95 [-1.50,+1.17], composition95 [-1.24,+1.17].
- latent minus pooled: -0.59; paired95 [-2.02,+0.85], composition95 [-1.95,+1.24].
- geometry minus plain: +5.47; paired95 [+3.06,+7.94], composition95 [+4.10,+6.77].
- latent minus plain: +5.27; paired95 [+2.86,+7.81], composition95 [+3.32,+7.29].

Prespecified relation-learning gate passed: **False**.
All6144 source/structure output records replay. Checkpoints restore, matching initial feedback weights and coordinate-only edge-head updates are verified.
A gain of geometry SC over plain is a known-technique baseline improvement, not evidence of new AI novelty. Primary learned-relation evidence requires gains over geometry-only and pooled-code feedback.
All inference arms use128 primitive denoiser calls. Training updates match but SC training compute is higher; component runtimes and forward counts are retained. The12 compositions are reused development conditions, not untouched confirmatory evidence. No chemical-bond semantics are assigned to the codes and no energy-law claim is made.
