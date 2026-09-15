# Independent conditional EDM: final bounded adaptation

The extended30000-update models are complete and audited. Same3000 isolated OMol25
FIT rows per continuation are reused for ten passes, with the published1e-4 learning
rate and unchanged conditioning/sampling. Exact3000-step checkpoints and optimizer
states were resumed for27000 additional steps. Both initial and final outcomes
remain archived. No further baseline sweep is planned.

Raw graph counts per768 attempts:128 calls103/100 (13.22% pooled),1001 calls93/106
(12.96%). LatentSpring harmonic has435/375 (52.73%) on the same compositions and
force update439/382 (53.45%), both at128 calls. The primary source attribution
remains the matched Gaussian/harmonic comparison, not this different-backbone
comparison. All3072 new final baseline outputs and structural records replay.

EDM continuation training seconds481.51/463.61 include the initial42-second phase;
LatentSpring source continuations took approximately570–590 seconds. EDM uses
2.38M parameters and one pass/update; LatentSpring5.90M and two. At128 calls EDM
generation takes48.54/47.94 seconds per768 draws; at1001 calls377.65/374.12 seconds.
Equal calls do not mean equal FLOPs. The warm-start histories and datasets differ;
GEOM pretraining overlap is not certified. These results describe a task-adapted
conditional baseline, not native EDM's published joint-generation performance or
matched total-pretraining superiority.

Job46546237_0/1 and audit `conditional_edm_extended_audit_v1.json` are complete.
No new physical oracle is queried by this baseline. The original pending-only
normal-GPU job46543268 was cancelled without training; initial actual training ran
as46543959_0/1. Source snapshots and all histories remain intact.
