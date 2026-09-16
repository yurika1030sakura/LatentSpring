# Matched physical adaptation of FM and GAGA

The raw feedback challenge is complete and audited. On the fresh32-composition
panel (1024 attempts per model seed), harmonic FM with ordinary distance
self-conditioning has155/165 valid outputs, versus151/131 for GAGA650.
Pooled rates are15.625% and13.7695%. The paired composition difference is+1.86pp,
with95 interval[-0.34,4.44]. Both seed point estimates are positive, but the
prespecified superiority gate does not pass. The global tree feedback has114/84
valid outputs (9.67% pooled), is worse, and is not adopted. These are matched
unpretrained EGNNs, not the primary FlowMol3 backbone.

The next fixed experiment addresses the review's remaining strong-baseline gap:
both validation-selected parent settings receive paired physical adaptation.
Source protocols are `matched_physical_s{0,1}_v1.json`; neither parent setting is
changed using the physical outcomes. The graph/force<=5 primary contrast compares
adapted FM with adapted GAGA under both eSEN and independent GFN2. Unadapted
outputs are reused, and all failures remain in joint-yield denominators.

Both parents originally consumed960000 retained supervised backbone example
passes. Each new student gets2000 such passes:1000 two-pass FM updates or2000
one-pass GAGA updates. Replay and physical students match within each parent;
coefficient1, original AdamW1e-4 and gradient clipping are fixed. Data presentations
and optimizer-step counts differ between families and are explicitly recorded;
there is no equal-wall-time claim. Each physical inference still uses128 calls.

Eight17-28-atom TRAIN compositions are selected solely by their frequency in the
common20000-row dataset (ties by fixed hash). Each parent generates64 FIT starts
per composition. The original eight-particle local teacher,300K temperature and
perturbation/shift bounds are retained, with no replenishment. The same physical
recipe and maximum preparation budget apply to both parents; native-anchor
support and actual query costs can differ. This is a fixed physical follow-up on
the already evaluated32-composition panel, not a new untouched test.

Job46669949 uses immutable source25745ee and runs both seeds with eSEN and GFN2
readouts. Audit46670954 uses source8e6fe03 and follows successful completion.
It replays the teacher geometry/support and target choices, verifies parameter
arithmetic and generation hashes, reparses GFN2 outputs and checks inversion
references. Results are under `runs/matched_physical_v1`. No adapted-GAGA
superiority claim may be made before the audit succeeds and the gate is checked.

The main source/physics factorial remains a completed positive result. Its
five-continuation source-budget follow-up runs separately as46656786. New evidence
must preserve the distinction between those pretrained models and these matched
unpretrained EGNNs.


## Qualification and correction

V1 finished as46669949 and its numerical replay audit46670954 passed. Inspection
of original parameter flags then exposed a missing invariant in that audit:
`requires_grad_(True)` unfroze the predefined diffusion schedule. GAGA's paired
schedule changes by up to0.008185/0.009170 across the two seeds; the FM schedule
is unused and remained identical. This prevents treating the GAGA v1 result as
the intended fixed-schedule benchmark. All artifacts and expenditure remain.

The FM outcome is a genuine negative result for this fixed adaptation setting:
raw graph support drops from155/165 to27/14, and GFN2 joint force<=5 yield drops
from4.785% to0.781%. The implementation issue does not explain that failure.
Do not scale or promote this full-strength EGNN recipe as established utility.

Source76225fe preserves the pretrained trainability flags and excludes the fixed
schedule from the optimizer and parameter update. A regression test passes and
checks both frozen-schedule equality and actual changes in trainable weights.
Correction46674037 and audit46674039 use the same GAGA parents,
cached teacher pools, exact target selections, learning rate, update counts and
coefficient. Only GAGA is retrained; FM's negative outputs remain unchanged.
New corrected GAGA outputs and physical calls are counted separately. These
studies do not alter the published main-backbone positive factorial.
