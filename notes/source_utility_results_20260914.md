# Source utility pilot: audited fresh generation

These12 compositions are held out from the source head, but originate from the flow training corpus. Both frozen decoders and all four prespecified methods are retained.

| Decoder | Fixed | Actual utility | Shuffled utility | Coordinate NLL |
|---|---:|---:|---:|---:|
| 1 | 461/768 | 462/768 | 423/768 | 476/768 |
| 2 | 410/768 | 392/768 | 410/768 | 409/768 |

Pooled actual-minus-control graph-rate differences; paired draw and descriptive size-stratified composition intervals are both retained:

- vs fixed: -1.11 pp; paired95 [-3.84, +1.76], composition95 [-3.58, +0.91].
- vs shuffled: +1.37 pp; paired95 [-1.63, +4.36], composition95 [-1.11, +3.91].
- vs nll: -2.02 pp; paired95 [-4.95, +0.98], composition95 [-5.47, +1.24].

Prespecified development gate passed: **False**. This is not an ICLR readiness or final target-distribution result.
All9216 source draws and structural assays replay. No new molecular oracle calls. Train-bank preparation and source fitting must be included in cost comparisons.
The existing negative coordinate-NLL, static-context and monomer studies remain unchanged. The pilot cannot establish broad failure of all source utility objectives or justify tuning this recipe on the observed validation outcomes.


The standalone adaptation cost includes at least7.71/7.90 minutes of bank generation
plus actual-head training per decoder (462.39/473.74 seconds). Corpus qualification
ran separately without a wall-time timer, so these are preparation lower bounds.
The shared experiment finished in24:02/24:43 scheduler elapsed. Its source networks
have1881 parameters each and all receive720 updates. All original frozen decoder
state tensors remain exactly unchanged after generation. The source/output KL bound
is respected on every held condition; it does not imply a useful utility gain.

The frozen-head FIT-bank importance gains for actual utility are0.02618/0.02204;
shuffled heads improve their own shuffled-bank objectives by0.02051/0.01545. Fresh
actual-minus-fixed utility is-0.00859, paired95[-0.03464,0.01745]. This is a
training-versus-transfer gap, not proof of a unique mechanism: finite-bank fitting,
composition shift, source-family capacity and optimization remain possible factors.
Do not call optimized-on-bank importance estimates unbiased performance estimates.

Decision: the prespecified gate FAILS. Do not expand this recipe's budget, relax
its trust limit, alter rewards, choose the favorable decoder or present it as a
successful neural increment. It does not prove that all utility-trained sources
must fail. SGFM/Noise PPO and other checked prior art also rule out originality
claims for generic source adaptation or frozen-decoder noise learning.

Next bounded question: is there a reproducible terminal-utility signal in the
current affinity family on FIT data? Before any further model run, specify a
within-FIT split/gradient or score-signal check that can distinguish reproducible
signal from the gains obtainable after label shuffling. Keep the12 newly evaluated
compositions/6144 outputs outside future fitting. A diagnostic is not itself a new
AI contribution. A different method would need an explicit missing capability,
a comparison against relevant source-adaptation work, and new frozen evidence.
Do not immediately add another context module or declare that simply more data
will solve the problem. No additional performance experiment is launched now.
