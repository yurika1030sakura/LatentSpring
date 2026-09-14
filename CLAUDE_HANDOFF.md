# Claude continuation — completed tree-prior studies, mixed evidence

Checkout: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`, branch
`iclr2027-arch-fix`. Read `research/NEXT.md`,
`research/TREE_PRIOR_STATE_20260914.json`, `notes/tree_prior_results_20260914.md`
and `notes/tree_mixture_prior_v1.md` before another experiment.

The user's endorsed direction is implemented: node/pair networks learn latent
connection affinities from composition and electronic conditions, fitting an
explicit normalized spatial prior without bond labels. The generator remains
flow matching. This is a candidate AI contribution; tree marginalization and
structured molecular priors have prior art. Extra learned utility is unproven.
Keep the user's priority on core usefulness and a bounded molecular demonstration.

All current studies are complete. Unrestricted-data graph counts/512 are78 warm,
76 Gaussian,94 fixed,92 node,93 pair. The actual prior-training sample contains
580/1000 disconnected structures. The next comparison uses the SAME train-only
connected/nonoverlap selection for every method: graph counts66,105,131,121,111.
Independent Gaussian/fixed continuation gives91/93, so the first+26 graph-pass
advantage does not clearly repeat. Disconnection falls192 to162, then243 to194.
The confirmation shares the warm checkpoint and development compositions; it
has new continuation/data-order and generation seeds, not independent pretraining.

All first-connected-run outputs received physical evaluation, including failures
and metals:5,120 raw eSEN calls. Node versus Gaussian mean E+ difference is-0.624 eV
[-1.084,-0.144]; node versus fixed is-0.229[-0.739,0.269]. These are marginal
conditional paired intervals on eight fixed development conditions, not replicated
training effects or composition-generalization intervals. Energy is not a thermal
distribution test. Retain the source framework; do not scale the present pair
head or declare a winning learned architecture.

Completed jobs:46333856 prior fit,46335714/46335903 unrestricted FM;46342237
connected prior fit,46342814/46342817 connected FM,46344611 audit;46352584_0/1
confirmation;46354739 energy. All completed0:0. Query live Slurm; no tree-study job
was active at the recorded check. Other projects were not changed. All6,144
source draws and structural outcomes and all5,120 stored physical rows were
audited. The audits do not retrain optimizers, regenerate final neural outputs or
independently re-query eSEN.53 targeted tests passed during implementation.
Auditv2 files strengthen provenance without changing outcomes; v1 is retained.

Use `research/evidence/generator_reference_registry_v2.json` for checkpoints and
source laws. A tree checkpoint embeds its prior separately from the vector-field
state. `prepare_research_backbone` restores the source-kind flag; generation still
needs matching explicit prior draws. Do not use implicit Gaussian sampling or a
Gaussian density for tree models. The old independent-Gaussian score proxy is
invalid for correlated pairing. T1-plus-noise outputs have no qualified density
or ESS. The experiment supplies composition and original charge/spin; it is not
a demonstrated joint generator of these conditions.

Next is one mechanism-based diagnosis of the gap between source fit and final
chemical utility, with fixed-tree and Gaussian controls. Scale/covariance matching
is relevant before topology attribution. See NEXT for constraints. No additional
protocol or jobs are currently frozen/queued. The user's broad authorization
permits useful bounded continuation without another permission question.

Keep722 reserved outcomes, old12/18 cohorts,2,560 orbit outputs and all6,144 tree
outputs outside fitting. Preserve bond-free OMol25, max_atoms200, original charge/
spin, separate environments and no home writes. Earlier failed routing, collision
and latent-mass sweeps remain closed. Existing data is enough for the initial
tests; no additional user data is currently needed. The ICLR goal is active and
unachieved, and the manuscript is not submission ready.
