# Claude continuation — monomer assay qualified, learned-source gain still absent

CURRENT OVERRIDE (2026-09-14): source-utility method IMPLEMENTED; controlled jobs
46434075_0/1 RUNNING from immutable commit0ad2193. Read
`research/SOURCE_UTILITY_STATE_20260914.json`, `notes/source_utility_method_v1.md`
and `research/evidence/source_utility_protocol_v1.json`. Prior “not implemented”
statements below describe the completed monomer checkpoint, not current work.
There is no new validated utility gain yet.24 FIT plus12 source-head held-out
compositions come from the real TRAINING corpus; do not confuse this split with
the older21 composition-disjoint evaluation molecules.9216 planned attempts,
actual/shuffled/NLL/fixed controls, two fixed decoders, no oracle calls. Re-query
Slurm; finish and audit this bounded run before proposing another method. Generic
frozen-decoder noise learning is prior art (Noise PPO); originality remains open.


Checkout `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`, branch
`iclr2027-arch-fix`. Read `research/NEXT.md`, `research/MONOMER_STATE_20260914.json`,
`notes/monomer_results_20260914.md` and `notes/source_utility_learning_brief_v1.md`.

The prospective single-molecule task is now explicit: neutral singlet organic
structures8-40 atoms. Metadata/reference criteria were fixed before generation.
Of78 eligible original references,31 qualified; fixed hashes selected21 in size
bins5/8/8. All21 reference assays pass. No equal elemental-count vector occurs
in3,902,107 train or39,415 legacy-val records from checksum-verified processed
corpora. Fast hashes are followed by exact checks, with deliberate collisions
and real-corpus positive controls. This does not certify arbitrary unrecorded
pretraining or complete parent-trajectory relationships.

All10,752 outputs are COMPLETE and audited. First Gaussian/shell/harmonic/node/pair
counts/1,344 are581/627/575/603/561; second Gaussian/shell/harmonic521/557/598.
Node/pair source models exist only for the first continuation. Node minus shell
is-1.79 pp[-5.21,1.64]; pair minus shell-4.91 pp[-8.41,-1.41]. No learned increment
is established. Shell geometric gains repeat, but its paired graph-gain intervals
span zero; keep the descriptive composition intervals too. No per-seed winner
selection or post hoc removal of failed outcomes. The previous static-context
adapter remains a negative result and is not scaled.

The implemented candidate is a learnable explicit-density source without supplied
bonds. It has more method content than only adding a loss, but generic tree models,
harmonic priors and latent conditioning have prior art. ICLR-level originality and
usefulness are not established. Do not say the new benchmark produced a neural win.

Next is one hypothesis: learn source probabilities from actual sample utility with
the decoder frozen. The brief gives the source-space importance identity and an
optional analytic tree-KL bound on raw output distribution shift under a common
fixed kernel. It is mathematical/design work only, not an implemented or successful
algorithm. Validate related work, support/estimator/trust assumptions and a bounded
FIT-bank protocol first. Use actual/shuffled utility and unadapted controls; an
objective comparison also needs an NLL source under the same frozen decoder.
Do not fit any old evaluation data or assert that the old NLL objective is proven
to be the unique cause of failure. User authorization permits bounded continuation
without another permission question.

New infrastructure: `cfm_mol/source_checkpoint.py` restores the exact saved source,
including node/pair priors;17 targeted restoration/tree tests pass. The evaluation
runner's optional checkpoint-prior path leaves older default protocols unchanged.
Jobs46415396_0/1 completed0:0 in34:26/20:28. Re-query Slurm. All source draws, reference
assays, training-row matching and structural readouts replay; optimizer trajectories
and final neural integration are not independently rerun. No new training or oracle
queries occurred. Total tree/monomer outputs are30,208; earlier tree-energy calls5120.

Protect722 reserved outcomes, old evaluated cohorts,2,560 orbit outputs,30,208
new-stage outputs,32 older references and78 monomer-pool references from fitting.
Keep bond-free OMol25, max_atoms200, original charge/spin, separate environments and
no home writes. The noisy final flow has no qualified density, ESS or Boltzmann law.
All old routing/collision/latent-mass/static-context sweeps stay closed. The internal
manuscript and abstract preserve the complete evidence; they are not submission
ready. User handles authorship/submission; ICLR goal active and unachieved.
