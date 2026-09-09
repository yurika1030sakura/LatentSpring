# Global-refresh SMC control

Previous SMC mutations used only local random-walk or Langevin proposals.
Even accurate weights and preserved initial ancestry do not guarantee movement
between distinct molecular arrangements. This control adds an independence-MH
proposal from the fixed, normalized initial proposal q0 at each bridge.

Write r(x)=log gamma(x)-log q0(x) and gamma_b=q0^(1-b) gamma^b.
For y sampled from q0, the independence proposal ratio is

    log acceptance ratio = b [r(y)-r(x)].

Detailed balance follows from equality of the accepted fluxes
min(gamma_b(x) q0(y), gamma_b(y) q0(x)). A subsequent MALA step preserves the
same bridge, so their composition is invariant even if the composition is
not itself reversible. The standard AIS/SMC weighting therefore still applies.
This is established independence Metropolis combined with local MCMC, not a
new sampling identity. Relevant precedent includes
https://pmc.ncbi.nlm.nih.gov/articles/PMC8915891/ .

The fixed experiment uses 64 particles, 16 annealing stages and two moves per
stage. Compare two local MALA moves with one global independence move followed
by one MALA move, for both the confinement Gaussian and defensive symmetry
proposals. Each arm uses 64*(1+16*2)=2,112 potential calls per seed, plus the
separately reported common FM pilot. Seeds 9031--9033 and all failures remain.
No online proposal or temperature tuning is introduced.

Initial ancestry is deliberately not relabeled as independence after an
accepted global move. Separate last-global-proposal IDs count distinct refresh
events; acceptance depends on the existing chain state, so these IDs are not
effective independent sample counts. Report normalizer estimates, observable
changes, acceptance and geometry as well as ancestry and weight ESS.

Analytic tests check Gaussian target moments/normalizers for independence and
hybrid kernels, the stationary all-accepted limit, proposal accounting and
rejection of an unmatched one-move hybrid configuration. The independent
triatomic reference and existing simple-Gaussian controls remain comparison
targets. Passing these tests does not establish a molecular advantage.
