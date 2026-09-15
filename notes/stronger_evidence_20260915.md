# Focused follow-up after the completed manuscript

The user authorized the proposed physical-quality comparison and a bounded test
of the remaining Jarzynski learning bottleneck. The published manuscript and all
existing generator checkpoints remain frozen while these experiments run.

## First experiment: physical quality of matched generators

Reuse all 16,384 primary 128-call outputs of the four unpretrained EGNN methods,
with their original two initialization seeds and all 64 compositions. No new
training, generation, selection by energy, or geometry optimization is involved.
The protocol is frozen before any physical readout. eSEN scores every output and
its reflection; independent GFN2 scores every graph-valid output and both
reference parities. All attempts remain in joint validity/quality denominators.
There are 33,024 planned eSEN evaluations and 2,198 GFN2 attempts, including
reference checks duplicated across seed reports for independent provenance.

Primary reporting uses full prespecified energy and force threshold curves,
with graph validity as a required joint event. This avoids turning a selected
valid subset's lower mean energy into a claim of higher usable-output yield.
Reference-relative energy compares the same composition, not necessarily the
same chemical graph; it is not an isomer-specific strain energy. Means on each
method's own valid subset and their coverage are secondary. No artificial
pairing of diffusion and FM output indices will be used for conditional means.

## Work-weight diagnostic

Inspection of all sixteen existing teacher files found substantial deviations
from the anchor's linear energy model: within-file median residuals were about
0.47--0.66 eV, with broad within-file distributions at kT=0.025852 eV. Typical
positive secant curvatures were about 41--64 eV/A^2 and source widths about
0.011--0.015 A. These are exploratory FIT-only diagnostics and suggest testing
curvature-aware affine escorts. They do not by themselves establish the cause
of low ESS or a benefit of a new method. The target restraint must stay fixed;
any new map must include its full density/Jacobian correction and independent
production draws. Pilot energy/force queries must be counted in equal-cost
comparisons. Exact quadratic checks and a molecular pilot precede any neural
training claim.

The new campaign may improve the evidence, but strong acceptance remains a
review judgment, not an experimental stopping condition or promised outcome.
