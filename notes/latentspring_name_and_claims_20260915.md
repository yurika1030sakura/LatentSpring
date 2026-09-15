# LatentSpring

Working method name: **LatentSpring**.

Paper title: **LatentSpring: Physics-Informed Molecular Flow Matching from Atomic Composition**.

“Latent” refers to unobserved spatial dependence trees, not a claim that the fixed
source affinities are learned chemical bonds. “Spring” refers to the Gaussian
harmonic source and physically motivated atomic scales. The method retains flow
matching; it is not a new force field or a proven global Boltzmann generator.

Use the name for the framework and specify the evaluated variant:

| Paper description | Existing immutable experiment identifier |
|---|---|
| Isotropic Gaussian reference | gaussian |
| LatentSpring harmonic source | harmonic_tree |
| LatentSpring with force update | escort_delta |
| LatentSpring with complete-work update | work_delta |

Do not rename stored models, protocols or source-code modules to change their
provenance. The canonical working manuscript remains `paper/tree_working.tex`.
A name search did not surface a matching molecular/ML method named LatentSpring;
this is a working research label, not a claim of globally unique usage.

Compared with the original BGFM loss proposal, the current research has a clearer
specified source, audited probability/physics boundaries, controlled source
comparisons and a tested physics-learning extension. This is an improvement in
research substance and evidence. Current Gaussian-source gains are not a direct,
matched performance estimate against the original BGFM model.

The method-level contribution is the particular molecular source and learning
construction plus its demonstrated utility. Random-tree priors, self-conditioning,
force guidance, escorted work/Jarzynski, energy-weighted FM and task arithmetic
are established ingredients. Applying sophisticated physics is not itself proof
of new physics or sufficient ICLR novelty. Work-specific superiority and global
thermal sampling must not be inferred from a favorable generic physics update.
