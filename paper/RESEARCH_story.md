# RESEARCH_story.md — how neighbouring papers organise their story, and what BGFM should copy

Prepared for the BGFM/HBFM ICLR 2026 writing pass. Everything below is read from the
actual papers (arXiv HTML / abs pages), August 2026. Where I could **not** verify a claim
from primary source, it is flagged `[UNVERIFIED — must check before citing]`.

Headline lesson, stated once up front:

> **None of the strong papers in this neighbourhood wins a crowded leaderboard.**
> They win by (i) naming a regime where the incumbents are structurally inapplicable,
> (ii) owning a second, physics-grounded metric table that the incumbents never reported,
> and (iii) being conspicuously honest about what broke. All three moves are available to
> us with the results we already have.

---

## Part 1 — Per-paper structural summaries

### A. FlowMol3 — our backbone
arXiv:2508.12629 (Aug 2025) · Digital Discovery 2026 · https://arxiv.org/abs/2508.12629

**Intro = 6 paragraphs, and the last two do all the work.**
1. Stakes: "Deep generative models that can directly sample molecular structures with desired
   properties have the potential to accelerate chemical discovery…" — therapeutics/proteins/materials.
2. Scope narrowing: "In this work we focus on unconditional generation of 3D, organic, small,
   drug-like molecules." + justification that the unconditional task is foundational.
3. Historical arc: SMILES → 2D graphs → 3D.
4. Problem diagnosis: diffusion changed the landscape, **but** "generated molecules differ
   substantially from 'real' molecules" in validity, geometry, functional-group composition.
5. **Rejects the obvious answer.** "More recent works have argued that relying on simplified,
   well-tested transformer-style architectures and scaling the size of the model will be
   essential…" → they say no, the pathology is *distribution drift*.
6. Method reveal: three architecture-agnostic techniques (self-conditioning, fake atoms,
   train-time geometry distortion), then previews the wins.

**Contribution bullets: none.** Contributions live in prose + abstract (~4 claims). Notable
because it's a Digital Discovery/chemistry-venue habit; **ICLR expects bullets, so don't copy this.**

**Related work: no dedicated section.** Absorbed into intro ¶3–4 plus §3.5 "Comparison to
Existing Methods" (six baselines: MiDi, JODO, EQGAT-Diff, Megalodon, SemlaFlow, ADiT).

**Experiments order — setup-heavy, then results:**
§3.1 data → §3.2 metric *definitions* → §3.3 sampling protocol → §3.4 ablation *design* →
§3.5 baseline list; then §4.1 SOTA table → §4.2 ablation results → §4.3 **mechanism**
(trajectory analysis of *why* the features work) → §4.4 chemistry deep-dive.

**Comparability moves worth stealing:**
- Every baseline re-scored with "the same script in the FlowMol repository."
- Explicitly flags the one asymmetry: 5000 samples for all baselines, but "For ADiT we used a
  collection of 10 thousand molecules provided by the authors."
- Introduces *new* metrics (FG deviation, OOD ring rate, PoseBusters) rather than fighting on
  the saturated ones — i.e. changes the axis of evaluation.
- Differentiates on **mechanism**, not scale: competitors scale, we diagnose.

**Negative results, kept in the main text (this is the model for our force-term result):**
- "Phenol esters are an example of a functional group whose representation are made worse by
  the addition of self-correcting features."
- "FlowMol3 achieves the best performance across all evaluations … with the exception of the
  median energy change and RMSD from energy minimization."
- Discussion: "FlowMol3 does not eliminate all gaps between generated and real molecules…"
  and "the difference in chemistry between generated molecules and 'real' molecules is still
  practically significant."

**Capability matrix: none** (only a results table).

---

### B. ET-Flow — the standard conformer-generation write-up
arXiv:2410.22388 · NeurIPS 2024 · https://arxiv.org/abs/2410.22388

**Intro = 4 paragraphs (tight, ICLR-shaped):**
1. "Predicting low-energy molecular conformations given a molecular graph is an important but
   challenging task in computational drug discovery." — task in sentence one.
2. What existing ML does (diffusion over conformer fields; torsion-angle diffusion).
3. Diagnose their cost: "Early approaches based on diffusion faced challenges such as lengthy
   inference and training times as well as having lower accuracy compared to cheminformatics methods."
4. "In this paper, we propose Equivariant Transformer Flow (ET-Flow), a simple yet powerful
   flow-matching model … with minimal assumptions."

**Contribution bullets: 3.** Note the *types*:
1. SOTA precision → "resulting in more physically realistic and reliable molecules for practitioners."
2. **A mechanistic insight, not a number**: "We highlight the effectiveness of incorporating
   equivariance and more informed priors … in our simple yet well-engineered method."
3. Efficiency: fewer sampling steps than GeoDiff, far fewer parameters than MCF.

**Comparability moves:**
- Re-evaluates an unfairly-configured baseline: "GeoDiff initially utilized a limited subset of
  the GEOM-DRUGS dataset; thus, for a fair comparison, we consider its re-evaluated performance
  as presented in (Jing et al., 2022)."
- Foregrounds a parameter-count asymmetry in their own favour (8.3M vs MCF's 13M–242M).
- Volunteers a caveat *against* themselves on wall-clock: "While ET-Flow may not achieve the
  fastest raw inference times (potentially due to MCF variants benefiting from optimized CUDA
  kernels for attention)…"

**Limitations: explicit paragraph, three concrete weaknesses** — recall/diversity metrics are
weaker; they need "an additional chirality correction step"; on GEOM-XL they are only
"comparable to MCF-S and TorsionDiff."

**Related work: 4 subsections inside "Background"** — diffusion models, flow matching,
conformer generation, equivariant architectures for atomistic systems.

**Experiments order:** 4.1 setup → 4.2 ensemble RMSD (COV/AMR main table) → 4.3 coverage-threshold
plots → **4.4 ensemble properties via GFN2-xTB (E, μ, HOMO-LUMO gap, E_min)** → 4.5 inference-step
ablation → 4.6 sampling efficiency. Design ablations + GEOM-XL + OOD are in Appendix D.

**The single most transferable move in this whole document:** ET-Flow's Table 3 is a
*physics-grounded, xTB-computed ensemble-property table*, separate from the crowded COV/MAT
table. It is how they demonstrate quality on an axis the leaderboard doesn't cover. **Our
per-group Boltzmann r against independent GFN2-xTB is exactly this table.** Structure the paper
so that table is the centrepiece and COV/MAT is context.

---

### C. Torsional Diffusion — the "capability first" precedent
arXiv:2206.01729 · NeurIPS 2022 · https://arxiv.org/abs/2206.01729

Abstract sells a capability, not a leaderboard row: the model "provides exact likelihoods,
which we employ to build **the first generalizable Boltzmann generator**." Note the shape:
*exact likelihood* → *therefore* a new capability → *therefore* ensemble properties. No claim
that they beat everyone at RMSD. This is the licence for our Table 3 argument.

---

### D. Adjoint Sampling — our nearest neighbour (MLIP energy → sampler)
arXiv:2504.11713 · ICML 2025 (Meta) · https://arxiv.org/abs/2504.11713

**Intro = 5 paragraphs:**
1. "Sampling from complex, high-dimensional distributions underlies many important problems."
2. Traditional MCMC/SMC: "suffer from slow mixing and poor scalability to high-dimensional settings."
3. Why naive generative modelling fails: ground-truth data is "unavailable" in molecular
   simulation, so prior methods are "highly inefficient in terms of energy function evaluations."
4. Trace the SOC line of work and name the *precise* inefficiency: existing methods "require
   computationally expensive simulation of the diffusion process per gradient update" and
   "at least one—sometimes many—energy evaluations per gradient update."
5. "We introduce *Adjoint Sampling*, a novel and extremely efficient variational inference framework…"

**Contribution bullets: 4, and bullet 4 is the one to copy:**
1. Efficiency: "the first on-policy approach to allow far more gradient updates per model sample
   and energy evaluation."
2. Theoretically grounded: objective "implicitly projects the model onto a set of optimal controls"
   with no "corrective measures such as importance sampling."
3. Structure: graph + Euclidean symmetries, periodic boundary conditions for torsions.
4. **New benchmarks: "Amortized molecule sampling benchmarks that challenge new methods to be
   applicable at scale."** and, crucially, the justification-by-consequence: "Being successful at
   these benchmarks directly drives progress in chemistry."

→ **This is the canonical answer to "we can't win the existing benchmark": define the benchmark,
and justify it by scientific consequence rather than by novelty.**

**Related work: 5 method-family buckets** — Learning-Augmented MCMC / MCMC-reliant Diffusion
Samplers / SOC-based Diffusion Samplers / Off-policy Methods / **Molecule Conformer Generative
Models**. That last bucket is where they park the entire data-driven conformer literature as a
*different regime* rather than as competitors they failed to beat.

**Experiments:** §5.1 synthetic energies (DW-4, LJ-13, LJ-55) with a rigid
Baselines / Evaluation / Results sub-structure; §5.2 real conformers from a **neural energy
(eSEN)** on SPICE and GEOM-DRUGS with the same sub-structure. Metrics: Geometric-W₂, Energy-W₂,
Path-ESS. Reciprocal-projection ablation in-text, detail in Appendix H.4.

**Honesty style: no limitations section, but plain-spoken theory/practice gaps** — "we differ
slightly and use a replay buffer … and we do not perform step 2 until convergence. We find this
helps smoothen the optimization."

**Why this matters to us:** they use eSEN, the same MLIP family as our OMol25 potential, on a
*conditional* task (conformers of a given molecule). One sentence in our intro must state the
difference: they sample conformers of a fixed graph; we generate composition + geometry
unconditionally, bond-free, across 83 elements.

---

### E. Transferable Boltzmann Generators — the terminology weapon
arXiv:2406.14426 · NeurIPS 2024 · https://arxiv.org/abs/2406.14426

**Intro = 6 paragraphs:**
1. Generative models in the physical sciences → narrow: "In this work, we will focus on the
   latter for molecular systems."
2. Formalise the sampling problem; contrast with "sequential sampling algorithms" (MD/MCMC).
3. **Introduce and *enforce a distinction*:** Boltzmann Generators vs — "we refer to them as
   **Boltzmann Emulators**" for models that produce plausible structures without unbiased
   equilibrium sampling.
4. Justify CNFs: "Recently, flow matching has emerged as an alternative training method."
5. Name the gap: "Thus far, Boltzmann Generators have been limited by the requirement to train them
   [per system]."
6. "In this work, we present a framework for transferable Boltzmann Generators."

**Contribution bullets: 3.**
1. "To the best of our knowledge, we introduce the first transferable Boltzmann Generator."
2. "We outline a general framework for training and sampling with transferable BGs based on
   continuous normalizing flows."
3. **"We conduct several ablation studies to investigate the effects of different architectures,
   training set sizes, and biasing."** ← internal validity promoted to a headline contribution.
   Directly relevant: our Y-scrambling + force-ablation work is a contribution of this type.

**How they handle "we don't beat everyone" — they *disqualify by capability*, not by number:**
- vs Timewarp: "However, in contrast to our work, they do not generate uncorrelated samples."
- vs torsion-space methods: "they are unable to generate samples from the full Boltzmann
  distribution in Euclidean space."
- And the categorical separation: "Boltzmann Emulators are analogous to Boltzmann Generators,
  yet they are not designed to generate unbiased equilibrium samples."

**Limitations: explicit and forward-looking** — "Scaling transferable Boltzmann Generators to
larger systems remains for future work. Notably, this usually requires large amounts of
computational resources"; "Scaling to larger systems often involves coarsegraining, which
typically results in the loss of an explicit energy function"; "we did not pursue the impact of a
training set comprising a smaller number of peptides."

**Related work organised by axes**, not by chronology: application domain (molecules vs lattice
systems) → coordinate representation (internal vs Cartesian) → adjacent concepts (emulators,
molecular generation, dynamics) → historical progression into the gap.

**Experiments:** single-system sanity check (alanine dipeptide, validates the architecture) →
transferability (train 200 dipeptides → 8 held-out peptides, the main claim) → comparison to
Timewarp via Wasserstein. Ablations inside §5.2: biased vs unbiased training data, trajectory
length (50 ns → 5 ns → 500 ps), architecture variants (TBG / TBG+backbone / TBG+full).

**Steal this: terminology as a comparability device.** Coining a clean two-category distinction
lets you place the entire incumbent literature in a different box *without* claiming to beat it.

---

### F. Sequential Boltzmann Generators — the "previously intractable regime" close
arXiv:2502.18462 · ICML 2025 · https://arxiv.org/abs/2502.18462

Two-contribution abstract (architecture + inference-time scaling). Two moves to copy:

- **A negative/contrarian design choice sold as a principle:** "In contrast to the equivariant
  continuous flows of prior methods, we leverage exactly invertible non-equivariant architectures
  which are highly efficient during both sample generation and likelihood evaluation."
- **The closing sentence is a capability-frontier claim, not a win:** "…demonstrating the first
  equilibrium sampling in Cartesian coordinates of tri-, tetra- and hexa-peptides that were thus
  far intractable for prior Boltzmann generators."

---

### G. Autoregressive Boltzmann Generators — current (2026) convention in this exact subfield
arXiv:2606.27361 · https://arxiv.org/abs/2606.27361

**Contribution bullets: 4.** Bullet 2 is a *comparative finding* presented as knowledge:
"We investigate various proposal formulations, demonstrating that discrete binning offers
superior training stability and scalability compared to continuous mixture models." Bullet 4
introduces a named 132M transferable model with zero-shot generalisation.

**Limitations: explicit, specific, self-inflicted** — "While ArBG enables a drastically different
approach to BGs, it comes with a few notable limitations. Firstly, AR models impose a specific
ordering over dimensions, while molecules themselves do not possess a natural ordering… Secondly,
the use of uniform binning bounds the precision of the model by Δ…"

**Experiments:** single systems (§4.1) → transferable model (§4.2) → **ablations last (§4.3)**.
Table 1 is results (E-W₂, T-W₂), *not* a capability matrix.

Takeaway: in 2026 this subfield expects (a) 4 bullets, (b) a named limitations section with two
or three concrete, technical weaknesses, (c) ablations last.

---

### H. EDM / GeoLDM — the de-novo 3D generation template
- **EDM**, arXiv:2203.17003, ICML 2022 · https://arxiv.org/abs/2203.17003
  Abstract shape: *what it is* (E(3)-equivariant diffusion jointly over continuous coordinates and
  categorical types) → *extra capability offered as a bonus* ("In addition, we provide a
  probabilistic analysis which admits likelihood computation of molecules using our model") →
  *result claim*. Even EDM sells likelihood as a capability, not a benchmark.
- **GeoLDM**, arXiv:2305.01140, ICML 2023 · https://arxiv.org/abs/2305.01140
  Template = "**first** X" + one headline number + one extra capability: first latent DM for
  molecular geometry; point-structured latent space preserving roto-translational equivariance;
  up to 7% better validity on large biomolecules; "higher capacity for controllable generation."

These are the papers whose *tables* we inherit and whose *task* we share. They are also the
papers whose densities are uncalibrated — i.e. the "emulator" column of our capability table.

---

### I. Zatom-1 — the one published OMol25-trained generator (our only external Table-2 comparison)
arXiv:2602.22251 · also ICLR 2026 FM4Science workshop version ·
https://arxiv.org/abs/2602.22251 · https://github.com/Zatom-AI/zatom

What I could verify from the arXiv version:
- Unified multimodal flow foundation model over 3D **molecules and materials**; deliberately
  simplified Transformer with Query-Key Normalization; multimodal flow-matching objective over
  discrete atom types + continuous coordinates.
- Trained on **OMol25 (4M)** and MPtrj; claims 94% PoseBusters validity on QM9/GEOM-Drugs;
  competitive on MP20.
- **Compute framing as a selling point:** "jointly trained Zatom-1 (80M) achieves its results
  using ∼400 GPU training hours" vs "ADiT (180M), which requires ∼1,200 GPU pretraining +
  finetuning hours"; ~100 GPU hours of property-prediction finetuning.
- 3 contribution bullets ("first-of-its-kind foundation model for diverse chemical tasks";
  LeMat-GenBench materials results; the standardised-Transformer architecture).
- Reports positive **generative→predictive transfer** (materials pretraining helps molecular
  property prediction) and a PLATOM-1 O(3)-equivariant variant (1.6× faster convergence, 71%
  fewer parameters).
- **No dedicated Limitations section.** Nearest thing: "Zatom-1's novelty and SUN rates and model
  scaling benefits could likely be improved by training on a larger set of diverse 3D materials…"
- **No capability matrix.**

`[UNVERIFIED — must check before citing]` Two things our Table 2 depends on and that I could
**not** confirm from primary source:
1. An **OMol25 de-novo generation results table**. In the arXiv HTML I could reach, OMol25 appears
   in Appendix E for *energy and force prediction* (MLIP), not for generation metrics. The
   generation numbers in our fact sheet (RDKit-valid 30.4 / connected 17.0 / PoseBusters 15.1 /
   uniqueness 93.5) must be traced to a specific version + table number.
2. The **"non-converged, 80 epochs"** author statement. The OpenReview workshop PDF is behind a
   bot check and I could not read it; the arXiv version contains GPU-hour figures but no epoch
   count or non-convergence admission that I could find.
   → **Action for the writing agent:** locate the exact sentence and cite `version + §/Table`.
   If it exists only in the workshop version, say so explicitly ("the workshop version of
   Zatom-1, from which these numbers are taken, states …"). If it cannot be located, downgrade
   the wording to the verifiable form: *"at the training budget reported by the authors
   (∼400 GPU-hours for the jointly trained 80M model)"* — which is still a legitimate,
   citable caveat and is enough to keep us honest. Do **not** write "non-converged" without
   a locatable author statement; that would be an unsupported characterisation of someone
   else's work, which is a worse failure mode than a weaker claim.

---

### J. Energy-Weighted Flow Matching (EWFM) — how to claim a narrow win honestly
arXiv:2509.03726 · https://arxiv.org/abs/2509.03726

Energy-only training of CNFs by reweighting the conditional-FM objective via importance
sampling; iterative (iEWFM) and annealed (aEWFM) variants. Two lessons:
- **Pick one axis you dominate and quantify it**: "up to three orders of magnitude fewer energy
  evaluations."
- **Use "competitive" for everything else.** The abstract claims competitiveness with
  "established energy-only methods" rather than dominance. This is the register we should use
  for Table 2 outside the columns we actually win.

Also the closest *method*-level neighbour to our λ₂ term: both make the energy function the
training signal for a CNF without target-distribution data. Our differentiator is that we keep
L_FM on real data and add energy as a *consistency* term, and that we run at 83-element,
4M-molecule scale rather than LJ-55.

---

### K. "An evaluation of unconditional 3D molecular generation methods" — cite this for incomparability
arXiv:2505.00518 · https://arxiv.org/abs/2505.00518

This is the paper to cite whenever we refuse to paste someone else's number into our table.
Verified content:
- Benchmarks are saturated: "recent molecular generation methods report saturated benchmarks,"
  while established benchmarks (GuacaMol, MOSES) omit "3D assessments."
- Recommends geometry-based validation via **PoseBusters** (bond lengths/angles within ±25% of
  experimental, planar aromatics, steric clashes at 30% vdW overlap, **strain-energy ratio
  threshold 100 under UFF**) plus FCD and ECFP4 diversity.
- **Explicit cross-paper incomparability:** models use different splits — "EQGAT-diff, FlowMol,
  and SemlaFlow use the training … splits generated by Vignac et al. (2023) … GCDM and GeoLDM
  use the splits generated by Anderson et al. (2019)" — and different hydrogen handling: "GCDM
  and GeoLDM do not add all hydrogens without post-processing."
- Frames unconditional generation as "a stepping stone towards the conditional tasks."

Three ways we use it: (1) justify reporting PoseBusters; (2) justify re-scoring everything
ourselves with one script; (3) justify *not* reproducing a 22-row COV/MAT leaderboard.

---

## Part 2 — Cross-paper statistics (calibration for our own choices)

| Paper | Intro ¶s | Contribution bullets | Dedicated Related Work | Dedicated Limitations | Ablations placed |
|---|---|---|---|---|---|
| FlowMol3 | 6 | 0 (prose) | no (in intro + §3.5) | no (in Discussion) | §4.2, right after main result |
| ET-Flow | 4 | 3 | yes (4 subsections in Background) | **yes** | §4.5 + Appendix D |
| Adjoint Sampling | 5 | 4 | yes (5 method families) | no (theory/practice gaps in text) | in-text + Appendix H.4 |
| Transferable BG | 6 | 3 | yes (organised by axes) | **yes** | inside §5.2 |
| Sequential BG | — | 2 (in abstract) | yes | — | — |
| ArBG (2026) | — | 4 | yes | **yes, very specific** | **last (§4.3)** |
| Zatom-1 | ~3 | 3 | yes (by domain) | no | appendices |

**Calibration for BGFM:** 7–8 intro paragraphs (we have more to set up than any of these:
two literatures + an MLIP + a negative result + an evaluation-validity argument), **5 bullets**,
a dedicated Related Work organised by method family *plus* a capability matrix, and a dedicated
**Limitations** section (non-negotiable for us — it is where the Y-scrambling n=120 status lives).

---

## Part 3 — Recommended Intro paragraph-function sequence for BGFM

Target: ~1.25 pages, 8 paragraphs, ending in 5 bullets. Each paragraph has exactly one job.

**P1 — Stakes, stated as an ensemble problem (3–4 sentences).**
Do *not* open with "generative models are transforming chemistry." Open with the observation
that the object of chemical interest is a **Boltzmann ensemble**, not a single structure: free
energies, spectra, reactivity and most measurable properties are ensemble averages, so a
generator is only as useful as its density is calibrated. State the goal in one sentence: a
single generator whose learned density equals `p(x) ∝ exp(−E(x)/kT)`, across the periodic table.
*Model: Adjoint Sampling ¶1 + TBG ¶1.*

**P2 — The two literatures that do not meet (the gap paragraph).**
Half a paragraph each. (a) 3D molecular generators (EDM, GeoLDM, FlowMol3, Zatom-1) learn from
data, cover drug-like organics, usually need bond labels, and — the load-bearing clause — carry
**no calibrated density**; borrow Klein & Noé's vocabulary and call them *emulators*.
(b) Boltzmann generators and diffusion samplers (TBG, SBG, ArBG, Adjoint Sampling, EWFM) do have
tractable densities but operate per-system or on small peptides, in torsion or Cartesian
coordinates, and do not generate composition. Close with one sentence naming the gap: *no
generator is simultaneously de-novo, periodic-table-scale, and Boltzmann-calibrated.*
*Model: TBG ¶3+¶5 (the emulator/generator split, then the gap).*

**P3 — What changed and made this possible now.**
Universal MLIPs (OMol25 / eSEN) supply a transferable, differentiable `E` and `F` at
near-DFT quality over 83 elements. That converts "energy supervision" from a per-system
luxury into a **data-scale training signal**. Cite Adjoint Sampling as the nearest use of an
MLIP energy for sampling, and immediately state the difference in one sentence: *they sample
conformers of a given molecule; we generate composition and geometry unconditionally,
bond-free.* Handling the nearest neighbour explicitly and early is what stops a reviewer
writing "this is Adjoint Sampling with a different backbone."
*Model: Adjoint Sampling ¶3–4 (name the precise inefficiency of the prior regime).*

**P4 — The formulation, in prose plus one inline equation.**
`L = L_FM + λ₁ L_force + λ₂ L_energy`. One clause each: the FM velocity admits a **closed-form
score read-out** `s_θ(x_t,t) = (t·v_θ − x_t)/((1−t)σ²)`, giving a *local gradient* condition
against `F/kT`; the variance form `Var(log p + E/kT)` gives a *global shape* condition in which
the partition function cancels, so no `Z` is ever needed. One sentence on the theorem
(joint optimum ⇒ model density = Boltzmann density, Appendix A). One sentence on the cost
(FFJORD reverse-time ODE + Hutchinson divergence, second-order and memory-hungry) — naming the
cost here buys credibility for the stability section later.

**P5 — What we actually found, including the surprise (the differentiating paragraph).**
This paragraph is why the paper gets accepted. Three sentences, three findings, with numbers:
(i) the **global** energy term works — per-group Boltzmann r rises 0.075 → 0.430 (Δ = +0.355,
Welch t ≈ 10.8); (ii) the **local** force term, the term anyone would reach for first,
*degrades* calibration (0.223 → 0.109 alone; energy+force 0.294 < energy-only 0.420); (iii) a
label-shuffling negative control shows roughly two thirds of the energy gain is generic density
regularisation and one third is Boltzmann-specific pairing information. Say plainly that we
report (ii) rather than dropping it.
*Model: FlowMol3 ¶5 (reject the obvious answer) + ArBG bullet 2 (a comparative finding is a
contribution) + FlowMol3's phenol-ester honesty.*

**P6 — Evaluation validity, and the incomparability disclosure.**
Define the primary metric — per-group Boltzmann r against an **independent** GFN2-xTB reference
(independent of the OMol25 potential we trained against; this is the single strongest
methodological point we have, so say the word "independent"). Say in one clause why grouping is
mandatory (pooling across molecules mixes per-molecule log Z and produces a Simpson's-paradox
artefact). Then the disclosure, in our own voice before a reviewer says it: GEOM COV/MAT is
reported as **context only** because those baselines solve a different, conditional problem;
and no number is copied across papers because splits, hydrogen handling and sample counts
differ (cite arXiv:2505.00518). Name our actual comparison set: Zatom-1 (at its authors'
reported budget) and a matched, converged bond-free FlowMol3 control we trained ourselves.
*Model: ET-Flow's volunteered caveats + FlowMol3's uniform-scoring-script paragraph.*

**P7 — Scope and transfer, with the wording discipline enforced.**
tmQM result (force-only −0.158 → energy +0.365, median +0.429, 37% of molecules r > 0.5,
270/270 xTB single points converged so there is no survivor bias). In the same breath, the
honest scope clause: these transition metals **are** in OMol25 training data, in its sparse tail
(0.34% of atoms; 1.5×10⁴–5.1×10⁴ atoms per metal), so this is transfer under **distribution
shift** to a different TM corpus — *not* zero-shot generalisation to unseen elements. Putting
the caveat in the intro rather than the appendix is what makes the claim believable.

**P8 — Contribution bullets** (below).

---

## Part 4 — Contribution bullet draft (5 bullets)

Style rules taken from the survey: ≤2 sentences each; lead with a bolded 2–4-word type label
(Adjoint Sampling does this: *Efficiency / Theoretically Grounded / Structure / New Benchmarks*);
put a number in every bullet that has one; put the negative result in a bullet of its own so a
reviewer cannot claim it was buried. All placeholder numbers wrapped in `\pending{}`.

> **Formulation and guarantee.** We introduce Boltzmann-Guided Flow Matching (BGFM), a bond-free
> 3D generator trained against a universal neural potential, in which the flow-matching velocity
> field is read out in closed form as a score to impose a *local* gradient condition against DFT
> forces, and a variance-form energy objective imposes a *global* shape condition in which the
> partition function cancels. We prove that at the joint optimum the model density equals the
> Boltzmann density (Theorem ~1, Appendix A).

> **Energy supervision measurably calibrates the density.** Across 120 held-out molecules and
> four seeds per arm, energy supervision raises the per-group Boltzmann correlation from
> r = 0.075 ± 0.004 to r = 0.430 ± 0.033 (Δ = +0.355, Welch t ≈ 10.8), scored against
> GFN2-xTB — a reference *independent* of the potential used for training.

> **A negative result that should change practice.** Force/score matching — the local condition
> most naturally suggested by the score read-out — *degrades* Boltzmann calibration rather than
> improving it, both alone (r = 0.109 vs 0.223 for the physics-free control) and in combination
> (r = 0.294 for force+energy vs 0.420 for energy alone), and the same ordering reproduces on
> transition-metal complexes (−0.158 vs +0.365). We report and analyse this rather than
> presenting only the winning configuration.

> **Mechanism isolated by a negative control.** Shuffling the geometry↔energy pairing *within*
> each parent molecule — which leaves the energy set and the within-parent energy spread exactly
> unchanged (77.6 kcal/mol in both arms) and alters only the correspondence — retains ≈63% of the
> gain, attributing that share to generic density regularisation and the remaining ≈37%
> (Δ ≈ \pending{+0.13}, t ≈ \pending{3.9}, p < \pending{0.05}) to Boltzmann-specific information.

> **A reproducible failure mode, diagnosed and fixed.** Single-parent estimation of the energy
> objective has such high variance (observed jumps 48 → 974) that gradient spikes drive weights
> to NaN in 3 of 9 seeds, and a conventional NaN guard silently zeroes the loss and continues
> training — masking the failure for a GPU-day. Increasing the parent batch to 8, capping the
> energy loss, and aborting on non-finite weights yields 2/2 stable seeds.

*Optional 6th bullet, only if page budget allows — otherwise fold into §8:*

> **Generation quality and a capability frontier.** On bond-free OMol25 de-novo generation BGFM
> exceeds the only published generator trained on the same corpus on every reported column
> (RDKit-valid 43.0 vs 30.4, connected 32.0 vs 17.0, PoseBusters 23.0 vs 15.1, uniqueness 100 vs
> 93.5), and is the only model in that comparison for which an exact log-density — and therefore
> a Boltzmann-calibration measurement at all — is defined.

---

## Part 5 — Sentence templates for honest incomparability (paste-and-edit)

**T1 — the context table (GEOM COV/MAT, fact [P4]).**
> "Table 1 places our numbers alongside the conditional conformer-generation literature for
> context only. Those methods receive the molecular graph and are asked to produce its
> low-energy conformers; BGFM receives nothing and must produce composition, geometry, and
> (post hoc) bonds jointly. COV and AMR are defined per reference molecule and therefore score a
> strictly conditional problem; a direct ranking would be uninformative in either direction and
> we make no such claim."

**T2 — the capability table (Table 3, fact (3) of the wording discipline).**
> "Entries marked N/A in Table 3 are not failures. Those methods do not define a tractable exact
> log p_θ, so the quantity in that column cannot be computed for them at all. Table 3 therefore
> states which questions each model class can be asked, not which model is better."

**T3 — protocol incomparability / why we re-scored everything.**
> "We do not transcribe numbers across papers. Published results in this area differ in
> train/test split, hydrogen handling, and sample count (Zhang et al., 2025), so every entry in
> Table 2 was recomputed by us with a single scoring script on n = 100 samples per model. Where a
> number is author-reported we mark it as such and state the training budget the authors report."

**T4 — naming the regime instead of claiming victory (the Klein–Noé move).**
> "Following the distinction drawn for Boltzmann generators (Klein & Noé, 2024), we call a model
> a *structure emulator* if it produces chemically plausible geometries without a calibrated
> density, and *Boltzmann-consistent* if its density provably tracks exp(−E/kT). Essentially all
> 3D molecular generators are emulators by construction. The contribution of this paper is not to
> out-emulate them, but to make a periodic-table-scale de-novo generator Boltzmann-consistent."

**T5 — the honest scope of our baseline set (fact (2)).**
> "Our external comparisons are deliberately narrow, and we state their scope rather than imply
> generality: one published generator trained on the same corpus (Zatom-1, at the training budget
> its authors report) and one matched, fully converged bond-free FlowMol3 control that we trained
> ourselves under identical data, schedule, and compute. We do not claim superiority over the
> broader 3D-generation literature, whose models are trained on different corpora with bond
> supervision and evaluated under different protocols."

**T6 — distribution shift, not unseen elements (fact (1) — the hardest line not to cross).**
> "We stress that these transition metals are present in the OMol25 training data, albeit only in
> its sparse tail: 0.34% of all training atoms, between 1.5×10⁴ and 5.1×10⁴ atoms per metal. The
> tmQM experiment therefore measures transfer under distribution shift to a different
> transition-metal corpus, and we make no zero-shot claim with respect to element coverage."

**T7 — the negative result as a contribution, not an apology (fact (4)).**
> "We report the force term's failure in full because the negative result is the informative one:
> the local gradient condition is the term a reader would adopt first, and our evidence is that
> at 83-element scale it trades away precisely the global calibration the energy term buys."

**T8 — the statistically-honest secondary effect (fact [C]; pairs with the placeholder rule).**
> "We separate the two claims by strength. The primary effect — energy supervision versus no
> physics — is large and unambiguous (Δ = +0.355, t ≈ 10.8). The decomposition into
> regularisation and Boltzmann-specific components rests on a second-order contrast that is
> sensitive to the number of evaluation groups: at n = 120 it is not statistically significant
> (Δ = +0.092, t = 1.77), and the n = \pending{240} figures reported above are
> \pending{provisional}. We therefore state the decomposition as a hypothesis supported by a
> consistent point estimate, not as an established result."

---

## Part 6 — Recommended section order, and where each result lives

Rationale: for us the ablations **are** the main result, so the ArBG/ET-Flow convention of
ablations-last is wrong. Use FlowMol3's convention (main table, then ablation, then mechanism)
and promote mechanism to a top-level section.

| § | Content | Precedent |
|---|---|---|
| 1 | Intro (8 ¶s + 5 bullets, per Part 3–4) | — |
| 2 | Background: flow matching, score read-out, MLIPs. Keep short. | ET-Flow "Background" |
| 3 | Related work in 5 method-family buckets, each ending in a one-sentence "how BGFM differs" clause; **capability matrix here or in §8** | Adjoint Sampling §4 buckets + TBG's disqualify-by-capability clauses |
| 4 | Method: three losses, the theorem statement, the FFJORD cost | — |
| 5 | **Evaluation protocol as its own section**: per-group Boltzmann r, the independent-xTB argument, why grouping is required, the incomparability disclosure (T1/T3) | FlowMol3 §3.2 defining metrics before results |
| 6 | Main result: energy supervision works ([A]) | FlowMol3 §4.1 |
| 7 | **Negative results and mechanism** ([B] force harmful, [C] Y-scrambling, [G] decomposition) — flag in the first line that §6–7 are the core of the paper | FlowMol3 §4.2–4.3; TBG contribution bullet 3 |
| 8 | Generation quality + transfer: Table 2 vs Zatom-1 + our own control ([E]), tmQM ([D]), capability table ([Table 3] with T2 caption) | ET-Flow §4.4 xTB ensemble properties |
| 9 | Numerical stability as an engineering contribution ([F]) | Adjoint Sampling's plain-spoken practice notes |
| 10 | **Limitations** — explicit, 3–4 concrete items in ArBG's register (see below) | ET-Flow, TBG, ArBG |
| 11 | Conclusion (short) | — |

**Limitations section content (write these, do not soften them):**
1. The Y-scrambling decomposition is a second-order effect and is not significant at n = 120
   (t = 1.77); the reported n = \pending{240} contrast is \pending{provisional}.
2. The force/score term is harmful in our setting; we characterise the effect empirically but do
   not have a complete explanation (candidate: the t→1 divergence of the score read-out combined
   with per-atom clamping biases the local term).
3. Energy supervision is expensive and was numerically fragile until stabilised; we report the
   fix, not a proof of robustness.
4. Baseline coverage is narrow (see T5); Symphony / EDM / GeoLDM retrains on OMol25 are
   \pending{in progress} and the corresponding Table 2 entries are placeholders.
5. Element coverage is 83 elements *in the training distribution*, with heavy-element performance
   measured only under distribution shift (T6).

---

## Part 7 — Designing our Table 3 (the capability matrix)

**Important finding from the survey:** *none* of the papers I read has a literal checkmark
capability matrix. TBG and Torsional Diffusion make the identical argument **in prose**. So
building one is (a) a genuine differentiator and (b) low-risk — provided the caption carries T2,
because an unlabelled checkmark grid reads as a performance claim and would violate wording
discipline (3).

Proposed columns (each must be a *capability*, verifiable from the cited paper, never a score):

| | elements | needs bond labels | unconditional de-novo composition | tractable exact log p_θ | trained against a transferable energy | amortised across molecules (no per-system retraining) | Boltzmann calibration reported |
|---|---|---|---|---|---|---|---|
| EDM / GeoLDM | ~5 organic | yes/implicit | yes | yes (NLL) | no | yes | no |
| FlowMol3 (and our bond-free control) | ~10 organic | yes (0 for our control) | yes | approximate | no | yes | no |
| Zatom-1 | OMol25 coverage | no | yes | not reported | no (generative FM) | yes | no |
| Torsional Diffusion | organic | yes (graph given) | no (conditional) | yes (exact) | no | yes | yes (ensemble) |
| Transferable BG | peptide alphabet | n/a | no | yes (exact) | classical FF | yes (within peptides) | yes |
| SBG / ArBG | peptide alphabet | n/a | no | yes | classical FF | partly | yes |
| Adjoint Sampling | organic (SPICE/DRUGS) | graph given | no (conditional) | yes | **yes (eSEN)** | yes | yes |
| **BGFM (ours)** | **83** | **no** | **yes** | **yes** | **yes (OMol25)** | **yes** | **yes** |

Only the bottom row has the full pattern. That is the whole argument, and it is a *capability*
argument — exactly what T2's caption must say. **Every cell must be checkable against the cited
paper; a single wrong checkmark in a table like this is the kind of thing a reviewer verifies
and it would cost more credibility than the table earns.**

---

## Part 8 — Abstract shape (bonus, derived from the same survey)

Best-performing pattern across SBG / GeoLDM / Adjoint Sampling:
1. One sentence of stakes, phrased as the ensemble problem.
2. One sentence naming the gap ("emulators have no calibrated density; Boltzmann generators do
   not compose molecules").
3. Two sentences of method, including the `Z`-cancellation clause and the guarantee.
4. **One sentence of headline number** (0.075 → 0.430, t ≈ 10.8, independent xTB reference).
5. **One sentence of the negative result** — unusual in an abstract, and precisely why it will be
   remembered.
6. One closing capability-frontier sentence in SBG's register: e.g. *"…yielding, to our
   knowledge, the first de-novo 3D molecular generator with a measurable Boltzmann calibration
   across 83 elements."* (Check "first" carefully before committing.)
