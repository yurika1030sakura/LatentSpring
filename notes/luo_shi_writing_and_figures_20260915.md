# Luo and Shi: scientific writing and figure review

We identified 32 AI-for-science work families using the authors' websites, Mila, coauthor publication lists and bibliographic discovery. Conference/preprint duplicates were grouped. The review covers 29 paper PDFs, one full article XML, and two primary-author overviews where full text was inaccessible (RDE and OmegaFold). The per-paper JSON records access depth and document hashes. A lecture deck mentioning OmegaFold was detected and excluded as a primary paper.

The review focuses on how each paper defines the task, names its model inputs and outputs, motivates its components and presents molecular evidence. It does not independently reproduce their experiments.

## Changes applied to LatentSpring

1. Explain the scientific task before notation: atomic identities and electronic state are fixed; coordinates are generated.
2. Separate task input, random source, neural input, neural prediction and returned structure. The network reads current coordinates, flow time and composition, and predicts a movement vector for each atom.
3. Keep learned velocity, physical force and statistical work distinct. Their meanings, stages and units differ.
4. Use our actual unoptimized molecular structures. The main diagram shows a recorded flow, the gallery shows distinct inferred connectivities from one composition, and the rotor figure shows the atoms that rotate.
5. Give every arrow a causal role: sampling coordinates, predicting movement, producing physical targets, fitting students, or combining parameter updates.
6. Show component benefits near the claims, with full controls and budgets in the appendix. FAFE illustrates why the issue is the problem solved and the evidence, not whether a contribution is a loss or a network.
7. Keep a shared element palette, restrained module colors, readable labels and consistent molecular cameras. Provide static process snapshots in the PDF and actual recorded steps in the supplementary animation.
8. Preserve scientific differences between tasks: graph-conditioned conformer models, pocket-conditioned growth, sequence design and composition-conditioned coordinate generation have different inputs and outputs.

## Per-paper interface and visual lesson

| Work | Input | Learned output / returned object | Figure or writing lesson |
| --- | --- | --- | --- |
| [confgf](https://proceedings.mlr.press/v139/shi21b/shi21b.pdf) | Chemical graph, noisy atom coordinates and noise level | Coordinate score vectors; Langevin sampling returns conformations | A small molecule makes the chain-rule conversion and per-atom arrows concrete. |
| [cgcf](https://arxiv.org/pdf/2102.10240v1) | Chemical graph and Gaussian latent distances | Distance samples, reconstructed coordinates, then energy-based refinement | Label representations between stages and show the molecule after each conversion. |
| [dgsm](https://proceedings.neurips.cc/paper/2021/file/a45a1d12ee0fb7f1f872ab91da18f899-Paper.pdf) | Chemical graph and current spatial coordinates | Coordinate scores using local and changing spatial neighborhoods | Show concrete molecular systems and the changing neighborhood that motivates the model. |
| [confvae](https://proceedings.mlr.press/v139/xu21f/xu21f.pdf) | Chemical graph and latent variables; coordinates provide training targets | Conformations through differentiable distance reconstruction | Place actual molecular inputs and outputs around the nested optimization diagram. |
| [sbdd](https://arxiv.org/pdf/2203.10446) | Protein-pocket atoms and a partial generated molecule | Spatial atom-occurrence probabilities and sequential atom placements | Repeated atom-level snapshots expose what changes at each generation step. |
| [pocket2mol](https://proceedings.mlr.press/v162/peng22b/peng22b.pdf) | Protein pocket and partial molecular structure | Frontier, relative position, atom type and bond-type predictions | Number the stages and distinguish location prediction from type/bond prediction. |
| [diffab](https://proceedings.neurips.cc/paper_files/paper/2022/file/3fa7d76a0dc1179f1e98d1bc62403756-Paper-Conference.pdf) | Antigen and antibody framework context | CDR sequence and structure through coupled diffusion | Explain the biological object and local geometric variables before the architecture. |
| [rde](https://openreview.net/forum?id=_X9Yl1K2mD) | Protein local environment and residue/mutation information | Rotamer densities, then mutation-effect estimates | A global complex view and a local rotamer zoom connect density to a physical interpretation. |
| [ebmfold](https://arxiv.org/pdf/2105.04771) | Protein geometric context and perturbed coordinate/distance representations | Denoising correction fields used for backbone sampling | Show the molecule, its numerical representation, and the predicted correction together. |
| [omegafold](https://www.biorxiv.org/content/10.1101/2022.07.21.500999v1) | A single amino-acid sequence | A predicted protein structure through language and geometric modules | Make the input restriction visible and put a concrete structure at the output. |
| [oagnn](https://arxiv.org/pdf/2201.13299) | Protein coordinates and local residue orientations | Geometric features for graph-level or residue-level tasks | Use a global protein view plus an atomic coordinate-frame zoom. |
| [pepflow](https://arxiv.org/pdf/2406.00735) | Protein receptor context and multimodal noisy peptide representations | Peptide sequence, backbone geometry and side-chain variables | Color and label each generated variable rather than collapsing all channels into a generic latent. |
| [pephar](https://arxiv.org/pdf/2411.18463v1) | Target protein context, optionally supplied hotspot information | Hotspots, growing peptide fragments and an assembled peptide | Tie each stage to a concrete problem: binding sites, bond geometry and assembly. |
| [fafe](https://proceedings.mlr.press/v235/wu24g/wu24g.pdf) | Protein sequences and predicted/reference residue frames during training | Improved complex structures after loss-based fine-tuning | Pair an actual structural failure with the mathematical mechanism and its measured effect. |
| [msm_mut](https://proceedings.neurips.cc/paper_files/paper/2024/file/5828b7516d6f117e8301120519366cdb-Paper-Conference.pdf) | Protein local structure motifs and a mutation query | Retrieved motif features and mutation-effect predictions | Zoom from full structure to the local learned unit, then show the data flow. |
| [chemprojector](https://arxiv.org/pdf/2406.04628) | An input molecular graph | A synthesis-path representation and a synthesizable molecular analogue | Separate the scientific search space, encoded representation and final molecule. |
| [synformer](https://arxiv.org/pdf/2410.03494) | Molecular context when supplied, and reaction/building-block libraries | Synthetic pathways and their resulting molecules | Show both the token sequence and the chemically meaningful output it encodes. |
| [prexsyn](https://arxiv.org/pdf/2512.00384) | Property queries, including logical compositions of conditions | Synthesis-path tokens and property-conditioned molecules | Expose conditioning, data generation, network prediction and oracle feedback as separate stages. |
| [antibody_pnas](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8931377/fullTextXML) | Antibody-antigen structures and candidate mutations | Predicted mutation effects and experimentally tested antibody variants | Separate the computational model diagram from the experimental evidence panels. |
| [graphaf](https://arxiv.org/pdf/2001.09382) | Partial molecular graph and Gaussian latent variables | New atom and bond types in an autoregressive graph sequence | Use numbered stages and a legend for distinct kinds of edges and operations. |
| [graph2graphs](https://arxiv.org/pdf/2003.12725) | A product molecular graph | Reaction centers, synthons and reactant molecular graphs | Use a real reaction to make a one-to-many output interface obvious. |
| [mars](https://arxiv.org/pdf/2103.10432) | A molecular graph and property objectives | Proposed graph edits and accepted molecular candidates | Show proposal, evaluation and acceptance as distinct operations. |
| [nerm](https://arxiv.org/pdf/2106.07801) | Mapped reactant molecular graphs | Electron redistribution and the resulting product graph | Use a concrete chemical change to explain the predicted mathematical object. |
| [geodiff](https://arxiv.org/pdf/2203.02923) | Chemical graph and noisy conformer coordinates | Equivariant reverse-transition predictions and conformations | Use consistent views across noisy and clean molecular states. |
| [torchdrug](https://arxiv.org/pdf/2202.08320) | Molecular, protein or biomedical graph data, depending on task | Task-dependent features, predictions or generated graphs | A hierarchy diagram is useful for software interfaces; molecular papers still need concrete structures. |
| [protseed](https://arxiv.org/pdf/2210.08761v1) | Protein context features plus current sequence and structure | Joint sequence/structure updates and final protein designs | Show distinct tasks by their specific inputs and outputs before a shared architecture. |
| [e3bind](https://arxiv.org/pdf/2210.06069) | A ligand chemical graph and protein structure | Iteratively updated ligand pose coordinates | Draw the geometric state before and after the update, with context alongside it. |
| [neural_physical](https://arxiv.org/pdf/2402.10433) | Pretrained protein sampler and initial conformations for a target | Simulation-derived fine-tuning targets and a target-specific sampler | Separate physical data production from the trained model used afterward. |
| [atomic_fm](https://arxiv.org/pdf/2409.12080v1) | A 2D ligand graph and noisy biotoken coordinates | Joint ligand conformations and protein backbone structures | Name each generated object and illustrate why unknown binding pose matters. |
| [slm](https://arxiv.org/pdf/2410.18403) | Protein sequence and discrete structure-token representations | Sampled structure tokens decoded into protein conformations | Draw the conversion between latent representation and physical structure explicitly. |
| [protalign](https://arxiv.org/pdf/2603.06748) | Protein backbones and property-scored preference pairs | Aligned inverse-folding sequence distributions | Separate sampling, scoring, pair construction and training in a feedback loop. |
| [atmos](https://arxiv.org/pdf/2603.17633) | Current atomic configuration, chemical context and temporal state | Temporally coupled subsequent atomic configurations | Use distinct encoders, state transitions and physical output frames with clear interfaces. |

## Scientific distinctions retained

- **confgf**: Known chemical graph; its learned score is not our physical force or FM velocity.
- **cgcf**: Its inference includes coordinate reconstruction and MCMC refinement; ours reports raw coordinate-flow outputs.
- **dgsm**: Spatial contact edges, chemical bonds and our latent source tree have different roles.
- **confvae**: Bilevel coordinate reconstruction is part of its generator; not a postprocessing step in our method.
- **sbdd**: Pocket-conditioned atom growth differs from fixed-composition coordinate generation.
- **pocket2mol**: It explicitly predicts bonds; we must not borrow that input/output description.
- **diffab**: Sequence/structure co-design with supplied protein context is a different task.
- **rde**: Density estimation of side chains is not composition-only molecular generation; reviewed through primary overview/code/figure only.
- **ebmfold**: Protein score-based folding does not validate our sampler as physical molecular dynamics.
- **omegafold**: Single-sequence protein prediction differs from sampling molecular compositions; full-paper access was limited.
- **oagnn**: A representation-learning architecture is not itself a molecular generator.
- **pepflow**: Its atom/residue variables and receptor conditioning differ from our fixed atomic identities.
- **pephar**: Geometric correction is an explicit stage there; it cannot be implied in our raw-output evaluation.
- **fafe**: A well-motivated new loss can be AI novelty; novelty is not determined by whether a new network is added.
- **msm_mut**: Retrieval-conditioned property prediction is not a baseline for raw 3D generation.
- **chemprojector**: Synthesis-path validity is a different notion from our perceived-graph validity.
- **synformer**: Its output is a route and molecule, not an unoptimized coordinate array.
- **prexsyn**: Programmable property control requires explicit training and testing; our temperature is not a learned generation condition.
- **antibody_pnas**: Wet-lab validation in this study cannot be implied by our approximate energy evaluations.
- **graphaf**: Its chemical-rule filtering and bond generation are not part of our inference.
- **graph2graphs**: Reaction graph translation is not coordinate flow matching.
- **mars**: Its MCMC and property-oracle calls belong to generation; our physical queries are training/evaluation costs.
- **nerm**: Electron flow here is not the coordinate-transport field used by our FM model.
- **geodiff**: The graph is given; diffusion-time snapshots are not automatically physical-time dynamics.
- **torchdrug**: Platform capabilities are not a claim that one generator solves all those tasks.
- **protseed**: Its supplied secondary structure/contact or backbone context differs from atomic composition alone.
- **e3bind**: Docking with a known ligand graph is different from inferring molecular connectivity.
- **neural_physical**: Simulation-assisted physical fine-tuning is prior work; our local teacher and paired update need their own evidence.
- **atomic_fm**: The paper does not require a bound ligand pose as input; our review corrected that initial assumption.
- **slm**: Structure-token diversity and physical trajectory fidelity are different claims.
- **protalign**: Improved property scores do not require inventing a new physical law.
- **atmos**: It learns molecular-dynamics trajectories; our animation shows generation progress only.

Source catalog: `research/evidence/luo_shi_science_corpus_v1.json`. Downloaded research papers and inspection images remain under the ignored `runs/editorial_review_20260915/literature/` directory; external figures are not included in our manuscript artwork.
