# Source controls under geometry self-conditioning

| Training seed | Unit Gaussian | Shell tree | Harmonic tree |
|---|---:|---:|---:|
| 1 | 357/768 | 417/768 | 436/768 |
| 2 | 377/768 | 342/768 | 388/768 |

All arms have identical geometry-SC structure,3000 two-pass updates and128 primitive denoiser calls per output. Four control models trained; two existing shell models and1536 shell outputs reused. New outputs3072. All4608 comparison records replay.

- fixed minus gaussian: +1.63pp; conditional paired95[-1.50,+4.88], descriptive composition95[-2.08,+5.99].
- fixed minus harmonic_tree: -4.23pp; conditional paired95[-7.29,-1.11], descriptive composition95[-8.53,-0.20].
- harmonic_tree minus gaussian: +5.86pp; conditional paired95[+2.67,+8.98], descriptive composition95[+0.52,+12.17].

The prespecified shell-superiority gate fails. Harmonic tree is a stronger candidate on this reused12-composition development panel; it improves over unit Gaussian in both seeds, with unequal gain sizes. Its radial Gaussian edges match shell covariance analytically. This is a post-development candidate choice, not proof of originality or a Boltzmann law.

Next essential test: freeze both existing harmonic/Gaussian models and evaluate all10 remaining qualified pool compositions,17-28 atoms, without further fitting or parameter changes. These compositions have zero matches in both verified processed corpora. No reserved722 outcomes are queried.
