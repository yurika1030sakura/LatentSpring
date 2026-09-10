# Endpoint-entropy learning prototype

This is a prospective alternative objective, not validated molecular performance
or established novelty. It targets a different failure mechanism from refitting
an auxiliary path distribution. VSD/DMD already use target-minus-generated
score information for generator updates; DMD2 studies inaccurate/stale critics.
Do not claim the score-difference gradient or two-timescale training as new.
Primary sources:
https://openaccess.thecvf.com/content/CVPR2024/html/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.html
https://proceedings.neurips.cc/paper_files/paper/2023/hash/1a87980b9853e84dfb295855b425c262-Abstract-Conference.html
https://arxiv.org/abs/2405.14867

## Why this is a different learning question

The standard chain rule gives
KL(Q_theta(z,y)||pi(y)L_phi(z|y))
=KL(q_theta(y)||pi(y))+E_(y~q_theta) KL(Q_theta(z|y)||L_phi(z|y)).
The second term is an inference approximation gap. Reducing the joint objective
need not reduce endpoint KL at each update, and a restricted reverse family can
change the best forward generator. This is a standard variational decomposition.
It is a mechanism hypothesis for our molecular failures, not a demonstrated cause.

A simple exact counterexample is y=a*z+sigma*epsilon with independent standard
normal z,epsilon, target pi=N(0,tau^2), tau^2>sigma^2, and restricted L(z|y)=phi(z).
Joint reverse-KL has its optimum at a=0, hence endpoint variance sigma^2. Marginal
KL is minimized at a^2=tau^2-sigma^2. A fully flexible reverse Gaussian removes
this particular gap, so this example must not be represented as a proof that the
trained molecular reverse network is similarly restricted.

## Gradient and exact sampler semantics

For y=F_theta(z)+sigma*epsilon with theta-independent sigma>0, regularity permits
 grad_theta KL(q_theta||pi)=E[(s_q(y)-s_pi(y)) dot d_theta y].
The explicit parameter-score term has zero expectation. Hold both score vectors
fixed during the generator update. The dot-product scalar used in code is a
gradient surrogate, not a KL value; finite-differencing its changing scalar value
would test the wrong quantity. Compare its derivative with the true known KL.

For our stated target, s_pi is (force_eSEN-kappa*x)/kT, projected to the same
orthonormal COM coordinates. The actual final Gaussian innovation epsilon gives
a denoising regression E||sigma*s_phi(y)+epsilon||^2 whose conditional optimum
is the score of the actual noisy endpoint law. Inputs to the critic are y and
physical condition, not its generating latent/noise. This does not use the
linear-FM score formula for an arbitrarily fine-tuned vector field. Changing
sigma or adding extra noise changes the distribution being scored and must not
silently be paired with the original unsmoothed physical target.

## First implementation and gates

The unit tests compare the surrogate gradient against finite differences of an
analytic Gaussian KL, verify the DSM optimum, and verify that actor updates do
not backpropagate into critic parameters. A scalar learning experiment uses
five seeds9095--9099,200 updates at .02, actor batch512, fitted-critic batch8192,
sigma=.2, target variance .25 and initial a=1.2. Retain fixed-reverse joint-KL,
exact endpoint score, fitted linear DSM score and deliberately stale score arms.
The linear critic uses a closed-form regression fit, not a neural molecule model.
This is an inexpensive mechanism check; there are no molecular oracle queries.

A molecular actor run has not yet passed scientific qualification. First
train and validate a frozen-proposal score estimator using fresh endpoint/noise
pairs and original charge/spin, with independent heldout denoising and nonlinear
Stein-moment checks. These checks are necessary but not a proof of global score
accuracy. Report zero/stale-critic controls and score-estimation error where a
known reference permits it. The true sampler's small terminal sigma makes this
potentially difficult; an apparently decreasing DSM loss is not enough.
Only after this gate should a bounded molecular actor comparison be proposed.
Reserved evaluation conditions remain untouched. This prototype alone does not
supply the method novelty needed for an ICLR submission.
