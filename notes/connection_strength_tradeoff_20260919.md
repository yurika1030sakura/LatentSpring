# Correction-strength follow-up

The previous validation study selected the largest allowed correction strength
for both FM and GAGA. The new calibration added strengths2 and4 for both
algorithms, reusing the same eight validation compositions, exact RNG seeds,
frozen parents and fitted physical heads. It generated1024 new raw validation
outputs and scored all of them with fixed-coordinate GFN2, without new training
or eSEN queries. The32 future test compositions were selected by metadata before
this calibration and remain separate from all prior evaluation panels.

The original extended protocol required keeping pooled graph validity within1pp
and each fit within3pp of strength1. Neither stronger setting met that condition
for either algorithm. The strict-guard decision is FALSE and remains unchanged.
Under that protocol, no fresh test samples were generated.

There was nevertheless a joint-quality tradeoff worth testing under the user's
goal of improving the GAGA comparison. Selecting each algorithm's maximum
validation joint yield over the same six strengths gives4 for both. FM has
31.64% joint yield and33.59% graph validity; equally selected GAGA has28.91% and
30.08%. FM's own strength1 graph validity was35.94%, so this is not an unqualified
improvement over its previous variant.

A separate protocol, connection_tradeoff_v1.json, was frozen after that
validation result and before any fresh test output. It explicitly changes the
objective to joint yield, reports graph validity alongside it, retains both
strength1 models as controls, and uses exactly the already-selected32 untouched
compositions. Its primary comparison is selected FM versus equally selected
GAGA; both fitted parents and all outputs are retained. The failed graph guard
is not reinterpreted as having passed.

This calibration is not a new network or an additional AI novelty claim. Its
purpose is to establish the usable quality tradeoff of the already-validated
learned physical correction. Stronger inference guidance scales the instantaneous
vector bound; it does not prove molecular validity, global Boltzmann sampling,
or lower energy for every individual generated structure.
