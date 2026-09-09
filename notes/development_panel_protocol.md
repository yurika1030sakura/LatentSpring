# Frozen development panel

Eight conditions are selected from the audited 664-condition new-development
manifest, using seed20270919. There is one condition per 2--12/13--24 atom bin,
neutral/charged status and singlet/open-shell status. Choose the minimum fixed
hash of seed, candidate index and source path within each stratum, excluding
previously selected compositions. Energies and method outputs do not rank or
filter candidates. Selection replays byte-for-byte from the source manifest.

The selected candidate indices are194,224,147,169,120,29,109,53, with10,12,9,12,
17,18,14,20 atoms. They include Re and Pt complexes and original nonminimum spin
states. Failures must be retained, not used to replace selected conditions.
This is development data; no reserved condition is exposed by this panel.

The first baseline samples32 geometries per condition, with batch16 and seed9063,
from the frozen global-state FM checkpoint. Model input kT remains its trained
constant1, not an inferred thermal label. Midpoint16/64 use32/128 field calls
per sample, and all coordinate changes are retained. Gaussian initial draws
are independent of reference coordinates. No FM density/importance weights are
claimed. The generator takes a condition manifest, not raw reference positions.

Raw references are exported separately in the omol25 environment and validated
against original numbers, charge, spin, source identifier and energy. They are
assessment controls only. A CPU interface check with two samples per condition
completed all eight conditions; this is not a quality/convergence result.

Future physics training and independent evaluation must keep these selected
conditions and account for every attempt. A larger final evaluation remains
separate and requires a frozen method protocol. The model's general max_atoms
stays200; these size bins are an initial development panel, not a new model cap.
