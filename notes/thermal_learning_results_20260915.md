# Physics-learning and local-work result

The direct-distillation adoption gate fails. Complete-work training does improve energy and force versus its matched unweighted escort, but all direct students are worse than the frozen harmonic generator.

A single coefficient1 parameter-arithmetic test reconstructs theta_frozen+(theta_physical-theta_replay). It preserves most observed validity while reducing energy and force in both continuations. It uses no new optimization or extra inference network. Parameter arithmetic is prior art and is not exact functional or KL cancellation.

| Model | First stream | Second stream |
|---|---:|---:|
| Frozen harmonic | 418/640 | 377/640 |
| Direct escort | 330/640 | 371/640 |
| Direct work | 344/640 | 362/640 |
| Escort difference | 407/640 | 387/640 |
| Work difference | 406/640 | 382/640 |

escort_delta minus frozen:
- energy_per_atom_eV, all_outputs: -0.015868, conditional95[-0.021499,-0.010145].
- energy_per_atom_eV, common_graph_supported: -0.015538, conditional95[-0.020840,-0.010313].
- force_rms_eV_A, all_outputs: -0.411074, conditional95[-0.491495,-0.334178].
- force_rms_eV_A, common_graph_supported: -0.366332, conditional95[-0.446546,-0.292304].

work_delta minus frozen:
- energy_per_atom_eV, all_outputs: -0.015647, conditional95[-0.022526,-0.008787].
- energy_per_atom_eV, common_graph_supported: -0.016042, conditional95[-0.021877,-0.010130].
- force_rms_eV_A, all_outputs: -0.437437, conditional95[-0.541248,-0.330161].
- force_rms_eV_A, common_graph_supported: -0.384033, conditional95[-0.481243,-0.289321].

work_delta minus escort_delta:
- energy_per_atom_eV, all_outputs: +0.000221, conditional95[-0.005568,+0.006493].
- energy_per_atom_eV, common_graph_supported: -0.002432, conditional95[-0.007555,+0.002644].
- force_rms_eV_A, all_outputs: -0.026363, conditional95[-0.116166,+0.066740].
- force_rms_eV_A, common_graph_supported: -0.029760, conditional95[-0.093772,+0.040912].

Work-transfer versus frozen gate: True. Extra-work-over-escort gate: False.
The observed work-transfer graph loss is0.55pp; its conditional interval[-2.58,+1.56] does not prove statistical noninferiority at2pp. The gate used an explicitly declared observed-rate tolerance.
Both physical transfers improve the eSEN readout. Work-specific superiority after useful transfer is unestablished; the cheaper unweighted escort is at least as credible for the next confirmation. Teacher preparation requires702 anchor queries for that variant, versus6232 for full work. Actual research queried all6232 once. The original harmonic models remain frozen.
Direct study:512 FIT generations,5120 evaluation generations,6232 teacher and10240 evaluation oracle queries. Transfer:2560 evaluation generations and5120 oracle queries. Preliminary source energy readout:5160 oracle queries. Total this turn:8192 new generation records and26752 raw oracle calls.
All local work, source draws, support, selections, model restoration and stored physical readouts pass audits. Local particle ESS is1.27–1.87 out of8 at composition-average level. Work estimates and their normalized finite-particle teacher do not establish global density, equilibrium chemical-isomer weights or a Boltzmann generator.
This intervention was chosen after the first result, on reused development compositions. Independent new-composition and energy-model confirmation remain essential before adopting it as a central performance claim. No coefficient sweep was performed.
