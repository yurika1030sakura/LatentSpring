"""Product-manifold constrained flow matching for 3D molecules.

Implements methods_derivation.tex Section 3 + Appendix A.

Public API:
  domain.py  - Appendix A Defs 1-4 (valence / steric / connectivity manifolds)
  fibre.py   - tangent projection, retraction, CFM interpolant on the steric fibre
  flow.py    - CFM regression loss + toy velocity net for pipeline testing
  flow_model - (TODO) fork of FlowMol3 wired to use the fibre projection + retraction
  sampling   - (TODO) Euler ODE integrator with discrete flow + gluing
  train      - (TODO) training entry point
"""
