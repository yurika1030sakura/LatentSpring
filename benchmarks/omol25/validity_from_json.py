"""Shared, model-agnostic xyz2mol validity on a samples JSON ([{atomic_numbers,
positions, charge?}]). Geometry-based (rdkit DetermineBonds) so it works across
all 83 elements / transition metals, unlike each repo's QM9/GEOM valence tables.
Reports the standard de-novo-3D-gen original metrics: validity + connectivity.
Run in any env with rdkit (e.g. envs/edm or envs/flowmol)."""
import argparse, json
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def check(Z, pos, charge=0):
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(Z))
    for i, z in enumerate(Z):
        rw.AddAtom(Chem.Atom(int(z)))
        conf.SetAtomPosition(i, (float(pos[i][0]), float(pos[i][1]), float(pos[i][2])))
    m = rw.GetMol()
    m.AddConformer(conf)
    try:
        rdDetermineBonds.DetermineBonds(m, charge=int(charge))
        Chem.SanitizeMol(m)
        nfrag = len(Chem.GetMolFrags(m))
        return True, (nfrag == 1)
    except Exception:
        return False, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = json.load(open(a.samples_json))
    n = len(recs)
    nv = nc = 0
    for r in recs:
        ok, conn = check(r["atomic_numbers"], r["positions"], r.get("charge", 0))
        nv += ok
        nc += conn
    res = {"n": n, "frac_valid": nv / max(1, n), "frac_connected": nc / max(1, n)}
    json.dump(res, open(a.out, "w"))
    print(f"[validity] {res}")


if __name__ == "__main__":
    main()
