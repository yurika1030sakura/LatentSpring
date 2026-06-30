"""Shared, model-agnostic xyz2mol validity on a samples JSON ([{atomic_numbers,
positions, charge?}]). Geometry-based (rdkit DetermineBonds) so it works across
all 83 elements / transition metals, unlike each repo's QM9/GEOM valence tables.
Reports the standard de-novo-3D-gen original metrics: validity + connectivity.
Run in any env with rdkit (e.g. envs/edm or envs/flowmol)."""
import argparse, json
import multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# rdDetermineBonds bond perception is combinatorial and can hang for minutes on
# large (~150+ atom) unphysical generated geometries. It is a C++ call, so a
# Python signal.alarm cannot interrupt it -- we run each molecule in a worker
# process and HARD-KILL it past TIMEOUT_S (a geometry whose bonds cannot be
# perceived in time is treated as invalid).
TIMEOUT_S = 15


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


def _worker(Z, pos, charge, q):
    q.put(check(Z, pos, charge))


def check_timeout(Z, pos, charge=0, timeout=TIMEOUT_S):
    """Returns (valid, connected, timed_out). timed_out molecules count as invalid
    but are reported separately so the timeout rate is auditable."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(Z, pos, charge, q))
    p.start()
    p.join(timeout)
    if p.is_alive():            # C++ hang -> hard kill, count as invalid
        p.terminate(); p.join()
        return False, False, True
    try:
        ok, conn = q.get_nowait()
        return ok, conn, False
    except Exception:
        return False, False, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = json.load(open(a.samples_json))
    n = len(recs)
    nv = nc = n_timeout = 0
    for i, r in enumerate(recs):
        ok, conn, to = check_timeout(r["atomic_numbers"], r["positions"], r.get("charge", 0))
        nv += ok
        nc += conn
        n_timeout += to
    res = {"n": n, "frac_valid": nv / max(1, n), "frac_connected": nc / max(1, n),
           "n_timeout": n_timeout}
    json.dump(res, open(a.out, "w"))
    print(f"[validity] {res}")


if __name__ == "__main__":
    main()
