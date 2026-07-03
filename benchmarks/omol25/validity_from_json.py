"""Shared, model-agnostic validity + PoseBusters + uniqueness on a samples JSON
([{atomic_numbers, positions, charge?}]). Geometry-based (rdkit DetermineBonds)
so it works across all 83 elements / transition metals, unlike each repo's
QM9/GEOM valence tables. Reports the standard de-novo-3D-gen metrics plus the
ones needed to compare against the OMol25 generator Zatom-1:
  - frac_valid       : rdDetermineBonds + SanitizeMol succeeds
  - frac_connected   : single fragment (atoms-connected)
  - frac_posebusters : PoseBusters config='mol' ALL-checks pass (Zatom-1's metric)
  - uniqueness       : distinct canonical SMILES among the VALID molecules
Run in an env with rdkit + posebusters (envs/flowmol). Pass --no_posebusters to
skip PoseBusters in envs that lack it (frac_posebusters is then null)."""
import argparse, json
import multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# PoseBusters adds per-molecule cost on top of bond perception; give the worker
# more headroom before the hard-kill.
TIMEOUT_S = 30

# PoseBusters "mol" config = de-novo molecule sanity (no reference/protein
# needed). Import once here so forked workers inherit it; absent in some envs
# (edm/omol25) -> frac_posebusters reported as null.
try:
    from posebusters import PoseBusters as _PoseBusters
except Exception:
    _PoseBusters = None

_PB = None  # lazily-created per worker process (fork-safe)


def _posebusters_pass(mol, use_pb=True):
    """True/False if PoseBusters ran, None if unavailable/errored."""
    global _PB
    if not use_pb or _PoseBusters is None:
        return None
    try:
        if _PB is None:
            _PB = _PoseBusters(config="mol", max_workers=0)
        df = _PB.bust(mol)
        cols = df.select_dtypes(include=bool)
        if cols.shape[1] == 0:
            return None
        return bool(cols.iloc[0].all())
    except Exception:
        return None


def check(Z, pos, charge=0, use_pb=True):
    """Returns (valid, connected, canonical_smiles|None, posebusters_pass|None)."""
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
        try:
            smi = Chem.MolToSmiles(Chem.RemoveHs(m))
        except Exception:
            smi = Chem.MolToSmiles(m)
        pb = _posebusters_pass(m, use_pb=use_pb)
        return True, (nfrag == 1), smi, pb
    except Exception:
        return False, False, None, False


def _worker(Z, pos, charge, use_pb, q):
    q.put(check(Z, pos, charge, use_pb))


def check_timeout(Z, pos, charge=0, use_pb=True, timeout=TIMEOUT_S):
    """Returns (valid, connected, smiles, pb_pass, timed_out). timed_out
    molecules count as invalid but are reported separately."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(Z, pos, charge, use_pb, q))
    p.start()
    p.join(timeout)
    if p.is_alive():            # C++/PoseBusters hang -> hard kill, count invalid
        p.terminate(); p.join()
        return False, False, None, False, True
    try:
        ok, conn, smi, pb = q.get_nowait()
        return ok, conn, smi, pb, False
    except Exception:
        return False, False, None, False, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_posebusters", action="store_true",
                    help="skip PoseBusters (e.g. in envs without it)")
    a = ap.parse_args()
    use_pb = not a.no_posebusters
    recs = json.load(open(a.samples_json))
    n = len(recs)
    nv = nc = n_timeout = n_pb = 0
    pb_available = False
    valid_smiles = []
    for i, r in enumerate(recs):
        ok, conn, smi, pb, to = check_timeout(
            r["atomic_numbers"], r["positions"], r.get("charge", 0), use_pb)
        nv += ok
        nc += conn
        n_timeout += to
        if ok and smi is not None:
            valid_smiles.append(smi)
        if pb is not None:
            pb_available = True
            n_pb += int(pb)
    uniqueness = (len(set(valid_smiles)) / len(valid_smiles)) if valid_smiles else 0.0
    res = {"n": n,
           "n_valid": nv,
           "frac_valid": nv / max(1, n),
           "frac_connected": nc / max(1, n),
           "uniqueness": uniqueness,           # distinct canonical SMILES / valid
           "frac_posebusters": (n_pb / max(1, n)) if pb_available else None,
           "n_timeout": n_timeout}
    json.dump(res, open(a.out, "w"))
    print(f"[validity] {res}")


if __name__ == "__main__":
    main()
