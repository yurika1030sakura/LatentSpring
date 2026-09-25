"""Generate raw molecular coordinates from a release checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from rdkit import Chem

from cfm_mol.release import Generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--condition", type=Path, default=Path("examples/condition.json"))
    parser.add_argument("--model", default="latentspring_s0")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=79151)
    parser.add_argument("--stream", type=int, default=0)
    parser.add_argument("--hydrogen", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.out.exists():
        raise FileExistsError(args.out)
    condition = json.loads(args.condition.read_text())
    generator = Generator(args.weights, device=args.device)
    result = generator.generate(condition["numbers"], model=args.model,
        charge=condition.get("charge", 0), spin_multiplicity=condition.get("spin_multiplicity", 1),
        samples=args.samples, seed=args.seed, stream=args.stream, hydrogen=args.hydrogen)
    args.out.mkdir(parents=True)
    torch.save(result, args.out / "samples.pt")
    periodic = Chem.GetPeriodicTable()
    with (args.out / "samples.xyz").open("w") as stream:
        for index, x in enumerate(result["positions"].numpy()):
            stream.write(f"{len(x)}\n{args.model} sample {index}; coordinates in angstroms\n")
            for z, xyz in zip(condition["numbers"], x):
                stream.write(f"{periodic.GetElementSymbol(z)} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n")
    (args.out / "report.json").write_text(json.dumps({k:result[k] for k in ["settings", "costs"]}, indent=2) + "\n")
    print(json.dumps(result["costs"], indent=2))


if __name__ == "__main__":
    main()
