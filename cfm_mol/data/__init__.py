"""OOD data-loader modules for the ICLR E1 experiment.

Each module exposes:
  download(dest_dir: Path) -> None
    One-shot acquisition (CLI url, DOI, GitHub release, etc.).
  process(raw_dir: Path, processed_dir: Path, atom_map: list[str]) -> None
    Convert raw to FlowMol3-compatible processed format
    (positions / atom_types / atom_charges / bond_types / bond_idxs /
     node_idx_array / edge_idx_array + train_data_marginal_dists.pt).
  slice_name : str
    Short tag used in filenames / result CSVs.

These are skeletons -- download URLs + atom-map specifics to be filled in
when running the actual OOD experiment.
"""
