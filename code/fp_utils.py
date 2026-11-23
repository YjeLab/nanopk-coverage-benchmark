"""Morgan fingerprints + descriptors with a disk cache (project-local utility)."""
import hashlib
import json
from pathlib import Path

import numpy as np

DESC_NAMES = ["MolWt", "MolLogP", "HBD", "HBA", "TPSA", "RotBonds",
              "FracCSP3", "AromaticRings", "Rings", "HeavyAtoms"]


def fingerprints(smiles, n_bits=1024, radius=2, cache_dir=None):
    smiles = list(smiles)
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    cache_dir = Path(cache_dir)
    key = hashlib.md5(
        json.dumps([len(smiles), n_bits, radius, smiles[:50], smiles[-50:]]).encode()
    ).hexdigest()[:16]
    cache = cache_dir / f"fp_{n_bits}_{key}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["X"], z["Xd"], z["keep"]

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    fps, desc, keep = [], [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits))
        desc.append([
            Descriptors.MolWt(m), Descriptors.MolLogP(m),
            rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumRotatableBonds(m),
            rdMolDescriptors.CalcFractionCSP3(m),
            rdMolDescriptors.CalcNumAromaticRings(m),
            rdMolDescriptors.CalcNumRings(m),
            rdMolDescriptors.CalcNumHeavyAtoms(m),
        ])
        keep.append(i)
    X = np.array([list(f) for f in fps], dtype=np.float32)
    Xd = np.array(desc, dtype=np.float32)
    keep = np.array(keep)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, Xd=Xd, keep=keep)
    except Exception:
        pass
    return X, Xd, keep