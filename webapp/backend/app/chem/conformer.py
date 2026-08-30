"""3D structure generation for the Step 2 instant profile. Pure RDKit, no model, no
network call -- see ARCHITECTURE.md #6 for why this needs no inference backend at all.
"""

from __future__ import annotations

from typing import Optional

from rdkit import Chem

from ..vendor.smiles_to_conformers import smiles_to_unimol_input


def smiles_to_mol_block(smiles: str) -> Optional[str]:
    """Generates one 3D conformer and returns it as a MOL block string for 3Dmol.js.

    Reuses smiles_to_unimol_input() for the actual embedding/optimization fallback
    chain (ETKDGv3 -> ETKDGv2 -> distance geometry, then MMFF94/UFF) so the viewer
    shows exactly the same conformer geometry the Uni-Mol models would see -- not a
    separately-tuned "just for display" embedding.
    """
    result = smiles_to_unimol_input(smiles)
    if result is None:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() != len(result["atoms"]):
        # remove_hs=True (the default) leaves heavy-atom ordering identical to
        # Chem.MolFromSmiles(smiles) -- RDKit's AddHs/RemoveHs round trip only
        # appends/removes trailing H atoms, it never reorders heavy atoms. A
        # mismatch here means something upstream changed; bail out rather than
        # silently attaching coordinates to the wrong atoms.
        return None

    coords = result["coordinates"]
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)

    return Chem.MolToMolBlock(mol)
