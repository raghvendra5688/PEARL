"""Molecular formula display. Pure RDKit, instant -- unlike the PubChem name lookup
in resolve/pubchem.py, this never depends on network access, so it's always shown
even when a name can't be found.
"""

from __future__ import annotations

from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def molecular_formula(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.CalcMolFormula(mol)
