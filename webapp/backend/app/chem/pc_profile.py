"""Human-readable physicochemical summary for the Step 2 instant profile.

Deliberately independent of extract_pc_features() (used only for Caco-2's model
input) -- see ARCHITECTURE.md #5 for why: two of these seven display fields are
filtered out of that 473-dim vector for multicollinearity reasons before training,
so they aren't reliably present there as named columns.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import Optional


def pc_profile_summary(smiles: str) -> Optional[dict]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "h_bond_donors": Descriptors.NumHDonors(mol),
        "h_bond_acceptors": Descriptors.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "aromatic_rings": Descriptors.NumAromaticRings(mol),
    }
