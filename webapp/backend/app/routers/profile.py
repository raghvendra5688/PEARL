"""GET /api/profile -- Step 2 instant profile: 3D structure, PC summary, ZINC
neighbours, formula + best-effort common name. See ARCHITECTURE.md #6/#7/#13.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel
from rdkit import Chem

from ..chem.conformer import smiles_to_mol_block
from ..chem.formula import molecular_formula
from ..chem.pc_profile import pc_profile_summary
from ..chem.zinc_neighbours import nearest_neighbours
from ..errors import AppError
from ..resolve.pubchem import lookup_names

router = APIRouter()


class Neighbour(BaseModel):
    smiles: str
    formula: Optional[str]
    name: Optional[str]
    similarity: float
    logp: float
    qed: float
    sas: float


class ProfileResponse(BaseModel):
    smiles: str
    formula: Optional[str]
    name: Optional[str]
    mol_block: str | None
    pc_profile: dict | None
    neighbours: list[Neighbour] | None


@router.get("/profile", response_model=ProfileResponse)
async def profile(smiles: str = Query(..., min_length=1)) -> ProfileResponse:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise AppError("That doesn't look like a valid SMILES string.", status_code=400)

    canonical = Chem.MolToSmiles(mol)

    # Each of these is CPU-bound (conformer embedding, PC featurization, a 250K-row
    # ZINC scan); running them via asyncio.to_thread keeps the event loop free
    # instead of blocking every other in-flight request. See ARCHITECTURE.md #6/#7.
    mol_block_task = asyncio.to_thread(smiles_to_mol_block, canonical)
    pc_profile_task = asyncio.to_thread(pc_profile_summary, canonical)
    neighbours_task = asyncio.to_thread(nearest_neighbours, canonical, 5)

    mol_block, pc_profile, neighbours = await asyncio.gather(
        mol_block_task, pc_profile_task, neighbours_task
    )

    # Name lookup is network I/O, not CPU work, so it runs on the event loop directly.
    lookup_smiles = [canonical] + [n["smiles"] for n in (neighbours or [])]
    names = await lookup_names(lookup_smiles)

    if neighbours:
        for n in neighbours:
            n["name"] = names.get(n["smiles"])

    return ProfileResponse(
        smiles=canonical,
        formula=molecular_formula(canonical),
        name=names.get(canonical),
        mol_block=mol_block,
        pc_profile=pc_profile,
        neighbours=neighbours,
    )
