"""POST /api/resolve -- SMILES or formula -> canonical SMILES (+ disambiguation list
for a formula matching multiple compounds). Name lookup intentionally out of scope --
see ARCHITECTURE.md #12.
"""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel
from rdkit import Chem

from ..errors import AppError
from ..resolve.pubchem import FormulaNotFound, InvalidFormula, resolve_formula

router = APIRouter()


class ResolveRequest(BaseModel):
    input_type: Literal["smiles", "formula"]
    value: str


class ResolveCandidate(BaseModel):
    cid: int
    smiles: str
    name: Optional[str] = None


class ResolveResponse(BaseModel):
    status: Literal["resolved", "disambiguation"]
    smiles: Optional[str] = None
    candidates: Optional[list[ResolveCandidate]] = None


def _canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise AppError("That doesn't look like a valid SMILES string.", status_code=400)
    return Chem.MolToSmiles(mol)


@router.post("/resolve", response_model=ResolveResponse)
async def resolve(req: ResolveRequest) -> ResolveResponse:
    value = req.value.strip()
    if not value:
        raise AppError("Please enter a SMILES string or a molecular formula.")

    if req.input_type == "smiles":
        return ResolveResponse(status="resolved", smiles=_canonicalize(value))

    try:
        candidates = await resolve_formula(value)
    except FormulaNotFound:
        raise AppError(f"No known compounds match the formula '{value}'.", status_code=404)
    except InvalidFormula:
        raise AppError(f"'{value}' doesn't look like a valid molecular formula.", status_code=400)

    if len(candidates) == 1:
        return ResolveResponse(status="resolved", smiles=_canonicalize(candidates[0]["smiles"]))

    return ResolveResponse(
        status="disambiguation",
        candidates=[
            ResolveCandidate(cid=c["cid"], smiles=_canonicalize(c["smiles"]), name=c["name"])
            for c in candidates
        ],
    )
