"""PubChem PUG REST lookups: formula resolution for the input path (see
ARCHITECTURE.md #12 for why *input* is scoped to SMILES + formula only -- that
restriction is about what a visitor can type, not about what the profile display can
look up), and a best-effort common-name lookup used only for display.
"""

from __future__ import annotations

import asyncio
from typing import Optional
from urllib.parse import quote

import httpx

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
MAX_CANDIDATES = 10
_TIMEOUT = httpx.Timeout(10.0)


class FormulaNotFound(Exception):
    """PubChem returns 404 for a syntactically valid formula with no matches."""


class InvalidFormula(Exception):
    """PubChem returns 500 for malformed input (e.g. an unrecognised element)."""


async def resolve_formula(formula: str) -> list[dict]:
    """Resolves a molecular formula (e.g. 'C9H8O4') to one or more candidate
    compounds. Returns a list of {"cid": int, "smiles": str, "name": str | None}.
    A formula is not unique, so callers decide what to do with >1 result -- see
    ARCHITECTURE.md #10.

    Raises FormulaNotFound on zero matches, InvalidFormula on malformed input. Any
    other PubChem failure propagates as an httpx exception, turned into a generic
    client-facing message by the global error handler -- see ARCHITECTURE.md #9.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        cids_resp = await client.get(
            f"{PUBCHEM_BASE}/compound/fastformula/{formula}/cids/JSON",
            params={"MaxRecords": MAX_CANDIDATES},
        )
        if cids_resp.status_code == 404:
            raise FormulaNotFound(formula)
        if cids_resp.status_code >= 400:
            raise InvalidFormula(formula)
        cids_resp.raise_for_status()

        cids = cids_resp.json().get("IdentifierList", {}).get("CID", [])
        if not cids:
            raise FormulaNotFound(formula)

        cid_list = ",".join(str(c) for c in cids[:MAX_CANDIDATES])
        props_resp = await client.get(
            f"{PUBCHEM_BASE}/compound/cid/{cid_list}/property/IsomericSMILES,IUPACName/JSON"
        )
        props_resp.raise_for_status()
        properties = props_resp.json().get("PropertyTable", {}).get("Properties", [])

    # PubChem quirk: requesting the "IsomericSMILES" property returns it under the
    # response key "SMILES", not "IsomericSMILES".
    return [
        {
            "cid": p["CID"],
            "smiles": p.get("SMILES"),
            "name": p.get("IUPACName"),
        }
        for p in properties
        if p.get("SMILES")
    ]


# Display-only common-name lookup (query molecule + ZINC neighbours). Independent
# of the input-resolution path above -- it never gates what a visitor can type, it
# only enriches what's shown once a molecule is already known. Cached process-wide
# since the same ZINC neighbours recur across queries. PubChem occasionally drops
# one request out of a concurrent burst even for well-known molecules, so each
# lookup gets one retry (see _lookup_name_uncached below).

_name_cache: dict[str, Optional[str]] = {}
_NAME_LOOKUP_TIMEOUT = httpx.Timeout(5.0)
_NAME_LOOKUP_CONCURRENCY = 4


async def _fetch_title(client: httpx.AsyncClient, smiles: str) -> Optional[str]:
    resp = await client.get(
        f"{PUBCHEM_BASE}/compound/smiles/{quote(smiles, safe='')}/property/Title/JSON"
    )
    if resp.status_code != 200:
        return None
    properties = resp.json().get("PropertyTable", {}).get("Properties", [])
    return properties[0].get("Title") if properties else None


async def _lookup_name_uncached(client: httpx.AsyncClient, smiles: str) -> Optional[str]:
    for attempt in range(2):
        try:
            return await _fetch_title(client, smiles)
        except Exception:
            if attempt == 0:
                await asyncio.sleep(0.5)
                continue
            # Best-effort: a slow/unreachable PubChem never breaks the profile.
            return None
    return None


async def lookup_names(smiles_list: list[str]) -> dict[str, Optional[str]]:
    """Resolves a common name for each SMILES in smiles_list, in parallel, bounded by
    _NAME_LOOKUP_CONCURRENCY, with one retry per lookup on failure. Returns
    {smiles: name | None} for every input, including cache hits -- never raises, so
    it can't take down /api/profile if PubChem is slow or unavailable.
    """
    uncached = [s for s in smiles_list if s not in _name_cache]
    if uncached:
        semaphore = asyncio.Semaphore(_NAME_LOOKUP_CONCURRENCY)

        async def _bounded(client: httpx.AsyncClient, smi: str) -> None:
            async with semaphore:
                _name_cache[smi] = await _lookup_name_uncached(client, smi)

        async with httpx.AsyncClient(timeout=_NAME_LOOKUP_TIMEOUT) as client:
            await asyncio.gather(*(_bounded(client, smi) for smi in uncached))

    return {s: _name_cache.get(s) for s in smiles_list}
