"""The Mode 2 endpoint: upload a lease, get one honest analysis back.

``POST /contract/analyze`` takes a single multipart file (a born-digital PDF or a
``.docx``), reads its bytes in memory and runs the contract pipeline to a terminal
:class:`ContractAnalysis` -- an analyzed ficha with its summary and anchored
clauses, or an honest stop (out of scope, or not analyzable). The endpoint only
wires the request's collaborators and the clock; the whole pipeline lives in the
Mode 2 package. The uploaded bytes are never written to disk or a database.
"""

from datetime import date

from fastapi import APIRouter, Depends, File, UploadFile

from lexme.api.dependencies import get_mode2_deps, get_today
from lexme.mode2 import ContractAnalysis, Mode2Deps, analyze_contract

router = APIRouter()


@router.post("/contract/analyze", response_model=ContractAnalysis)
async def analyze(
    file: UploadFile = File(...),
    deps: Mode2Deps = Depends(get_mode2_deps),
    today: date = Depends(get_today),
) -> ContractAnalysis:
    """Analyze an uploaded contract, or stop honestly at the first gate it fails."""
    content = await file.read()
    return analyze_contract(
        filename=file.filename or "",
        content=content,
        deps=deps,
        today=today,
    )
