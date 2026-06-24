"""GET /mfr/dataset/table-of-contents — load MFR dataset and return TOC."""

import core
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/mfr", tags=["MFR"])


@router.get("/dataset/table-of-contents")
def mfr_table_of_contents():
    """Load the configured MFR dataset (HOOPS_AI_MFR_FLOW_NAME) and return its table of contents."""
    try:
        return core.get_mfr_table_of_contents()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MFR dataset load failed: {exc}") from exc
