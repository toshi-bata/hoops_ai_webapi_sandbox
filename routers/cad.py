"""POST /cad/load — upload a CAD file and receive BRep information."""

import pathlib
import shutil
import uuid

import core
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(prefix="/cad", tags=["CAD"])


@router.post("/load")
def cad_load(file: UploadFile = File(...)):
    """Upload a CAD file, load it with HOOPSLoader, and return BRep face/edge attributes."""
    if not file.filename:
        raise HTTPException(status_code=422, detail="Uploaded file must have a filename.")

    # Save to a temporary path under uploads/
    core.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = pathlib.PurePath(file.filename).name
    tmp_path = core.UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"

    try:
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        return core.load_brep_info(tmp_path)

    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CAD load failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
