"""POST /mfr/inference — upload a CAD file and receive MFR feature predictions."""

import pathlib
import shutil
import uuid

import core
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(prefix="/mfr", tags=["MFR"])


@router.post("/inference")
def mfr_inference(file: UploadFile = File(...)):
    """Upload a CAD file, run MFR inference, and return per-face feature predictions."""
    if not file.filename:
        raise HTTPException(status_code=422, detail="Uploaded file must have a filename.")

    core.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = pathlib.PurePath(file.filename).name
    tmp_path = core.UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"

    try:
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        return core.run_mfr_inference(tmp_path)

    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MFR inference failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
