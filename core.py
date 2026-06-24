"""
Minimal helpers shared across routers.

Environment variables (place in .env):
  HOOPS_AI_LICENSE          - license key (required)
  HOOPS_AI_NOTEBOOK_DIR     - path to the notebooks directory (required for MFR)
  HOOPS_AI_MFR_MODEL_NAME   - trained model filename under packages/trained_ml_models/ (required for MFR)
"""

import io
import os
import pathlib
import shutil
import ssl
import uuid
from contextlib import redirect_stdout
from typing import Any

APP_ROOT = pathlib.Path(__file__).resolve().parent
ENV_FILE_PATH = APP_ROOT / ".env"
UPLOAD_DIR = APP_ROOT / "uploads"

_mfr_inference_model = None

DEFAULT_MFR_LABELS: dict[int, str] = {
    0: "no-label",
    1: "rectangular_through_slot",
    2: "triangular_through_slot",
    3: "rectangular_passage",
    4: "triangular_passage",
    5: "6sides_passage",
    6: "rectangular_through_step",
    7: "2sides_through_step",
    8: "slanted_through_step",
    9: "rectangular_blind_step",
    10: "triangular_blind_step",
    11: "rectangular_blind_slot",
    12: "rectangular_pocket",
    13: "triangular_pocket",
    14: "6sides_pocket",
    15: "chamfer",
    16: "circular_through_slot",
    17: "through_hole",
    18: "circular_blind_step",
    19: "horizontal_circular_end_blind_slot",
    20: "vertical_circular_end_blind_slot",
    21: "circular_end_pocket",
    22: "o-ring",
    23: "blind_hole",
    24: "fillet",
}


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def load_env_file(path: pathlib.Path = ENV_FILE_PATH) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


def get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Environment variable '{name}' is required. "
            f"Set it in {ENV_FILE_PATH.name} or the environment."
        )
    return value


# ---------------------------------------------------------------------------
# HOOPS license
# ---------------------------------------------------------------------------

def init_hoops_license() -> None:
    import hoops_ai

    load_env_file()
    license_key = get_required_env("HOOPS_AI_LICENSE")
    hoops_ai.set_license(license_key, validate=True)
    print(f"[hoops_ai] License initialized.")


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


# ---------------------------------------------------------------------------
# CAD: load file and extract BRep info
# ---------------------------------------------------------------------------

def load_brep_info(cad_file_path: pathlib.Path) -> dict[str, Any]:
    """Load a CAD file with HOOPSLoader and return BRep face/edge attributes."""
    from hoops_ai.cadaccess import HOOPSLoader, HOOPSTools
    from hoops_ai.cadencoder import BrepEncoder

    cad_loader = HOOPSLoader()
    model = cad_loader.create_from_file(str(cad_file_path))

    tools = HOOPSTools()
    tools.adapt_brep(model)

    brep_encoder = BrepEncoder(model.get_brep())

    [face_types, face_areas, face_centroids, face_loops], face_types_descr = (
        brep_encoder.push_face_attributes()
    )
    [edge_types, edge_lengths, edge_dihedrals, edge_convexities], edge_types_descr = (
        brep_encoder.push_edge_attributes()
    )

    return {
        "filename": cad_file_path.name,
        "faces": {
            "count": len(face_types) if hasattr(face_types, "__len__") else None,
            "types": json_safe(face_types),
            "areas": json_safe(face_areas),
            "centroids": json_safe(face_centroids),
            "loops": json_safe(face_loops),
            "types_description": json_safe(face_types_descr),
        },
        "edges": {
            "count": len(edge_types) if hasattr(edge_types, "__len__") else None,
            "types": json_safe(edge_types),
            "lengths": json_safe(edge_lengths),
            "dihedrals": json_safe(edge_dihedrals),
            "convexities": json_safe(edge_convexities),
            "types_description": json_safe(edge_types_descr),
        },
    }


# ---------------------------------------------------------------------------
# MFR: inference
# ---------------------------------------------------------------------------

def _get_mfr_inference_model():
    global _mfr_inference_model
    if _mfr_inference_model is None:
        _mfr_inference_model = _create_mfr_inference_model()
    return _mfr_inference_model


def _create_mfr_inference_model():
    from hoops_ai.cadaccess import HOOPSLoader
    from hoops_ai.ml.EXPERIMENTAL import FlowInference, GraphNodeClassification

    load_env_file()
    notebooks_dir = pathlib.Path(get_required_env("HOOPS_AI_NOTEBOOK_DIR"))
    model_name = get_required_env("HOOPS_AI_MFR_MODEL_NAME")
    trained_model = notebooks_dir.parent / "packages" / "trained_ml_models" / model_name
    output_dir = notebooks_dir / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = HOOPSLoader()
    inference_model = FlowInference(
        cad_loader=loader,
        flowmodel=GraphNodeClassification(result_dir=str(output_dir)),
    )
    inference_model.load_from_checkpoint(trained_model)
    return inference_model


def run_mfr_inference(cad_file_path: pathlib.Path) -> dict[str, Any]:
    """Run MFR inference on a CAD file and return per-face feature predictions."""
    model = _get_mfr_inference_model()
    ml_input = model.preprocess(str(cad_file_path))
    predictions, probabilities = model.predict_and_postprocess(ml_input)

    preds = json_safe(predictions)
    named = [DEFAULT_MFR_LABELS.get(int(p), str(p)) for p in preds]

    return {
        "filename": cad_file_path.name,
        "face_count": len(preds),
        "predictions": preds,
        "feature_names": named,
        "probabilities": json_safe(probabilities),
    }
