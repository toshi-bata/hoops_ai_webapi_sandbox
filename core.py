"""
Minimal helpers shared across routers.

Environment variables (place in .env):
  HOOPS_AI_LICENSE          - license key (required)
  HOOPS_AI_NOTEBOOK_DIR     - path to the notebooks directory (required for MFR)
  HOOPS_AI_MFR_FLOW_NAME    - MFR flow name, e.g. ETL_CADSYNTH_training_b2 (required for MFR)
"""

import io
import os
import pathlib
import shutil
import ssl
from contextlib import redirect_stdout
from typing import Any

APP_ROOT = pathlib.Path(__file__).resolve().parent
ENV_FILE_PATH = APP_ROOT / ".env"
UPLOAD_DIR = APP_ROOT / "uploads"


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
# MFR: dataset explorer
# ---------------------------------------------------------------------------

def _create_mfr_dataset_explorer():
    """Instantiate DatasetExplorer for the configured MFR flow."""
    if os.name == "nt":
        # Suppress SSL certificate loading issues on Windows
        original_load_default_certs = ssl.SSLContext.load_default_certs
        ssl.SSLContext.load_default_certs = lambda self, purpose=ssl.Purpose.SERVER_AUTH: None
        try:
            from hoops_ai.dataset import DatasetExplorer
        finally:
            ssl.SSLContext.load_default_certs = original_load_default_certs
    else:
        from hoops_ai.dataset import DatasetExplorer

    notebooks_dir = pathlib.Path(get_required_env("HOOPS_AI_NOTEBOOK_DIR"))
    flow_name = get_required_env("HOOPS_AI_MFR_FLOW_NAME")
    flow_root = notebooks_dir / "out" / "flows" / flow_name

    return DatasetExplorer(
        merged_store_path=str(flow_root / f"{flow_name}.dataset"),
        parquet_file_path=str(flow_root / f"{flow_name}.infoset"),
        parquet_file_attribs=str(flow_root / f"{flow_name}.attribset"),
        dask_client_params={"processes": False},
    )


def get_mfr_table_of_contents() -> dict[str, Any]:
    """Load the MFR dataset and return the table of contents."""
    load_env_file()
    explorer = _create_mfr_dataset_explorer()

    output = io.StringIO()
    with redirect_stdout(output):
        result = explorer.print_table_of_contents()

    response: dict[str, Any] = {"table_of_contents": output.getvalue()}
    if result is not None:
        response["result"] = json_safe(result)
    return response
