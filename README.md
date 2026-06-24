# hoops_ai_webapi_sandbox

A minimal FastAPI sandbox for verifying HOOPS AI WebAPI capabilities.

## Overview

| Endpoint | Description |
|---|---|
| `POST /cad/load` | Upload a CAD file and receive BRep face/edge attributes (tests Exchange via HOOPSLoader) |
| `POST /mfr/inference` | Upload a CAD file and receive per-face MFR feature predictions |
| `GET /` | Health check |

## Prerequisites

- Python 3.12
- `hoops_ai` package (must be installed separately with a valid license)
- HOOPS AI license key

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> `hoops_ai` is not included in `requirements.txt`. Install it separately following the Tech Soft 3D installation instructions.

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `HOOPS_AI_LICENSE` | ✅ | HOOPS AI license key |
| `HOOPS_AI_NOTEBOOK_DIR` | MFR only | Path to the notebooks directory |
| `HOOPS_AI_MFR_MODEL_NAME` | MFR only | Trained model filename under `packages/trained_ml_models/` (e.g. `ts3d_162k_mfr.ckpt`) |

When using the MFR endpoint, the trained model must exist at:

```
<HOOPS_AI_NOTEBOOK_DIR>/../packages/trained_ml_models/<HOOPS_AI_MFR_MODEL_NAME>
```

## Running the server

Use the Python interpreter from the HOOPS AI virtual environment:

**Windows**
```powershell
C:\path\to\hoops_ai\.venv\Scripts\python.exe main.py
```

**Linux / macOS**
```bash
/path/to/hoops_ai/.venv/bin/python main.py
```

Options:

**Windows**
```powershell
C:\path\to\hoops_ai\.venv\Scripts\python.exe main.py --host 0.0.0.0 --port 8000 --reload
```

**Linux / macOS**
```bash
/path/to/hoops_ai/.venv/bin/python main.py --host 0.0.0.0 --port 8000 --reload
```

> **VS Code users:** Select the HOOPS AI venv as the Python interpreter (`Ctrl+Shift+P` → *Python: Select Interpreter*). The integrated terminal will then activate it automatically, and `python main.py` will work without a full path.
>
> **Linux systemd users:** Use the full path in `ExecStart` — see the systemd example below.

Once started, the interactive API docs (Swagger UI) are available at `http://localhost:8000/docs`.

## Testing from the terminal

### Health check

```bash
curl http://localhost:8000/
```

### `POST /cad/load`

**Linux / macOS**
```bash
curl -X POST http://localhost:8000/cad/load -F "file=@/path/to/part.stp"
```

**Windows (PowerShell)**
```powershell
curl.exe -X POST http://localhost:8000/cad/load -F "file=@C:\path\to\part.stp"
```

### `POST /mfr/inference`

**Linux / macOS**
```bash
curl -X POST http://localhost:8000/mfr/inference -F "file=@/path/to/part.stp"
```

**Windows (PowerShell)**
```powershell
curl.exe -X POST http://localhost:8000/mfr/inference -F "file=@C:\path\to\part.stp"
```

> You can also use the Swagger UI at `http://localhost:8000/docs` to test interactively.

## Endpoint details

### `POST /cad/load`

Upload a CAD file (e.g. STEP, IGES, STL). The file is loaded with `HOOPSLoader` and BRep face/edge attributes are returned.

**Request:** `multipart/form-data` with a `file` field containing the CAD file.

**Response example:**

```json
{
  "filename": "part.stp",
  "faces": {
    "count": 24,
    "types": [...],
    "areas": [...],
    "centroids": [...],
    "loops": [...],
    "types_description": [...]
  },
  "edges": {
    "count": 48,
    "types": [...],
    "lengths": [...],
    "dihedrals": [...],
    "convexities": [...],
    "types_description": [...]
  }
}
```

### `POST /mfr/inference`

Upload a CAD file. MFR inference is run using a pre-trained `GraphNodeClassification` model, returning per-face feature label IDs and names.

**Request:** `multipart/form-data` with a `file` field containing the CAD file.

**Response example:**

```json
{
  "filename": "part.stp",
  "face_count": 24,
  "predictions": [0, 17, 17, 23, 15, ...],
  "feature_names": ["no-label", "through_hole", "through_hole", "blind_hole", "chamfer", ...],
  "probabilities": [[0.91, 0.02, ...], ...]
}
```

## Project structure

```
hoops_ai_webapi_sandbox/
├── main.py              # FastAPI application entry point
├── core.py              # Shared helpers (.env loading, HOOPS init, BRep/MFR logic)
├── routers/
│   ├── cad.py           # POST /cad/load
│   └── mfr.py           # GET /mfr/dataset/table-of-contents
├── requirements.txt
├── .env.example
└── README.md
```
