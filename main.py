"""
hoops_ai_webapi_sandbox — minimal FastAPI sandbox for HOOPS AI WebAPI verification.

Endpoints:
  POST /cad/load                        Upload a CAD file → BRep info (tests Exchange)
  GET  /mfr/dataset/table-of-contents   Load MFR dataset → table of contents
"""

import sys
from contextlib import asynccontextmanager

import core
from fastapi import FastAPI
from routers import cad, mfr


@asynccontextmanager
async def lifespan(app: FastAPI):
    core.load_env_file()
    core.init_hoops_license()
    yield


app = FastAPI(
    title="HOOPS AI WebAPI Sandbox",
    description=(
        "Minimal sandbox to verify HOOPS AI WebAPI capabilities.\n\n"
        "- `POST /cad/load` — upload a CAD file, load it with HOOPSLoader, receive BRep attributes\n"
        "- `GET /mfr/dataset/table-of-contents` — load the configured MFR dataset and return its TOC"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(cad.router)
app.include_router(mfr.router)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "HOOPS AI WebAPI Sandbox"}


if __name__ == "__main__":
    import argparse
    import socket
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the HOOPS AI WebAPI Sandbox server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", args.port)) == 0:
            print(
                f"Error: port {args.port} is already in use. Use --port <number>.",
                file=sys.stderr,
            )
            sys.exit(1)

    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)
