
import logging
import logging.config
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routers.oem_agent_router import router as oem_router

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_CFG = Path(__file__).parent / "logging.yaml"
if _LOG_CFG.exists():
    with open(_LOG_CFG) as f:
        logging.config.dictConfig(yaml.safe_load(f))
else:
    logging.basicConfig(level=logging.INFO)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OEM Financial Agent API",
    description="AI agent for automotive OEM financial report analysis",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    # In production everything is same-origin, so this list only matters locally
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routes (must be registered BEFORE the catch-all SPA route) ────────────
app.include_router(oem_router, prefix="/oem-agent", tags=["OEM Financial Agent"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "oem-financial-agent"}


# ── Serve built React frontend ────────────────────────────────────────────────
# In Docker the Dockerfile copies frontend/dist/ to /app/frontend/dist/.
# main.py lives at /app/backend/main.py, so we go one level up.
_DIST = Path(__file__).parent.parent / "frontend" / "dist"

if _DIST.exists():
    # /assets/* — Vite hashes these filenames, safe to cache forever
    app.mount("/assets", StaticFiles(directory=str(_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Return index.html for every route not matched above.
        Required for React client-side routing — the browser can navigate
        directly to any URL and React handles it on the client.
        """
        return FileResponse(str(_DIST / "index.html"))