"""OEM Financial Agent — FastAPI entrypoint."""

import logging
import logging.config
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.oem_agent_router import router as oem_router

_LOG_CFG = Path(__file__).parent / "logging.yaml"
if _LOG_CFG.exists():
    with open(_LOG_CFG) as f:
        logging.config.dictConfig(yaml.safe_load(f))
else:
    logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="OEM Financial Agent API",
    description="AI agent for automotive OEM financial report analysis",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(oem_router, prefix="/oem-agent", tags=["OEM Financial Agent"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "oem-financial-agent"}
