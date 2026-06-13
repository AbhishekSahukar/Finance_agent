"""OEM Agent FastAPI router."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.config.settings import Settings, get_settings
from api.controllers.oem_agent_controller import OEMAgentController
from api.models.oem_models import RunAgentRequest, RunAgentResponse
from api.services.langfuse_callback_service import LangfuseCallbackService, get_langfuse_service
from api.services.oem_agent.extraction_tools import KPI_SYNONYMS

logger = logging.getLogger(__name__)
router = APIRouter()


def _ctrl(
    settings: Annotated[Settings, Depends(get_settings)],
    langfuse:  Annotated[LangfuseCallbackService, Depends(get_langfuse_service)],
) -> OEMAgentController:
    return OEMAgentController(settings=settings, langfuse=langfuse)


@router.post("/run", response_model=RunAgentResponse, summary="Run agent (JSON / base64)")
async def run_agent(
    request: RunAgentRequest,
    ctrl: Annotated[OEMAgentController, Depends(_ctrl)],
) -> RunAgentResponse:
    return await ctrl.run_agent(request)


@router.post("/upload", response_model=RunAgentResponse, summary="Run agent (multipart PDF upload)")
async def upload_and_run(
    files: list[UploadFile] = File(...),
    company_names: str = Form(..., description="Comma-separated, e.g. 'BMW,BMW,Mercedes,Mercedes'"),
    report_types:  str = Form(..., description="Comma-separated FY or Q, e.g. 'FY,Q,FY,Q'"),
    session_id:    str = Form(default=""),
    ctrl: OEMAgentController = Depends(_ctrl),
) -> RunAgentResponse:
    companies = [c.strip() for c in company_names.split(",")]
    types     = [t.strip() for t in report_types.split(",")]
    if len(files) != len(companies) or len(files) != len(types):
        raise HTTPException(400, "files, company_names and report_types must have the same count.")

    encoded = []
    for f, co, rt in zip(files, companies, types):
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"'{f.filename}' must be a PDF.")
        encoded.append(await OEMAgentController.encode_file(f, co, rt))

    return await ctrl.run_agent(RunAgentRequest(files=encoded, session_id=session_id))


@router.get("/schema", summary="KPI synonym map")
async def kpi_schema() -> dict:
    return {
        "canonical_kpis": list(KPI_SYNONYMS.keys()),
        "synonym_map": KPI_SYNONYMS,
    }
