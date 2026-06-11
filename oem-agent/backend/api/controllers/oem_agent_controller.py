"""
OEM Agent controller — validates input, invokes LangGraph, serialises response.

Fix: _build_response now derives company names and periods directly from
     table columns (which are "{company} {period_label}") instead of
     trying to split on the last word, which broke for names like
     "Mercedes-Benz FY2025".
"""

import base64
import logging
from typing import Annotated

from fastapi import Depends, HTTPException, UploadFile

from api.config.settings import get_settings, Settings
from api.models.oem_models import (
    RunAgentRequest, RunAgentResponse, KPISubstitutionNote, SummaryTableCell,
)
from api.services.langfuse_callback_service import LangfuseCallbackService, get_langfuse_service
from api.services.oem_agent.oem_generation_service import build_oem_agent_graph, KPI_ROW_ORDER

logger = logging.getLogger(__name__)

# Known period suffixes — used to split "Company Name FY2025" correctly
_PERIOD_SUFFIXES = ("FY2025", "Q4 2025", "Q3 2025", "Q2 2025", "Q1 2025",
                    "Q4 2024", "FY2024", "H1 2025", "H2 2025")


def _split_column_key(col: str):
    """
    Split a column key like 'BMW Group FY2025' → ('BMW Group', 'FY2025').
    Works for multi-word company names.
    """
    for suffix in _PERIOD_SUFFIXES:
        if col.endswith(suffix):
            company = col[: -len(suffix)].strip()
            return company, suffix
    # fallback: last token is period
    parts = col.rsplit(" ", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (col, "")


class OEMAgentController:
    def __init__(
        self,
        settings: Annotated[Settings, Depends(get_settings)],
        langfuse:  Annotated[LangfuseCallbackService, Depends(get_langfuse_service)],
    ):
        self._settings = settings
        self._langfuse = langfuse
        self._graph = build_oem_agent_graph(langfuse)

    async def run_agent(self, request: RunAgentRequest) -> RunAgentResponse:
        if not request.files:
            raise HTTPException(400, "At least one PDF file is required.")
        if len(request.files) > 6:
            raise HTTPException(400, "Maximum 6 files (2 per company).")

        logger.info("[Controller] run_agent — %d files", len(request.files))

        initial = {
            "files": request.files,
            "session_id": request.session_id,
            "messages": [],
            "extractions": [],
            "substitution_notes": [],
            "validation_warnings": [],
            "summary_table": None,
            "executive_narrative": "",
            "status": "pending",
            "error": None,
        }

        try:
            final = await self._graph.ainvoke(initial)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("[Controller] Graph failed: %s", exc, exc_info=True)
            raise HTTPException(500, f"Agent execution failed: {exc}") from exc

        if final.get("error"):
            raise HTTPException(500, final["error"])

        return self._build_response(final)

    def _build_response(self, state: dict) -> RunAgentResponse:
        table_raw = state.get("summary_table") or {}
        columns: list[str] = table_raw.get("columns", [])
        rows_raw: dict = table_raw.get("rows", {})

        # Build typed table preserving KPI row order
        typed_table: dict[str, dict] = {}
        for _, label in KPI_ROW_ORDER:
            if label in rows_raw:
                typed_table[label] = {
                    col_key: SummaryTableCell(**cell)
                    for col_key, cell in rows_raw[label].items()
                }

        # Derive unique companies and periods from actual column keys
        companies = list(dict.fromkeys(
            _split_column_key(col)[0] for col in columns
        ))
        periods = list(dict.fromkeys(
            _split_column_key(col)[1] for col in columns
        ))

        return RunAgentResponse(
            status=state.get("status", "complete"),
            companies=companies,
            periods=periods,
            table={"columns": columns, "rows": typed_table},
            substitution_notes=[
                KPISubstitutionNote(**n) for n in state.get("substitution_notes", [])
            ],
            executive_narrative=state.get("executive_narrative", ""),
            warnings=state.get("validation_warnings", []),
            langfuse_trace_url=self._langfuse.get_trace_url(),
        )

    @staticmethod
    async def encode_file(file: UploadFile, company: str, report_type: str) -> dict:
        content = await file.read()
        if not content:
            raise HTTPException(400, f"File {file.filename} is empty.")
        return {
            "company": company,
            "report_type": report_type,
            "filename": file.filename or f"{company}_{report_type}.pdf",
            "content_b64": base64.b64encode(content).decode(),
        }