

import base64
import logging
import re
from typing import Annotated

from fastapi import Depends, HTTPException, UploadFile

from api.config.settings import get_settings, Settings
from api.models.oem_models import (
    RunAgentRequest, RunAgentResponse, KPISubstitutionNote, SummaryTableCell,
)
from api.services.langfuse_callback_service import LangfuseCallbackService, get_langfuse_service
from api.services.oem_agent.oem_generation_service import build_oem_agent_graph, KPI_ROW_ORDER

logger = logging.getLogger(__name__)

# Matches any period label produced by _detect_period()
_PERIOD_RE = re.compile(
    r'(FY\d{4}|Q[1-4]\s+\d{4}|H[12]\s+\d{4}|9M\s+\d{4}|\d{1,2}M\s+\d{4})$'
)


def _split_column_key(col: str):
    """Split 'BMW Group FY2025' → ('BMW Group', 'FY2025'). Works for any OEM name."""
    m = _PERIOD_RE.search(col)
    if m:
        return col[:m.start()].strip(), m.group(1)
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

        typed_table: dict[str, dict] = {}
        for _, label in KPI_ROW_ORDER:
            if label in rows_raw:
                typed_table[label] = {
                    col_key: SummaryTableCell(**cell)
                    for col_key, cell in rows_raw[label].items()
                }

        companies = list(dict.fromkeys(_split_column_key(col)[0] for col in columns))
        periods   = list(dict.fromkeys(_split_column_key(col)[1] for col in columns))

        # FIX: note field may be None when a KPI was not_disclosed/not_reported.
        # KPISubstitutionNote.note is declared as str, so coerce None → "".
        def _safe_note(n: dict) -> KPISubstitutionNote:
            return KPISubstitutionNote(
                company=n.get("company", ""),
                period=n.get("period", ""),
                canonical_name=n.get("canonical_name", ""),
                found_as=n.get("found_as", ""),
                note=n.get("note") or "",   # ← None → ""
            )

        return RunAgentResponse(
            status=state.get("status", "complete"),
            companies=companies,
            periods=periods,
            table={"columns": columns, "rows": typed_table},
            substitution_notes=[
                _safe_note(n) for n in state.get("substitution_notes", [])
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