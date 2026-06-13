"""Pydantic models for OEM financial KPI extraction."""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class KPIValue(BaseModel):
    canonical_name: str
    value: Optional[float] = None
    formatted_value: str = "N/A"
    unit: str = ""
    period: str
    found_as: str = ""
    is_substitute: bool = False
    substitute_note: Optional[str] = None
    source_page: Optional[int] = None
    not_reported: bool = False


class CompanyExtraction(BaseModel):
    company: str
    report_type: str
    period_label: str
    revenue: KPIValue
    ebit: KPIValue
    ebit_margin: KPIValue
    cash_kpi: KPIValue
    net_liquidity: KPIValue
    return_on_capital: KPIValue
    cost_of_capital: Optional[KPIValue] = None
    eps: Optional[KPIValue] = None
    dividend_per_share: Optional[KPIValue] = None
    market_cap_at_100eur: Optional[KPIValue] = None
    extraction_warnings: list[str] = Field(default_factory=list)


class RunAgentRequest(BaseModel):
    files: list[dict] = Field(description="[{company, report_type, content_b64, filename}]")
    session_id: str = ""


class KPISubstitutionNote(BaseModel):
    company: str
    period: str
    canonical_name: str
    found_as: str
    note: str


class SummaryTableCell(BaseModel):
    value: str
    is_substitute: bool = False
    substitute_note: Optional[str] = None
    not_reported: bool = False


class RunAgentResponse(BaseModel):
    status: str
    companies: list[str]
    periods: list[str]
    table: dict
    substitution_notes: list[KPISubstitutionNote]
    executive_narrative: str
    warnings: list[str] = Field(default_factory=list)
    langfuse_trace_url: Optional[str] = None
