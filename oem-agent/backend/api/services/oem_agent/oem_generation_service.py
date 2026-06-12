

import base64
import json
import logging
import re
from typing import TypedDict, Annotated, Optional
from operator import add

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from api.config.settings import get_settings
from api.models.prompt_id import PromptId
from api.services.langfuse_callback_service import LangfuseCallbackService
from api.services.oem_agent.extraction_tools import build_extraction_tools

logger = logging.getLogger(__name__)

KPI_ROW_ORDER = [
    ("revenue",              "Revenue"),
    ("ebit",                 "EBIT / Operating Result"),
    ("ebit_margin",          "EBIT Margin"),
    ("cash_kpi",             "Cash KPI (preferred)"),
    ("net_liquidity",        "Net Liquidity"),
    ("return_on_capital",    "Return on Capital (ROIC/ROCE)"),
    ("cost_of_capital",      "Cost of Capital / WACC"),
    ("eps",                  "EPS"),
    ("dividend_per_share",   "Dividend per Share"),
    ("market_cap_at_100eur", "Market Cap @ €100/share"),
]


# ── Period detection ──────────────────────────────────────────────────────────

# Ordered from most-specific to least-specific so the first match wins.
_PERIOD_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "Full Year 2025", "Full-Year 2024", "FY 2025", "FY2025"
    (re.compile(r'\bfull[-\s]?year\s+(\d{4})\b', re.I),        "FY{year}"),
    (re.compile(r'\bfy\s*(\d{4})\b',              re.I),        "FY{year}"),
    (re.compile(r'\bannual\s+report\s+(\d{4})\b', re.I),        "FY{year}"),
    # "Q1 2026", "First Quarter 2026", "1st Quarter 2025"
    (re.compile(r'\bq1\s+(\d{4})\b',              re.I),        "Q1 {year}"),
    (re.compile(r'\bfirst\s+quarter\s+(\d{4})\b', re.I),        "Q1 {year}"),
    (re.compile(r'\b1st\s+quarter\s+(\d{4})\b',   re.I),        "Q1 {year}"),
    # Q2
    (re.compile(r'\bq2\s+(\d{4})\b',              re.I),        "Q2 {year}"),
    (re.compile(r'\bsecond\s+quarter\s+(\d{4})\b',re.I),        "Q2 {year}"),
    (re.compile(r'\b2nd\s+quarter\s+(\d{4})\b',   re.I),        "Q2 {year}"),
    # Q3
    (re.compile(r'\bq3\s+(\d{4})\b',              re.I),        "Q3 {year}"),
    (re.compile(r'\bthird\s+quarter\s+(\d{4})\b', re.I),        "Q3 {year}"),
    (re.compile(r'\b3rd\s+quarter\s+(\d{4})\b',   re.I),        "Q3 {year}"),
    # Q4
    (re.compile(r'\bq4\s+(\d{4})\b',              re.I),        "Q4 {year}"),
    (re.compile(r'\bfourth\s+quarter\s+(\d{4})\b',re.I),        "Q4 {year}"),
    (re.compile(r'\b4th\s+quarter\s+(\d{4})\b',   re.I),        "Q4 {year}"),
    # "nine months ended", "six months ended" → H1/9M labels
    (re.compile(r'\bnine\s+months?\s+(?:ended\s+)?(\d{4})\b',   re.I), "9M {year}"),
    (re.compile(r'\bsix\s+months?\s+(?:ended\s+)?(\d{4})\b',    re.I), "H1 {year}"),
    # Bare year as last resort — only used when report_type=FY
    (re.compile(r'\b(20[2-9]\d)\b'),                             "FY{year}"),
]


def _detect_period(text: str, report_type: str) -> str:
    """
    Scan the first 3000 chars of extracted PDF text and return a clean period
    label like "FY2025", "Q1 2026", "Q3 2025".

    Falls back to safe defaults:
      FY  → "FY2025"
      Q   → "Q4 2025"
    """
    sample = text[:3000]

    if report_type == "FY":
        # For full-year reports use only FY / annual / bare-year patterns
        fy_patterns = [p for p, t in _PERIOD_PATTERNS if "{year}" in t and t.startswith("FY")]
        for pattern, template in zip(fy_patterns, [t for _, t in _PERIOD_PATTERNS if t.startswith("FY")]):
            m = pattern.search(sample)
            if m:
                label = template.replace("{year}", m.group(1))
                logger.info("[Period] Detected '%s' from FY report text", label)
                return label
        logger.warning("[Period] Could not detect FY period — defaulting to FY2025")
        return "FY2025"

    else:
        # For quarterly reports try Q1-Q4 patterns first
        q_patterns = [(p, t) for p, t in _PERIOD_PATTERNS
                      if t.startswith("Q") or t.startswith("H") or t.startswith("9M")]
        for pattern, template in q_patterns:
            m = pattern.search(sample)
            if m:
                label = template.replace("{year}", m.group(1))
                logger.info("[Period] Detected '%s' from Q report text", label)
                return label
        logger.warning("[Period] Could not detect Q period — defaulting to Q4 2025")
        return "Q4 2025"


# ── PDF text extraction ───────────────────────────────────────────────────────

# Smart chunking limits — see _extract_pdf_text docstring for rationale.
_MAX_CHARS_QUARTERLY = 100_000
_MAX_CHARS_FY_HEAD   =  40_000
_MAX_CHARS_FY_TAIL   =  80_000


def _extract_pdf_text(content_b64: str, report_type: str = "Q") -> str:
    """
    Decode a base64 PDF and extract all text via PyMuPDF.

    Root cause of FY2025 Net Liquidity / Market Cap being Not Reported:
      BMW FY2025 annual report = 1,695,884 chars. Previous code truncated at
      80,000 chars (head-only). BMW balance-sheet items — Automotive Net
      Financial Assets (net liquidity) and shares outstanding (market cap) —
      appear in the financial-statements section which starts ~1.2M chars into
      the document. The LLM never saw them.

    Strategy by report type:

      Quarterly (report_type="Q"):
        Short documents (< 100k typically). Send full text up to 100k.

      Full-year (report_type="FY"):
        HEAD (40k) + TAIL (80k) = 120k chars total.
          HEAD: cover page, period detection, Revenue, EBIT, FCF — always early.
          TAIL: financial statements, balance sheet, shares outstanding,
                net liquidity, RoCE tables — always late in annual reports.
        A separator line tells the LLM the middle was omitted.
    """
    try:
        import fitz  # PyMuPDF
        raw = base64.b64decode(content_b64)
        doc = fitz.open(stream=raw, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        full_text = "\n\n".join(pages)
        total_chars = len(full_text)

        if report_type == "FY" and total_chars > (_MAX_CHARS_FY_HEAD + _MAX_CHARS_FY_TAIL):
            head = full_text[:_MAX_CHARS_FY_HEAD]
            tail = full_text[-_MAX_CHARS_FY_TAIL:]
            result = (
                head
                + "\n\n[... DOCUMENT MIDDLE OMITTED FOR CONTEXT WINDOW — "
                  "FINANCIAL TABLES CONTINUE BELOW ...]\n\n"
                + tail
            )
            logger.info(
                "[PDF] FY smart-chunk: %d total → head %d + tail %d = %d chars sent",
                total_chars, len(head), len(tail), len(result),
            )
        else:
            max_chars = _MAX_CHARS_QUARTERLY
            if total_chars > max_chars:
                logger.warning(
                    "[PDF] Q report truncated %d → %d chars", total_chars, max_chars
                )
                full_text = full_text[:max_chars]
            result = full_text

        return result.strip()

    except ImportError:
        logger.error("[PDF] PyMuPDF not installed — pip install pymupdf")
        return "[ERROR: PyMuPDF not installed. Run: pip install pymupdf]"
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("[PDF] Extraction failed: %s", exc)
        return f"[ERROR extracting PDF text: {exc}]"


# ── LangGraph state ───────────────────────────────────────────────────────────

class OEMGraphState(TypedDict):
    files: list[dict]
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    extractions: Annotated[list[dict], add]
    substitution_notes: Annotated[list[dict], add]
    validation_warnings: Annotated[list[str], add]
    summary_table: Optional[dict]
    executive_narrative: str
    status: str
    error: Optional[str]


# ── Generation service ────────────────────────────────────────────────────────

class OEMGenerationService:
    def __init__(self, langfuse: LangfuseCallbackService):
        self._settings = get_settings()
        self._langfuse = langfuse
        self._tools = build_extraction_tools()
        self._llm = self._build_llm()

    def _build_llm(self) -> ChatOpenAI:
        llm = ChatOpenAI(
            model=self._settings.OPENROUTER_MODEL,
            openai_api_key=self._settings.OPENROUTER_API_KEY,
            openai_api_base=self._settings.OPENROUTER_BASE_URL,
            temperature=0.0,
            max_tokens=4096,
        )
        return llm.bind_tools(self._tools)

    # ── Node 1: ingest ────────────────────────────────────────────────────────

    async def ingest_pdfs(self, state: OEMGraphState) -> dict:
        """Extract real text + detect period label for each PDF."""
        logger.info("[Agent] ingest_pdfs — %d files", len(state["files"]))
        enriched = []
        for f in state["files"]:
            text = _extract_pdf_text(f["content_b64"], report_type=f.get("report_type", "Q"))
            period_label = _detect_period(text, f.get("report_type", "FY"))
            logger.info(
                "[Agent] %s (%s): %d chars, period='%s'",
                f.get("filename", "?"), f.get("company", "?"),
                len(text), period_label,
            )
            enriched.append({**f, "extracted_text": text, "period_label": period_label})
        return {"files": enriched, "status": "ingested"}

    # ── Node 2: extract ───────────────────────────────────────────────────────

    async def extract_metrics(self, state: OEMGraphState) -> dict:
        """One focused LLM call per file with detected period label."""
        logger.info("[Agent] extract_metrics — %d files", len(state["files"]))

        base_system_prompt = await self._langfuse.aget_prompt(
            PromptId.OEM_EXTRACTION_SYSTEM_PROMPT
        )
        callbacks = []
        cb = self._langfuse.get_langchain_handler(session_id=state.get("session_id", ""))
        if cb:
            callbacks.append(cb)

        config = {"callbacks": callbacks} if callbacks else {}
        all_extractions: list[dict] = []

        for f in state["files"]:
            company      = f.get("company", "Unknown")
            report_type  = f.get("report_type", "FY")
            filename     = f.get("filename", "unknown.pdf")
            period_label = f.get("period_label") or (
                "FY2025" if report_type == "FY" else "Q4 2025"
            )
            extracted_text = f.get("extracted_text", "")

            if not extracted_text or extracted_text.startswith("[ERROR"):
                logger.warning("[Agent] No text for %s — stub", filename)
                all_extractions.append(self._stub(f, period_label))
                continue

            report_desc = (
                f"Full Year ({period_label})"
                if report_type == "FY"
                else f"Quarterly ({period_label})"
            )

            focused_system = (
                f"{base_system_prompt}\n\n"
                f"=== CURRENT FILE ===\n"
                f"Company: {company}\n"
                f"Report type: {report_desc}\n"
                f"Period label to use in ALL tool calls: {period_label}\n"
                f"Filename: {filename}\n\n"
                f"IMPORTANT RULES FOR THIS FILE:\n"
                f"1. Use period='{period_label}' in every single tool call — no other value.\n"
                f"2. Pass the `unit` field exactly as written in the report "
                f"(e.g. 'EUR m', 'EUR bn', 'in millions') — the system will convert automatically.\n"
                f"3. Call every tool. If a KPI is absent, call the tool with not_reported=true.\n"
                f"4. Do NOT skip tools for missing data."
            )

            messages = [
                SystemMessage(content=focused_system),
                HumanMessage(content=(
                    f"Extract all financial KPIs from this {company} "
                    f"{'full-year' if report_type == 'FY' else 'quarterly'} report:\n\n"
                    f"{extracted_text}"
                )),
            ]

            try:
                response = await self._llm.ainvoke(messages, config=config)
                extraction = self._parse_single_file_response(
                    response, company, report_type, period_label
                )
                all_extractions.append(extraction)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("[Agent] LLM call failed for %s: %s", filename, exc)
                all_extractions.append(self._stub(f, period_label))

        return {"extractions": all_extractions, "status": "extracted"}

    def _parse_single_file_response(
        self,
        response,
        company: str,
        report_type: str,
        period_label: str,
    ) -> dict:
        tool_calls = getattr(response, "tool_calls", []) or []
        tool_results: dict = {}

        logger.debug("[Agent] %s (%s): %d tool calls", company, period_label, len(tool_calls))

        for tc in tool_calls:
            name = tc.get("name", "")
            args = dict(tc.get("args", {}))
            # Always enforce the correct period
            if "period" in args:
                args["period"] = period_label

            tool_fn = next((t for t in self._tools if t.name == name), None)
            if not tool_fn:
                logger.warning("[Agent] Unknown tool: %s", name)
                continue
            try:
                result = tool_fn.func(**args)
                tool_results[name] = result
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("[Agent] Tool %s failed: %s (args=%s)", name, exc, args)

        return self._assemble(company, report_type, period_label, tool_results)

    def _assemble(self, company, report_type, period_label, tool_results) -> dict:
        def _not_reported_stub(canonical: str) -> dict:
            return dict(
                canonical_name=canonical, value=None,
                formatted_value="Not Reported", unit="",
                period=period_label, found_as="",
                is_substitute=False, substitute_note=None,
                source_page=None, not_reported=True, derived=False,
            )

        def _get(tool_name: str, canonical: str) -> dict:
            r = tool_results.get(tool_name)
            if r and isinstance(r, dict):
                return r
            return _not_reported_stub(canonical)

        eps_div = tool_results.get("extract_eps_dividend") or {}
        if not isinstance(eps_div, dict):
            eps_div = {}

        revenue    = _get("extract_revenue",           "Revenue")
        ebit       = _get("extract_ebit",              "EBIT")
        ebit_margin = _get("extract_ebit_margin",      "EBIT Margin")

        # ── EBIT Margin derivation ────────────────────────────────────────────
        # If the tool call for EBIT Margin failed (not_reported=True) but we
        # successfully extracted both EBIT and Revenue, compute the margin
        # ourselves and flag it as derived so the table shows a ≈ badge.
        if (
            ebit_margin.get("not_reported")
            and ebit.get("value") is not None
            and revenue.get("value") is not None
            and revenue["value"] != 0
        ):
            derived_pct = round((ebit["value"] / revenue["value"]) * 100, 1)
            ebit_margin = dict(
                canonical_name="EBIT Margin",
                value=derived_pct,
                formatted_value=f"{derived_pct:.1f}%",
                unit="%",
                period=period_label,
                found_as="Derived: EBIT ÷ Revenue",
                is_substitute=True,
                substitute_note=(
                    f"EBIT Margin derived as EBIT ÷ Revenue "
                    f"({ebit['formatted_value']} ÷ {revenue['formatted_value']} × 100)"
                ),
                source_page=None,
                not_reported=False,
                derived=True,
            )
            logger.info(
                "[Agent] EBIT Margin derived for %s (%s): %.1f%%",
                company, period_label, derived_pct,
            )

        return {
            "company":             company,
            "report_type":         report_type,
            "period_label":        period_label,
            "revenue":             revenue,
            "ebit":                ebit,
            "ebit_margin":         ebit_margin,
            "cash_kpi":            _get("extract_cash_kpi",          "Free Cash Flow"),
            "net_liquidity":       _get("extract_net_liquidity",     "Net Liquidity"),
            "return_on_capital":   _get("extract_return_on_capital", "ROIC"),
            "cost_of_capital":     tool_results.get("extract_cost_of_capital"),
            "eps":                 eps_div.get("eps"),
            "dividend_per_share":  eps_div.get("dividend_per_share"),
            "market_cap_at_100eur":tool_results.get("extract_market_cap"),
            "extraction_warnings": [],
        }

    def _stub(self, file: dict, period_label: str = "") -> dict:
        if not period_label:
            period_label = "FY2025" if file.get("report_type") == "FY" else "Q4 2025"

        def s(n):
            return dict(
                canonical_name=n, value=None, formatted_value="Not Reported",
                unit="", period=period_label, found_as="",
                is_substitute=False, substitute_note=None,
                source_page=None, not_reported=True,
            )

        return {
            "company":             file.get("company", "Unknown"),
            "report_type":         file.get("report_type", "FY"),
            "period_label":        period_label,
            "revenue":             s("Revenue"),
            "ebit":                s("EBIT"),
            "ebit_margin":         s("EBIT Margin"),
            "cash_kpi":            s("Free Cash Flow"),
            "net_liquidity":       s("Net Liquidity"),
            "return_on_capital":   s("ROIC"),
            "cost_of_capital":     None,
            "eps":                 None,
            "dividend_per_share":  None,
            "market_cap_at_100eur":None,
            "extraction_warnings": ["PDF could not be parsed by LLM"],
        }

    # ── Extraction quality check ──────────────────────────────────────────────

    @staticmethod
    def _extraction_has_data(extractions: list[dict]) -> bool:
        """
        Return True if at least one extraction has at least one real (non-stub)
        KPI value.  A real value means not_reported=False and value is not None.
        """
        monetary_fields = [
            "revenue", "ebit", "cash_kpi", "net_liquidity", "return_on_capital",
        ]
        for ex in extractions:
            for field in monetary_fields:
                kpi = ex.get(field)
                if (
                    kpi
                    and isinstance(kpi, dict)
                    and not kpi.get("not_reported", True)
                    and kpi.get("value") is not None
                ):
                    return True
        return False

    # ── Node 3: validate ──────────────────────────────────────────────────────

    async def validate(self, state: OEMGraphState) -> dict:
        extractions = state["extractions"]
        logger.info("[Agent] validate — %d extractions", len(extractions))

        # Early-exit: if every extraction is a stub, skip downstream work
        if not self._extraction_has_data(extractions):
            logger.error(
                "[Agent] All extractions are empty stubs — no real KPI data found. "
                "Check PyMuPDF installation and that PDFs contain selectable text."
            )
            return {
                "substitution_notes":  [],
                "validation_warnings": [
                    "Extraction produced no data. Possible causes: "
                    "(1) PyMuPDF not installed, (2) scanned/image PDFs with no selectable text, "
                    "(3) LLM API key missing or quota exceeded, "
                    "(4) Model returned no tool calls. "
                    "Check backend logs for details."
                ],
                "status": "extraction_failed",
            }
        notes, warnings = [], []
        fields = [f for f, _ in KPI_ROW_ORDER]

        for ex in state["extractions"]:
            co     = ex.get("company", "?")
            period = ex.get("period_label", "?")
            for field in fields:
                kpi = ex.get(field)
                if not kpi or not isinstance(kpi, dict):
                    continue
                if kpi.get("is_substitute"):
                    notes.append(dict(
                        company=co, period=period,
                        canonical_name=kpi["canonical_name"],
                        found_as=kpi["found_as"],
                        note=kpi.get("substitute_note", ""),
                    ))
                fv = kpi.get("formatted_value", "")
                if kpi.get("not_reported") and not str(fv).startswith("N/A"):
                    warnings.append(
                        f"{co} ({period}): '{kpi['canonical_name']}' not reported."
                    )

        return {
            "substitution_notes":  notes,
            "validation_warnings": warnings,
            "status": "validated",
        }

    # ── Node 4: synthesize ────────────────────────────────────────────────────

    async def synthesize_summary(self, state: OEMGraphState) -> dict:
        logger.info("[Agent] synthesize_summary")
        extractions = state["extractions"]

        # Propagate extraction failure cleanly to the API response
        if state.get("status") == "extraction_failed" or not self._extraction_has_data(extractions):
            return {
                "summary_table": {"columns": [], "rows": {}},
                "executive_narrative": (
                    "Extraction failed — no financial data could be read from the uploaded PDFs. "
                    "Please verify that: (1) the PDFs contain selectable text (not scanned images), "
                    "(2) PyMuPDF is installed (`pip install pymupdf`), and "
                    "(3) your OpenRouter API key is valid and has available quota."
                ),
                "status": "extraction_failed",
            }

        columns = list(dict.fromkeys(
            f"{ex['company']} {ex['period_label']}" for ex in extractions
        ))

        rows: dict[str, dict] = {}
        for field, label in KPI_ROW_ORDER:
            rows[label] = {}
            for ex in extractions:
                col = f"{ex['company']} {ex['period_label']}"
                kpi = ex.get(field)
                if kpi and isinstance(kpi, dict):
                    rows[label][col] = dict(
                        value=kpi.get("formatted_value", "Not Reported"),
                        is_substitute=kpi.get("is_substitute", False),
                        substitute_note=kpi.get("substitute_note"),
                        not_reported=kpi.get("not_reported", False),
                    )
                else:
                    rows[label][col] = dict(
                        value="N/A", is_substitute=False,
                        substitute_note=None, not_reported=False,
                    )

        narrative = await self._generate_narrative(state, rows, columns)
        return {
            "summary_table": {"columns": columns, "rows": rows},
            "executive_narrative": narrative,
            "status": "complete",
        }

    async def _generate_narrative(self, state, table, columns) -> str:
        try:
            system_prompt = await self._langfuse.aget_prompt(
                PromptId.OEM_SYNTHESIS_SYSTEM_PROMPT
            )
            llm = ChatOpenAI(
                model=self._settings.OPENROUTER_MODEL,
                openai_api_key=self._settings.OPENROUTER_API_KEY,
                openai_api_base=self._settings.OPENROUTER_BASE_URL,
                temperature=0.3,
                max_tokens=600,
            )
            resp = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=(
                    f"Columns (companies + periods): {columns}\n\n"
                    f"Substitutions: {json.dumps(state.get('substitution_notes', []))}\n\n"
                    f"KPI table:\n{json.dumps(table, indent=2)}"
                )),
            ])
            return resp.content
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("[Agent] Narrative failed: %s", exc)
            return "Executive narrative unavailable — see the KPI table below."


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_oem_agent_graph(langfuse: LangfuseCallbackService):
    svc = OEMGenerationService(langfuse)
    builder = StateGraph(OEMGraphState)
    builder.add_node("ingest_pdfs",        svc.ingest_pdfs)
    builder.add_node("extract_metrics",    svc.extract_metrics)
    builder.add_node("validate",           svc.validate)
    builder.add_node("synthesize_summary", svc.synthesize_summary)
    builder.set_entry_point("ingest_pdfs")
    builder.add_edge("ingest_pdfs",        "extract_metrics")
    builder.add_edge("extract_metrics",    "validate")
    builder.add_edge("validate",           "synthesize_summary")
    builder.add_edge("synthesize_summary", END)
    return builder.compile()