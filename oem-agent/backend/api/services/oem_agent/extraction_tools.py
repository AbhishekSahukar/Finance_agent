
import logging
from typing import Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Synonym registry ──────────────────────────────────────────────────────────

KPI_SYNONYMS: dict[str, list[str]] = {
    "Revenue": [
        "revenue", "total revenue", "sales", "net revenue", "net sales",
        "group revenue", "turnover", "revenues", "group sales", "consolidated revenue",
    ],
    "EBIT": [
        "ebit", "operating profit", "operating result", "operating income",
        "earnings before interest and taxes", "profit from operations",
        "operating profit/(loss)", "adjusted ebit", "adjusted operating profit",
        "ebit (loss)", "profit/(loss) from operations",
    ],
    "EBIT Margin": [
        "ebit margin", "operating margin", "return on sales", "ros",
        "operating profit margin", "ebit as % of revenue", "return on sales (ros)",
        "ebit margin in the automotive segment", "automotive ebit margin",
        "ebit margin (automotive segment)",
    ],
    "Free Cash Flow": [
        "free cash flow", "fcf", "cash generation", "industrial free cash flow",
        "adjusted fcf", "automotive free cash flow", "automotive fcf",
        "adjusted industrial free cash flow", "free cash flow before dividends",
        "free cash flow (automotive segment)",
    ],
    "Net Liquidity": [
        # Generic
        "net liquidity", "net cash position", "net cash", "net financial position",
        "net liquidity / net financial position", "net debt", "net cash / net debt",
        "net cash and cash equivalents", "net liquidity position",
        # BMW-specific (from real PDF logs)
        "automotive net financial assets", "automotive net financial position",
        "net financial assets", "industrial net liquidity",
        "net financial assets (automotive segment)",
        "automotive segment net financial assets",
    ],
    "ROIC": [
        # Generic
        "roic", "return on invested capital", "roce", "return on capital employed",
        "return on net assets", "rona", "capital efficiency", "return on capital",
        "adjusted roic",
        # BMW-specific — the parenthetical forms the LLM actually sends
        "return on capital employed (roce)",
        "return on capital employed (roic)",
        "roce (automotive)", "roic (automotive)",
        "return on equity (roe)",          # only if no better match
        "roce automotive segment",
    ],
    "Cost of Capital": [
        "wacc", "weighted average cost of capital", "hurdle rate",
        "cost of capital", "required return", "minimum return", "threshold return",
    ],
    "EPS": [
        "eps", "earnings per share", "basic eps", "diluted eps",
        "earnings per ordinary share", "net income per share",
        "earnings per share (basic)", "earnings per share (diluted)",
        "earnings per ordinary share (basic)", "earnings per ordinary share (diluted)",
    ],
    "Dividend per Share": [
        "dividend per share", "dps", "dividend", "proposed dividend",
        "dividend per ordinary share", "annual dividend",
        "dividend per share (proposed)", "dividend per common share",
    ],
    "Shares Outstanding": [
        "shares outstanding", "number of shares", "ordinary shares",
        "shares issued", "weighted average shares", "total shares", "shares in issue",
    ],
}


# ── Unit helpers ──────────────────────────────────────────────────────────────

_MONETARY_UNIT  = "EUR bn"
_PERCENT_UNIT   = "%"
_PER_SHARE_UNIT = "EUR"

# Strings that mean the raw value is in millions
_MILLION_SIGNALS = (
    "eur m", "eur million", "eur millions", "€m", "€ m",
    "million eur", "millions eur", "m eur", "meur",
    "million €", "millions €", "in millions", "mio", "mio.",
    "€ million", "€ millions", "million", "millions",
)

# Strings that mean the raw value is already in billions
_BILLION_SIGNALS = (
    "eur bn", "eur billion", "eur billions", "€bn", "€ bn",
    "billion eur", "bn eur", "billion €", "in billions",
    "€ billion", "€ billions",
)


def _infer_monetary_unit(raw_unit: Optional[str], found_as: str = "") -> str:
    """
    Determine whether raw_value is in millions or billions, returning
    the normalised signal string for _normalise_monetary().

    Priority:
      1. raw_unit if it contains a known signal
      2. found_as context (e.g. "Net Liquidity in EUR million")
      3. Default: treat as billions (OEM headline figures are always bn-scale)
    """
    for source in (raw_unit or "", found_as):
        sl = source.strip().lower()
        if any(sig in sl for sig in _MILLION_SIGNALS):
            return "EUR m"          # signals: divide by 1000
        if any(sig in sl for sig in _BILLION_SIGNALS):
            return "EUR bn"         # signals: keep as-is
    # No unit signal found → assume billions (safe for OEM financials)
    logger.debug("[Unit] No unit signal in '%s' / '%s' — assuming EUR bn", raw_unit, found_as)
    return "EUR bn"


def _normalise_monetary(raw_value: float, raw_unit: str) -> tuple[float, str]:
    """Convert to EUR bn. Divides by 1000 if unit signals millions."""
    unit_lower = raw_unit.strip().lower()
    if any(sig in unit_lower for sig in _MILLION_SIGNALS):
        logger.debug("[Unit] %s %s → EUR bn (/1000)", raw_value, raw_unit)
        return raw_value / 1000.0, _MONETARY_UNIT
    return raw_value, _MONETARY_UNIT


# ── Synonym resolution ────────────────────────────────────────────────────────

def _resolve(raw_label: str) -> tuple[str, bool]:
    """
    Return (canonical_name, is_substitute).
    is_substitute=True when the raw label differs from the canonical name.
    Strips parenthetical suffixes for broader matching before exact lookup.
    """
    norm = raw_label.strip().lower()

    # Exact lookup first
    for canonical, synonyms in KPI_SYNONYMS.items():
        if norm in synonyms:
            return canonical, norm != canonical.lower()

    # Strip trailing parenthetical (e.g. "return on capital employed (roce)" →
    # try "return on capital employed" as a fallback)
    stripped = norm.rsplit("(", 1)[0].strip()
    if stripped != norm:
        for canonical, synonyms in KPI_SYNONYMS.items():
            if stripped in synonyms:
                return canonical, True   # always a substitute — label was modified

    return raw_label.strip(), False


# ── Core builder ──────────────────────────────────────────────────────────────

def _build(
    canonical: str,
    raw_value: Optional[float],
    unit: Optional[str],
    period: str,
    found_as: str,
    source_page: Optional[int] = None,
    not_reported: bool = False,
    kpi_family: str = "monetary",   # "monetary" | "percent" | "per_share"
    derived: bool = False,
    derived_note: Optional[str] = None,
) -> dict:
    """
    Build a KPIValue-compatible dict with full provenance.

    unit may now be None — the function infers the correct unit from
    kpi_family when the LLM omits it.
    """
    resolved, is_sub = _resolve(found_as) if found_as else (canonical, False)

    # Build substitution note
    if derived and derived_note:
        note = derived_note
    elif is_sub:
        note = f"'{found_as}' used as equivalent for '{canonical}' (KPI synonym map)"
    else:
        note = None

    # Determine display unit from family when missing
    if unit is None:
        unit = (
            _MONETARY_UNIT  if kpi_family == "monetary"  else
            _PERCENT_UNIT   if kpi_family == "percent"   else
            _PER_SHARE_UNIT
        )

    display_unit = (
        _MONETARY_UNIT  if kpi_family == "monetary"  else
        _PERCENT_UNIT   if kpi_family == "percent"   else
        _PER_SHARE_UNIT
    )

    if not_reported:
        return dict(
            canonical_name=canonical, value=None,
            formatted_value="Not Reported",
            unit=display_unit, period=period,
            found_as=found_as or canonical,
            is_substitute=is_sub or derived,
            substitute_note=note,
            source_page=source_page,
            not_reported=True,
            derived=derived,
        )

    if raw_value is not None:
        try:
            v = float(str(raw_value).replace(",", ".").replace(" ", ""))

            if kpi_family == "monetary":
                effective_unit = _infer_monetary_unit(unit, found_as)
                v, display_unit = _normalise_monetary(v, effective_unit)
                fmt = f"€{v:,.2f} bn"

            elif kpi_family == "percent":
                display_unit = _PERCENT_UNIT
                # Guard: LLM may return 0.085 instead of 8.5
                if 0 < abs(v) < 1:
                    v = v * 100
                    logger.debug("[Unit] Decimal percent %s → %.1f%%", raw_value, v)
                fmt = f"{v:.1f}%"

            else:  # per_share
                display_unit = _PER_SHARE_UNIT
                fmt = f"€{v:.2f}"

        except (ValueError, TypeError):
            v, fmt, display_unit = None, str(raw_value), unit
    else:
        v, fmt = None, "N/A"

    return dict(
        canonical_name=canonical, value=v, formatted_value=fmt,
        unit=display_unit, period=period,
        found_as=found_as or canonical,
        is_substitute=is_sub or derived,
        substitute_note=note,
        source_page=source_page,
        not_reported=False,
        derived=derived,
    )


# ── Pydantic schemas ──────────────────────────────────────────────────────────
# ALL fields that the LLM may omit are Optional with safe defaults.
# This is the primary fix: Pydantic validates the LLM args before our
# function is called, so if a field is required in the schema it raises
# ValidationError (which we see as TypeError in the logs).

class _MonetaryBase(BaseModel):
    raw_value:   Optional[float] = Field(default=None,     description="Numeric value as found in the report. Omit or null if not_reported=true.")
    unit:        Optional[str]   = Field(default=None,     description="Unit exactly as written: 'EUR bn', 'EUR m', '€ million'. Omit if unknown — system will infer.")
    period:      str
    found_as:    str             = Field(default="",       description="Exact label from the report.")
    source_page: Optional[int]   = Field(default=None)
    not_reported: bool           = Field(default=False,    description="Set true if this KPI does not appear in the report.")

class _PercentBase(BaseModel):
    raw_value:   Optional[float] = Field(default=None,     description="Percentage value, e.g. 8.5 for 8.5%. Omit or null if not_reported=true.")
    unit:        Optional[str]   = Field(default="%",      description="Always '%'. Omit entirely — system always uses %.")
    period:      str
    found_as:    str             = Field(default="",       description="Exact label from the report.")
    source_page: Optional[int]   = Field(default=None)
    not_reported: bool           = Field(default=False)

class ExtractRevenueInput(_MonetaryBase): pass
class ExtractEBITInput(_MonetaryBase): pass
class ExtractEBITMarginInput(_PercentBase): pass
class ExtractCashKPIInput(_MonetaryBase): pass
class ExtractNetLiquidityInput(_MonetaryBase): pass
class ExtractReturnOnCapitalInput(_PercentBase): pass

class ExtractCostOfCapitalInput(_PercentBase):
    not_disclosed: bool = Field(default=False, description="Set true if WACC/hurdle rate is not published in the report.")

class ExtractEPSDividendInput(BaseModel):
    """
    EPS and Dividend are extracted in one call.
    Either value may be absent independently — missing one must not fail the other.
    """
    eps_value:       Optional[float] = Field(default=None, description="EPS value. Null if not available.")
    dividend_value:  Optional[float] = Field(default=None, description="Dividend per share. Null if not declared/available.")
    unit:            Optional[str]   = Field(default="EUR")
    period:          str
    eps_found_as:    str             = Field(default="", description="Exact EPS label from report.")
    dividend_found_as: str           = Field(default="", description="Exact dividend label from report.")
    source_page:     Optional[int]   = Field(default=None)
    not_applicable:  bool            = Field(default=False, description="True for non-listed entities (no shares/EPS).")
    not_reported:    bool            = Field(default=False)

class ExtractMarketCapInput(BaseModel):
    shares_outstanding_millions: Optional[float] = Field(default=None, description="Shares outstanding in millions. Null if not_applicable or not_reported.")
    period:         str
    source_page:    Optional[int]  = Field(default=None)
    not_applicable: bool           = Field(default=False)
    not_reported:   bool           = Field(default=False)


# ── Tool functions ────────────────────────────────────────────────────────────
# All positional args now have defaults matching the schema.
# Calling convention: all args passed as kwargs by LangChain ToolNode.

def extract_revenue(
    period: str,
    raw_value: Optional[float] = None,
    unit: Optional[str] = None,
    found_as: str = "",
    source_page: Optional[int] = None,
    not_reported: bool = False,
) -> dict:
    """Extract Revenue / Sales / Group Revenue / Turnover. Always normalised to EUR bn."""
    return _build("Revenue", raw_value, unit, period, found_as, source_page, not_reported, "monetary")


def extract_ebit(
    period: str,
    raw_value: Optional[float] = None,
    unit: Optional[str] = None,
    found_as: str = "",
    source_page: Optional[int] = None,
    not_reported: bool = False,
) -> dict:
    """Extract EBIT / Operating Profit / Operating Result / Operating Income. Always EUR bn."""
    return _build("EBIT", raw_value, unit, period, found_as, source_page, not_reported, "monetary")


def extract_ebit_margin(
    period: str,
    raw_value: Optional[float] = None,
    unit: Optional[str] = "%",
    found_as: str = "",
    source_page: Optional[int] = None,
    not_reported: bool = False,
) -> dict:
    """Extract EBIT Margin / Operating Margin / Return on Sales. Value as % (8.5 for 8.5%)."""
    return _build("EBIT Margin", raw_value, unit, period, found_as, source_page, not_reported, "percent")


def extract_cash_kpi(
    period: str,
    raw_value: Optional[float] = None,
    unit: Optional[str] = None,
    found_as: str = "",
    source_page: Optional[int] = None,
    not_reported: bool = False,
) -> dict:
    """Extract preferred cash metric: FCF / Industrial FCF / Automotive FCF. Always EUR bn."""
    return _build("Free Cash Flow", raw_value, unit, period, found_as, source_page, not_reported, "monetary")


def extract_net_liquidity(
    period: str,
    raw_value: Optional[float] = None,
    unit: Optional[str] = None,
    found_as: str = "",
    source_page: Optional[int] = None,
    not_reported: bool = False,
) -> dict:
    """
    Extract Net Liquidity / Net Cash Position / Automotive Net Financial Assets /
    Industrial Net Liquidity. Always EUR bn.
    """
    return _build("Net Liquidity", raw_value, unit, period, found_as, source_page, not_reported, "monetary")


def extract_return_on_capital(
    period: str,
    raw_value: Optional[float] = None,
    unit: Optional[str] = "%",
    found_as: str = "",
    source_page: Optional[int] = None,
    not_reported: bool = False,
) -> dict:
    """
    Extract ROIC / ROCE / Return on Capital Employed / Return on Net Assets.
    Value as percentage (9.0 for 9.0%).
    BMW reports this as 'RoCE' or 'Return on capital employed (RoCE)'.
    """
    return _build("ROIC", raw_value, unit, period, found_as, source_page, not_reported, "percent")


def extract_cost_of_capital(
    period: str,
    raw_value: Optional[float] = None,
    unit: Optional[str] = "%",
    found_as: str = "",
    source_page: Optional[int] = None,
    not_reported: bool = False,
    not_disclosed: bool = False,
) -> dict:
    """
    Extract WACC / Cost of Capital / Hurdle Rate.
    Set not_disclosed=True when the company does not publish this figure.
    raw_value and unit are not required when not_disclosed=True.
    """
    if not_disclosed:
        return _build(
            "Cost of Capital", None, "%", period,
            found_as or "Not disclosed", source_page,
            not_reported=True, kpi_family="percent",
        )
    return _build("Cost of Capital", raw_value, unit, period, found_as, source_page, not_reported, "percent")


def extract_eps_dividend(
    period: str,
    unit: Optional[str] = "EUR",
    eps_value: Optional[float] = None,
    dividend_value: Optional[float] = None,
    eps_found_as: str = "",
    dividend_found_as: str = "",
    source_page: Optional[int] = None,
    not_applicable: bool = False,
    not_reported: bool = False,
) -> dict:
    """
    Extract EPS and Dividend per Share in one call.
    eps_value and dividend_value are independently optional:
      - EPS present, no dividend → dividend marked Not Reported
      - Dividend present, no EPS → EPS marked Not Reported
      - not_applicable=True → both marked N/A (non-listed entity)
    """
    # Each leg is independently flagged
    eps_nr  = not_reported or (eps_value is None and not not_applicable)
    div_nr  = not_reported or (dividend_value is None and not not_applicable)

    eps = _build(
        "EPS", eps_value, unit, period,
        eps_found_as or "Earnings per Share",
        source_page, eps_nr, "per_share",
    )
    div = _build(
        "Dividend per Share", dividend_value, unit, period,
        dividend_found_as or "Dividend per Share",
        source_page, div_nr, "per_share",
    )

    if not_applicable:
        eps["formatted_value"] = "N/A"
        div["formatted_value"] = "N/A"
        eps["not_reported"] = False   # N/A ≠ not_reported
        div["not_reported"] = False

    return {"eps": eps, "dividend_per_share": div}


def extract_market_cap(
    period: str,
    shares_outstanding_millions: Optional[float] = None,
    source_page: Optional[int] = None,
    not_applicable: bool = False,
    not_reported: bool = False,
) -> dict:
    """
    Compute hypothetical market cap at €100/share.
    Requires shares outstanding in millions.
    Set not_applicable=True for non-listed entities.
    """
    if not_applicable:
        r = _build("Market Cap @100 EUR", None, _MONETARY_UNIT, period,
                   "N/A — not listed", source_page, kpi_family="monetary")
        r["formatted_value"] = "N/A"
        return r
    if not_reported or shares_outstanding_millions is None:
        return _build(
            "Market Cap @100 EUR", None, _MONETARY_UNIT, period,
            "Shares Outstanding", source_page, not_reported=True, kpi_family="monetary",
        )
    cap_bn = (shares_outstanding_millions * 100) / 1000.0
    return {
        "canonical_name": "Market Cap @100 EUR",
        "value": cap_bn,
        "formatted_value": f"€{cap_bn:,.2f} bn",
        "unit": _MONETARY_UNIT, "period": period,
        "found_as": "Shares Outstanding",
        "is_substitute": False,
        "substitute_note": f"Calculated: {shares_outstanding_millions:.1f}M shares × €100",
        "source_page": source_page,
        "not_reported": False,
        "derived": False,
    }


# ── Tool registry ─────────────────────────────────────────────────────────────

def build_extraction_tools() -> list[StructuredTool]:
    """Return the full list of LangChain StructuredTools for KPI extraction."""
    return [
        StructuredTool.from_function(
            func=extract_revenue, name="extract_revenue",
            description=(
                "Extract Revenue / Sales / Group Revenue / Turnover. "
                "Pass `unit` exactly as written in the report (e.g. 'EUR bn', 'EUR m', '€ million'). "
                "Omit `unit` if not clear — system will infer. "
                "Set not_reported=true if revenue is absent from this report."
            ),
            args_schema=ExtractRevenueInput,
        ),
        StructuredTool.from_function(
            func=extract_ebit, name="extract_ebit",
            description=(
                "Extract EBIT / Operating Profit / Operating Result / Operating Income. "
                "Pass `unit` exactly as written. Omit `unit` if not clear."
            ),
            args_schema=ExtractEBITInput,
        ),
        StructuredTool.from_function(
            func=extract_ebit_margin, name="extract_ebit_margin",
            description=(
                "Extract EBIT Margin / Operating Margin / Return on Sales (ROS). "
                "Pass raw_value as a percentage number (5.3 for 5.3%). "
                "Do NOT pass `unit` — system always uses %. "
                "BMW reports this as 'EBIT margin in the Automotive segment'."
            ),
            args_schema=ExtractEBITMarginInput,
        ),
        StructuredTool.from_function(
            func=extract_cash_kpi, name="extract_cash_kpi",
            description=(
                "Extract the company's preferred cash metric: Free Cash Flow / FCF / "
                "Industrial Free Cash Flow / Automotive FCF / Cash Generation. "
                "Pass `unit` exactly as written."
            ),
            args_schema=ExtractCashKPIInput,
        ),
        StructuredTool.from_function(
            func=extract_net_liquidity, name="extract_net_liquidity",
            description=(
                "Extract Net Liquidity / Net Cash Position / Net Financial Assets. "
                "BMW reports this as 'Automotive Net Financial Assets' or "
                "'Industrial Net Liquidity' — these are valid equivalents. "
                "Pass `unit` exactly as written. "
                "Set not_reported=true ONLY if the figure genuinely does not exist — "
                "check for 'Automotive Net Financial Assets' before marking not_reported."
            ),
            args_schema=ExtractNetLiquidityInput,
        ),
        StructuredTool.from_function(
            func=extract_return_on_capital, name="extract_return_on_capital",
            description=(
                "Extract ROIC / ROCE / Return on Capital Employed / Return on Net Assets. "
                "BMW reports this as 'RoCE' or 'Return on capital employed (RoCE)'. "
                "Pass raw_value as a percentage number (9.0 for 9.0%). "
                "Do NOT pass `unit` — system always uses %. "
                "Set not_reported=true if absent from quarterly reports."
            ),
            args_schema=ExtractReturnOnCapitalInput,
        ),
        StructuredTool.from_function(
            func=extract_cost_of_capital, name="extract_cost_of_capital",
            description=(
                "Extract WACC / Cost of Capital / Hurdle Rate. "
                "Set not_disclosed=true if not published (do NOT pass raw_value or unit). "
                "Do NOT pass `unit` — system always uses %."
            ),
            args_schema=ExtractCostOfCapitalInput,
        ),
        StructuredTool.from_function(
            func=extract_eps_dividend, name="extract_eps_dividend",
            description=(
                "Extract EPS (Earnings per Share) and Dividend per Share in one call. "
                "eps_value and dividend_value are INDEPENDENTLY optional: "
                "pass whichever values are available, null for the other. "
                "Do not fail if only one is available. "
                "Set not_applicable=true for non-listed entities."
            ),
            args_schema=ExtractEPSDividendInput,
        ),
        StructuredTool.from_function(
            func=extract_market_cap, name="extract_market_cap",
            description=(
                "Calculate market cap at €100/share from shares outstanding in millions. "
                "Set not_applicable=true for non-listed companies (no shares data). "
                "Set not_reported=true if shares outstanding cannot be found."
            ),
            args_schema=ExtractMarketCapInput,
        ),
    ]