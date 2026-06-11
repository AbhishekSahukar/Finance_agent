"""
LangChain structured extraction tools — one per KPI with full synonym resolution.

Fix: unit normalisation.
  - The LLM may return values in EUR million (e.g. 142,700 EUR m) or EUR billion
    (142.7 EUR bn). We normalise everything to a single display unit per KPI
    family so the table is consistent.
  - Normalisation rules (applied in _build before formatting):
      * Monetary KPIs (revenue, EBIT, FCF, net liquidity, market cap):
        always display in EUR bn.  If the LLM passes EUR m / EUR million,
        divide by 1000 automatically.
      * Percentage KPIs (margins, ROIC, WACC): keep as %.
        Guard against decimal fractions (0.085 → 8.5%).
      * Per-share KPIs (EPS, DPS): keep as EUR.
  - The raw `unit` string the LLM passes in is inspected and normalised;
    the stored `unit` field always reflects the final display unit.
"""

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
    ],
    "EBIT Margin": [
        "ebit margin", "operating margin", "return on sales", "ros",
        "operating profit margin", "ebit as % of revenue", "return on sales (ros)",
    ],
    "Free Cash Flow": [
        "free cash flow", "fcf", "cash generation", "industrial free cash flow",
        "adjusted fcf", "automotive free cash flow", "automotive fcf",
        "adjusted industrial free cash flow", "free cash flow before dividends",
    ],
    "Net Liquidity": [
        "net liquidity", "net cash position", "net cash", "net financial position",
        "net liquidity / net financial position", "net debt", "net cash / net debt",
        "net cash and cash equivalents", "net liquidity position",
    ],
    "ROIC": [
        "roic", "return on invested capital", "roce", "return on capital employed",
        "return on net assets", "rona", "capital efficiency", "return on capital",
        "adjusted roic",
    ],
    "Cost of Capital": [
        "wacc", "weighted average cost of capital", "hurdle rate",
        "cost of capital", "required return", "minimum return", "threshold return",
    ],
    "EPS": [
        "eps", "earnings per share", "basic eps", "diluted eps",
        "earnings per ordinary share", "net income per share",
        "earnings per share (basic)", "earnings per share (diluted)",
    ],
    "Dividend per Share": [
        "dividend per share", "dps", "dividend", "proposed dividend",
        "dividend per ordinary share", "annual dividend",
    ],
    "Shares Outstanding": [
        "shares outstanding", "number of shares", "ordinary shares",
        "shares issued", "weighted average shares", "total shares", "shares in issue",
    ],
}

# ── Unit normalisation ────────────────────────────────────────────────────────

_MONETARY_UNIT  = "EUR bn"
_PERCENT_UNIT   = "%"
_PER_SHARE_UNIT = "EUR"

_MILLION_SIGNALS = (
    "eur m", "eur million", "eur millions", "€m", "€ m",
    "million eur", "millions eur", "m eur", "meur",
    "million €", "millions €", "in millions", "mio", "mio.",
    "million", "millions",
)

def _normalise_monetary(raw_value: float, raw_unit: str) -> tuple[float, str]:
    """Convert to EUR bn. Divides by 1000 if unit signals millions."""
    unit_lower = raw_unit.strip().lower()
    if any(sig in unit_lower for sig in _MILLION_SIGNALS):
        logger.debug("[Unit] %s %s → EUR bn (/1000)", raw_value, raw_unit)
        return raw_value / 1000.0, _MONETARY_UNIT
    return raw_value, _MONETARY_UNIT


def _resolve(raw_label: str) -> tuple[str, bool]:
    norm = raw_label.strip().lower()
    for canonical, synonyms in KPI_SYNONYMS.items():
        if norm in synonyms:
            return canonical, norm != canonical.lower()
    return raw_label.strip(), False


def _build(
    canonical: str,
    raw_value: Optional[float],
    unit: str,
    period: str,
    found_as: str,
    source_page: Optional[int] = None,
    not_reported: bool = False,
    kpi_family: str = "monetary",
) -> dict:
    resolved, is_sub = _resolve(found_as) if found_as else (canonical, False)
    note = (
        f"'{found_as}' used as equivalent for '{canonical}' (KPI synonym map)"
        if is_sub else None
    )

    display_unit = (
        _MONETARY_UNIT if kpi_family == "monetary"
        else _PERCENT_UNIT if kpi_family == "percent"
        else _PER_SHARE_UNIT
    )

    if not_reported:
        return dict(
            canonical_name=canonical, value=None,
            formatted_value="Not Reported",
            unit=display_unit, period=period,
            found_as=found_as or canonical,
            is_substitute=is_sub, substitute_note=note,
            source_page=source_page, not_reported=True,
        )

    if raw_value is not None:
        try:
            v = float(str(raw_value).replace(",", ".").replace(" ", ""))

            if kpi_family == "monetary":
                v, display_unit = _normalise_monetary(v, unit)
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
        v, fmt, display_unit = None, "N/A", display_unit

    return dict(
        canonical_name=canonical, value=v, formatted_value=fmt,
        unit=display_unit, period=period,
        found_as=found_as or canonical,
        is_substitute=is_sub, substitute_note=note,
        source_page=source_page, not_reported=False,
    )


# ── Input schemas ─────────────────────────────────────────────────────────────

class _MonetaryBase(BaseModel):
    raw_value: Optional[float] = None
    unit: str = Field(
        default="EUR bn",
        description="Unit exactly as written in the report: 'EUR bn', 'EUR m', 'EUR million', 'in millions', etc."
    )
    period: str
    found_as: str
    source_page: Optional[int] = None
    not_reported: bool = False

class _PercentBase(BaseModel):
    raw_value: Optional[float] = None
    unit: str = Field(default="%")
    period: str
    found_as: str
    source_page: Optional[int] = None
    not_reported: bool = False

class ExtractRevenueInput(_MonetaryBase): pass
class ExtractEBITInput(_MonetaryBase): pass
class ExtractEBITMarginInput(_PercentBase): pass
class ExtractCashKPIInput(_MonetaryBase): pass
class ExtractNetLiquidityInput(_MonetaryBase): pass
class ExtractReturnOnCapitalInput(_PercentBase): pass

class ExtractCostOfCapitalInput(_PercentBase):
    not_disclosed: bool = False

class ExtractEPSDividendInput(BaseModel):
    eps_value: Optional[float] = None
    dividend_value: Optional[float] = None
    unit: str = Field(default="EUR")
    period: str
    eps_found_as: str = ""
    dividend_found_as: str = ""
    source_page: Optional[int] = None
    not_applicable: bool = False
    not_reported: bool = False

class ExtractMarketCapInput(BaseModel):
    shares_outstanding_millions: Optional[float] = None
    period: str
    source_page: Optional[int] = None
    not_applicable: bool = False
    not_reported: bool = False


# ── Tool functions ────────────────────────────────────────────────────────────

def extract_revenue(raw_value, unit, period, found_as, source_page=None, not_reported=False):
    """Extract Revenue / Sales / Group Revenue / Turnover. Always stored in EUR bn."""
    return _build("Revenue", raw_value, unit, period, found_as, source_page, not_reported, "monetary")

def extract_ebit(raw_value, unit, period, found_as, source_page=None, not_reported=False):
    """Extract EBIT / Operating Profit / Operating Result. Always EUR bn."""
    return _build("EBIT", raw_value, unit, period, found_as, source_page, not_reported, "monetary")

def extract_ebit_margin(raw_value, unit, period, found_as, source_page=None, not_reported=False):
    """Extract EBIT Margin / Operating Margin / Return on Sales. Value as % (8.5 for 8.5%)."""
    return _build("EBIT Margin", raw_value, unit, period, found_as, source_page, not_reported, "percent")

def extract_cash_kpi(raw_value, unit, period, found_as, source_page=None, not_reported=False):
    """Extract preferred cash metric: FCF / Industrial FCF / Automotive FCF. Always EUR bn."""
    return _build("Free Cash Flow", raw_value, unit, period, found_as, source_page, not_reported, "monetary")

def extract_net_liquidity(raw_value, unit, period, found_as, source_page=None, not_reported=False):
    """Extract Net Liquidity / Net Cash Position / Net Financial Position. Always EUR bn."""
    return _build("Net Liquidity", raw_value, unit, period, found_as, source_page, not_reported, "monetary")

def extract_return_on_capital(raw_value, unit, period, found_as, source_page=None, not_reported=False):
    """Extract ROIC / ROCE / Return on Net Assets. Value as %."""
    return _build("ROIC", raw_value, unit, period, found_as, source_page, not_reported, "percent")

def extract_cost_of_capital(raw_value, unit, period, found_as, source_page=None, not_reported=False, not_disclosed=False):
    """Extract WACC / Cost of Capital / Hurdle Rate. Set not_disclosed=True if absent."""
    if not_disclosed:
        return _build("Cost of Capital", None, "%", period, found_as, source_page, not_reported=True, kpi_family="percent")
    return _build("Cost of Capital", raw_value, unit, period, found_as, source_page, not_reported, "percent")

def extract_eps_dividend(eps_value, dividend_value, unit, period, eps_found_as="",
                         dividend_found_as="", source_page=None, not_applicable=False, not_reported=False):
    """Extract EPS and Dividend per Share. Set not_applicable=True for non-listed entities."""
    flag = not_applicable or not_reported
    eps = _build("EPS", eps_value, unit, period,
                 eps_found_as or "Earnings per Share", source_page, flag, "per_share")
    div = _build("Dividend per Share", dividend_value, unit, period,
                 dividend_found_as or "Dividend per Share", source_page, flag, "per_share")
    if not_applicable:
        eps["formatted_value"] = "N/A"
        div["formatted_value"] = "N/A"
    return {"eps": eps, "dividend_per_share": div}

def extract_market_cap(shares_outstanding_millions, period, source_page=None,
                       not_applicable=False, not_reported=False):
    """Compute market cap at €100/share. Requires shares outstanding in millions."""
    if not_applicable:
        r = _build("Market Cap @100 EUR", None, _MONETARY_UNIT, period,
                   "N/A", source_page, kpi_family="monetary")
        r["formatted_value"] = "N/A"
        return r
    if not_reported or shares_outstanding_millions is None:
        return _build("Market Cap @100 EUR", None, _MONETARY_UNIT, period,
                      "Shares Outstanding", source_page, not_reported=True, kpi_family="monetary")
    cap_bn = (shares_outstanding_millions * 100) / 1000.0
    return {
        "canonical_name": "Market Cap @100 EUR",
        "value": cap_bn,
        "formatted_value": f"€{cap_bn:,.2f} bn",
        "unit": _MONETARY_UNIT, "period": period,
        "found_as": "Shares Outstanding", "is_substitute": False,
        "substitute_note": f"Calculated: {shares_outstanding_millions:.1f}M shares × €100",
        "source_page": source_page, "not_reported": False,
    }


# ── Tool list ─────────────────────────────────────────────────────────────────

def build_extraction_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(func=extract_revenue,           name="extract_revenue",           description="Extract Revenue / Sales / Turnover. Pass `unit` exactly as written in report (e.g. 'EUR bn' or 'EUR m').", args_schema=ExtractRevenueInput),
        StructuredTool.from_function(func=extract_ebit,              name="extract_ebit",              description="Extract EBIT / Operating Profit / Operating Result. Pass `unit` exactly as written.", args_schema=ExtractEBITInput),
        StructuredTool.from_function(func=extract_ebit_margin,       name="extract_ebit_margin",       description="Extract EBIT Margin / Operating Margin / Return on Sales. Value as percentage number (8.5 for 8.5%).", args_schema=ExtractEBITMarginInput),
        StructuredTool.from_function(func=extract_cash_kpi,          name="extract_cash_kpi",          description="Extract preferred cash metric: FCF / Industrial FCF / Automotive FCF. Pass `unit` exactly as written.", args_schema=ExtractCashKPIInput),
        StructuredTool.from_function(func=extract_net_liquidity,     name="extract_net_liquidity",     description="Extract Net Liquidity / Net Cash Position / Net Financial Position. Pass `unit` exactly as written.", args_schema=ExtractNetLiquidityInput),
        StructuredTool.from_function(func=extract_return_on_capital, name="extract_return_on_capital", description="Extract ROIC / ROCE / Return on Net Assets. Value as percentage number.", args_schema=ExtractReturnOnCapitalInput),
        StructuredTool.from_function(func=extract_cost_of_capital,   name="extract_cost_of_capital",   description="Extract WACC / Cost of Capital / Hurdle Rate. Set not_disclosed=True if absent.", args_schema=ExtractCostOfCapitalInput),
        StructuredTool.from_function(func=extract_eps_dividend,      name="extract_eps_dividend",      description="Extract EPS and Dividend per Share. Set not_applicable=True for non-listed entities.", args_schema=ExtractEPSDividendInput),
        StructuredTool.from_function(func=extract_market_cap,        name="extract_market_cap",        description="Calculate market cap at EUR 100/share using shares outstanding in millions.", args_schema=ExtractMarketCapInput),
    ]