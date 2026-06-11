
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../" * 6 + "backend"))

from api.services.oem_agent.extraction_tools import (
    extract_ebit_margin,
    extract_net_liquidity,
    extract_return_on_capital,
    extract_cost_of_capital,
    extract_eps_dividend,
    extract_revenue,
    extract_ebit,
    extract_cash_kpi,
    extract_market_cap,
    _resolve,
    _infer_monetary_unit,
    KPI_SYNONYMS,
)


def approx(a: float, b: float, tol: float = 0.001) -> bool:
    return abs(a - b) < tol


# ═══════════════════════════════════════════════════════════════════════════════
# Group 1 — Missing unit (the primary real-world failure)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMissingUnit:
    """Exact reproduction of the LLM args that caused TypeError in the BMW logs."""

    def test_ebit_margin_missing_unit_exact_bmw_args(self):
        """
        Real failure: extract_ebit_margin() missing 1 required positional argument: 'unit'
        args={'period': 'FY2025', 'found_as': 'EBIT margin in the Automotive segment',
              'raw_value': 5.3, 'source_page': 9}
        """
        r = extract_ebit_margin(
            period="FY2025",
            found_as="EBIT margin in the Automotive segment",
            raw_value=5.3,
            source_page=9,
            # unit intentionally omitted — must not raise
        )
        assert r["value"] is not None, "Should have extracted a value"
        assert approx(r["value"], 5.3)
        assert r["formatted_value"] == "5.3%"
        assert r["unit"] == "%"
        assert r["not_reported"] is False
        assert r["is_substitute"] is True   # synonym used
        assert "EBIT margin in the Automotive segment" in r["found_as"]

    def test_return_on_capital_missing_unit_exact_bmw_args(self):
        """
        Real failure: extract_return_on_capital() missing 1 required positional argument: 'unit'
        args={'period': 'FY2025', 'found_as': 'RoCE', 'raw_value': 9.0, 'source_page': 9}
        """
        r = extract_return_on_capital(
            period="FY2025",
            found_as="RoCE",
            raw_value=9.0,
            source_page=9,
            # unit intentionally omitted
        )
        assert approx(r["value"], 9.0)
        assert r["formatted_value"] == "9.0%"
        assert r["unit"] == "%"
        assert r["not_reported"] is False

    def test_monetary_missing_unit_defaults_to_bn(self):
        """When unit is omitted for a monetary KPI, system assumes EUR bn."""
        r = extract_revenue(period="FY2025", raw_value=133.5, found_as="Revenue")
        assert approx(r["value"], 133.5)
        assert r["formatted_value"] == "€133.50 bn"
        assert r["unit"] == "EUR bn"

    def test_monetary_unit_eur_million_converts(self):
        """When unit signals millions, value is divided by 1000."""
        r = extract_revenue(period="FY2025", raw_value=133453.0,
                            unit="€ million", found_as="Revenue")
        assert approx(r["value"], 133.453)
        assert "133.45" in r["formatted_value"]

    def test_percent_missing_unit_defaults_to_pct(self):
        """Omitting unit for a percent KPI should default to %."""
        r = extract_ebit_margin(period="FY2025", raw_value=10.3, found_as="EBIT Margin")
        assert r["unit"] == "%"
        assert r["formatted_value"] == "10.3%"


# ═══════════════════════════════════════════════════════════════════════════════
# Group 2 — not_reported=True without raw_value / unit
# ═══════════════════════════════════════════════════════════════════════════════

class TestNotReportedWithoutValue:
    """
    Real failure: extract_net_liquidity() missing 2 required positional arguments:
    'raw_value' and 'unit'
    args={'period': 'FY2025', 'found_as': 'Net Liquidity', 'not_reported': True}
    """

    def test_net_liquidity_not_reported_exact_bmw_args(self):
        r = extract_net_liquidity(
            period="FY2025",
            found_as="Net Liquidity",
            not_reported=True,
            # raw_value intentionally omitted
            # unit intentionally omitted
        )
        assert r["not_reported"] is True
        assert r["value"] is None
        assert r["formatted_value"] == "Not Reported"

    def test_return_on_capital_not_reported_no_value(self):
        """Q1 2026 RoCE: not_reported=True, no raw_value, no unit."""
        r = extract_return_on_capital(
            period="Q1 2026",
            found_as="Return on capital employed (RoCE)",
            not_reported=True,
        )
        assert r["not_reported"] is True
        assert r["formatted_value"] == "Not Reported"

    def test_ebit_not_reported_no_value(self):
        r = extract_ebit(period="Q1 2026", found_as="EBIT", not_reported=True)
        assert r["not_reported"] is True
        assert r["value"] is None

    def test_cash_kpi_not_reported_no_value(self):
        r = extract_cash_kpi(period="Q1 2026", found_as="Free Cash Flow", not_reported=True)
        assert r["not_reported"] is True

    def test_revenue_not_reported_no_value(self):
        r = extract_revenue(period="Q1 2026", found_as="Revenue", not_reported=True)
        assert r["not_reported"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# Group 3 — not_disclosed=True without raw_value / unit
# ═══════════════════════════════════════════════════════════════════════════════

class TestNotDisclosedWithoutValue:
    """
    Real failure: extract_cost_of_capital() missing 2 required positional arguments:
    'raw_value' and 'unit'
    args={'period': 'FY2025', 'found_as': 'WACC', 'not_disclosed': True}
    """

    def test_cost_of_capital_not_disclosed_exact_bmw_args(self):
        r = extract_cost_of_capital(
            period="FY2025",
            found_as="WACC",
            not_disclosed=True,
            # raw_value intentionally omitted
            # unit intentionally omitted
        )
        assert r["not_reported"] is True
        assert r["value"] is None
        assert r["formatted_value"] == "Not Reported"

    def test_cost_of_capital_not_disclosed_q1(self):
        r = extract_cost_of_capital(
            period="Q1 2026",
            found_as="WACC",
            not_disclosed=True,
        )
        assert r["not_reported"] is True

    def test_cost_of_capital_with_value_still_works(self):
        r = extract_cost_of_capital(
            period="FY2025",
            raw_value=8.0,
            found_as="WACC",
        )
        assert approx(r["value"], 8.0)
        assert r["formatted_value"] == "8.0%"
        assert r["not_reported"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# Group 4 — EPS without dividend / dividend without EPS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEPSDividendIndependentlyOptional:
    """
    Real failure: extract_eps_dividend() missing 1 required positional argument: 'dividend_value'
    args={'period': 'Q1 2026', 'eps_value': 2.68, ..., 'not_reported': False, 'unit': 'EUR'}
    """

    def test_eps_only_no_dividend_exact_bmw_args(self):
        """Q1 2026: EPS available, no dividend declared in quarterly report."""
        r = extract_eps_dividend(
            period="Q1 2026",
            eps_found_as="Earnings per ordinary share",
            eps_value=2.68,
            dividend_found_as="Dividend per share",
            not_reported=False,
            unit="EUR",
            source_page=4,
            # dividend_value intentionally omitted
        )
        assert r["eps"]["value"] is not None
        assert approx(r["eps"]["value"], 2.68)
        assert r["eps"]["formatted_value"] == "€2.68"
        assert r["eps"]["not_reported"] is False

        # Dividend should be Not Reported, not a crash
        assert r["dividend_per_share"]["not_reported"] is True
        assert r["dividend_per_share"]["formatted_value"] == "Not Reported"

    def test_dividend_only_no_eps(self):
        """Dividend declared, EPS not in this report."""
        r = extract_eps_dividend(
            period="FY2025",
            dividend_value=5.90,
            dividend_found_as="Dividend per share",
            unit="EUR",
            # eps_value intentionally omitted
        )
        assert approx(r["dividend_per_share"]["value"], 5.90)
        assert r["dividend_per_share"]["formatted_value"] == "€5.90"
        assert r["eps"]["not_reported"] is True

    def test_both_present(self):
        r = extract_eps_dividend(
            period="FY2025",
            eps_value=11.60,
            dividend_value=5.90,
            unit="EUR",
            eps_found_as="Earnings per share",
            dividend_found_as="Dividend per share",
        )
        assert approx(r["eps"]["value"], 11.60)
        assert approx(r["dividend_per_share"]["value"], 5.90)
        assert r["eps"]["not_reported"] is False
        assert r["dividend_per_share"]["not_reported"] is False

    def test_not_applicable_both_na(self):
        """Non-listed entity: both marked N/A."""
        r = extract_eps_dividend(period="FY2025", not_applicable=True)
        assert r["eps"]["formatted_value"] == "N/A"
        assert r["dividend_per_share"]["formatted_value"] == "N/A"
        assert r["eps"]["not_reported"] is False   # N/A ≠ not_reported

    def test_both_absent_not_reported(self):
        r = extract_eps_dividend(period="FY2025", not_reported=True)
        assert r["eps"]["not_reported"] is True
        assert r["dividend_per_share"]["not_reported"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# Group 5 — EBIT Margin derivation
# ═══════════════════════════════════════════════════════════════════════════════

class TestEBITMarginDerivation:
    """
    When extract_ebit_margin tool call fails, _assemble() must derive the margin
    from EBIT and Revenue if both are present.
    """

    def _run_assemble(self, tool_results: dict) -> dict:
        from unittest.mock import MagicMock
        from api.services.oem_agent.oem_generation_service import OEMGenerationService
        from api.services.oem_agent.extraction_tools import build_extraction_tools
        svc = OEMGenerationService.__new__(OEMGenerationService)
        svc._settings = MagicMock(OPENROUTER_API_KEY="x", OPENROUTER_BASE_URL="x", OPENROUTER_MODEL="x")
        svc._langfuse = MagicMock()
        svc._tools = build_extraction_tools()
        return svc._assemble("BMW Group", "FY", "FY2025", tool_results)

    def test_margin_derived_when_ebit_margin_not_reported(self):
        """EBIT and Revenue present, EBIT Margin absent → derived."""
        revenue_kpi = extract_revenue(period="FY2025", raw_value=133.453, found_as="Revenue")
        ebit_kpi    = extract_ebit(period="FY2025", raw_value=10.186, found_as="EBIT")

        result = self._run_assemble({
            "extract_revenue": revenue_kpi,
            "extract_ebit":    ebit_kpi,
            # extract_ebit_margin intentionally absent (simulates tool failure)
        })

        margin = result["ebit_margin"]
        assert margin["not_reported"] is False, "Should not be Not Reported"
        assert margin["derived"] is True, "Should be flagged as derived"
        assert margin["is_substitute"] is True, "derived should set is_substitute for ≈ badge"
        assert margin["value"] is not None
        expected = round((10.186 / 133.453) * 100, 1)
        assert approx(margin["value"], expected)
        assert "EBIT Margin derived" in margin["substitute_note"]

    def test_margin_not_derived_when_revenue_missing(self):
        """Cannot derive margin without revenue."""
        ebit_kpi = extract_ebit(period="FY2025", raw_value=10.186, found_as="EBIT")
        result = self._run_assemble({"extract_ebit": ebit_kpi})
        assert result["ebit_margin"]["not_reported"] is True

    def test_margin_not_derived_when_ebit_missing(self):
        """Cannot derive margin without EBIT."""
        rev_kpi = extract_revenue(period="FY2025", raw_value=133.453, found_as="Revenue")
        result = self._run_assemble({"extract_revenue": rev_kpi})
        assert result["ebit_margin"]["not_reported"] is True

    def test_tool_margin_takes_priority(self):
        """When EBIT Margin tool call succeeds, use that value (not derived)."""
        revenue_kpi = extract_revenue(period="FY2025", raw_value=133.453, found_as="Revenue")
        ebit_kpi    = extract_ebit(period="FY2025", raw_value=10.186, found_as="EBIT")
        margin_kpi  = extract_ebit_margin(period="FY2025", raw_value=7.6, found_as="EBIT Margin")

        result = self._run_assemble({
            "extract_revenue":    revenue_kpi,
            "extract_ebit":       ebit_kpi,
            "extract_ebit_margin": margin_kpi,
        })
        assert result["ebit_margin"]["derived"] is False
        assert approx(result["ebit_margin"]["value"], 7.6)

    def test_margin_derivation_with_bmw_real_values(self):
        """
        BMW FY2025 real values from logs:
          Revenue: 133,453 EUR million → 133.453 EUR bn
          EBIT:     10,186 EUR million → 10.186 EUR bn
          Expected margin: 10.186 / 133.453 * 100 = 7.63%
        """
        revenue_kpi = extract_revenue(
            period="FY2025", raw_value=133453.0,
            unit="€ million", found_as="Revenue",
        )
        ebit_kpi = extract_ebit(
            period="FY2025", raw_value=10186.0,
            unit="€ million", found_as="EBIT",
        )
        result = self._run_assemble({
            "extract_revenue": revenue_kpi,
            "extract_ebit":    ebit_kpi,
        })
        margin = result["ebit_margin"]
        assert margin["derived"] is True
        expected = round((10.186 / 133.453) * 100, 1)
        assert approx(margin["value"], expected, tol=0.05)


# ═══════════════════════════════════════════════════════════════════════════════
# Group 6 — ROCE / RoCE synonym handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestROCESynonyms:
    """BMW uses 'RoCE' and 'Return on capital employed (RoCE)' — both must resolve."""

    def test_roce_lowercase(self):
        c, is_sub = _resolve("roce")
        assert c == "ROIC"

    def test_RoCE_mixed_case(self):
        """Exact label from BMW FY2025 log."""
        c, is_sub = _resolve("RoCE")
        assert c == "ROIC"
        assert is_sub is True   # differs from canonical "ROIC"

    def test_return_on_capital_employed_roce_parenthetical(self):
        """Exact label from BMW Q1 2026 log."""
        c, is_sub = _resolve("Return on capital employed (RoCE)")
        assert c == "ROIC"
        assert is_sub is True

    def test_roce_automotive(self):
        c, _ = _resolve("RoCE (Automotive)")
        assert c == "ROIC"

    def test_return_on_capital_employed_without_parens(self):
        c, _ = _resolve("Return on capital employed")
        assert c == "ROIC"

    def test_extract_return_on_capital_with_roce_label(self):
        """End-to-end: real BMW args with RoCE label."""
        r = extract_return_on_capital(
            period="FY2025",
            found_as="RoCE",
            raw_value=9.0,
            source_page=9,
        )
        assert approx(r["value"], 9.0)
        assert r["canonical_name"] == "ROIC"
        assert r["is_substitute"] is True
        assert r["not_reported"] is False

    def test_extract_return_on_capital_parenthetical_not_reported(self):
        """Q1 2026 exact args from log."""
        r = extract_return_on_capital(
            period="Q1 2026",
            found_as="Return on capital employed (RoCE)",
            not_reported=True,
        )
        assert r["not_reported"] is True
        assert r["canonical_name"] == "ROIC"


# ═══════════════════════════════════════════════════════════════════════════════
# Group 7 — Net Liquidity / Automotive Net Financial Assets synonyms
# ═══════════════════════════════════════════════════════════════════════════════

class TestNetLiquiditySynonyms:

    def test_automotive_net_financial_assets(self):
        c, is_sub = _resolve("Automotive Net Financial Assets")
        assert c == "Net Liquidity"
        assert is_sub is True

    def test_industrial_net_liquidity(self):
        c, is_sub = _resolve("Industrial Net Liquidity")
        assert c == "Net Liquidity"
        assert is_sub is True

    def test_net_financial_assets(self):
        c, _ = _resolve("Net Financial Assets")
        assert c == "Net Liquidity"

    def test_net_cash_position(self):
        c, _ = _resolve("Net Cash Position")
        assert c == "Net Liquidity"

    def test_extract_net_liquidity_with_automotive_label(self):
        """BMW may report net liquidity as 'Automotive Net Financial Assets'."""
        r = extract_net_liquidity(
            period="FY2025",
            found_as="Automotive Net Financial Assets",
            raw_value=32.3,
            unit="EUR bn",
        )
        assert approx(r["value"], 32.3)
        assert r["canonical_name"] == "Net Liquidity"
        assert r["is_substitute"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# Group 8 — Unit inference
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnitInference:

    def test_infer_millions_from_unit(self):
        assert _infer_monetary_unit("EUR m") == "EUR m"
        assert _infer_monetary_unit("€ million") == "EUR m"
        assert _infer_monetary_unit("in millions") == "EUR m"

    def test_infer_billions_from_unit(self):
        assert _infer_monetary_unit("EUR bn") == "EUR bn"
        assert _infer_monetary_unit("€ billion") == "EUR bn"

    def test_infer_from_found_as_context(self):
        """Unit absent but found_as contains a hint."""
        assert _infer_monetary_unit(None, "Net Liquidity in EUR million") == "EUR m"
        assert _infer_monetary_unit(None, "Revenue in billions") == "EUR bn"

    def test_default_to_bn_when_no_signal(self):
        """No unit info anywhere → default to EUR bn."""
        assert _infer_monetary_unit(None, "Revenue") == "EUR bn"
        assert _infer_monetary_unit("", "") == "EUR bn"

    def test_decimal_percent_guard(self):
        """LLM returns 0.054 instead of 5.4 — should be corrected."""
        r = extract_ebit_margin(period="FY2025", raw_value=0.054, found_as="EBIT Margin")
        assert approx(r["value"], 5.4)
        assert r["formatted_value"] == "5.4%"

    def test_normal_percent_unchanged(self):
        r = extract_ebit_margin(period="FY2025", raw_value=10.3, found_as="EBIT Margin")
        assert approx(r["value"], 10.3)


# ═══════════════════════════════════════════════════════════════════════════════
# Group 9 — Synonym map completeness
# ═══════════════════════════════════════════════════════════════════════════════

class TestSynonymMapCompleteness:

    def test_required_canonicals_present(self):
        required = [
            "Revenue", "EBIT", "EBIT Margin", "Free Cash Flow",
            "Net Liquidity", "ROIC", "Cost of Capital", "EPS",
            "Dividend per Share",
        ]
        for name in required:
            assert name in KPI_SYNONYMS, f"Missing canonical: {name}"

    def test_no_duplicate_synonyms_across_canonicals(self):
        seen = {}
        for canonical, synonyms in KPI_SYNONYMS.items():
            for syn in synonyms:
                if syn in seen:
                    pytest.fail(
                        f"Synonym '{syn}' appears in both "
                        f"'{seen[syn]}' and '{canonical}'"
                    )
                seen[syn] = canonical

    def test_ebit_margin_automotive_segment_variant(self):
        c, _ = _resolve("EBIT margin in the Automotive segment")
        assert c == "EBIT Margin"

    def test_wacc_resolves(self):
        c, _ = _resolve("WACC")
        assert c == "Cost of Capital"

    def test_earnings_per_ordinary_share_resolves(self):
        c, _ = _resolve("Earnings per ordinary share")
        assert c == "EPS"