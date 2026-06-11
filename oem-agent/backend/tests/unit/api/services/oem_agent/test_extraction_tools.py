"""Unit tests for KPI extraction tools — AAA pattern."""

import pytest
from api.services.oem_agent.extraction_tools import (
    _resolve, extract_ebit, extract_revenue, extract_cash_kpi,
    extract_market_cap, extract_eps_dividend, KPI_SYNONYMS,
)


class TestResolve:
    def test_exact_canonical_not_substitute(self):
        c, s = _resolve("EBIT"); assert c == "EBIT"; assert s is False

    def test_operating_result_maps_to_ebit(self):
        c, s = _resolve("Operating Result"); assert c == "EBIT"; assert s is True

    def test_operating_profit_maps_to_ebit(self):
        c, s = _resolve("Operating Profit"); assert c == "EBIT"; assert s is True

    def test_ros_maps_to_ebit_margin(self):
        c, s = _resolve("Return on Sales"); assert c == "EBIT Margin"; assert s is True

    def test_wacc_maps_to_cost_of_capital(self):
        c, s = _resolve("WACC"); assert c == "Cost of Capital"; assert s is True

    def test_industrial_fcf_maps_to_fcf(self):
        c, s = _resolve("Industrial Free Cash Flow"); assert c == "Free Cash Flow"; assert s is True

    def test_case_insensitive(self):
        c, s = _resolve("operating income"); assert c == "EBIT"; assert s is True

    def test_unknown_passthrough(self):
        c, s = _resolve("Some Unknown Metric"); assert c == "Some Unknown Metric"; assert s is False


class TestExtractEBIT:
    def test_exact(self):
        r = extract_ebit(7.7, "EUR bn", "FY2025", "EBIT")
        assert r["is_substitute"] is False; assert r["value"] == 7.7

    def test_operating_result_flagged(self):
        r = extract_ebit(5.1, "EUR bn", "FY2025", "Operating Result")
        assert r["is_substitute"] is True
        assert "Operating Result" in r["substitute_note"]

    def test_not_reported(self):
        r = extract_ebit(None, "EUR bn", "Q4 2025", "EBIT", not_reported=True)
        assert r["formatted_value"] == "Not Reported"; assert r["not_reported"] is True


class TestExtractRevenue:
    def test_group_sales_is_substitute(self):
        r = extract_revenue(142.7, "EUR bn", "FY2025", "Group Sales")
        assert r["is_substitute"] is True

    def test_formatted_value(self):
        r = extract_revenue(142.7, "EUR bn", "FY2025", "Revenue")
        assert "142.70" in r["formatted_value"]; assert "EUR bn" in r["formatted_value"]


class TestExtractCashKPI:
    def test_automotive_fcf_substitute(self):
        r = extract_cash_kpi(3.2, "EUR bn", "FY2025", "Automotive FCF")
        assert r["is_substitute"] is True

    def test_exact_fcf(self):
        r = extract_cash_kpi(8.0, "EUR bn", "FY2025", "Free Cash Flow")
        assert r["is_substitute"] is False


class TestMarketCap:
    def test_calculation(self):
        r = extract_market_cap(600.0, "FY2025")
        assert r["value"] == pytest.approx(60.0, rel=1e-3)

    def test_not_applicable(self):
        r = extract_market_cap(None, "FY2025", not_applicable=True)
        assert r["formatted_value"] == "N/A"


class TestEPSDividend:
    def test_both_fields_returned(self):
        r = extract_eps_dividend(5.2, 2.5, "EUR", "FY2025")
        assert r["eps"]["value"] == 5.2; assert r["dividend_per_share"]["value"] == 2.5

    def test_not_applicable(self):
        r = extract_eps_dividend(None, None, "EUR", "FY2025", not_applicable=True)
        assert r["eps"]["formatted_value"] == "N/A"
        assert r["dividend_per_share"]["formatted_value"] == "N/A"


class TestSynonymMap:
    def test_required_canonicals_present(self):
        for name in ["Revenue","EBIT","EBIT Margin","Free Cash Flow",
                     "Net Liquidity","ROIC","Cost of Capital","EPS","Dividend per Share"]:
            assert name in KPI_SYNONYMS

    def test_no_duplicate_synonyms(self):
        seen = {}
        for canonical, syns in KPI_SYNONYMS.items():
            for s in syns:
                assert s not in seen, f"'{s}' duplicated in {seen[s]} and {canonical}"
                seen[s] = canonical
