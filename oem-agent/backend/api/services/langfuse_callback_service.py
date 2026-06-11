"""
Langfuse callback service with graceful fallback to hardcoded prompts.

Key fix: the fallback extraction prompt now explicitly tells the LLM
to call tools for EVERY KPI it finds, pass the exact period label it
was given, and never skip a tool call just because a value is absent.
"""

import logging
from functools import lru_cache
from typing import Optional

from api.config.settings import get_settings

logger = logging.getLogger(__name__)

FALLBACK_PROMPTS = {
    "OEM_EXTRACTION_SYSTEM_PROMPT": """\
You are a financial data extraction specialist for automotive OEM annual and quarterly reports.

CANONICAL KPI NAMES AND SYNONYMS — every synonym is treated as equivalent to its canonical name:
- Revenue:          "Revenue", "Total Revenue", "Sales", "Net Revenue", "Group Revenue",
                    "Turnover", "Group Sales", "Net Sales", "Revenues"
- EBIT:             "EBIT", "Operating Profit", "Operating Result", "Operating Income",
                    "Profit from Operations", "Operating Profit/(Loss)", "Adjusted EBIT"
- EBIT Margin:      "EBIT Margin", "Operating Margin", "Return on Sales", "ROS",
                    "Operating Profit Margin", "EBIT as % of Revenue"
- Free Cash Flow:   "Free Cash Flow", "FCF", "Industrial Free Cash Flow",
                    "Automotive Free Cash Flow", "Automotive FCF", "Cash Generation",
                    "Adjusted FCF", "Adjusted Industrial Free Cash Flow",
                    "Free Cash Flow before Dividends"
- Net Liquidity:    "Net Liquidity", "Net Cash Position", "Net Cash", "Net Financial Position",
                    "Net Cash / Net Debt", "Net Financial Debt (negative = liquidity)",
                    "Net Liquidity Position"
- ROIC:             "ROIC", "ROCE", "Return on Invested Capital",
                    "Return on Capital Employed", "Return on Net Assets", "RONA",
                    "Capital Efficiency", "Return on Capital", "Adjusted ROIC"
- Cost of Capital:  "WACC", "Weighted Average Cost of Capital", "Hurdle Rate",
                    "Cost of Capital", "Required Return", "Minimum Return"
- EPS:              "EPS", "Earnings per Share", "Basic EPS", "Diluted EPS",
                    "Earnings per Ordinary Share", "Net Income per Share"
- Dividend:         "Dividend per Share", "DPS", "Dividend", "Proposed Dividend",
                    "Dividend per Ordinary Share", "Annual Dividend"

EXTRACTION RULES — follow every rule exactly:

1. Call the appropriate tool for EVERY KPI you can find in the report text.
2. Use the EXACT period label you were given in the "CURRENT FILE" section for ALL tool calls.
   Do NOT invent a period string — copy it verbatim.
3. Record the EXACT label from the report in the `found_as` field.
4. Set `is_substitute: true` when found_as differs from the canonical KPI name.
5. If a KPI is not in this report, still call the tool and set `not_reported: true`.
   Do NOT simply omit tool calls for missing KPIs — call the tool and flag it.
6. For EPS and Dividend, use `not_applicable: true` for non-listed entities
   (e.g. Stellantis N.V. subsidiaries, private companies).
7. For Cost of Capital / WACC, use `not_disclosed: true` if absent from the report.
8. For Market Cap: extract shares outstanding (in millions) to compute cap at €100/share.
   Use `not_applicable: true` for non-listed entities.
9. Always include units in the formatted value (EUR bn / EUR m / %).
10. EBIT Margin: express as a percentage number, e.g. 8.5 for 8.5%.
11. Make one tool call per KPI. Do not batch multiple KPIs into a single call.
""",

    "OEM_SYNTHESIS_SYSTEM_PROMPT": """\
You are a senior financial analyst writing a concise executive summary for C-suite readers.

Given extracted KPIs for automotive OEMs (full-year + quarterly where available):
- Write 4–5 sentences comparing profitability (EBIT margin), cash generation, and
  capital efficiency (ROIC) across the companies.
- Reference specific numbers from the table.
- Note any companies where terminology substitutions were made (e.g. "Operating Result"
  used instead of "EBIT").
- Use a professional financial register. No markdown headers or bullet points.
- If quarterly data is mostly "Not Reported", focus the comparison on full-year figures.

Respond ONLY with the narrative text.
""",
}


class LangfuseCallbackService:
    def __init__(self):
        self._settings = get_settings()
        self._client = None
        self._available = False
        self._init()

    def _init(self):
        if not self._settings.LANGFUSE_PUBLIC_KEY:
            logger.info("[Langfuse] No key configured — using fallback prompts.")
            return
        try:
            from langfuse import Langfuse  # pylint: disable=import-outside-toplevel
            self._client = Langfuse(
                public_key=self._settings.LANGFUSE_PUBLIC_KEY,
                secret_key=self._settings.LANGFUSE_SECRET_KEY,
                host=self._settings.LANGFUSE_HOST,
            )
            self._available = True
            logger.info("[Langfuse] Connected to %s", self._settings.LANGFUSE_HOST)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("[Langfuse] Init failed: %s — using fallback prompts.", exc)

    def get_prompt(self, prompt_id: str, **kwargs) -> str:
        if self._available and self._client:
            try:
                obj = self._client.get_prompt(prompt_id)
                text = obj.compile(**kwargs) if kwargs else obj.prompt
                return text
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("[Langfuse] get_prompt(%s) failed: %s", prompt_id, exc)
        text = FALLBACK_PROMPTS.get(prompt_id, f"[Prompt '{prompt_id}' not found]")
        for k, v in kwargs.items():
            text = text.replace(f"{{{k}}}", str(v))
        return text

    async def aget_prompt(self, prompt_id: str, **kwargs) -> str:
        return self.get_prompt(prompt_id, **kwargs)

    def get_langchain_handler(self, session_id: str = ""):
        if not self._available or not self._client:
            return None
        try:
            from langfuse.callback import CallbackHandler  # pylint: disable=import-outside-toplevel
            return CallbackHandler(
                public_key=self._settings.LANGFUSE_PUBLIC_KEY,
                secret_key=self._settings.LANGFUSE_SECRET_KEY,
                host=self._settings.LANGFUSE_HOST,
                session_id=session_id,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("[Langfuse] Handler failed: %s", exc)
            return None

    def get_trace_url(self) -> Optional[str]:
        return None


@lru_cache(maxsize=1)
def get_langfuse_service() -> LangfuseCallbackService:
    return LangfuseCallbackService()