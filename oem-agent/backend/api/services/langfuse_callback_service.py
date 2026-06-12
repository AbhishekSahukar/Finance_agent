
import logging
from functools import lru_cache
from typing import Optional

from api.config.settings import get_settings

logger = logging.getLogger(__name__)


FALLBACK_PROMPTS: dict[str, str] = {
    "OEM_EXTRACTION_SYSTEM_PROMPT": """\
You are a financial data extraction specialist for automotive OEM annual and quarterly reports.

CANONICAL KPI NAMES AND SYNONYMS — every synonym is treated as equivalent to its canonical name:
- Revenue:          "Revenue", "Total Revenue", "Sales", "Net Revenue", "Group Revenue",
                    "Turnover", "Group Sales", "Net Sales", "Revenues"
- EBIT:             "EBIT", "Operating Profit", "Operating Result", "Operating Income",
                    "Profit from Operations", "Operating Profit/(Loss)", "Adjusted EBIT"
- EBIT Margin:      "EBIT Margin", "Operating Margin", "Return on Sales", "ROS",
                    "Operating Profit Margin", "EBIT as % of Revenue",
                    "EBIT margin in the Automotive segment"
- Free Cash Flow:   "Free Cash Flow", "FCF", "Industrial Free Cash Flow",
                    "Automotive Free Cash Flow", "Automotive FCF", "Cash Generation",
                    "Adjusted FCF", "Adjusted Industrial Free Cash Flow"
- Net Liquidity:    "Net Liquidity", "Net Cash Position", "Net Cash", "Net Financial Position",
                    "Automotive Net Financial Assets", "Industrial Net Liquidity",
                    "Net Financial Assets"
- ROIC:             "ROIC", "ROCE", "RoCE", "Return on Invested Capital",
                    "Return on Capital Employed", "Return on Capital Employed (RoCE)",
                    "Return on Net Assets", "RONA", "Adjusted ROIC"
- Cost of Capital:  "WACC", "Weighted Average Cost of Capital", "Hurdle Rate",
                    "Cost of Capital", "Required Return"
- EPS:              "EPS", "Earnings per Share", "Basic EPS", "Diluted EPS",
                    "Earnings per Ordinary Share", "Net Income per Share"
- Dividend:         "Dividend per Share", "DPS", "Dividend", "Proposed Dividend"

EXTRACTION RULES — follow every rule exactly:

1. Call the appropriate tool for EVERY KPI you can find in the report text.
2. Use the EXACT period label given in the CURRENT FILE section for ALL tool calls.
3. Record the EXACT label from the report in the `found_as` field.
4. Set `is_substitute: true` when found_as differs from the canonical KPI name.
5. If a KPI is not in this report, still call the tool with `not_reported: true`.
6. For EPS and Dividend, use `not_applicable: true` for non-listed entities.
7. For Cost of Capital / WACC, use `not_disclosed: true` if absent from the report.
8. For Market Cap: extract shares outstanding (in millions) to compute cap at €100/share.
9. Pass `unit` exactly as written in the report (e.g. 'EUR m', 'EUR bn', '€ million').
10. EBIT Margin: pass raw_value as a percentage number (5.3 for 5.3%). Do NOT pass unit.
11. Make one tool call per KPI. Do not batch multiple KPIs into a single call.
12. COST OF CAPITAL — extract only genuine WACC disclosures:
    ONLY extract Cost of Capital when the report explicitly discloses a WEIGHTED AVERAGE
    cost of capital (WACC) as a single blended percentage (e.g. Mercedes-Benz WACC = 9.5%).
    Do NOT extract any of these as Cost of Capital:
      - "Minimum rate of return" / "Mindestrendite" (BMW EVA input = cost of equity, NOT WACC)
      - Goodwill impairment discount rates (these are segment-specific, not company WACC)
      - Project hurdle rates
    If the percentage you found is described as "minimum return on equity" or appears in
    an EVA / value-added section, set not_disclosed=true instead.
13. For extract_eps_dividend: NEVER set not_reported=true if you have found eps_value.
    not_reported=true means the value does not exist in the report at all.
    If EPS is present but no dividend is declared (e.g. quarterly report), pass
    eps_value with the found number and simply omit dividend_value — do not set
    not_reported=true for the whole call.
""",

    "OEM_SYNTHESIS_SYSTEM_PROMPT": """\
You are a senior financial analyst writing a concise executive summary for C-suite readers.

Given extracted KPIs for automotive OEMs (full-year + quarterly where available),
write 4-5 sentences of flowing prose.

STRICT RULES — you must follow every one of these:
1. Only reference KPI values that are explicitly present in the table with a real number.
   Do NOT reference values marked "Not Reported" or "N/A".
2. Do NOT mention Cost of Capital, WACC, or hurdle rate unless the table contains
   an actual numeric value for it. If all companies show "Not Reported" for Cost of Capital,
   do not mention it at all.
3. Do NOT state or imply whether ROIC is above or below cost of capital unless both
   ROIC and Cost of Capital are present as extracted numeric values.
4. Do NOT invent, estimate, or infer any value not explicitly in the table.
5. Compare profitability (EBIT margin), cash generation, and capital efficiency (ROIC/ROCE)
   only where those values are present.
6. If EBIT Margin is flagged as derived (calculated as EBIT / Revenue), note it as such.
7. Note any terminology substitutions (e.g. "Operating Result" used instead of "EBIT").
8. Use a professional financial register. No markdown, no bullet points, no headers.

Respond ONLY with the narrative paragraph.
""",
}


class LangfuseCallbackService:
    """
    Wrapper around the Langfuse SDK for prompt retrieval and LangChain tracing.

    Behaviour:
      - When Langfuse is reachable and the prompt exists → uses Langfuse prompt.
      - When Langfuse is unreachable, the prompt is missing, or any error occurs
        → logs a WARNING with the full exception and uses the hardcoded fallback.
      - Every successful Langfuse fetch is logged at DEBUG so you can confirm
        which source was used.
    """

    def __init__(self):
        self._settings = get_settings()
        self._client = None
        self._available = False
        self._init()

    def _init(self) -> None:
        if not self._settings.LANGFUSE_PUBLIC_KEY:
            logger.info("[Langfuse] No public key configured — using fallback prompts.")
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
            logger.warning(
                "[Langfuse] Initialisation failed: %s — all prompts will use fallback.", exc
            )

    # ── Prompt retrieval ──────────────────────────────────────────────────────

    def get_prompt(self, prompt_id: str, **kwargs) -> str:
        """
        Fetch prompt by ID. Always tries Langfuse first, falls back to
        FALLBACK_PROMPTS on any error.

        Uses compile() in all cases (never obj.prompt) for SDK-version safety.
        """
        if self._available and self._client:
            try:
                prompt_obj = self._client.get_prompt(prompt_id)

                # Always call compile() — stable API across Langfuse SDK v2 and v3.
                # compile(**{}) with no template variables returns the plain prompt text.
                text = prompt_obj.compile(**kwargs)

                # compile() on a ChatPromptClient returns a list of message dicts.
                # We only use text prompts; convert to string defensively.
                if isinstance(text, list):
                    text = "\n".join(
                        m.get("content", "") if isinstance(m, dict) else str(m)
                        for m in text
                    )

                logger.debug(
                    "[Langfuse] ✓ get_prompt('%s') → %d chars (source: Langfuse)",
                    prompt_id, len(text),
                )
                return text

            except Exception as exc:  # pylint: disable=broad-except
                # Log the FULL exception so it is visible in the backend logs.
                # Common causes:
                #   - Prompt name not found in Langfuse project (case-sensitive)
                #   - Network timeout
                #   - SDK version mismatch (compile() signature changed)
                logger.warning(
                    "[Langfuse] ✗ get_prompt('%s') failed: %s — using fallback prompt.",
                    prompt_id, exc,
                )

        # ── Fallback ──────────────────────────────────────────────────────────
        text = FALLBACK_PROMPTS.get(prompt_id)
        if text is None:
            logger.error(
                "[Langfuse] Prompt '%s' not found in FALLBACK_PROMPTS either. "
                "Check prompt ID spelling.",
                prompt_id,
            )
            return f"[Prompt '{prompt_id}' not found — check prompt ID]"

        # Apply any template variables
        for k, v in kwargs.items():
            text = text.replace(f"{{{k}}}", str(v))

        logger.debug(
            "[Langfuse] get_prompt('%s') → %d chars (source: FALLBACK)",
            prompt_id, len(text),
        )
        return text

    async def aget_prompt(self, prompt_id: str, **kwargs) -> str:
        """Async wrapper — safe to call from FastAPI route handlers."""
        return self.get_prompt(prompt_id, **kwargs)

    # ── LangChain tracing ─────────────────────────────────────────────────────

    def get_langchain_handler(self, session_id: str = ""):
        """
        Return a LangChain CallbackHandler for Langfuse tracing.
        Returns None if Langfuse is not available.
        """
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
            logger.warning("[Langfuse] CallbackHandler creation failed: %s", exc)
            return None

    def get_trace_url(self) -> Optional[str]:
        return None


@lru_cache(maxsize=1)
def get_langfuse_service() -> LangfuseCallbackService:
    """Singleton — same instance reused across all requests."""
    return LangfuseCallbackService()