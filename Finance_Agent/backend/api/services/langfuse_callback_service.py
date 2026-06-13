
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
                    "Net Financial Assets", "Automotive Net Financial Position"
- ROIC:             "ROIC", "ROCE", "RoCE", "Return on Invested Capital",
                    "Return on Capital Employed", "Return on Capital Employed (RoCE)",
                    "Return on Net Assets", "RONA", "Adjusted ROIC"
- Cost of Capital:  "WACC", "Weighted Average Cost of Capital",
                    "Cost of Capital", "Kapitalkostensatz", "Kapitalkostensatz (WACC)"
- EPS:              "EPS", "Earnings per Share", "Basic EPS", "Diluted EPS",
                    "Earnings per Ordinary Share", "Net Income per Share",
                    "Earnings per share (in euros)"
- Dividend:         "Dividend per Share", "DPS", "Dividend", "Proposed Dividend"

EXTRACTION RULES:

1. YOU MUST CALL ALL 9 EXTRACTION TOOLS — exactly once each — in every response.
   If a KPI does not appear in this report, call the tool with not_reported=true.
   Skipping any tool is an error.
2. Use the EXACT period label given in the CURRENT FILE section for ALL tool calls.
3. The report text arrives in up to 4 sections separated by
   '[... SECTION OMITTED FOR CONTEXT WINDOW ...]' markers.
   Search EVERY section for each KPI before marking it not_reported.
4. Net Liquidity synonyms (search all sections):
   'Automotive Net Financial Assets', 'Net Financial Assets',
   'Industrial Net Liquidity', 'Net Cash Position', 'Net Liquidity'.
5. Market Cap: search all sections for shares outstanding using:
   'share capital', 'shares issued', 'number of shares',
   'weighted average shares', 'shares outstanding'.
   Extract the number in millions and pass to extract_market_cap.
6. Record the EXACT label from the report in found_as.
7. Set is_substitute=true when found_as differs from the canonical KPI name.
8. Pass unit exactly as written (e.g. 'EUR m', 'EUR bn'). Omit if unclear.
9. EBIT Margin: raw_value as percentage number (5.3 for 5.3%). No unit.
10. COST OF CAPITAL — extract ONLY an after-tax group-level WACC explicitly labelled as
    'WACC', 'Kapitalkostensatz (WACC)', or 'cost of capital rate' for the whole group.
    Examples of values that look like WACC but are NOT:
      - BMW: any 'minimum rate of return' or 'Mindestrendite' — this is cost of equity only
      - Any rate described as 'before taxes' / 'vor Steuern' — use not_disclosed for these
      - Goodwill impairment discount rates for specific segments (e.g. 13.0%, 13.7%)
      - Debt cost rates (Fremdkapitalkostensatz)
    VALID examples: Mercedes-Benz 'Group, after taxes: 9.5%'.
    If you are unsure, use not_disclosed=true.
11. For extract_eps_dividend:
    - If EPS is found, pass eps_value. Omit dividend_value if not declared.
    - QUARTERLY REPORTS: set dividend_value=null unless this specific quarterly report
      explicitly announces a NEW dividend. A dividend mentioned in passing (e.g. as
      prior-year comparison or reference to the annual figure) does NOT count.
      Interim/quarterly reports almost never declare dividends — default to omitting it.
    - ANNUAL REPORTS: extract the proposed or declared dividend per share.
    - Never set not_reported=true for the whole call just because dividend is absent.
12. Output ONLY tool calls. No plain text or explanation whatsoever.
""",

    "OEM_SYNTHESIS_SYSTEM_PROMPT": """\
You are a senior financial analyst writing a concise executive summary for C-suite readers.

Given extracted KPIs for automotive OEMs (full-year + quarterly where available),
write 4-5 sentences of flowing prose.

STRICT RULES:

1. Only reference values explicitly present with a real number.
   Skip any KPI marked "Not Reported" or "N/A" for a given company.
2. Do NOT compare companies on a KPI unless BOTH have a real value for it.
   If one company is missing a KPI, do not make relative statements about it.
3. Do NOT mention Cost of Capital or WACC unless a real numeric value exists in the table.
4. Do NOT imply ROIC vs cost of capital unless both are present.
5. Do NOT invent, estimate, or infer anything not in the table.
6. EBIT Margin wording:
   - If found_as contains "Derived: EBIT / Revenue" — say "calculated from EBIT/Revenue".
   - If found_as is a synonym label (e.g. "EBIT margin in the Automotive segment") — it is
     a reported figure; do NOT call it derived. BMW EBIT margin is always reported directly.
   - Mercedes-Benz EBIT Margin is calculated from EBIT/Revenue; note this when present.
7. If a company has all KPIs as Not Reported, omit it from comparisons and note
   at the end that extraction was incomplete for that company.
8. Professional register. No markdown, no bullets, no headers.

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
                text = prompt_obj.compile(**kwargs)

                # compile() on a ChatPromptClient returns a list of message dicts.
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
        """Return a LangChain CallbackHandler for Langfuse tracing."""
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