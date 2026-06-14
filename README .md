# OEM Financial Benchmarking Agent

An AI agent that extracts, normalises, and compares financial KPIs from automotive OEM annual and quarterly reports. Upload PDFs for up to three companies and receive a structured comparison table, an executive summary, and a downloadable PDF report — all generated from the actual document text, with no hardcoded values.

---

## What it does

1. **Extracts text** from uploaded PDFs using PyMuPDF. Large annual reports are split into targeted chunks (cover, management report, deep notes, financial statements) so that balance-sheet items buried deep in the document are still captured.
2. **Calls an LLM** (DeepSeek V4 Flash via OpenRouter) with a structured tool-calling prompt. The model reads the extracted text and calls one tool per KPI — `extract_revenue`, `extract_ebit`, `extract_net_liquidity`, etc.
3. **Normalises units** automatically. Values reported in EUR millions are converted to EUR billions. Percentage KPIs have a decimal-fraction guard (0.054 → 5.4%).
4. **Maps synonyms** — BMW's "Automotive Net Financial Assets" → Net Liquidity, "RoCE" → ROIC, "EBIT margin in the Automotive segment" → EBIT Margin, and 60+ other aliases — so terminology differences across OEMs don't cause gaps.
5. **Derives missing values** — if EBIT Margin is absent but EBIT and Revenue are both extracted, the margin is computed as EBIT ÷ Revenue × 100 and flagged as calculated.
6. **Generates an executive summary** via a second LLM call that is strictly constrained to only reference values that were actually extracted.
7. **Produces a PDF report** in the browser (no backend involvement) containing the executive summary, the full KPI table, and all metric substitution notes.

---

## KPIs extracted

| KPI | Canonical name | Example synonyms handled |
|---|---|---|
| Revenue | Revenue | Group Sales, Turnover, Net Revenue |
| EBIT | EBIT | Operating Result, Operating Profit, Operating Income |
| EBIT Margin | EBIT Margin | Return on Sales, EBIT margin in the Automotive segment |
| Cash KPI | Free Cash Flow | Industrial FCF, Automotive FCF, Cash Generation |
| Net Liquidity | Net Liquidity | Automotive Net Financial Assets, Industrial Net Liquidity |
| Return on Capital | ROIC | ROCE, RoCE, Return on Capital Employed, RONA |
| Cost of Capital | Cost of Capital | WACC (after-tax group rate only — segment impairment rates excluded) |
| EPS | EPS | Earnings per Ordinary Share, Earnings per share (in euros) |
| Dividend per Share | Dividend per Share | DPS, Proposed Dividend |
| Market Cap @ €100/share | Market Cap | Calculated from shares outstanding × €100 |

---

## Architecture

```
frontend/          React 18 + Vite + Tailwind CSS
  src/
    App.tsx        Upload UI, results table, PDF download
    types/api.ts   TypeScript types for the API response

backend/
  main.py          FastAPI app, CORS, router mounting
  api/
    routers/       POST /oem-agent/upload
    controllers/   Request validation, graph invocation, response serialisation
    services/
      oem_agent/
        oem_generation_service.py   LangGraph 4-node workflow
        extraction_tools.py         9 LangChain StructuredTools + synonym map
      langfuse_callback_service.py  Prompt retrieval (Langfuse → fallback)
    models/        Pydantic request/response models
    config/        Settings (reads .env)
  tests/
    unit/          81 unit tests for extraction tools and chunking logic
```

### LangGraph workflow

```
ingest_pdfs → extract_metrics → validate → synthesize_summary
```

| Node | What it does |
|---|---|
| `ingest_pdfs` | Decodes base64 PDFs, extracts text with PyMuPDF, applies 5-chunk strategy for large annual reports, detects period label (FY2025, Q1 2026, etc.) from document text |
| `extract_metrics` | Runs all files in parallel (`asyncio.gather`). Each file gets its own LLM call with a focused system prompt. Retries with a trimmed 60k-char context if the model returns 0 tool calls |
| `validate` | Collects substitution notes (deduped by company + canonical name + normalised found-as label), builds data-gap warnings |
| `synthesize_summary` | Second LLM call generates the executive narrative from a compacted table (Not Reported values stripped before sending) |

### PDF chunking for large annual reports

Annual reports from BMW, Mercedes-Benz, and Volkswagen range from 1.2 M to 2.2 M characters. A naive head-truncation misses balance-sheet items that appear deep in the financial statements. The agent uses a 5-window strategy:

```
HEAD   (0 – 30k)          cover page, period detection, headline KPIs
NEAR   (~15% centroid, 80k)  management report — EPS, KPI summaries, dividends
MID-A  (~70% centroid, 60k)  BMW Automotive Net Financial Assets (~char 1,184k)
MID-B  (~85% centroid, 60k)  balance-sheet notes, shares outstanding
TAIL   (last 110k)         financial statement notes, Mercedes shares (~char 1,140k)
```

Total sent per FY report: ~340k characters (~85k tokens).

---

## Stack

| Layer | Technology |
|---|---|
| LLM | DeepSeek V4 Flash via OpenRouter (`deepseek/deepseek-v4-flash`) |
| Agent framework | LangGraph 0.2 + LangChain 0.3 |
| Tool calling | LangChain StructuredTools with Pydantic schemas |
| PDF extraction | PyMuPDF (`pymupdf`) |
| Prompt management | Langfuse (with hardcoded fallback if unavailable) |
| API | FastAPI 0.115 + Uvicorn |
| Frontend | React 18 + Vite 5 + Tailwind CSS 3 |
| PDF export | jsPDF 2.5 + jspdf-autotable 3.8 (browser-side, no backend) |
| Tests | Pytest (81 unit tests) |

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- An [OpenRouter](https://openrouter.ai) API key with access to `deepseek/deepseek-v4-flash` or any other model of your choice.
- (Optional) A [Langfuse](https://cloud.langfuse.com) project for prompt management and tracing

---

## Setup

### 1. Clone and configure

```bash
git clone <repo-url>
cd oem-agent
```

Create `backend/.env`:

```dotenv
# Required
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=deepseek/deepseek-v4-flash or any other model of your choice.

# Optional — Langfuse prompt management and tracing
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional — defaults shown
MAX_PDF_SIZE_MB=50
AGENT_TIMEOUT_SECONDS=300
```

### 2. Backend

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

---

## Usage

1. Enter a company name in each column (e.g. `BMW`, `Mercedes-Benz`, `VW`).
2. Upload the full-year PDF report under **FY** and the quarterly report under **Q** for each company. At least one PDF is required to enable the Run button.
3. Click **Run extraction**. All six LLM calls run in parallel — typical total time is 2–5 minutes, bounded by the largest annual report.
4. Review the executive summary and KPI table. Hover over ≈ badges to see what synonym was used.
5. Click **Download PDF** to save a formatted report.


---

## Langfuse prompt management (optional)

If Langfuse keys are configured, the agent fetches its system prompts from your Langfuse project at runtime. The prompt names must match exactly:

| Prompt name | Purpose |
|---|---|
| `OEM_EXTRACTION_SYSTEM_PROMPT` | Instructs the LLM which tools to call, synonym mappings, CoC exclusion rules, Q-report dividend rules |
| `OEM_SYNTHESIS_SYSTEM_PROMPT` | Instructs the LLM to write the executive narrative using only extracted values |

If Langfuse is unavailable or the prompt is not found, the agent automatically falls back to the hardcoded prompts in `langfuse_callback_service.py`. Every fetch is logged:

```
[Langfuse] ✓ get_prompt('OEM_EXTRACTION_SYSTEM_PROMPT') → 2341 chars (source: Langfuse)
[Langfuse] ✗ get_prompt('OEM_SYNTHESIS_SYSTEM_PROMPT') failed: ... → using fallback
```

---

## Running tests

```bash
cd backend
pytest tests/unit/ -v
```

The unit test suite (81 tests) covers:

- Unit normalisation (EUR m → EUR bn, decimal percent guard)
- Synonym resolution for all 9 KPIs including BMW/Mercedes/VW-specific labels
- `not_reported=True` without `raw_value` for all tool paths
- `not_disclosed=True` for Cost of Capital
- EPS extracted when `not_reported=True` is set at call level (regression for BMW Q1)
- EBIT Margin derivation from EBIT ÷ Revenue
- ROCE / RoCE parenthetical synonym matching
- Period detection — English and German formats, text buried past char 3000
- PDF 5-chunk coverage verification for BMW and Mercedes

---

## Key design decisions

**No retrieval (RAG).** KPIs are few and well-defined; targeted chunking finds them more reliably than semantic search over dense financial tables. Retrieval was considered but adds complexity without solving the core problem (known KPI labels, known document structure).

**One LLM call per file, all in parallel.** Giving the model one company at a time eliminates the "Unknown company" problem that occurs when all six files share a context. `asyncio.gather` runs them concurrently so total time ≈ max(slowest file) rather than sum(all files).

**Structured tool calling over free-text extraction.** Each KPI has its own typed tool with a Pydantic schema. Unit conversion, synonym resolution, and derivation happen deterministically in Python — not in the LLM's unverifiable output.

**Period detection from PDF text.** The period label (FY2025, Q1 2026, etc.) is detected from the first 8000 characters of extracted text using 30+ regex patterns. It is then injected into every tool call, so the LLM cannot hallucinate a period.

**Company names from the upload form only.** No production logic references specific OEM names. Any company whose PDF contains selectable text can be analysed.

---

## Environment variables reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | ✅ | — | OpenRouter API key |
| `OPENROUTER_BASE_URL` | | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `OPENROUTER_MODEL` | | `deepseek/deepseek-v4-flash` | Model identifier |
| `LANGFUSE_PUBLIC_KEY` | | — | Langfuse public key (enables prompt management) |
| `LANGFUSE_SECRET_KEY` | | — | Langfuse secret key |
| `LANGFUSE_HOST` | | `https://cloud.langfuse.com` | Langfuse instance URL |
| `MAX_PDF_SIZE_MB` | | `50` | Maximum PDF upload size |
| `AGENT_TIMEOUT_SECONDS` | | `300` | Per-request timeout |

---

## Project structure

```
oem-agent/
├── backend/
│   ├── api/
│   │   ├── config/settings.py          Pydantic settings (reads .env)
│   │   ├── controllers/                Request handling, response serialisation
│   │   ├── models/oem_models.py        Pydantic request/response models
│   │   ├── routers/oem_agent_router.py POST /oem-agent/upload
│   │   └── services/
│   │       ├── oem_agent/
│   │       │   ├── oem_generation_service.py  LangGraph workflow, PDF chunking
│   │       │   └── extraction_tools.py        9 tools, synonym map, unit conversion
│   │       └── langfuse_callback_service.py   Prompt retrieval with fallback
│   ├── tests/unit/                     81 unit tests
│   ├── main.py                         FastAPI app
│   ├── logging.yaml                    Structured logging config
│   └── requirements.txt
└── frontend/
    └── src/
        ├── App.tsx                     Upload UI, KPI table, PDF download
        └── types/api.ts                TypeScript API types
```
## License

This project is provided for demonstration and portfolio purposes only. All rights reserved. Unauthorized copying, modification, distribution, or commercial use is prohibited without prior permission.