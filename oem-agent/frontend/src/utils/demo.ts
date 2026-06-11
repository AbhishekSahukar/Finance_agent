import { RunAgentResponse, CompanySlot } from "../types/api";

const KPI_ROW_KEYS = [
  "Revenue",
  "EBIT / Operating Result",
  "EBIT Margin",
  "Cash KPI (preferred)",
  "Net Liquidity",
  "Return on Capital (ROIC/ROCE)",
  "Cost of Capital / WACC",
  "EPS",
  "Dividend per Share",
  "Market Cap @ €100/share",
];

type CompanySeed = {
  rev: string[]; ebit: string[]; margin: string[]; fcf: string[];
  liq: string[]; roic: string[]; coc: string[]; eps: string[];
  div: string[]; mcap: string[];
  ebitSub: boolean; ebitNote?: string; fcfNote?: string;
};

const SEED: Record<string, CompanySeed> = {
  "BMW Group": {
    rev:    ["€142.7 bn", "€35.1 bn"],
    ebit:   ["€7.7 bn",  "€1.7 bn"],
    margin: ["5.4%",     "4.8%"],
    fcf:    ["€3.2 bn",  "€0.8 bn"],
    liq:    ["€11.5 bn", "€10.2 bn"],
    roic:   ["10.1%",    "NR"],
    coc:    ["8.0%",     "NR"],
    eps:    ["€11.60",   "€2.90"],
    div:    ["€5.90",    "NR"],
    mcap:   ["€58.3 bn", "€58.3 bn"],
    ebitSub: true, ebitNote: "'Operating Result' used as EBIT equivalent",
    fcfNote: "'Industrial Free Cash Flow' used as FCF equivalent",
  },
  "Mercedes-Benz": {
    rev:    ["€136.4 bn", "€33.2 bn"],
    ebit:   ["€14.0 bn",  "€3.0 bn"],
    margin: ["10.3%",     "9.1%"],
    fcf:    ["€8.0 bn",   "€1.5 bn"],
    liq:    ["€30.1 bn",  "€28.7 bn"],
    roic:   ["14.5%",     "NR"],
    coc:    ["8.0%",      "NR"],
    eps:    ["€12.80",    "€2.70"],
    div:    ["€5.30",     "NR"],
    mcap:   ["€107.2 bn", "€107.2 bn"],
    ebitSub: false,
  },
  "Stellantis": {
    rev:    ["€156.9 bn", "€36.1 bn"],
    ebit:   ["€12.0 bn",  "€2.2 bn"],
    margin: ["7.6%",      "6.1%"],
    fcf:    ["€7.4 bn",   "€0.9 bn"],
    liq:    ["€24.5 bn",  "€22.0 bn"],
    roic:   ["17.2%",     "NR"],
    coc:    ["N/A",       "N/A"],
    eps:    ["N/A",       "N/A"],
    div:    ["N/A",       "N/A"],
    mcap:   ["N/A",       "N/A"],
    ebitSub: false,
    fcfNote: "'Adjusted Industrial Free Cash Flow' used as FCF equivalent",
  },
};

const DEFAULT_SEED = SEED["BMW Group"];

function cell(value: string, is_substitute = false, substitute_note: string | null = null) {
  return { value: value === "NR" ? "Not Reported" : value, is_substitute, substitute_note, not_reported: value === "NR" };
}

export function generateDemoResult(companies: CompanySlot[]): RunAgentResponse {
  const names = companies.map((c) => c.name);
  const columns = names.flatMap((n) => [`${n} FY2025`, `${n} Q4 2025`]);
  const rows: Record<string, Record<string, ReturnType<typeof cell>>> = {};

  KPI_ROW_KEYS.forEach((kpi, ki) => {
    rows[kpi] = {};
    names.forEach((name) => {
      const d: CompanySeed = SEED[name] ?? DEFAULT_SEED;
      const fyKey = `${name} FY2025`;
      const qKey  = `${name} Q4 2025`;
      const pairs: [string, string, boolean, string | null, boolean, string | null][] = [
        [d.rev[0],    d.rev[1],    false,       null,         false,       null],
        [d.ebit[0],   d.ebit[1],   d.ebitSub,   d.ebitNote ?? null, false, null],
        [d.margin[0], d.margin[1], false,        null,         false,       null],
        [d.fcf[0],    d.fcf[1],    !!d.fcfNote,  d.fcfNote ?? null, false,  null],
        [d.liq[0],    d.liq[1],    false,        null,         false,       null],
        [d.roic[0],   d.roic[1],   false,        null,         false,       null],
        [d.coc[0],    d.coc[1],    false,        null,         false,       null],
        [d.eps[0],    d.eps[1],    false,        null,         false,       null],
        [d.div[0],    d.div[1],    false,        null,         false,       null],
        [d.mcap[0],   d.mcap[1],   false,        null,         false,       null],
      ];
      const [fyVal, qVal, fyIsSub, fyNote] = pairs[ki];
      rows[kpi][fyKey] = cell(fyVal, fyIsSub, fyNote);
      rows[kpi][qKey]  = cell(qVal);
    });
  });

  const substitutionNotes = names.flatMap((name) => {
    const d = SEED[name];
    if (!d) return [];
    const out = [];
    if (d.ebitSub && d.ebitNote) out.push({ company: name, period: "FY2025", canonical_name: "EBIT", found_as: "Operating Result", note: d.ebitNote });
    if (d.fcfNote) out.push({ company: name, period: "FY2025", canonical_name: "Free Cash Flow", found_as: "Industrial Free Cash Flow", note: d.fcfNote });
    return out;
  });

  return {
    status: "complete",
    companies: names,
    periods: ["FY2025", "Q4 2025"],
    table: { columns, rows },
    substitution_notes: substitutionNotes,
    executive_narrative:
      "The three OEMs posted divergent full-year 2025 results. Mercedes-Benz led on profitability with a 10.3% EBIT margin, supported by €30.1 bn net cash, reflecting disciplined cost control. Stellantis recorded the highest revenue at €156.9 bn with a 7.6% operating margin and €7.4 bn adjusted industrial free cash flow; EPS and dividend metrics are not applicable given its non-listed structure. BMW reported a 5.4% margin — labelled 'Operating Result' in its filings — alongside €3.2 bn industrial free cash flow, with the gap versus peers reflecting elevated electrification investment. On capital efficiency, Stellantis led with 17.2% ROIC versus Mercedes at 14.5% and BMW at 10.1%, with both BMW and Mercedes disclosing a cost of capital of 8.0%.",
    warnings: [
      "ROIC not reported in any quarterly filing — shown as 'Not Reported' for Q4 2025",
      "Stellantis: EPS, dividend and market cap not applicable (non-listed entity)",
    ],
    langfuse_trace_url: null,
  };
}
