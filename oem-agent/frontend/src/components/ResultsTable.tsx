import React from "react";
import { TrendingUp, Info, AlertTriangle, ExternalLink } from "lucide-react";
import clsx from "clsx";
import { RunAgentResponse, SummaryTableCell } from "../types/api";
import { SubstituteTag } from "./SubstituteTag";

const KPI_ROWS = [
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

// Known period suffixes – must stay in sync with backend _PERIOD_SUFFIXES
const PERIOD_SUFFIXES = [
  "FY2025", "Q4 2025", "Q3 2025", "Q2 2025", "Q1 2025",
  "FY2024", "Q4 2024", "H1 2025", "H2 2025",
];

function splitColumnKey(col: string): [string, string] {
  for (const suffix of PERIOD_SUFFIXES) {
    if (col.endsWith(suffix)) {
      return [col.slice(0, -suffix.length).trim(), suffix];
    }
  }
  const parts = col.split(" ");
  return [parts.slice(0, -1).join(" "), parts[parts.length - 1]];
}

interface CellProps {
  cell: SummaryTableCell | undefined;
}

function Cell({ cell }: CellProps) {
  if (!cell) return <span className="text-slate-700 font-mono text-xs">—</span>;
  const isBlank = cell.value === "N/A" || cell.not_reported || cell.value === "Not Reported";
  return (
    <span className="inline-flex items-center">
      <span className={clsx("font-mono text-xs", isBlank ? "text-slate-600 italic" : "text-slate-200")}>
        {cell.value}
      </span>
      {cell.is_substitute && <SubstituteTag note={cell.substitute_note} />}
    </span>
  );
}

interface Props {
  result: RunAgentResponse;
}

export function ResultsTable({ result }: Props) {
  const { table, substitution_notes, warnings, executive_narrative, langfuse_trace_url } = result;

  // Derive companies from columns using the same splitting logic as the backend
  const columns = table.columns ?? [];
  const companies = Array.from(
    new Map(columns.map((col) => [splitColumnKey(col)[0], true])).keys()
  );

  // For each company, find its FY column and Q column by scanning table.columns
  function getCell(kpi: string, company: string, periodType: "FY" | "Q"): SummaryTableCell | undefined {
    const col = columns.find((c) => {
      const [co, period] = splitColumnKey(c);
      if (co !== company) return false;
      return periodType === "FY" ? period.startsWith("FY") : !period.startsWith("FY");
    });
    if (!col) return undefined;
    return table.rows[kpi]?.[col];
  }

  function getColLabel(company: string, periodType: "FY" | "Q"): string {
    const col = columns.find((c) => {
      const [co, period] = splitColumnKey(c);
      if (co !== company) return false;
      return periodType === "FY" ? period.startsWith("FY") : !period.startsWith("FY");
    });
    if (!col) return periodType === "FY" ? "FY 2025" : "Q4 2025";
    return splitColumnKey(col)[1];
  }

  return (
    <div className="space-y-6 animate-fadeUp">

      {/* Langfuse trace */}
      {langfuse_trace_url && (
        <a
          href={langfuse_trace_url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
        >
          <ExternalLink size={12} />
          View Langfuse trace
        </a>
      )}

      {/* Executive narrative */}
      <section className="rounded-xl border border-indigo-500/20 bg-indigo-500/5 p-5">
        <div className="flex items-center gap-2 mb-3">
          <TrendingUp size={14} className="text-indigo-400" />
          <span className="text-[11px] font-semibold text-indigo-400 uppercase tracking-wider">
            Executive summary
          </span>
        </div>
        <p className="text-sm text-slate-300 leading-relaxed">{executive_narrative}</p>
      </section>

      {/* KPI table */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
            KPI extraction table
          </h3>
          <div className="flex items-center gap-4 text-[10px] text-slate-600">
            <span className="flex items-center gap-1">
              <span className="w-3.5 h-3.5 rounded-full bg-amber-500/20 border border-amber-500/40 text-amber-400 text-[9px] font-bold flex items-center justify-center">
                ≈
              </span>
              Equivalent metric used
            </span>
            <span className="italic">Not Reported = absent from Q filing</span>
          </div>
        </div>

        <div className="rounded-xl border border-slate-700/50 overflow-hidden">
          <div className="overflow-x-auto scrollbar-thin">
            <table className="w-full text-left border-collapse min-w-[700px]">
              <thead>
                <tr className="bg-surface-800">
                  <th className="px-4 py-3 text-[11px] font-semibold text-slate-500 uppercase tracking-wider border-b border-slate-700/60 w-52 sticky left-0 bg-surface-800 z-10">
                    KPI
                  </th>
                  {companies.map((co) => (
                    <React.Fragment key={co}>
                      <th className="px-4 py-3 border-b border-slate-700/60 min-w-[120px]">
                        <div className="text-[10px] text-slate-500 font-medium mb-0.5 truncate max-w-[140px]">{co}</div>
                        <div className="text-xs font-semibold text-indigo-400">{getColLabel(co, "FY")}</div>
                      </th>
                      <th className="px-4 py-3 border-b border-slate-700/60 border-r border-slate-700/30 last:border-r-0 min-w-[120px]">
                        <div className="text-[10px] text-slate-500 font-medium mb-0.5">&nbsp;</div>
                        <div className="text-xs font-semibold text-amber-400">{getColLabel(co, "Q")}</div>
                      </th>
                    </React.Fragment>
                  ))}
                </tr>
              </thead>
              <tbody>
                {KPI_ROWS.map((kpi, ri) => (
                  <tr
                    key={kpi}
                    className={clsx(
                      "border-b border-slate-800/50 last:border-b-0 transition-colors",
                      ri % 2 === 0 ? "bg-surface-900/30" : "bg-transparent",
                      "hover:bg-surface-700/30"
                    )}
                  >
                    <td className="px-4 py-2.5 text-xs text-slate-400 font-medium sticky left-0 bg-inherit z-10">
                      {kpi}
                    </td>
                    {companies.map((co) => (
                      <React.Fragment key={co}>
                        <td className="px-4 py-2.5">
                          <Cell cell={getCell(kpi, co, "FY")} />
                        </td>
                        <td className="px-4 py-2.5 border-r border-slate-700/30 last:border-r-0">
                          <Cell cell={getCell(kpi, co, "Q")} />
                        </td>
                      </React.Fragment>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Substitution notes */}
      {substitution_notes.length > 0 && (
        <section className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-5">
          <div className="flex items-center gap-2 mb-4">
            <Info size={13} className="text-amber-400" />
            <span className="text-[11px] font-semibold text-amber-400 uppercase tracking-wider">
              KPI substitution notes
            </span>
          </div>
          <div className="space-y-3">
            {substitution_notes.map((s, i) => (
              <div key={i} className="flex gap-3 items-start">
                <div className="flex gap-1.5 items-center shrink-0 mt-0.5">
                  <span className="text-[10px] font-bold text-amber-400/80 bg-amber-500/10 border border-amber-500/20 px-1.5 py-0.5 rounded font-mono">
                    {s.company}
                  </span>
                  <span className="text-[10px] text-slate-600">{s.period}</span>
                </div>
                <div>
                  <p className="text-xs text-slate-300">
                    <span className="text-slate-400 font-medium">{s.canonical_name}</span>
                    <span className="text-slate-600 mx-1.5">←</span>
                    <span className="text-amber-400 font-mono">"{s.found_as}"</span>
                  </p>
                  <p className="text-[11px] text-slate-600 mt-0.5">{s.note}</p>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Warnings */}
      {warnings.length > 0 && (
        <section className="rounded-xl border border-red-500/20 bg-red-500/5 p-4">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle size={13} className="text-red-400" />
            <span className="text-[11px] font-semibold text-red-400 uppercase tracking-wider">
              Data gaps
            </span>
          </div>
          <ul className="space-y-1">
            {warnings.map((w, i) => (
              <li key={i} className="text-xs text-slate-400">{w}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}