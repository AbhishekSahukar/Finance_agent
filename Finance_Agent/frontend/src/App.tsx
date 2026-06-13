import React, { useState, useCallback } from "react";
import { Upload, FileText, X, Loader2, AlertTriangle } from "lucide-react";
import clsx from "clsx";
import { RunAgentResponse, CompanySlot } from "./types/api";

const DEFAULT_COMPANIES: CompanySlot[] = [
  { id: 1, name: "", fy: null, q: null },
  { id: 2, name: "", fy: null, q: null },
  { id: 3, name: "", fy: null, q: null },
];

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

const PERIOD_RE = /^(FY\d{4}|Q[1-4]\s+\d{4}|H[12]\s+\d{4}|9M\s+\d{4}|\d{1,2}M\s+\d{4})$/;

function splitCol(col: string): [string, string] {
  const words = col.split(" ");
  for (let n = 2; n >= 1; n--) {
    const suffix = words.slice(-n).join(" ");
    if (PERIOD_RE.test(suffix)) return [words.slice(0, -n).join(" "), suffix];
  }
  return [words.slice(0, -1).join(" "), words[words.length - 1]];
}

function FileSlot({ label, badge, file, onFile }: {
  label: string; badge: string; file: File | null; onFile: (f: File) => void;
}) {
  const ref = React.useRef<HTMLInputElement>(null);
  return (
    <div
      onClick={() => ref.current?.click()}
      className={clsx(
        "flex items-center gap-2.5 px-3 py-2 rounded-lg border cursor-pointer text-sm transition-colors select-none",
        file
          ? "border-slate-600 bg-slate-800 text-slate-300"
          : "border-dashed border-slate-700 bg-slate-900 text-slate-500 hover:border-slate-600 hover:text-slate-400"
      )}
    >
      <input ref={ref} type="file" accept=".pdf" className="hidden"
        onChange={e => { const f = e.target.files?.[0]; if (f) onFile(f); }} />
      <span className={clsx(
        "text-[10px] font-bold tracking-widest shrink-0 px-1.5 py-0.5 rounded",
        badge === "FY"
          ? "bg-indigo-900/60 text-indigo-400 border border-indigo-700/40"
          : "bg-amber-900/40 text-amber-400 border border-amber-700/40"
      )}>{badge}</span>
      {file
        ? <><FileText size={13} className="text-slate-400 shrink-0" /><span className="truncate max-w-[140px]">{file.name}</span></>
        : <><Upload size={12} className="shrink-0" /><span>{label}</span></>}
    </div>
  );
}

function CellDisplay({ cell }: {
  cell: { value: string; is_substitute: boolean; substitute_note: string | null; not_reported: boolean } | undefined;
}) {
  const [tip, setTip] = React.useState(false);
  if (!cell) return <span className="text-slate-700 font-mono text-xs">—</span>;
  const blank = cell.not_reported || cell.value === "Not Reported" || cell.value === "N/A";
  return (
    <span className="inline-flex items-center gap-1">
      <span className={clsx("font-mono text-xs", blank ? "text-slate-600 italic" : "text-slate-200")}>
        {cell.value}
      </span>
      {cell.is_substitute && cell.substitute_note && (
        <span className="relative inline-flex"
          onMouseEnter={() => setTip(true)} onMouseLeave={() => setTip(false)}>
          <span className="text-amber-500 font-bold text-[10px] cursor-help">≈</span>
          {tip && (
            <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 z-50 w-52 bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-[11px] text-slate-300 shadow-xl leading-relaxed pointer-events-none whitespace-normal">
              {cell.substitute_note}
            </span>
          )}
        </span>
      )}
    </span>
  );
}

export default function App() {
  const [companies, setCompanies] = useState<CompanySlot[]>(DEFAULT_COMPANIES);
  const [running, setRunning]     = useState(false);
  const [result, setResult]       = useState<RunAgentResponse | null>(null);
  const [error, setError]         = useState<string | null>(null);

  const setFile = useCallback((id: number, field: "fy" | "q", file: File) =>
    setCompanies(prev => prev.map(c => c.id === id ? { ...c, [field]: file } : c)), []);

  const setName = useCallback((id: number, name: string) =>
    setCompanies(prev => prev.map(c => c.id === id ? { ...c, name } : c)), []);

  // Run is disabled until at least one PDF is uploaded
  const hasFiles = companies.some(c => c.fy || c.q);

  const runAgent = async () => {
    setRunning(true);
    setResult(null);
    setError(null);
    try {
      const form = new FormData();
      const names: string[] = [], types: string[] = [];
      for (const c of companies) {
        const name = c.name.trim() || `Company ${c.id}`;
        if (c.fy) { form.append("files", c.fy); names.push(name); types.push("FY"); }
        if (c.q)  { form.append("files", c.q);  names.push(name); types.push("Q");  }
      }
      form.append("company_names", names.join(","));
      form.append("report_types",  types.join(","));

      const res = await fetch("/oem-agent/upload", { method: "POST", body: form });
      if (!res.ok) throw new Error(`Server ${res.status}: ${await res.text()}`);
      setResult(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  };

  const resultCompanies = result
    ? Array.from(new Map((result.table.columns ?? []).map(c => [splitCol(c)[0], true])).keys())
    : [];

  const getCell = (kpi: string, company: string, type: "FY" | "Q") => {
    const col = (result?.table.columns ?? []).find(c => {
      const [co, p] = splitCol(c);
      return co === company && (type === "FY" ? p.startsWith("FY") : !p.startsWith("FY"));
    });
    return col ? result?.table.rows[kpi]?.[col] : undefined;
  };

  const getColLabel = (company: string, type: "FY" | "Q") => {
    const col = (result?.table.columns ?? []).find(c => {
      const [co, p] = splitCol(c);
      return co === company && (type === "FY" ? p.startsWith("FY") : !p.startsWith("FY"));
    });
    return col ? splitCol(col)[1] : (type === "FY" ? "FY" : "Q");
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans"
         style={{ fontFamily: "Inter, system-ui, sans-serif" }}>

      <header className="border-b border-slate-800 px-8 h-12 flex items-center gap-3">
        <div className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
        <span className="text-sm font-medium text-slate-300">OEM Financial Agent</span>
      </header>

      <main className="max-w-5xl mx-auto px-8 py-10 space-y-10">

        {/* Upload */}
        <section className="space-y-5">
          <h1 className="text-lg font-semibold text-slate-100">Upload reports</h1>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {companies.map((c, i) => (
              <div key={c.id} className="space-y-2">
                <input
                  value={c.name}
                  placeholder={`Company ${i + 1} name`}
                  onChange={e => setName(c.id, e.target.value)}
                  className="w-full bg-transparent text-sm font-medium text-slate-200 placeholder-slate-600 outline-none border-b border-slate-800 focus:border-slate-600 pb-1 transition-colors"
                />
                <FileSlot badge="FY" label="Full-year report" file={c.fy}
                          onFile={f => setFile(c.id, "fy", f)} />
                <FileSlot badge="Q"  label="Quarterly report"  file={c.q}
                          onFile={f => setFile(c.id, "q",  f)} />
              </div>
            ))}
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={runAgent}
              disabled={running || !hasFiles}
              className={clsx(
                "flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-medium transition-all",
                running || !hasFiles
                  ? "bg-indigo-800/30 text-indigo-500 cursor-not-allowed"
                  : "bg-indigo-600 hover:bg-indigo-500 text-white active:scale-[0.98]"
              )}
            >
              {running
                ? <><Loader2 size={14} className="animate-spin" />Extracting…</>
                : "Run extraction"}
            </button>
            {result && (
              <button onClick={() => { setResult(null); setError(null); }}
                className="flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-300 transition-colors">
                <X size={13} /> Clear
              </button>
            )}
            {!hasFiles && !running && (
              <span className="text-xs text-slate-600">Upload at least one PDF to run</span>
            )}
          </div>
        </section>

        {/* Error */}
        {error && (
          <div className="flex gap-2 items-start rounded-lg border border-red-900/50 bg-red-950/30 px-4 py-3 text-sm text-red-400">
            <AlertTriangle size={14} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Results */}
        {result && (
          <section className="space-y-8" style={{ animation: "fadeIn 0.3s ease both" }}>
            <style>{`@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}`}</style>

            <div>
              <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Executive summary</h2>
              <p className="text-sm text-slate-300 leading-relaxed max-w-3xl">{result.executive_narrative}</p>
            </div>

            <div>
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">KPI table</h2>
                <span className="text-[10px] text-slate-600 italic">
                  <span className="text-amber-500 font-bold mr-1">≈</span>equivalent metric &nbsp;·&nbsp; italic = Not Reported / N/A
                </span>
              </div>
              <div className="overflow-x-auto rounded-xl border border-slate-800">
                <table className="w-full border-collapse text-sm min-w-[640px]">
                  <thead>
                    <tr className="bg-slate-900 border-b border-slate-800">
                      <th className="text-left px-4 py-2.5 text-[11px] font-semibold text-slate-500 uppercase tracking-wide w-44">KPI</th>
                      {resultCompanies.map(co => (
                        <React.Fragment key={co}>
                          <th className="text-left px-4 py-2.5 border-l border-slate-800">
                            <div className="text-[10px] text-slate-600 truncate max-w-[110px]">{co}</div>
                            <div className="text-xs font-semibold text-indigo-400">{getColLabel(co, "FY")}</div>
                          </th>
                          <th className="text-left px-4 py-2.5 border-l border-slate-800/50">
                            <div className="text-[10px] text-slate-600">&nbsp;</div>
                            <div className="text-xs font-semibold text-amber-400">{getColLabel(co, "Q")}</div>
                          </th>
                        </React.Fragment>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {KPI_ROWS.map((kpi, ri) => (
                      <tr key={kpi} className={clsx(
                        "border-b border-slate-800/40 last:border-b-0 hover:bg-slate-800/30 transition-colors",
                        ri % 2 === 0 ? "bg-transparent" : "bg-slate-900/20"
                      )}>
                        <td className="px-4 py-2 text-xs text-slate-400 font-medium">{kpi}</td>
                        {resultCompanies.map(co => (
                          <React.Fragment key={co}>
                            <td className="px-4 py-2 border-l border-slate-800">
                              <CellDisplay cell={getCell(kpi, co, "FY")} />
                            </td>
                            <td className="px-4 py-2 border-l border-slate-800/50">
                              <CellDisplay cell={getCell(kpi, co, "Q")} />
                            </td>
                          </React.Fragment>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {result.substitution_notes.length > 0 && (
              <div>
                <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Metric substitutions</h2>
                <div className="space-y-2">
                  {result.substitution_notes.map((s, i) => (
                    <div key={i} className="text-xs text-slate-400 flex gap-2 items-baseline">
                      <span className="text-amber-500 font-bold shrink-0">≈</span>
                      <span>
                        <span className="text-slate-300 font-medium">{s.company}</span>
                        {" · "}{s.canonical_name} reported as{" "}
                        <span className="text-amber-400 font-mono">"{s.found_as}"</span>
                        {s.note ? ` — ${s.note}` : ""}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {result.warnings.length > 0 && (
              <div className="space-y-1">
                {result.warnings.map((w, i) => (
                  <p key={i} className="text-xs text-slate-600">{w}</p>
                ))}
              </div>
            )}
          </section>
        )}

        {/* Empty state — shown before any run */}
        {!running && !result && !error && (
          <div className="text-center py-20 text-slate-600">
            <Upload size={28} className="mx-auto mb-4 opacity-30" />
            <p className="text-sm">Upload PDF reports above, then click <span className="text-slate-500">Run extraction</span>.</p>
          </div>
        )}

      </main>
    </div>
  );
}