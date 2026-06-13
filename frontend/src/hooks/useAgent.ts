import { useState, useCallback } from "react";
import { RunAgentResponse, CompanySlot } from "../types/api";
import { generateDemoResult } from "../utils/demo";

export type PipelineStep = "ingest" | "extract" | "validate" | "synthesize" | null;

const STEPS: PipelineStep[] = ["ingest", "extract", "validate", "synthesize"];

export function useAgent() {
  const [running, setRunning]       = useState(false);
  const [activeStep, setActiveStep] = useState<PipelineStep>(null);
  const [result, setResult]         = useState<RunAgentResponse | null>(null);
  const [error, setError]           = useState<string | null>(null);

  const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

  const runDemo = useCallback(async (companies: CompanySlot[]) => {
    for (const step of STEPS) {
      setActiveStep(step);
      await delay(700 + Math.random() * 400);
    }
    setResult(generateDemoResult(companies));
  }, []);

  const runReal = useCallback(async (companies: CompanySlot[]) => {
    const form = new FormData();
    const names: string[] = [], types: string[] = [];
    for (const c of companies) {
      if (c.fy) { form.append("files", c.fy); names.push(c.name); types.push("FY"); }
      if (c.q)  { form.append("files", c.q);  names.push(c.name); types.push("Q");  }
    }
    form.append("company_names", names.join(","));
    form.append("report_types",  types.join(","));

    setActiveStep("ingest");
    const res = await fetch("/oem-agent/upload", { method: "POST", body: form });
    setActiveStep("validate");
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`Server ${res.status}: ${body}`);
    }
    setActiveStep("synthesize");
    await delay(300);
    const data: RunAgentResponse = await res.json();
    setResult(data);
  }, []);

  const run = useCallback(async (companies: CompanySlot[]) => {
    setRunning(true);
    setResult(null);
    setError(null);

    const hasReal = companies.some((c) => c.fy instanceof File || c.q instanceof File);

    try {
      if (hasReal) {
        await runReal(companies);
      } else {
        await runDemo(companies);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      await runDemo(companies);
    } finally {
      setRunning(false);
      setActiveStep(null);
    }
  }, [runReal, runDemo]);

  const clear = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { running, activeStep, result, error, run, clear };
}
