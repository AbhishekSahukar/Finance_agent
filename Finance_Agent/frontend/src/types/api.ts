export interface SummaryTableCell {
  value: string;
  is_substitute: boolean;
  substitute_note: string | null;
  not_reported: boolean;
}

export interface KPISubstitutionNote {
  company: string;
  period: string;
  canonical_name: string;
  found_as: string;
  note: string;
}

export interface RunAgentResponse {
  status: string;
  companies: string[];
  periods: string[];
  table: {
    columns: string[];
    rows: Record<string, Record<string, SummaryTableCell>>;
  };
  substitution_notes: KPISubstitutionNote[];
  executive_narrative: string;
  warnings: string[];
  langfuse_trace_url: string | null;
}

export interface CompanySlot {
  id: number;
  name: string;
  fy: File | null;
  q: File | null;
}
