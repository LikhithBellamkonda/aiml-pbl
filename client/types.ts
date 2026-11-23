export interface BootstrapResponse {
  medicationNames: string[];
  feasibility: {
    target_decisions?: Array<{
      outcome?: string;
      decision?: string;
      reason?: string;
    }>;
    training_readiness?: {
      status?: string;
      reason?: string;
    };
  };
  reproductiveToxMetrics?: {
    generated_at?: string;
    status?: string;
    clinical_pregnancy_model_trained?: boolean;
    source_file?: string;
    important_limitations?: string[];
    endpoints?: Record<
      string,
      {
        records: number;
        positive: number;
        negative: number;
        positive_percent: number;
        scaffold_groups: number;
        split_method: string;
        features: string;
        models: Record<
          string,
          {
            auroc: number;
            auprc: number;
            balanced_accuracy: number;
            brier_score: number;
            ece: number;
          }
        >;
      }
    >;
  };
  labelsPresent: boolean;
  evaluatedModel: boolean;
  filesPresent: {
    master: boolean;
    adr: boolean;
    sider?: boolean;
    faersAdr?: boolean;
    animalAdr: boolean;
    targets?: boolean;
    labels: boolean;
    checkpoint: boolean;
  };
}

export interface DrugCandidate {
  drug_id: string;
  name: string;
  resolution_status: string | null;
  pubchem_cid: number | null;
  inchikey: string | null;
}

export interface DrugProfile {
  query: string;
  matched: boolean;
  ambiguous: boolean;
  drug_id: string | null;
  drug_name: string | null;
  name_key: string | null;
  candidates: DrugCandidate[];
  warnings: string[];
  identity: {
    drug_id: string | null;
    name: string | null;
    active_ingredient?: string | null;
    pubchem_cid: number | null;
    inchikey: string | null;
    smiles: string | null;
    molecular_formula: string | null;
    molecular_weight: number | null;
    resolution_status: string | null;
    resolution_source: string | null;
  };
  molecular: {
    available: boolean;
    reason: string | null;
    canonical_smiles?: string;
    molecular_formula: string | null;
    formula_parts: Array<{ element: string; count: number }>;
    scaffold?: string | null;
    structure_png: string | null;
    descriptors: Record<string, number>;
    fingerprint: {
      type?: string;
      on_bits?: number;
      density?: number;
    };
    svg: string | null;
  };
  adr: {
    available: boolean;
    term_count: number;
    pregnancy_related_term_count?: number;
    pregnancy_related_terms?: Array<{
      term: string;
      meddra_id: string | null;
      records: number;
      pregnancy_related: boolean;
      sources: string;
    }>;
    faers_adr_vector_hits?: number;
    faers_adr_quality?: number;
    record_count?: number;
    source_datasets: string[];
    terms: Array<Record<string, unknown>>;
    reason?: string | null;
  };
  reproductive_toxicity: {
    available: boolean;
    record_count: number;
    species: string[];
    study_types: string[];
    study_flags?: string[];
    sources?: string[];
    records: Array<Record<string, unknown>>;
    reason?: string | null;
  };
  pregnancy: {
    available: boolean;
    record_count: number;
    reported_count_total?: number | null;
    category_counts?: Record<string, number>;
    outcome_totals?: Array<{
      category: string;
      outcome: string;
      records: number;
      reported_count: number | null;
    }>;
    sources?: string[];
    trimesters?: string[];
    unknown_trimester?: boolean;
    records: Array<Record<string, unknown>>;
    reason?: string | null;
  };
  tga_classification?: {
    available: boolean;
    category: string | null;
    safety_statement: string | null;
    therapeutic_classes: string[];
    records: Array<Record<string, unknown>>;
  };
  targets?: {
    available: boolean;
    count: number;
    targets: Array<{
      target_name: string;
      target_class: string;
      gene: string;
      action_type: string;
      organism: string;
      source: string;
    }>;
  };
  related_pregnancy_evidence: {
    available: boolean;
    reason?: string | null;
    similarity_method?: string;
    candidates: Array<{
      drug_id: string;
      drug_name: string;
      similarity: number;
      source_record_count: number;
      tga_category?: string | null;
      toxicity_record_count?: number;
      estimated_concern_signal?: number;
      outcomes: Array<{
        outcome: string;
        source_rows: number;
        reported_count: number | null;
      }>;
      sources: string[];
      timing: string;
    }>;
  };
  faers: {
    available: boolean;
    report_count?: number | null;
    outcome_count?: number | null;
    source_name?: string | null;
    reason?: string | null;
  };
  availability: {
    available: boolean;
    sources: Record<string, { status: string | null; metadata_completeness: number | null }>;
    missing_sources: string[];
  };
  evidence_counts: {
    adr_terms: number;
    pregnancy_records: number;
    nonclinical_toxicity_records: number;
    faers_reports: number | null;
  };
  research_priority: {
    available: boolean;
    score: number | null;
    max_score: number;
    components: Array<{
      domain: string;
      covered: boolean;
      gap_points: number;
    }>;
    reason?: string | null;
  };
}

export interface TrimesterScore {
  score: number | null;
  score_available: boolean;
  eligible_record_count: number;
  total_human_records?: number;
  source_counts?: Record<string, number>;
  has_nonclinical_evidence?: boolean;
}

export interface TrimesterResult {
  trimester: 'first_trimester' | 'second_trimester' | 'third_trimester';
  trimester_label: string;
  gestational_window: string;
  concern_score: number;
  score_label: string;
  priority_band: string;
  evidence_category: string;
  confidence_score: number;
  confidence_level: 'High' | 'Moderate' | 'Low' | 'Very Low';
  uncertainty_percent: number;
  adverse_outcomes: string[];
  supporting_evidence: string[];
  missing_evidence_notes: string[];
  active_tiers: Array<'H' | 'P' | 'A' | 'S' | 'T' | 'M'>;
  maternal_score: number;
  fetal_score: number;
}

export interface TierSummary {
  tier: 'H' | 'P' | 'A' | 'S' | 'T' | 'M';
  name: string;
  available: boolean;
  is_human_evidence: boolean;
  evidence_count: number;
  base_weight: number;
  effective_weight: number;
  scores_by_trimester: Record<'first_trimester' | 'second_trimester' | 'third_trimester', number | null>;
  summary: string;
  provenance: string[];
}

export interface ConflictFlag {
  severity: 'High' | 'Moderate' | 'Informational';
  title: string;
  detail: string;
  tiers_involved: string[];
}

export interface AiReasoningAssessment {
  triggered: boolean;
  trigger_reason: string;
  ml_score: number;
  ai_supplementary_score: number;
  blended_tier6_score: number;
  gap_interpretation: string;
  explanation: string;
  key_signals: string[];
  evidence_domains_synthesized: string[];
}

export interface MlModelAssessment {
  model_name: string;
  human_checkpoint_present: boolean;
  nonclinical_benchmark_present: boolean;
  ml_score: number;
  ml_score_basis: string;
  placental_transfer_propensity: string;
  structural_alerts: string[];
  target_alerts: string[];
  feature_completeness: number;
  artifact_note: string;
}

export interface DrugEvaluation {
  trimester: string;
  outcome_target: string;
  evidence_source_available: boolean;
  records: Array<Record<string, string>>;
  evidence_record_count: number;
  trimester_evidence_counts: Record<string, { label: string; record_count: number }>;
  provenance_badges: Array<Record<string, string>>;
  risk_score: number | null;
  risk_available: boolean;
  risk_scoring_status: string;
  trimester_results: {
    first_trimester: TrimesterResult;
    second_trimester: TrimesterResult;
    third_trimester: TrimesterResult;
  };
  tier_breakdown: TierSummary[];
  ml_assessment: MlModelAssessment;
  ai_reasoning: AiReasoningAssessment;
  confidence: {
    overall_confidence_score: number;
    overall_confidence_level: 'High' | 'Moderate' | 'Low' | 'Very Low';
    uncertainty_percent: number;
    note: string;
  };
  conflicts: ConflictFlag[];
  evidence_quality: number | null;
  evidence_gap: boolean;
  evidence_source_breakdown: Array<{
    source: string;
    label: string;
    available: boolean;
    records: number;
    observed_total: number | null;
    completeness: number | null;
    volume_score: number | null;
    quality: number;
  }>;
  evidence_quality_method: string;
  evidence_availability: {
    available_sources: string[];
    missing_sources: string[];
    q_cov: number | null;
  };
  maternal_priority: Record<string, TrimesterScore>;
  fetal_priority: Record<string, TrimesterScore>;
  overall_priority: Record<string, TrimesterScore>;
  evidence_status: string;
  status_detail: string;
  animal_adr_evidence: {
    available: boolean;
    message?: string;
    report_count?: number;
    reaction_count?: number;
    species_count?: number;
    serious_report_count?: number;
    top_species?: string;
    top_reaction_category?: string;
    record_completeness?: number;
    mapping_status?: string;
    report_date_min?: string;
    report_date_max?: string;
    sources?: string;
    source_url?: string;
    death_or_euthanasia_report_count?: number;
  };
  evidence_distinction: Array<{
    evidence_type: string;
    label: string;
    meaning: string;
  }>;
}

export interface DrugApiResponse {
  mode: 'matched' | 'ambiguous' | 'unmatched';
  query: string;
  selectedName?: string;
  drugId?: string | null;
  profile: DrugProfile;
  evaluation: DrugEvaluation | null;
  suggestions: string[];
}
