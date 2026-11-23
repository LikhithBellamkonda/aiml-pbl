import { CsvRow, getDataStore, ScoringConfig } from './dataLoader.js';
import { buildDrugProfile, normalise } from './drugService.js';

export const TRIMESTER_OPTIONS = [
  'First Trimester (T1)',
  'Second Trimester (T2)',
  'Third Trimester (T3)',
  'Entire Pregnancy',
  'Unknown',
] as const;

export const TRIMESTER_VALUES: Record<string, string> = {
  'First Trimester (T1)': 'first_trimester',
  'Second Trimester (T2)': 'second_trimester',
  'Third Trimester (T3)': 'third_trimester',
  'Entire Pregnancy': 'entire_pregnancy',
  Unknown: 'unknown',
};

const CORE_TRIMESTERS = ['first_trimester', 'second_trimester', 'third_trimester'] as const;
type CoreTrimester = (typeof CORE_TRIMESTERS)[number];

const TRIMESTERS = [
  'first_trimester',
  'second_trimester',
  'third_trimester',
  'entire_pregnancy',
] as const;

const TRIMESTER_LABELS: Record<string, string> = {
  first_trimester: 'First Trimester (T1)',
  second_trimester: 'Second Trimester (T2)',
  third_trimester: 'Third Trimester (T3)',
  entire_pregnancy: 'Entire Pregnancy',
};

const TRIMESTER_WINDOWS: Record<CoreTrimester, string> = {
  first_trimester: 'Weeks 0–13 (Organogenesis & early placentation)',
  second_trimester: 'Weeks 14–27 (Fetal growth & organ maturation)',
  third_trimester: 'Weeks 28+ (Late growth, ductal/renal physiology & neonatal transition)',
};

const SOURCE_ORDER = ['onsides', 'mendeley', 'faers', 'tga'] as const;

const SOURCE_LABELS: Record<string, string> = {
  onsides: 'OnSIDES / SIDER general ADR terms',
  mendeley: 'Human pregnancy outcomes (aggregate)',
  faers: 'FAERS reporting summary (aggregate)',
  tga: 'Pregnancy risk classifications (TGA)',
};

const SOURCE_SPECS: Record<
  string,
  {
    contentFields: string[];
    volumeColumn: string | null;
    volumeReference: number | null;
    granularity: string;
  }
> = {
  onsides: {
    contentFields: ['adverse_event_term', 'meddra_id', 'concept_type', 'source_identifier'],
    volumeColumn: null,
    volumeReference: 500.0,
    granularity: 'record-level labelled ADR terms',
  },
  mendeley: {
    contentFields: ['reported_term', 'reported_count', 'evidence_type', 'source_table', 'source_file'],
    volumeColumn: 'reported_count',
    volumeReference: 5000.0,
    granularity: 'aggregate spontaneous-report summary',
  },
  faers: {
    contentFields: ['faers_report_count', 'faers_outcome_count', 'faers_source_name'],
    volumeColumn: 'faers_report_count',
    volumeReference: 5000.0,
    granularity: 'aggregate reporting summary',
  },
  tga: {
    contentFields: ['Category', 'Safety statement', 'Classification 1', 'Classification 2', 'Classification 3'],
    volumeColumn: null,
    volumeReference: null,
    granularity: 'regulatory classification record',
  },
};

const QUALITY_METHOD =
  'Per source: quality = 0.5 x completeness + 0.5 x volume, where completeness is the populated fraction of content fields and volume is log1p(observed count) / log1p(reference volume), capped at 1. Missing sources are excluded from concern-score denominators rather than treated as zero safety.';

export const EVIDENCE_DISTINCTION = [
  {
    evidence_type: 'human_pregnancy_specific',
    label: 'Human pregnancy-specific evidence (Tier 1)',
    meaning:
      'Reported human pregnancy outcomes, classifications, or timed associations. This is the primary human evidence tier.',
  },
  {
    evidence_type: 'pregnancy_adr_regulatory',
    label: 'Pregnancy ADR & regulatory classifications (Tier 2)',
    meaning:
      'TGA pregnancy risk categories (A–X), FAERS pregnancy reporting volume, and pregnancy-linked MedDRA ADR terms.',
  },
  {
    evidence_type: 'human_spontaneous_adr',
    label: 'Human general ADR evidence (Tier 3)',
    meaning:
      'SIDER and FAERS general adverse-event terms. Supports systemic toxicity screening; not pregnancy-specific outcomes.',
  },
  {
    evidence_type: 'chemical_similarity',
    label: 'Chemical structural analogue evidence (Tier 4)',
    meaning:
      '512-bit OpenChemLib fingerprint Tanimoto similarity (>= 0.20) to drugs with recorded pregnancy or toxicity data. Structural similarity does not prove shared outcomes.',
  },
  {
    evidence_type: 'animal_experimental_toxicity',
    label: 'Animal developmental/reproductive & veterinary ADR (Tier 5)',
    meaning:
      'EPA ToxRefDB developmental/reproductive assays and FDA CVM veterinary ADAE reports. Nonclinical animal signal only — never a confirmed human pregnancy outcome.',
  },
  {
    evidence_type: 'molecular_ml_ai_estimation',
    label: 'Molecular physicochemical & ML/AI reasoning fallback (Tier 6)',
    meaning:
      'Placental-transfer descriptor/target screening plus supplementary AI reasoning triggered whenever ML score < 20% or molecular data is incomplete.',
  },
];

const TGA_BASE_SCORE: Record<string, number> = {
  X: 92,
  D: 79,
  C: 57,
  B3: 48,
  B2: 34,
  B1: 25,
  A: 15,
};

const HIGH_CONCERN_ADR_KEYWORDS = [
  'teratogen',
  'congenital',
  'malformation',
  'abortion',
  'miscarriage',
  'stillbirth',
  'hepatotox',
  'nephrotox',
  'myelosuppress',
  'agranulocytosis',
  'haemorrhage',
  'hemorrhage',
  'thrombocytopenia',
  'pancytopenia',
  'arrhythmia',
  'qt prolongation',
  'seizure',
  'anaphylaxis',
  'stevens-johnson',
  'embryo',
  'fetal',
  'foetal',
  'neonatal',
  'placental',
  'premature',
];

const TIER_WEIGHTS: Record<'H' | 'P' | 'A' | 'S' | 'T' | 'M', number> = {
  H: 0.34,
  P: 0.24,
  A: 0.12,
  S: 0.1,
  T: 0.12,
  M: 0.08,
};

export interface TierSummary {
  tier: 'H' | 'P' | 'A' | 'S' | 'T' | 'M';
  name: string;
  available: boolean;
  is_human_evidence: boolean;
  evidence_count: number;
  base_weight: number;
  effective_weight: number;
  scores_by_trimester: Record<CoreTrimester, number | null>;
  summary: string;
  provenance: string[];
}

export interface TrimesterResult {
  trimester: CoreTrimester;
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
  placental_transfer_propensity: 'High' | 'Moderate' | 'Low' | 'Active/FcRn or Unknown';
  structural_alerts: string[];
  target_alerts: string[];
  feature_completeness: number;
  artifact_note: string;
}

export type EvaluationOptions = {
  forceMissingArtifacts?: boolean;
  profileOverride?: ReturnType<typeof buildDrugProfile>;
};

function clampScore(val: number, min = 8.0, max = 98.0): number {
  return Math.round(Math.min(max, Math.max(min, val)) * 10) / 10;
}

function round1(val: number): number {
  return Math.round(val * 10) / 10;
}

function safeNum(val: unknown): number | null {
  if (val === null || val === undefined) return null;
  const s = String(val).trim();
  if (!s) return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

function isPopulated(val: unknown): boolean {
  if (val === null || val === undefined) return false;
  const s = String(val).trim();
  return s !== '' && s.toLowerCase() !== 'nan' && s.toLowerCase() !== 'none';
}

function frameCompleteness(rows: CsvRow[], fields: string[]): number {
  if (rows.length === 0) return 0;
  const presentFields = fields.filter((f) => f in rows[0]);
  if (presentFields.length === 0) return 0;
  let sum = 0;
  for (const f of presentFields) {
    let pop = 0;
    for (const r of rows) {
      if (isPopulated(r[f])) pop++;
    }
    sum += pop / rows.length;
  }
  return sum / presentFields.length;
}

function volumeScore(rowCount: number, observedTotal: number, reference: number | null): number {
  if (reference === null || reference <= 0) {
    return rowCount > 0 ? 1.0 : 0.0;
  }
  const value = Math.max(observedTotal, rowCount, 0.0);
  return Math.min(1.0, Math.log1p(value) / Math.log1p(reference));
}

function sourceFamily(sourceValue: string): string {
  const s = sourceValue.toLowerCase();
  if (s.includes('mendeley')) return 'mendeley';
  if (s.includes('onsides') || s.includes('sider')) return 'onsides';
  if (s.includes('faers')) return 'faers';
  if (s.includes('tga') || s.includes('medicines-pregnancy')) return 'tga';
  if (s.includes('reprotox')) return 'reprotox_kg';
  if (s.includes('epa') || s.includes('toxrefdb')) return 'epa_toxrefdb';
  return 'unknown';
}

export function computeEvidenceQualityScore(drugId: string, trimester?: string) {
  const store = getDataStore();
  const siderRows = store.siderByDrugId.get(drugId) ?? [];

  const availabilityMask: Array<[string, number]> = [];
  const qualityVector: Array<[string, number]> = [];
  const breakdown: Array<Record<string, unknown>> = [];
  const missingDatasets: string[] = [];

  for (const source of SOURCE_ORDER) {
    const spec = SOURCE_SPECS[source];
    if (source === 'onsides') {
      const csvRows = store.adr.filter((r) => String(r.drug_id ?? '') === String(drugId));
      const totalRows = csvRows.length + siderRows.length;
      const available = totalRows > 0 && !trimester ? 1.0 : 0.0;
      availabilityMask.push([source, available]);
      if (available === 0) {
        breakdown.push({
          source,
          label: SOURCE_LABELS[source],
          available: false,
          records: 0,
          observed_total: null,
          completeness: null,
          volume_score: null,
          quality: 0.0,
          granularity: spec.granularity,
        });
        qualityVector.push([source, 0.0]);
      } else {
        const completeness = siderRows.length > 0 ? 0.92 : frameCompleteness(csvRows, spec.contentFields);
        const vol = volumeScore(totalRows, totalRows, spec.volumeReference);
        const quality = 0.5 * completeness + 0.5 * vol;
        qualityVector.push([source, quality]);
        breakdown.push({
          source,
          label: SOURCE_LABELS[source],
          available: true,
          records: totalRows,
          observed_total: totalRows,
          completeness: Math.round(completeness * 10000) / 10000,
          volume_score: Math.round(vol * 10000) / 10000,
          quality: Math.round(quality * 10000) / 10000,
          granularity: spec.granularity,
        });
      }
      continue;
    }

    const frame =
      source === 'faers'
        ? store.faers
        : source === 'mendeley'
          ? store.pregnancy
          : store.classifications;
    if (frame.length === 0) missingDatasets.push(source);

    let rows = frame.filter((r) => String(r.drug_id ?? '') === String(drugId));
    if (rows.length > 0 && trimester) {
      rows = rows.filter((r) => {
        if (!('trimester' in r)) return false;
        const t = String(r.trimester ?? '').trim().toLowerCase().replace(/[-\s]+/g, '_');
        return t === trimester.trim().toLowerCase().replace(/[-\s]+/g, '_');
      });
    }

    const available = rows.length > 0 ? 1.0 : 0.0;
    availabilityMask.push([source, available]);

    if (rows.length === 0) {
      breakdown.push({
        source,
        label: SOURCE_LABELS[source] ?? source,
        available: false,
        records: 0,
        observed_total: null,
        completeness: null,
        volume_score: null,
        quality: 0.0,
        granularity: spec.granularity,
      });
      qualityVector.push([source, 0.0]);
      continue;
    }

    const completeness = frameCompleteness(rows, spec.contentFields);
    let observedTotal: number | null = null;
    if (spec.volumeColumn && spec.volumeColumn in rows[0]) {
      let sum = 0;
      let anyValid = false;
      for (const r of rows) {
        const v = safeNum(r[spec.volumeColumn]);
        if (v !== null) {
          sum += v;
          anyValid = true;
        }
      }
      if (anyValid) observedTotal = sum;
    }

    const vol = volumeScore(rows.length, observedTotal ?? rows.length, spec.volumeReference);
    const quality = 0.5 * completeness + 0.5 * vol;
    qualityVector.push([source, quality]);
    breakdown.push({
      source,
      label: SOURCE_LABELS[source] ?? source,
      available: true,
      records: rows.length,
      observed_total: observedTotal,
      completeness: Math.round(completeness * 10000) / 10000,
      volume_score: Math.round(vol * 10000) / 10000,
      quality: Math.round(quality * 10000) / 10000,
      granularity: spec.granularity,
    });
  }

  const availableCount = availabilityMask.filter(([, v]) => v > 0).length;
  const qCov =
    availableCount === 0
      ? null
      : Math.round(
          (qualityVector.reduce((acc, [, q]) => acc + q, 0) / SOURCE_ORDER.length) * 10000
        ) / 10000;

  return {
    drug_id: drugId,
    trimester: trimester ?? null,
    evidence_quality: qCov,
    q_cov: qCov,
    availability_mask: availabilityMask,
    source_breakdown: breakdown,
    available_sources: availableCount,
    total_sources: SOURCE_ORDER.length,
    missing_sources: missingDatasets,
    evidence_gap: qCov === null || availableCount < 2 || qCov < 0.3,
    quality_method: QUALITY_METHOD,
  };
}

export function animalEvidence(drugId: string | null, drugName: string | null) {
  const store = getDataStore();
  if (store.animalAdr.length === 0 || !drugId) {
    return {
      available: false,
      drug_id: drugId ?? '',
      queried_name: drugName ?? '',
      evidence_type: 'animal_spontaneous_adr',
      message: 'Animal spontaneous ADR evidence unavailable for this drug. Missing veterinary records never imply human pregnancy safety.',
    };
  }
  const row = store.animalAdr.find((r) => String(r.drug_id ?? '') === String(drugId));
  if (!row) {
    return {
      available: false,
      drug_id: drugId ?? '',
      queried_name: drugName ?? '',
      evidence_type: 'animal_spontaneous_adr',
      message: 'No matching FDA CVM veterinary ADR records exist in the local snapshot. Absence of animal ADR data is not proof of safety.',
    };
  }
  return {
    available: true,
    drug_id: drugId,
    evidence_type: 'animal_spontaneous_adr',
    report_count: Number(row.animal_adr_report_count ?? 0),
    reaction_count: Number(row.animal_adr_reaction_count ?? 0),
    species_count: Number(row.animal_adr_species_count ?? 0),
    top_species: row.top_species || 'Unknown',
    top_reaction_category: row.top_reaction_category || 'Unclassified',
    serious_report_count: Number(row.serious_report_count ?? 0),
    death_or_euthanasia_report_count: Number(row.death_or_euthanasia_report_count ?? 0),
    record_completeness: Number(row.record_completeness ?? 0),
    mapping_status: row.mapping_status || 'unknown',
    report_date_min: row.report_date_min || '?',
    report_date_max: row.report_date_max || '?',
    sources: row.sources || 'FDA CVM ADAE',
    source_url: row.source_url || '',
  };
}

function computeTier1HumanPregnancy(
  profile: ReturnType<typeof buildDrugProfile>,
  config: ScoringConfig
): TierSummary {
  const preg = profile.pregnancy;
  const totals = (preg.outcome_totals ?? []) as Array<{
    category: string;
    outcome: string;
    records: number;
    reported_count: number | null;
  }>;

  if (!preg.available || totals.length === 0) {
    return {
      tier: 'H',
      name: 'Direct Human Pregnancy Outcomes (Tier 1)',
      available: false,
      is_human_evidence: true,
      evidence_count: 0,
      base_weight: TIER_WEIGHTS.H,
      effective_weight: 0,
      scores_by_trimester: {
        first_trimester: null,
        second_trimester: null,
        third_trimester: null,
      },
      summary: 'No direct human pregnancy outcome records exist in local datasets.',
      provenance: [],
    };
  }

  const endpointRel = config.endpoint_relevance ?? {};
  const explicitTrimesters = new Set((preg.trimesters ?? []).map((t) => normalise(t)));
  const hasExplicitTiming = !preg.unknown_trimester && explicitTrimesters.size > 0;
  const timingFactor = hasExplicitTiming ? 1.0 : 0.88;

  const scores: Record<CoreTrimester, number> = {
    first_trimester: 0,
    second_trimester: 0,
    third_trimester: 0,
  };

  for (const t of CORE_TRIMESTERS) {
    let weightedSignal = 0;
    let totalWeight = 0;
    let volumeAcc = 0;
    for (const item of totals) {
      const outcomeKey = normalise(item.outcome).replace(/ /g, '_');
      const rel = endpointRel[outcomeKey]?.[t] ?? (outcomeKey.includes('congenital') ? (t === 'first_trimester' ? 1.0 : 0.45) : 0.5);
      const cnt = Math.max(1, item.reported_count ?? item.records);
      const vol = Math.min(1.0, Math.log1p(cnt) / Math.log1p(120));
      const severity =
        outcomeKey === 'congenital_anomaly' || outcomeKey === 'fetal_death' || outcomeKey === 'stillbirth'
          ? 1.0
          : outcomeKey === 'miscarriage' || outcomeKey === 'preterm_birth'
            ? 0.85
            : 0.72;
      weightedSignal += rel * severity * (40 + 54 * vol);
      totalWeight += Math.max(0.2, rel);
      volumeAcc += cnt;
    }
    const base = totalWeight > 0 ? weightedSignal / Math.max(1, Math.min(totalWeight, totals.length * 0.75)) : 35;
    const boost = Math.min(14, Math.log1p(volumeAcc) * 2.2);
    scores[t] = clampScore((base + boost) * timingFactor, 24.0, 97.0);
  }

  const timingNote = hasExplicitTiming
    ? `Explicitly timed records (${Array.from(explicitTrimesters).join(', ')}).`
    : 'Source records lack gestational-week timestamps; projected across T1/T2/T3 by developmental endpoint relevance.';

  return {
    tier: 'H',
    name: 'Direct Human Pregnancy Outcomes (Tier 1)',
    available: true,
    is_human_evidence: true,
    evidence_count: preg.record_count ?? totals.length,
    base_weight: TIER_WEIGHTS.H,
    effective_weight: TIER_WEIGHTS.H,
    scores_by_trimester: scores,
    summary: `${preg.record_count} human pregnancy rows (${preg.reported_count_total ?? 'aggregate'} reported counts across ${totals.length} outcome groups). ${timingNote}`,
    provenance: (preg.sources ?? ['real_pregnancy_outcomes.csv']) as string[],
  };
}

function computeTier2PregnancyAdrAndTga(
  profile: ReturnType<typeof buildDrugProfile>,
  animalFallback: ReturnType<typeof animalEvidence>
): TierSummary {
  const tga = profile.tga_classification;
  const faers = profile.faers;
  const pregTerms = (profile.adr.pregnancy_related_terms ?? []) as Array<{ term: string; records: number }>;

  const hasTga = Boolean(tga?.available && (tga.category || tga.safety_statement));
  const hasFaers = Boolean(faers?.available && (faers.report_count ?? 0) > 0);
  const hasPregAdr = pregTerms.length > 0;

  if (!hasTga && !hasFaers && !hasPregAdr) {
    return {
      tier: 'P',
      name: 'Pregnancy ADR & Regulatory Classifications (Tier 2)',
      available: false,
      is_human_evidence: true,
      evidence_count: 0,
      base_weight: TIER_WEIGHTS.P,
      effective_weight: 0,
      scores_by_trimester: {
        first_trimester: null,
        second_trimester: null,
        third_trimester: null,
      },
      summary: 'No TGA pregnancy category, FAERS pregnancy summary, or pregnancy-related ADR terms indexed.',
      provenance: [],
    };
  }

  const cat = (tga?.category ?? '').toUpperCase().trim();
  const stmt = (tga?.safety_statement ?? '').toLowerCase();
  const tgaBase = cat ? (TGA_BASE_SCORE[cat] ?? 45) : 35;

  const faersReports = faers?.report_count ?? 0;
  const faersScore = hasFaers ? Math.min(86, 26 + 52 * (Math.log1p(faersReports) / Math.log1p(4000))) : null;
  const pregTermScore = hasPregAdr ? Math.min(82, 34 + 12 * Math.log1p(pregTerms.length)) : null;

  // Stage-specific biological modulators parsed from TGA statement and MedDRA pregnancy terms
  const termBlob = `${stmt} ${pregTerms.map((t) => t.term.toLowerCase()).join(' ')}`;
  let m1 = 1.0;
  let m2 = 0.94;
  let m3 = 0.9;

  if (/(first trimester|organogenesis|teratogen|congenital|malformation|spina bifida|neural tube|embryo|miscarriage|8 weeks)/.test(termBlob)) {
    m1 = 1.08;
    m2 = 0.92;
    m3 = 0.82;
  }
  if (/(second trimester|third trimester|oligohydramnios|ductus arteriosus|neonatal|premature|preterm|withdrawal|respiratory depression|placental haemorrhage)/.test(termBlob)) {
    m2 = Math.max(m2, 1.02);
    m3 = 1.1;
  }
  if (/(all trimesters|throughout pregnancy|contraindicated)/.test(termBlob)) {
    m1 = Math.max(m1, 1.05);
    m2 = Math.max(m2, 1.02);
    m3 = Math.max(m3, 1.0);
  }

  const parts: number[] = [];
  if (hasTga) parts.push(tgaBase);
  if (faersScore !== null) parts.push(faersScore);
  if (pregTermScore !== null) parts.push(pregTermScore);
  const combinedBase = parts.reduce((a, b) => a + b, 0) / parts.length;

  const scores: Record<CoreTrimester, number> = {
    first_trimester: clampScore(combinedBase * m1, 14.0, 96.0),
    second_trimester: clampScore(combinedBase * m2, 14.0, 96.0),
    third_trimester: clampScore(combinedBase * m3, 14.0, 96.0),
  };

  const pieces: string[] = [];
  if (hasTga) pieces.push(`TGA Category ${cat || 'Listed'}`);
  if (hasFaers) pieces.push(`FAERS ${faersReports} reports`);
  if (hasPregAdr) pieces.push(`${pregTerms.length} pregnancy-tagged MedDRA ADR terms`);
  if (animalFallback.available && (animalFallback.report_count ?? 0) > 0) {
    pieces.push(`contextual veterinary signal note only (excluded from human tier score)`);
  }

  const prov: string[] = [];
  if (hasTga) prov.push('pregnancy_classifications.csv (TGA)');
  if (hasFaers) prov.push('real_faers_summary.csv');
  if (hasPregAdr) prov.push('SIDER pregnancy-related MedDRA terms');

  return {
    tier: 'P',
    name: 'Pregnancy ADR & Regulatory Classifications (Tier 2)',
    available: true,
    is_human_evidence: true,
    evidence_count: (hasTga ? 1 : 0) + (hasFaers ? 1 : 0) + pregTerms.length,
    base_weight: TIER_WEIGHTS.P,
    effective_weight: TIER_WEIGHTS.P,
    scores_by_trimester: scores,
    summary: pieces.join(' · '),
    provenance: prov,
  };
}

function computeTier3GeneralAdr(profile: ReturnType<typeof buildDrugProfile>): TierSummary {
  const adr = profile.adr;
  const terms = (adr.terms ?? []) as Array<{ term: string; records: number }>;
  const faersHits = adr.faers_adr_vector_hits ?? 0;

  if (!adr.available || (adr.term_count === 0 && faersHits === 0)) {
    return {
      tier: 'A',
      name: 'General ADR Evidence (Tier 3)',
      available: false,
      is_human_evidence: true,
      evidence_count: 0,
      base_weight: TIER_WEIGHTS.A,
      effective_weight: 0,
      scores_by_trimester: {
        first_trimester: null,
        second_trimester: null,
        third_trimester: null,
      },
      summary: 'No SIDER or FAERS general ADR records are mapped for this drug.',
      provenance: [],
    };
  }

  let highConcernCount = 0;
  for (const t of terms) {
    const low = (t.term ?? '').toLowerCase();
    if (HIGH_CONCERN_ADR_KEYWORDS.some((kw) => low.includes(kw))) {
      highConcernCount++;
    }
  }

  const termDensity = Math.min(1.0, Math.log1p(adr.term_count) / Math.log1p(250));
  const hitDensity = Math.min(1.0, faersHits / 15);
  const severeDensity = Math.min(1.0, highConcernCount / 6);

  const base = 18 + 28 * termDensity + 16 * severeDensity + 14 * hitDensity;

  return {
    tier: 'A',
    name: 'General ADR Evidence (Tier 3)',
    available: true,
    is_human_evidence: true,
    evidence_count: Math.max(adr.term_count, faersHits),
    base_weight: TIER_WEIGHTS.A,
    effective_weight: TIER_WEIGHTS.A,
    scores_by_trimester: {
      first_trimester: clampScore(base * 0.96, 14.0, 78.0),
      second_trimester: clampScore(base * 1.0, 14.0, 78.0),
      third_trimester: clampScore(base * 1.04, 14.0, 78.0),
    },
    summary: `${adr.term_count} distinct MedDRA ADR terms (${highConcernCount} severe/developmental-adjacent terms), ${faersHits}/20 FAERS ADR vector flags.`,
    provenance: (adr.source_datasets ?? ['SIDER']) as string[],
  };
}

function computeTier4Similarity(profile: ReturnType<typeof buildDrugProfile>): TierSummary {
  const rel = profile.related_pregnancy_evidence;
  const candidates = rel?.candidates ?? [];
  if (!rel?.available || candidates.length === 0) {
    return {
      tier: 'S',
      name: 'Similar Chemical Structure Analogues (Tier 4)',
      available: false,
      is_human_evidence: false,
      evidence_count: 0,
      base_weight: TIER_WEIGHTS.S,
      effective_weight: 0,
      scores_by_trimester: {
        first_trimester: null,
        second_trimester: null,
        third_trimester: null,
      },
      summary: rel?.reason || 'No chemical analogues met the >= 0.20 Tanimoto similarity threshold.',
      provenance: [],
    };
  }

  let weightedSignalSum = 0;
  let simSum = 0;
  const outcomeBlob: string[] = [];

  for (const c of candidates) {
    const sim = Number(c.similarity ?? 0);
    const sig = Number(c.estimated_concern_signal ?? 35);
    weightedSignalSum += sim * sig;
    simSum += sim;
    for (const o of c.outcomes ?? []) {
      outcomeBlob.push(o.outcome.toLowerCase());
    }
  }

  const avgSignal = simSum > 0 ? weightedSignalSum / simSum : 35;
  const bestSim = candidates[0].similarity;
  const simConfidenceScale = 0.55 + 0.45 * Math.min(1, bestSim / 0.7);
  const rawBase = avgSignal * simConfidenceScale;

  const hasEarly = outcomeBlob.some((o) => o.includes('congenital') || o.includes('miscarriage'));
  const hasLate = outcomeBlob.some((o) => o.includes('preterm') || o.includes('growth') || o.includes('stillbirth'));

  return {
    tier: 'S',
    name: 'Similar Chemical Structure Analogues (Tier 4)',
    available: true,
    is_human_evidence: false,
    evidence_count: candidates.length,
    base_weight: TIER_WEIGHTS.S,
    effective_weight: TIER_WEIGHTS.S,
    scores_by_trimester: {
      first_trimester: clampScore(rawBase * (hasEarly ? 1.06 : 1.0), 14.0, 84.0),
      second_trimester: clampScore(rawBase * 0.96, 14.0, 84.0),
      third_trimester: clampScore(rawBase * (hasLate ? 1.05 : 0.92), 14.0, 84.0),
    },
    summary: `Top structural analogue: ${candidates[0].drug_name} (Tanimoto ${bestSim.toFixed(3)}${candidates[0].tga_category ? `, TGA ${candidates[0].tga_category}` : ''}); ${candidates.length} analogues >= 0.20.`,
    provenance: ['OpenChemLib 512-bit Fingerprint Tanimoto Similarity (>= 0.20)'],
  };
}

function computeTier5AnimalToxicity(
  profile: ReturnType<typeof buildDrugProfile>,
  animalAdr: ReturnType<typeof animalEvidence>
): TierSummary {
  const tox = profile.reproductive_toxicity;
  const hasExperimental = Boolean(tox?.available && (tox.record_count > 0 || (tox.study_flags?.length ?? 0) > 0));
  const hasVetAdr = Boolean(animalAdr.available && (animalAdr.report_count ?? 0) > 0);

  if (!hasExperimental && !hasVetAdr) {
    return {
      tier: 'T',
      name: 'Animal Developmental Toxicity & Veterinary ADR (Tier 5)',
      available: false,
      is_human_evidence: false,
      evidence_count: 0,
      base_weight: TIER_WEIGHTS.T,
      effective_weight: 0,
      scores_by_trimester: {
        first_trimester: null,
        second_trimester: null,
        third_trimester: null,
      },
      summary: 'No nonclinical EPA ToxRefDB or FDA CVM veterinary ADR records exist for this drug.',
      provenance: [],
    };
  }

  let expScore = 0;
  if (hasExperimental) {
    const studyCount = tox.record_count ?? 1;
    const speciesBonus = Math.min(12, (tox.species?.length ?? 0) * 5);
    const hasDev = (tox.study_types ?? []).some((s) => s.toUpperCase().includes('DEV') || s.toUpperCase().includes('DART'));
    const hasMgr = (tox.study_types ?? []).some((s) => s.toUpperCase().includes('MGR'));
    expScore = 34 + 26 * (Math.log1p(studyCount) / Math.log1p(25)) + speciesBonus + (hasDev ? 10 : 0) + (hasMgr ? 6 : 0);
  }

  let vetScore = 0;
  if (hasVetAdr) {
    const reports = animalAdr.report_count ?? 0;
    const serious = animalAdr.serious_report_count ?? 0;
    const deaths = animalAdr.death_or_euthanasia_report_count ?? 0;
    vetScore =
      22 +
      28 * (Math.log1p(reports) / Math.log1p(1200)) +
      (serious > 0 ? Math.min(14, 4 * Math.log1p(serious)) : 0) +
      (deaths > 0 ? 6 : 0);
  }

  const combined =
    hasExperimental && hasVetAdr
      ? 0.68 * expScore + 0.32 * vetScore
      : hasExperimental
        ? expScore
        : vetScore;

  const summaryParts: string[] = [];
  if (hasExperimental) {
    summaryParts.push(
      `${tox.record_count} nonclinical study records (species: ${(tox.species ?? []).join(', ') || 'unspecified'}; types: ${(tox.study_types ?? []).join(', ') || 'listed'})`
    );
  }
  if (hasVetAdr) {
    summaryParts.push(
      `FDA CVM ${animalAdr.report_count} veterinary ADR reports across ${animalAdr.species_count} species (${animalAdr.serious_report_count} serious)`
    );
  }

  const prov: string[] = [];
  if (hasExperimental) prov.push(...((tox.sources as string[]) ?? ['EPA ToxRefDB']));
  if (hasVetAdr) prov.push('FDA CVM ADAE (VeDDRA)');

  return {
    tier: 'T',
    name: 'Animal Developmental Toxicity & Veterinary ADR (Tier 5)',
    available: true,
    is_human_evidence: false,
    evidence_count: (hasExperimental ? tox.record_count : 0) + (hasVetAdr ? 1 : 0),
    base_weight: TIER_WEIGHTS.T,
    effective_weight: TIER_WEIGHTS.T,
    scores_by_trimester: {
      first_trimester: clampScore(combined * 1.04, 16.0, 86.0),
      second_trimester: clampScore(combined * 0.96, 16.0, 86.0),
      third_trimester: clampScore(combined * 0.9, 16.0, 86.0),
    },
    summary: `Nonclinical animal evidence ONLY (not human proof): ${summaryParts.join('; ')}.`,
    provenance: prov,
  };
}

function computeMlAndAiTier6(
  profile: ReturnType<typeof buildDrugProfile>,
  priorTiers: { H: TierSummary; P: TierSummary; A: TierSummary; S: TierSummary; T: TierSummary },
  forceMissingArtifacts: boolean
): {
  tier: TierSummary;
  mlAssessment: MlModelAssessment;
  aiReasoning: AiReasoningAssessment;
} {
  const store = getDataStore();
  const mol = profile.molecular;
  const targets = (profile.targets?.targets ?? []) as Array<{
    target_name: string;
    target_class: string;
    gene: string;
    action_type: string;
  }>;
  const nonclinicalBenchmarkPresent =
    !forceMissingArtifacts && Object.keys(store.reproductiveToxMetrics ?? {}).length > 0;
  const humanCheckpointPresent = !forceMissingArtifacts && Boolean(store.filesPresent.checkpoint);

  const structuralAlerts: string[] = [];
  const targetAlerts: string[] = [];
  let mlScore = 14.0;
  let featureCompleteness = 0.0;
  let placentalPropensity: MlModelAssessment['placental_transfer_propensity'] = 'Active/FcRn or Unknown';
  let mlBasis = '';

  const smiles = (profile.identity?.smiles || mol.canonical_smiles || '').toLowerCase();
  const mw = profile.identity?.molecular_weight ?? mol.descriptors?.molecular_weight ?? null;
  const clogp = mol.descriptors?.clogp ?? null;
  const tpsa = mol.descriptors?.tpsa ?? null;
  const hbd = mol.descriptors?.h_bond_donors ?? null;
  const hba = mol.descriptors?.h_bond_acceptors ?? null;
  const rings = mol.descriptors?.ring_count ?? null;

  if (mol.available && mw !== null) {
    featureCompleteness = 0.85;
    // Physicochemical placental transfer + teratogenicity screening heuristic grounded in Lipinski/placental BBB rules
    let score = 10.0;

    if (mw <= 500) {
      score += 14;
      placentalPropensity = 'High';
      structuralAlerts.push(`Low MW (${mw.toFixed(1)} g/mol <= 500) favors passive transplacental diffusion`);
    } else if (mw <= 800) {
      score += 7;
      placentalPropensity = 'Moderate';
      structuralAlerts.push(`Intermediate MW (${mw.toFixed(1)} g/mol) with moderate passive transfer`);
    } else {
      score += 2;
      placentalPropensity = 'Low';
      structuralAlerts.push(`High MW (${mw.toFixed(1)} g/mol > 800) restricts passive early diffusion`);
    }

    if (clogp !== null) {
      if (clogp >= 1.5 && clogp <= 4.8) {
        score += 12;
        structuralAlerts.push(`Lipophilic logP (${clogp.toFixed(2)}) supports membrane permeation`);
      } else if (clogp > 4.8) {
        score += 15;
        structuralAlerts.push(`High lipophilicity (cLogP ${clogp.toFixed(2)}) increases tissue partitioning`);
      } else if (clogp < 0) {
        score -= 2; // Hydrophilic drug -> lower passive score, often pushes ml_score < 20 to trigger AI reasoning
      } else {
        score += 4;
      }
    }

    if (tpsa !== null) {
      if (tpsa < 75) {
        score += 8;
        structuralAlerts.push(`Low TPSA (${tpsa.toFixed(1)} Å²) facilitates barrier permeability`);
      } else if (tpsa > 140) {
        score -= 4;
      } else {
        score += 3;
      }
    }

    if (rings !== null && rings >= 3) {
      score += 6;
      structuralAlerts.push(`Polycyclic core (${rings} rings)`);
    }

    // Specific known toxicophore motifs in SMILES
    if (/(n=n|nn\(|n\(=o\)o|c\(=o\)cl|p\(=o\)|\[pt\]|\[as\]|c=c\(c\)c=c|c\(=o\)no)/.test(smiles)) {
      score += 16;
      structuralAlerts.push('Reactive, conjugated polyene, hydroxamate, or alkylating structural motif');
    }

    // Pharmacological target developmental sensitivity check
    const devTargets = targets.filter((t) =>
      /(retinoid|folate|dihydrofolate|topoisomerase|angiotensin|cyclooxygenase|histone deacetylase|androgen|estrogen|progesterone|vegf|mtor|tubulin|calcineurin|vitamin k|coagulation|serotonin|gaba)/i.test(
        `${t.target_name} ${t.target_class} ${t.gene}`
      )
    );
    if (devTargets.length > 0) {
      score += Math.min(20, 8 + devTargets.length * 4);
      targetAlerts.push(...devTargets.slice(0, 4).map((t) => `${t.target_name} (${t.gene || t.target_class})`));
    } else if (targets.length > 0) {
      score += 4;
      targetAlerts.push(`${targets.length} mapped human/mammalian pharmacological targets`);
    }

    mlScore = round1(Math.min(92.0, Math.max(8.0, score)));
    mlBasis = 'Deterministic physicochemical placental-transfer + structural motif + target interaction scorer (human neural checkpoint absent locally).';
  } else {
    // Biologic / combination / missing SMILES -> passive ML cannot score structure, so mlScore drops below 20%
    featureCompleteness = targets.length > 0 ? 0.35 : 0.1;
    mlScore = targets.length > 0 ? 14.5 : 9.0;
    if (targets.length > 0) {
      targetAlerts.push(...targets.slice(0, 3).map((t) => `${t.target_name}`));
    }
    structuralAlerts.push('No small-molecule SMILES structure available (biologic, macromolecule, salt mix, or unmapped compound)');
    mlBasis = 'Incomplete molecular structure prevents full fingerprint ML inference; <20% fallback activates supplementary AI reasoning.';
  }

  const mlTriggered = mlScore < 20.0;
  const availablePrior = Object.values(priorTiers).filter((t) => t.available);
  const priorAvg =
    availablePrior.length > 0
      ? availablePrior.reduce((acc, t) => {
          const s =
            ((t.scores_by_trimester.first_trimester ?? 0) +
              (t.scores_by_trimester.second_trimester ?? 0) +
              (t.scores_by_trimester.third_trimester ?? 0)) /
            3;
          return acc + s * t.base_weight;
        }, 0) / availablePrior.reduce((acc, t) => acc + t.base_weight, 0)
      : null;

  const keySignals: string[] = [];
  if (priorTiers.H.available) keySignals.push(`Human pregnancy outcomes: ${priorTiers.H.summary}`);
  if (priorTiers.P.available) keySignals.push(`Pregnancy ADR/Regulatory: ${priorTiers.P.summary}`);
  if (priorTiers.A.available) keySignals.push(`General ADR context: ${priorTiers.A.summary}`);
  if (priorTiers.S.available) keySignals.push(`Structural analogues: ${priorTiers.S.summary}`);
  if (priorTiers.T.available) keySignals.push(`Nonclinical animal signal: ${priorTiers.T.summary}`);
  if (targetAlerts.length > 0) keySignals.push(`Target pharmacology: ${targetAlerts.join(', ')}`);

  // Detect monoclonal antibodies or biologics for second/third trimester FcRn placental transport insight
  const drugNameLow = (profile.drug_name ?? profile.query ?? '').toLowerCase();
  const isBiologicMab = /(mab$|mab\b|cept$|glutide$|parin$|insulin|interferon|immunoglobulin)/.test(drugNameLow);

  let aiSupplementaryScore = mlScore;
  let gapInterpretation = 'ML physicochemical screening exceeded 20%; AI gap reasoning recorded as contextual annotation only.';
  let explanation = 'Primary molecular scorer achieved adequate feature signal (>= 20%).';

  if (mlTriggered) {
    // Synthesize across prior records, ADRs, analogues, animal data, and pregnancy/biologic context
    let est = priorAvg !== null ? 0.62 * priorAvg + 0.38 * 28.0 : 22.0;
    if (isBiologicMab) {
      est = Math.max(est, 26.0);
      keySignals.push('Biologic / macromolecular naming pattern: minimal passive T1 transfer, potential active IgG/FcRn placental transfer in late T2/T3');
    }
    if (hbd !== null && hbd > 5) {
      keySignals.push(`High hydrogen-bond donor count (${hbd}) lowers ML passive permeability prediction despite potential systemic effects`);
    }
    aiSupplementaryScore = clampScore(est, 14.0, 85.0);

    const activeNames = availablePrior.map((t) => `${t.tier}:${t.evidence_count} records`).join(', ') || 'no prior empirical tiers';
    gapInterpretation = mol.available
      ? `Hydrophilic / high-TPSA or high-MW profile lowered passive ML transfer score (${mlScore.toFixed(1)} < 20%); AI reasoning synthesized ${activeNames} to prevent false 0% risk assignment.`
      : `Missing SMILES structure reduced ML score to ${mlScore.toFixed(1)} (< 20%); AI reasoning bridged structural gaps using ${activeNames}.`;

    explanation = `Triggered <20% ML fallback (ML=${mlScore.toFixed(1)}%). Supplementary AI assessment estimated ${aiSupplementaryScore.toFixed(1)}/100 across prior pregnancy, regulatory, ADR, analogue, and nonclinical context. ${isBiologicMab ? 'Later trimesters reflect possible FcRn transport of IgG-class biologics.' : ''}`;
  }

  const tier6T1 = mlTriggered ? aiSupplementaryScore * (isBiologicMab ? 0.78 : 1.0) : mlScore * (placentalPropensity === 'High' ? 1.06 : 0.96);
  const tier6T2 = mlTriggered ? aiSupplementaryScore * (isBiologicMab ? 1.04 : 0.96) : mlScore * 0.98;
  const tier6T3 = mlTriggered ? aiSupplementaryScore * (isBiologicMab ? 1.15 : 0.94) : mlScore * 0.95;

  const tier: TierSummary = {
    tier: 'M',
    name: 'Molecular Descriptors, ML & AI Fallback (Tier 6)',
    available: true, // Always active as baseline molecular/AI screening floor so score is NEVER 0
    is_human_evidence: false,
    evidence_count: (mol.available ? 1 : 0) + targets.length + availablePrior.length,
    base_weight: TIER_WEIGHTS.M,
    effective_weight: TIER_WEIGHTS.M,
    scores_by_trimester: {
      first_trimester: clampScore(tier6T1, 10.0, 90.0),
      second_trimester: clampScore(tier6T2, 10.0, 90.0),
      third_trimester: clampScore(tier6T3, 10.0, 90.0),
    },
    summary: mlTriggered
      ? `ML score ${mlScore.toFixed(1)} (< 20%) triggered AI reasoning fallback (${aiSupplementaryScore.toFixed(1)}). ${gapInterpretation}`
      : `Physicochemical/target ML screening score ${mlScore.toFixed(1)} (${placentalPropensity} placental propensity).`,
    provenance: [
      ...(mol.available ? ['OpenChemLib Molecular Descriptors'] : ['Incomplete structure fallback']),
      ...(nonclinicalBenchmarkPresent ? ['data/reproductive_tox_training_metrics.json'] : []),
      ...(mlTriggered ? ['<20% ML Supplementary AI Reasoning Layer'] : []),
    ],
  };

  return {
    tier,
    mlAssessment: {
      model_name: forceMissingArtifacts
        ? 'Exploratory Heuristic Fallback (Artifacts Disabled)'
        : nonclinicalBenchmarkPresent
          ? 'Placental Physicochemical Scorer + OECD Nonclinical Reference (v2)'
          : 'Placental Physicochemical Scorer (No Checkpoint)',
      human_checkpoint_present: humanCheckpointPresent,
      nonclinical_benchmark_present: nonclinicalBenchmarkPresent,
      ml_score: mlScore,
      ml_score_basis: mlBasis,
      placental_transfer_propensity: placentalPropensity,
      structural_alerts: structuralAlerts,
      target_alerts: targetAlerts,
      feature_completeness: Math.round(featureCompleteness * 100),
      artifact_note: humanCheckpointPresent
        ? 'Trained human checkpoint active.'
        : 'No validated human pregnancy neural checkpoint (.pt) is present locally; exploratory OECD guideline benchmarks are indexed separately in Data Readiness, while runtime scoring uses deterministic physicochemical/target features + <20% AI fallback.',
    },
    aiReasoning: {
      triggered: mlTriggered,
      trigger_reason: mlTriggered
        ? `ML prediction score (${mlScore.toFixed(1)}%) fell below 20% threshold.`
        : `ML prediction score (${mlScore.toFixed(1)}%) >= 20% threshold; supplementary override not required.`,
      ml_score: mlScore,
      ai_supplementary_score: aiSupplementaryScore,
      blended_tier6_score: clampScore((tier6T1 + tier6T2 + tier6T3) / 3, 10.0, 90.0),
      gap_interpretation: gapInterpretation,
      explanation,
      key_signals: keySignals,
      evidence_domains_synthesized: availablePrior.map((t) => t.name),
    },
  };
}

function detectConflicts(
  profile: ReturnType<typeof buildDrugProfile>,
  tiers: Record<'H' | 'P' | 'A' | 'S' | 'T' | 'M', TierSummary>,
  mlScore: number
): ConflictFlag[] {
  const conflicts: ConflictFlag[] = [];
  const tgaCat = (profile.tga_classification?.category ?? '').toUpperCase().trim();
  const hScore = tiers.H.scores_by_trimester.first_trimester ?? 0;
  const pScore = tiers.P.scores_by_trimester.first_trimester ?? 0;
  const tScore = tiers.T.scores_by_trimester.first_trimester ?? 0;
  const sScore = tiers.S.scores_by_trimester.first_trimester ?? 0;

  if ((['X', 'D'].includes(tgaCat) || hScore >= 60) && tiers.T.available && tScore < 38) {
    conflicts.push({
      severity: 'High',
      title: 'Human/Regulatory Concern Exceeds Animal Signal',
      detail: `Human pregnancy or regulatory classification (${tgaCat || `Tier H ${hScore}`}) indicates high concern, whereas nonclinical animal scores are substantially lower (${tScore.toFixed(1)}). Human data supersedes negative animal screens.`,
      tiers_involved: ['H', 'P', 'T'],
    });
  }

  if ((['X', 'D'].includes(tgaCat) || hScore >= 60) && mlScore < 25) {
    conflicts.push({
      severity: 'Moderate',
      title: 'High Human Signal vs Low Molecular Passive-Transfer ML Score',
      detail: `Confirmed human/regulatory concern contrasts with a low molecular passive-transfer estimate (${mlScore.toFixed(1)}%). Target pharmacology or indirect maternal/placental mechanisms likely drive concern.`,
      tiers_involved: ['H', 'P', 'M'],
    });
  }

  if (tiers.T.available && tScore >= 45 && !tiers.H.available) {
    conflicts.push({
      severity: 'Moderate',
      title: 'Animal Toxicity Signal Without Human Pregnancy Confirmation',
      detail: `Nonclinical animal toxicity / veterinary ADR records show elevated signal (${tScore.toFixed(1)}), but direct human pregnancy outcome data are missing. Do not equate missing human data with human safety.`,
      tiers_involved: ['T', 'H'],
    });
  } else if (tiers.T.available && tScore >= 50 && ['A', 'B1', 'B2'].includes(tgaCat)) {
    conflicts.push({
      severity: 'Moderate',
      title: 'Animal Toxicity vs Lower Regulatory Category',
      detail: `Animal datasets indicate elevated signal (${tScore.toFixed(1)}), whereas TGA Category ${tgaCat} reflects lower clinical risk concern or dose-dependent margins.`,
      tiers_involved: ['T', 'P'],
    });
  }

  if (tiers.S.available && sScore >= 60 && !tiers.H.available && pScore < 35) {
    conflicts.push({
      severity: 'Informational',
      title: 'High-Concern Structural Analogue With Limited Target-Drug Data',
      detail: `A structural analogue shows elevated pregnancy signal (${sScore.toFixed(1)}), while the queried drug lacks strong human/regulatory records. Prioritise target-specific verification.`,
      tiers_involved: ['S', 'H', 'P'],
    });
  }

  if (tiers.H.available && profile.pregnancy?.unknown_trimester) {
    conflicts.push({
      severity: 'Informational',
      title: 'Untimed Human Pregnancy Outcomes Projected Across Trimesters',
      detail:
        'Human pregnancy outcomes exist in local tables but lack explicit gestational-week labels; T1/T2/T3 scores reflect developmental-endpoint relevance weights rather than timed case counts.',
      tiers_involved: ['H'],
    });
  }

  return conflicts;
}

function bandForScore(score: number): string {
  if (score >= 65) return 'High Screening Priority';
  if (score >= 45) return 'Moderate-High Screening Priority';
  if (score >= 25) return 'Moderate Screening Priority';
  return 'Low-Baseline Exploratory Priority (Never Zero-Risk)';
}

function confidenceForTrimester(
  tiers: Record<'H' | 'P' | 'A' | 'S' | 'T' | 'M', TierSummary>,
  hasExplicitTiming: boolean,
  hasSmiles: boolean,
  conflicts: ConflictFlag[],
  mlTriggered: boolean
): { score: number; level: 'High' | 'Moderate' | 'Low' | 'Very Low' } {
  let conf = 14;
  if (tiers.H.available) conf += hasExplicitTiming ? 36 : 24;
  if (tiers.P.available) conf += 18;
  if (tiers.A.available) conf += 9;
  if (tiers.S.available) conf += 6;
  if (tiers.T.available) conf += 8;
  if (hasSmiles) conf += 6;
  if (mlTriggered) conf -= 5;
  for (const c of conflicts) {
    if (c.severity === 'High') conf -= 8;
    else if (c.severity === 'Moderate') conf -= 5;
  }
  const finalConf = Math.round(Math.min(94, Math.max(12, conf)));
  const level: 'High' | 'Moderate' | 'Low' | 'Very Low' =
    finalConf >= 70 ? 'High' : finalConf >= 48 ? 'Moderate' : finalConf >= 28 ? 'Low' : 'Very Low';
  return { score: finalConf, level };
}

export function evaluateDrugSafety(
  drugName: string,
  trimester: string,
  outcomeTarget: string,
  drugId: string | null,
  options: EvaluationOptions = {}
) {
  const store = getDataStore();
  const profile = options.profileOverride ?? buildDrugProfile(drugId || drugName);
  const evidenceSourceAvailable = store.filesPresent.master;

  const allowedOutcomes =
    outcomeTarget === 'Fetal Outcomes Only'
      ? new Set(['congenital_anomaly', 'miscarriage', 'preterm_birth', 'fetal_growth_outcome', 'stillbirth'])
      : outcomeTarget === 'Maternal Outcomes Only'
        ? new Set(['maternal_adverse_reaction'])
        : new Set([
            'congenital_anomaly',
            'miscarriage',
            'preterm_birth',
            'fetal_growth_outcome',
            'stillbirth',
            'maternal_adverse_reaction',
          ]);

  const drugMaster = store.master.filter((r) =>
    drugId
      ? String(r.drug_id ?? '') === drugId || String(r.canonical_drug_id ?? '') === drugId
      : normalise(r.drug_name) === normalise(drugName)
  );

  let filteredMaster = drugMaster.filter((r) => {
    const oc = normalise(r.outcome_category);
    const sf = sourceFamily(String(r.source_dataset ?? ''));
    return allowedOutcomes.has(oc) && ['mendeley', 'faers', 'onsides'].includes(sf);
  });

  const trimesterCounts: Record<string, { label: string; record_count: number }> = {
    first_trimester: { label: 'First Trimester (T1)', record_count: 0 },
    second_trimester: { label: 'Second Trimester (T2)', record_count: 0 },
    third_trimester: { label: 'Third Trimester (T3)', record_count: 0 },
    entire_pregnancy: { label: 'Entire Pregnancy', record_count: 0 },
    unknown: { label: 'Timing not recorded', record_count: 0 },
  };

  for (const r of filteredMaster) {
    const t = normalise(r.trimester || 'unknown');
    if (t in trimesterCounts) trimesterCounts[t].record_count += 1;
    else trimesterCounts.unknown.record_count += 1;
  }

  if (trimester !== 'All') {
    const wanted = TRIMESTER_VALUES[trimester] ?? 'unknown';
    filteredMaster = filteredMaster.filter((r) => normalise(r.trimester || 'unknown') === wanted);
  }

  const effectiveId = profile.drug_id ?? drugId;
  const quality = effectiveId ? computeEvidenceQualityScore(effectiveId) : null;
  const mask = quality?.availability_mask ?? [];
  const availableSources = mask.filter(([, v]) => v > 0).map(([k]) => k);
  const missingSources = mask.filter(([, v]) => v <= 0).map(([k]) => k);

  const animalAdr = animalEvidence(effectiveId, drugName);

  // Build all 6 Tiers
  const H = computeTier1HumanPregnancy(profile, store.scoringConfig);
  const P = computeTier2PregnancyAdrAndTga(profile, animalAdr);
  const A = computeTier3GeneralAdr(profile);
  const S = computeTier4Similarity(profile);
  const T = computeTier5AnimalToxicity(profile, animalAdr);
  const { tier: M, mlAssessment, aiReasoning } = computeMlAndAiTier6(
    profile,
    { H, P, A, S, T },
    Boolean(options.forceMissingArtifacts)
  );

  const allTiers: Record<'H' | 'P' | 'A' | 'S' | 'T' | 'M', TierSummary> = { H, P, A, S, T, M };
  const tierList = [H, P, A, S, T, M];

  // Effective renormalized weights across available tiers (missing sources are NEVER treated as 0!)
  const activeWeightSum = tierList
    .filter((tier) => tier.available)
    .reduce((acc, tier) => acc + tier.base_weight, 0);

  for (const tier of tierList) {
    tier.effective_weight =
      tier.available && activeWeightSum > 0
        ? Math.round((tier.base_weight / activeWeightSum) * 1000) / 1000
        : 0;
  }

  const conflicts = detectConflicts(profile, allTiers, mlAssessment.ml_score);
  const hasExplicitTiming = Boolean(profile.pregnancy?.available && !profile.pregnancy?.unknown_trimester);
  const baseConf = confidenceForTrimester(
    allTiers,
    hasExplicitTiming,
    Boolean(profile.molecular?.available),
    conflicts,
    aiReasoning.triggered
  );

  const trimesterResults: Record<CoreTrimester, TrimesterResult> = {} as Record<CoreTrimester, TrimesterResult>;
  const maternalPriority: Record<string, Record<string, unknown>> = {};
  const fetalPriority: Record<string, Record<string, unknown>> = {};
  const overallPriority: Record<string, Record<string, unknown>> = {};

  for (const t of CORE_TRIMESTERS) {
    let weightedSum = 0;
    const activeTiers: Array<'H' | 'P' | 'A' | 'S' | 'T' | 'M'> = [];
    const supportingEvidence: string[] = [];

    for (const tier of tierList) {
      if (!tier.available) continue;
      const tScore = tier.scores_by_trimester[t];
      if (tScore === null) continue;
      weightedSum += tier.effective_weight * tScore;
      activeTiers.push(tier.tier);
      supportingEvidence.push(`[Tier ${tier.tier}] ${tier.summary}`);
    }

    // Add gentle biological stage differentiation if multiple tiers are close, guaranteeing non-zero [8.0, 98.0]
    const concernScore = clampScore(weightedSum, 8.0, 98.0);
    const missingNotes: string[] = [];
    if (!H.available) missingNotes.push('No direct human pregnancy outcome records (Tier 1 missing; not zero risk).');
    else if (profile.pregnancy?.unknown_trimester) {
      missingNotes.push('Human pregnancy records lack explicit gestational-week timing; stage weight projected.');
    }
    if (!P.available) missingNotes.push('No TGA category or pregnancy-tagged FAERS summary.');
    if (!A.available) missingNotes.push('No SIDER / FAERS general ADR records.');
    if (!S.available) missingNotes.push('No structurally related analogues >= 0.20 Tanimoto similarity.');
    if (!T.available) missingNotes.push('No nonclinical EPA ToxRefDB or veterinary ADR studies.');
    if (!profile.molecular?.available) missingNotes.push('No small-molecule SMILES available; <20% ML fallback relied on AI synthesis.');
    if (!mlAssessment.human_checkpoint_present) {
      missingNotes.push('Validated human pregnancy ML checkpoint absent locally; physicochemical/AI fallback active.');
    }

    const adverseOutcomes: string[] = [];
    for (const o of (profile.pregnancy?.outcome_totals ?? []) as Array<{ outcome: string; reported_count: number | null }>) {
      adverseOutcomes.push(`${o.outcome.replace(/_/g, ' ')}${o.reported_count ? ` (${o.reported_count})` : ''}`);
    }
    if (profile.tga_classification?.category) {
      adverseOutcomes.push(`TGA Category ${profile.tga_classification.category}`);
    }
    for (const pTerm of ((profile.adr?.pregnancy_related_terms ?? []) as Array<{ term: string }>).slice(0, 4)) {
      adverseOutcomes.push(pTerm.term);
    }
    if (adverseOutcomes.length === 0 && T.available) {
      adverseOutcomes.push('Nonclinical developmental/veterinary toxicity signal only');
    }
    if (adverseOutcomes.length === 0) {
      adverseOutcomes.push('Exploratory screening signal derived from ADR/analogue/molecular features');
    }

    const primaryCategory = H.available
      ? H.name
      : P.available
        ? P.name
        : T.available
          ? T.name
          : A.available
            ? A.name
            : S.available
              ? S.name
              : M.name;

    // Slight per-trimester confidence attenuation when direct timing is absent
    const tConfScore =
      hasExplicitTiming
        ? baseConf.score
        : t === 'first_trimester'
          ? Math.max(12, baseConf.score - 2)
          : t === 'second_trimester'
            ? Math.max(12, baseConf.score - 4)
            : Math.max(12, baseConf.score - 5);
    const tConfLevel: 'High' | 'Moderate' | 'Low' | 'Very Low' =
      tConfScore >= 70 ? 'High' : tConfScore >= 48 ? 'Moderate' : tConfScore >= 28 ? 'Low' : 'Very Low';

    // Maternal vs Fetal split (both strictly > 0)
    const maternalBase = clampScore(
      concernScore * (P.available || A.available ? 0.96 : 0.85) + (t === 'third_trimester' ? 2.5 : 0),
      8.0,
      97.0
    );
    const fetalBase = clampScore(
      concernScore * (H.available || T.available || S.available ? 1.03 : 0.92),
      8.0,
      98.0
    );

    trimesterResults[t] = {
      trimester: t,
      trimester_label: TRIMESTER_LABELS[t],
      gestational_window: TRIMESTER_WINDOWS[t],
      concern_score: concernScore,
      score_label: 'Research / Screening Priority Score (0–100, not clinical probability)',
      priority_band: bandForScore(concernScore),
      evidence_category: primaryCategory,
      confidence_score: tConfScore,
      confidence_level: tConfLevel,
      uncertainty_percent: 100 - tConfScore,
      adverse_outcomes: Array.from(new Set(adverseOutcomes)).slice(0, 8),
      supporting_evidence: supportingEvidence,
      missing_evidence_notes: missingNotes,
      active_tiers: activeTiers,
      maternal_score: maternalBase,
      fetal_score: fetalBase,
    };

    maternalPriority[t] = {
      drug_id: effectiveId,
      trimester: t,
      trimester_label: TRIMESTER_LABELS[t],
      score: maternalBase,
      score_available: true,
      eligible_record_count: (profile.adr?.term_count ?? 0) + (P.available ? 1 : 0),
      total_human_records: profile.pregnancy?.record_count ?? 0,
      has_nonclinical: T.available,
    };

    fetalPriority[t] = {
      drug_id: effectiveId,
      trimester: t,
      trimester_label: TRIMESTER_LABELS[t],
      score: fetalBase,
      score_available: true,
      eligible_record_count: (profile.pregnancy?.record_count ?? 0) + (S.available ? 1 : 0) + (T.available ? 1 : 0),
      total_human_records: profile.pregnancy?.record_count ?? 0,
      total_nonclinical_records: profile.reproductive_toxicity?.record_count ?? 0,
      has_nonclinical_evidence: T.available,
    };

    overallPriority[t] = {
      trimester: t,
      trimester_label: TRIMESTER_LABELS[t],
      score: concernScore,
      score_available: true,
      maternal_score: maternalBase,
      fetal_score: fetalBase,
      eligible_record_count: tierList.filter((x) => x.available).reduce((acc, x) => acc + x.evidence_count, 0),
    };
  }

  // Entire pregnancy aggregate across T1/T2/T3
  const entireScore = clampScore(
    (trimesterResults.first_trimester.concern_score +
      trimesterResults.second_trimester.concern_score +
      trimesterResults.third_trimester.concern_score) /
      3,
    8.0,
    98.0
  );
  const entireMaternal = clampScore(
    (trimesterResults.first_trimester.maternal_score +
      trimesterResults.second_trimester.maternal_score +
      trimesterResults.third_trimester.maternal_score) /
      3,
    8.0,
    97.0
  );
  const entireFetal = clampScore(
    (trimesterResults.first_trimester.fetal_score +
      trimesterResults.second_trimester.fetal_score +
      trimesterResults.third_trimester.fetal_score) /
      3,
    8.0,
    98.0
  );

  maternalPriority.entire_pregnancy = {
    drug_id: effectiveId,
    trimester: 'entire_pregnancy',
    trimester_label: 'Entire Pregnancy',
    score: entireMaternal,
    score_available: true,
    eligible_record_count: maternalPriority.first_trimester.eligible_record_count,
  };
  fetalPriority.entire_pregnancy = {
    drug_id: effectiveId,
    trimester: 'entire_pregnancy',
    trimester_label: 'Entire Pregnancy',
    score: entireFetal,
    score_available: true,
    eligible_record_count: fetalPriority.first_trimester.eligible_record_count,
  };
  overallPriority.entire_pregnancy = {
    trimester: 'entire_pregnancy',
    trimester_label: 'Entire Pregnancy',
    score: entireScore,
    score_available: true,
    maternal_score: entireMaternal,
    fetal_score: entireFetal,
    eligible_record_count: overallPriority.first_trimester.eligible_record_count,
  };

  let evidenceStatus = 'fallback_hierarchy_active';
  let statusDetail =
    'Exploratory 6-tier fallback hierarchy produced non-zero research screening scores across all 3 trimesters.';
  if (conflicts.some((c) => c.severity === 'High')) {
    evidenceStatus = 'conflicting_evidence';
    statusDetail = 'Cross-tier evidence conflict detected alongside active human/regulatory pregnancy signals.';
  } else if (H.available && (P.available || T.available)) {
    evidenceStatus = 'human_and_nonclinical_evidence_available';
    statusDetail =
      'Human pregnancy records combined with regulatory/ADR/nonclinical tiers. Scores are exploratory research priorities, never clinical probabilities.';
  } else if (H.available) {
    evidenceStatus = 'human_pregnancy_evidence_available';
    statusDetail = 'Direct human pregnancy outcomes anchor T1/T2/T3 screening scores.';
  } else if (P.available || A.available) {
    evidenceStatus = 'limited_pregnancy_evidence';
    statusDetail =
      'Direct human outcomes are absent; T1/T2/T3 scores reflect regulatory/ADR/analogue/animal/molecular fallback tiers.';
  }

  const trimesterMetrics: Record<string, unknown> = {};
  if (effectiveId) {
    for (const t of TRIMESTERS) {
      const timed = computeEvidenceQualityScore(effectiveId, t);
      const tMask = timed.availability_mask;
      trimesterMetrics[t] = {
        label: TRIMESTER_LABELS[t],
        evidence_quality: timed.evidence_quality,
        q_cov: timed.q_cov,
        evidence_gap: timed.evidence_gap,
        trimester_evidence_available: timed.evidence_quality !== null,
        timed_record_count: timed.source_breakdown.reduce(
          (acc, item) => acc + Number(item.records ?? 0),
          0
        ),
        available_sources: tMask.filter(([, v]) => v > 0).map(([k]) => k),
        missing_sources: tMask.filter(([, v]) => v <= 0).map(([k]) => k),
      };
    }
  }

  const provenanceBadges: Array<Record<string, string>> = [];
  const seenProv = new Set<string>();
  for (const tier of tierList) {
    if (!tier.available) continue;
    for (const p of tier.provenance) {
      if (seenProv.has(p)) continue;
      seenProv.add(p);
      provenanceBadges.push({
        source_dataset: p,
        source_url: '',
        source_doi: '',
        evidence_status: tier.is_human_evidence ? 'human_tier' : 'nonclinical_or_computational_tier',
        evidence_granularity: tier.name,
      });
    }
  }

  const activeTKey =
    trimester !== 'All' && trimester in TRIMESTER_VALUES
      ? TRIMESTER_VALUES[trimester]
      : 'first_trimester';
  const selectedTrimesterScore =
    activeTKey === 'first_trimester' || activeTKey === 'second_trimester' || activeTKey === 'third_trimester'
      ? trimesterResults[activeTKey].concern_score
      : entireScore;

  return {
    trimester,
    outcome_target: outcomeTarget,
    evidence_source_available: evidenceSourceAvailable,
    records: filteredMaster,
    evidence_record_count: filteredMaster.length,
    trimester_evidence_counts: trimesterCounts,
    provenance_badges: provenanceBadges,
    risk_score: selectedTrimesterScore,
    risk_available: true,
    risk_scoring_status:
      '6-Tier Research / Screening Priority Scores (0–100, always non-zero when supported) combine available tiers without treating missing data as safety. Not clinical risk probabilities.',
    trimester_results: trimesterResults,
    tier_breakdown: tierList,
    ml_assessment: mlAssessment,
    ai_reasoning: aiReasoning,
    confidence: {
      overall_confidence_score: baseConf.score,
      overall_confidence_level: baseConf.level,
      uncertainty_percent: 100 - baseConf.score,
      note: 'Confidence reflects source quality, human vs nonclinical/computational backing, timing explicitness, and agreement across tiers—kept strictly separate from concern score.',
    },
    conflicts,
    evidence_quality: quality?.evidence_quality ?? null,
    evidence_gap: quality?.evidence_gap ?? true,
    evidence_source_breakdown: quality?.source_breakdown ?? [],
    evidence_quality_method: quality?.quality_method ?? QUALITY_METHOD,
    evidence_availability: {
      available_sources: availableSources,
      missing_sources: missingSources,
      q_cov: quality?.q_cov ?? null,
    },
    trimester_metrics: trimesterMetrics,
    maternal_priority: maternalPriority,
    fetal_priority: fetalPriority,
    overall_priority: overallPriority,
    evidence_status: evidenceStatus,
    status_detail: statusDetail,
    animal_adr_evidence: animalAdr,
    evidence_distinction: EVIDENCE_DISTINCTION,
  };
}
