import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { parse } from 'csv-parse/sync';

export type CsvRow = Record<string, string>;

export interface SiderTermRecord {
  term: string;
  meddra_id: string;
  concept_type: string;
  source_dataset: string;
}

export interface TargetInteractionRecord {
  target_name: string;
  target_class: string;
  gene: string;
  action_type: string;
  organism: string;
  source: string;
}

export interface ScoringConfig {
  version: string;
  normalization_factor: number;
  overall_weights: { maternal: number; fetal: number };
  endpoint_relevance: Record<string, Record<string, number>>;
  source_reliability: Record<string, number>;
  timing_relevance: Record<string, number>;
  volume: { reference: number; cap: number };
  outcome_groups: { fetal: string[]; maternal: string[] };
  nonclinical_sources: string[];
  status_thresholds: { evidence_gap_q_cov: number; min_sources_for_human_pregnancy: number };
}

export interface DataStore {
  mapping: CsvRow[];
  pubchem: CsvRow[];
  adr: CsvRow[];
  faersAdr: CsvRow[];
  pregnancy: CsvRow[];
  pregnancyFeatures: CsvRow[];
  toxicity: CsvRow[];
  toxicityFlags: CsvRow[];
  faers: CsvRow[];
  availability: CsvRow[];
  classifications: CsvRow[];
  master: CsvRow[];
  animalAdr: CsvRow[];
  siderByDrugId: Map<string, SiderTermRecord[]>;
  targetsByDrugName: Map<string, TargetInteractionRecord[]>;
  feasibility: Record<string, unknown>;
  scoringConfig: ScoringConfig;
  metrics: Record<string, unknown>;
  metadata: Record<string, unknown>;
  reproductiveToxMetrics: Record<string, unknown>;
  filesPresent: {
    master: boolean;
    adr: boolean;
    sider: boolean;
    faersAdr: boolean;
    animalAdr: boolean;
    targets: boolean;
    labels: boolean;
    checkpoint: boolean;
  };
}

const ROOT = process.cwd();
const DATA_DIR = path.join(ROOT, 'data');
const DATASETS_DIR = path.join(ROOT, 'datasets');

function readCsvOptional(filePath: string): { rows: CsvRow[]; present: boolean } {
  if (!fs.existsSync(filePath)) {
    return { rows: [], present: false };
  }
  const content = fs.readFileSync(filePath, 'utf-8');
  if (!content.trim()) {
    return { rows: [], present: true };
  }
  const rows = parse(content, {
    columns: true,
    skip_empty_lines: true,
    relax_column_count: true,
    trim: false,
  }) as CsvRow[];
  return { rows, present: true };
}

function readJsonOptional<T>(filePath: string, fallback: T): T {
  if (!fs.existsSync(filePath)) {
    return fallback;
  }
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8')) as T;
  } catch {
    return fallback;
  }
}

function loadSiderIndex(mapping: CsvRow[]): {
  map: Map<string, SiderTermRecord[]>;
  present: boolean;
} {
  const gzPath = path.join(DATASETS_DIR, 'meddra_all_se.tsv.gz');
  const map = new Map<string, SiderTermRecord[]>();
  if (!fs.existsSync(gzPath)) {
    return { map, present: false };
  }

  const knownCids = new Set<string>();
  for (const row of mapping) {
    const id = (row.drug_id ?? '').trim();
    if (id.startsWith('CID')) knownCids.add(id);
  }

  const text = zlib.gunzipSync(fs.readFileSync(gzPath)).toString('utf-8');
  const lines = text.split('\n');
  const seenPerDrug = new Map<string, Set<string>>();

  for (const line of lines) {
    if (!line) continue;
    const parts = line.split('\t');
    if (parts.length < 6) continue;
    const flatRaw = parts[0] ?? '';
    const stereoId = parts[1] ?? '';
    const flatId = flatRaw.startsWith('CID')
      ? `CID${flatRaw.slice(4).padStart(9, '0')}`
      : '';

    const matchedIds: string[] = [];
    if (knownCids.has(stereoId)) matchedIds.push(stereoId);
    if (flatId && flatId !== stereoId && knownCids.has(flatId)) matchedIds.push(flatId);
    if (matchedIds.length === 0) continue;

    const conceptType = (parts[3] ?? '').trim();
    const meddraId = (parts[4] ?? parts[2] ?? '').trim();
    const term = (parts[5] ?? '').trim();
    if (!term) continue;

    for (const did of matchedIds) {
      let seen = seenPerDrug.get(did);
      if (!seen) {
        seen = new Set<string>();
        seenPerDrug.set(did, seen);
      }
      const sig = `${term.toLowerCase()}|${meddraId}|${conceptType}`;
      if (seen.has(sig)) continue;
      seen.add(sig);

      let list = map.get(did);
      if (!list) {
        list = [];
        map.set(did, list);
      }
      list.push({
        term,
        meddra_id: meddraId,
        concept_type: conceptType,
        source_dataset: 'SIDER meddra_all_se',
      });
    }
  }

  return { map, present: true };
}

function loadTargetInteractions(): {
  map: Map<string, TargetInteractionRecord[]>;
  present: boolean;
} {
  const gzPath = path.join(DATASETS_DIR, 'drug.target.interaction.tsv.gz');
  const map = new Map<string, TargetInteractionRecord[]>();
  if (!fs.existsSync(gzPath)) {
    return { map, present: false };
  }

  const text = zlib.gunzipSync(fs.readFileSync(gzPath)).toString('utf-8');
  const lines = text.split('\n');
  const stripQuotes = (s: string) => s.replace(/^"|"$/g, '').trim();

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (!line) continue;
    const cols = line.split('\t');
    if (cols.length < 20) continue;
    const drugName = stripQuotes(cols[0]).toLowerCase();
    const targetName = stripQuotes(cols[2]);
    if (!drugName || !targetName) continue;
    const targetClass = stripQuotes(cols[3]);
    const gene = stripQuotes(cols[5]);
    const actSource = stripQuotes(cols[11]) || stripQuotes(cols[14]) || 'DrugCentral/ChEMBL';
    const actionType = stripQuotes(cols[17]);
    const organism = stripQuotes(cols[19]);

    let list = map.get(drugName);
    if (!list) {
      list = [];
      map.set(drugName, list);
    }
    if (
      !list.some(
        (item) => item.target_name === targetName && item.gene === gene
      )
    ) {
      list.push({
        target_name: targetName,
        target_class: targetClass,
        gene,
        action_type: actionType,
        organism,
        source: actSource,
      });
    }
  }

  return { map, present: true };
}

const DEFAULT_SCORING_CONFIG: ScoringConfig = {
  version: '1.0',
  normalization_factor: 20.0,
  overall_weights: { maternal: 0.5, fetal: 0.5 },
  endpoint_relevance: {},
  source_reliability: {},
  timing_relevance: {},
  volume: { reference: 100.0, cap: 1.0 },
  outcome_groups: { fetal: [], maternal: [] },
  nonclinical_sources: ['reprotox_kg', 'epa_toxrefdb'],
  status_thresholds: { evidence_gap_q_cov: 0.3, min_sources_for_human_pregnancy: 2 },
};

let cachedStore: DataStore | null = null;

export function getDataStore(): DataStore {
  if (cachedStore) {
    return cachedStore;
  }

  const mapping = readCsvOptional(path.join(DATA_DIR, 'drug_mapping.csv'));
  const pubchem = readCsvOptional(path.join(DATA_DIR, 'real_pubchem.csv'));
  const adr = readCsvOptional(path.join(DATA_DIR, 'real_adr_evidence.csv'));
  const faersAdr = readCsvOptional(path.join(DATA_DIR, 'real_faers_adr.csv'));
  const pregnancy = readCsvOptional(path.join(DATA_DIR, 'real_pregnancy_outcomes.csv'));
  const pregnancyFeatures = readCsvOptional(path.join(DATA_DIR, 'real_pregnancy_features.csv'));
  const toxicity = readCsvOptional(path.join(DATA_DIR, 'real_toxicity_evidence.csv'));
  const toxicityFlags = readCsvOptional(path.join(DATA_DIR, 'real_toxicity.csv'));
  const faers = readCsvOptional(path.join(DATA_DIR, 'real_faers_summary.csv'));
  const availability = readCsvOptional(path.join(DATA_DIR, 'source_availability.csv'));
  const classifications = readCsvOptional(path.join(DATA_DIR, 'pregnancy_classifications.csv'));
  const master = readCsvOptional(path.join(DATA_DIR, 'standardized_master_evidence.csv'));
  const animalAdr = readCsvOptional(path.join(DATA_DIR, 'animal_adr_features.csv'));

  const sider = loadSiderIndex(mapping.rows);
  const targets = loadTargetInteractions();

  let feasibility = readJsonOptional<Record<string, unknown>>(
    path.join(DATA_DIR, 'dataset_integration_feasibility.json'),
    {}
  );
  if (Object.keys(feasibility).length === 0) {
    feasibility = readJsonOptional<Record<string, unknown>>(
      path.join(DATA_DIR, 'dataset_feasibility.json'),
      {}
    );
  }

  const scoringConfig = readJsonOptional<ScoringConfig>(
    path.join(DATA_DIR, 'scoring_config.json'),
    DEFAULT_SCORING_CONFIG
  );

  const metrics = readJsonOptional<Record<string, unknown>>(
    path.join(DATA_DIR, 'training_metrics.json'),
    {}
  );
  const metadata = readJsonOptional<Record<string, unknown>>(
    path.join(DATA_DIR, 'model_metadata.json'),
    {}
  );
  const reproductiveToxMetrics = readJsonOptional<Record<string, unknown>>(
    path.join(DATA_DIR, 'reproductive_tox_training_metrics.json'),
    {}
  );

  cachedStore = {
    mapping: mapping.rows,
    pubchem: pubchem.rows,
    adr: adr.rows,
    faersAdr: faersAdr.rows,
    pregnancy: pregnancy.rows,
    pregnancyFeatures: pregnancyFeatures.rows,
    toxicity: toxicity.rows,
    toxicityFlags: toxicityFlags.rows,
    faers: faers.rows,
    availability: availability.rows,
    classifications: classifications.rows,
    master: master.rows,
    animalAdr: animalAdr.rows,
    siderByDrugId: sider.map,
    targetsByDrugName: targets.map,
    feasibility,
    scoringConfig,
    metrics,
    metadata,
    reproductiveToxMetrics,
    filesPresent: {
      master: master.present,
      adr: adr.present,
      sider: sider.present,
      faersAdr: faersAdr.present,
      animalAdr: animalAdr.present,
      targets: targets.present,
      labels: fs.existsSync(path.join(DATA_DIR, 'pregnancy_labels.csv')),
      checkpoint: false,
    },
  };

  return cachedStore;
}

export function getMedicationNames(): string[] {
  const store = getDataStore();
  const names = new Set<string>();
  for (const row of store.mapping) {
    const name = (row.original_name ?? '').trim();
    if (name) {
      names.add(name);
    }
  }
  return Array.from(names).sort((a, b) =>
    a.toLowerCase().localeCompare(b.toLowerCase())
  );
}
