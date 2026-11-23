import { analyseMolecule, structuralAnalogueEvidence } from './chemistry.js';
import { CsvRow, getDataStore } from './dataLoader.js';

export const OUTCOME_CATEGORIES: Record<string, string> = {
  fetal_death: 'fetal',
  stillbirth: 'fetal',
  miscarriage: 'fetal',
  congenital_anomaly: 'fetal',
  fetal_growth_restriction: 'fetal',
  preterm_birth: 'neonatal',
  maternal_adverse_reaction: 'maternal',
};

const SALT_TOKENS = new Set([
  'hydrochloride',
  'hcl',
  'sodium',
  'potassium',
  'acetate',
  'tartrate',
  'succinate',
  'sulfate',
  'sulphate',
  'maleate',
  'mesylate',
  'phosphate',
  'calcium',
  'monohydrate',
  'dihydrate',
  'trihydrate',
  'citrate',
  'fumarate',
  'bromide',
  'chloride',
  'nitrate',
  'besylate',
  'tosylate',
  'furoate',
  'propionate',
  'valerate',
  'dipropionate',
  'hemisuccinate',
  'hemihydrate',
  'hydrate',
  'systemic',
  'topical',
]);

const PREGNANCY_ADR_KEYWORDS = [
  'abortion',
  'miscarriage',
  'stillbirth',
  'preterm',
  'premature',
  'congenital',
  'malformation',
  'teratogen',
  'anomaly',
  'growth',
  'placenta',
  'placental',
  'fetal',
  'foetal',
  'neonatal',
  'embryo',
  'pregnancy',
  'gestation',
  'intrauterine',
  'spina bifida',
  'cleft',
  'birth defect',
  'oligohydramnios',
];

export function normalise(value: unknown): string {
  if (value === null || value === undefined) return '';
  return String(value).toLowerCase().trim().replace(/\s+/g, ' ');
}

function stripSalts(name: string): string {
  const cleaned = name
    .toLowerCase()
    .replace(/[()[\]]/g, ' ')
    .replace(/[^a-z0-9/+\s-]+/g, ' ')
    .trim();
  const tokens = cleaned.split(/\s+/).filter((t) => t && !SALT_TOKENS.has(t));
  return tokens.join(' ').trim();
}

function cleanStr(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  const s = String(value).trim();
  return s === '' || s.toLowerCase() === 'nan' ? null : s;
}

function parseCid(value: unknown): number | null {
  const s = cleanStr(value);
  if (!s) return null;
  const num = Number(s);
  return Number.isFinite(num) ? Math.trunc(num) : null;
}

function parseNum(value: unknown): number | null {
  const s = cleanStr(value);
  if (!s) return null;
  const num = Number(s);
  return Number.isFinite(num) ? num : null;
}

export function pregnancyOutcomeCategory(outcome: string | null, group = ''): string {
  const key = normalise(outcome).replace(/ /g, '_');
  if (key in OUTCOME_CATEGORIES) {
    return OUTCOME_CATEGORIES[key];
  }
  const text = `${normalise(outcome)} ${normalise(group)}`;
  if (text.includes('maternal')) return 'maternal';
  if (text.includes('neonatal') || text.includes('preterm') || text.includes('prematur')) {
    return 'neonatal';
  }
  return 'fetal';
}

function drugRows(rows: CsvRow[], drugId: string | null, nameKey?: string | null): CsvRow[] {
  if (rows.length === 0) return [];
  if (drugId) {
    const byId = rows.filter((r) => String(r.drug_id ?? '') === String(drugId));
    if (byId.length > 0) return byId;
  }
  if (nameKey) {
    const nk = normalise(nameKey);
    if (nk) {
      return rows.filter((r) => normalise(r.name_key) === nk);
    }
  }
  return [];
}

function mappingMatches(mapping: CsvRow[], query: string): CsvRow[] {
  const key = normalise(query);
  if (!key) return [];
  let numericKey: number | null = null;
  const maybeNum = Number(key);
  if (Number.isFinite(maybeNum) && key !== '') {
    numericKey = Math.trunc(maybeNum);
  }
  let cidDigits: number | null = null;
  if (key.startsWith('cid')) {
    const trimmed = key.slice(3).replace(/^0+/, '');
    if (/^\d+$/.test(trimmed)) {
      cidDigits = Number.parseInt(trimmed, 10);
    }
  }

  const exact = mapping.filter((row) => {
    if (normalise(row.original_name) === key) return true;
    if (normalise(row.normalized_name) === key) return true;
    if (normalise(row.name_key) === key) return true;
    if (normalise(row.drug_id) === key) return true;
    if (normalise(row.inchikey) === key) return true;
    if (normalise(row.canonical_smiles) === key) return true;
    const rowCid = parseCid(row.pubchem_cid);
    if (rowCid !== null) {
      if (numericKey !== null && rowCid === numericKey) return true;
      if (cidDigits !== null && rowCid === cidDigits) return true;
    }
    return false;
  });
  if (exact.length > 0) return exact;

  // Salt-stripped and ingredient-level fallback (avoids unrelated false positives)
  const strippedQuery = stripSalts(key);
  if (strippedQuery && strippedQuery.length >= 4) {
    const saltMatches = mapping.filter((row) => {
      const strippedOrig = stripSalts(row.original_name ?? '');
      const strippedKey = stripSalts(row.name_key ?? '');
      return strippedOrig === strippedQuery || strippedKey === strippedQuery;
    });
    if (saltMatches.length > 0) return saltMatches;

    // Parenthetical synonym or combination ingredient match
    const ingredientMatches = mapping.filter((row) => {
      const raw = normalise(row.original_name);
      if (raw.includes(`(${strippedQuery})`)) return true;
      const ingredients = raw.split(/[;/+]/).map((p) => stripSalts(p));
      return ingredients.some((ing) => ing === strippedQuery);
    });
    if (ingredientMatches.length > 0) return ingredientMatches;
  }

  return [];
}

function evidenceIndexMatches(query: string) {
  const store = getDataStore();
  const key = normalise(query);
  if (!key) return [];
  const rows = store.pregnancy.filter(
    (r) => normalise(r.drug_name) === key || normalise(r.name_key) === key
  );
  const seen = new Set<string>();
  const candidates: Array<{
    drug_id: string;
    name: string;
    resolution_status: string;
    resolution_source: string;
    pubchem_cid: number | null;
    inchikey: string | null;
    smiles: string | null;
    name_key: string;
    warnings: string[];
  }> = [];

  for (const row of rows) {
    const name = cleanStr(row.drug_name) || query;
    const evidenceKey = normalise(cleanStr(row.name_key) || name);
    const rawId = cleanStr(row.drug_id);
    const drugId = rawId ?? `NAME:${encodeURIComponent(evidenceKey)}`;
    if (seen.has(drugId)) continue;
    seen.add(drugId);
    candidates.push({
      drug_id: drugId,
      name,
      resolution_status: 'name_keyed_evidence_only',
      resolution_source: 'data/real_pregnancy_outcomes.csv',
      pubchem_cid: null,
      inchikey: null,
      smiles: null,
      name_key: evidenceKey,
      warnings: [
        'Resolved from the human pregnancy evidence index; this name has no PubChem structure in the local mapping, so molecular and nonclinical joins are unavailable.',
      ],
    });
  }
  return candidates;
}

function identityFor(drugId: string | null, mappingRow: CsvRow | null) {
  const store = getDataStore();
  const pubchemRow = drugId
    ? store.pubchem.find((r) => String(r.drug_id ?? '') === String(drugId)) ?? null
    : null;
  const row = mappingRow ?? pubchemRow;
  if (!row) {
    return {
      drug_id: drugId,
      name: null,
      active_ingredient: null,
      pubchem_cid: null,
      inchikey: null,
      smiles: null,
      molecular_formula: null,
      molecular_weight: null,
      resolution_status: null,
      resolution_source: null,
    };
  }
  const pick = (col: string) =>
    cleanStr(row[col]) ?? (pubchemRow ? cleanStr(pubchemRow[col]) : null);
  const name = pick('original_name') || pick('title') || drugId;
  return {
    drug_id: drugId,
    name,
    active_ingredient: name ? stripSalts(name) || name : null,
    pubchem_cid: parseCid(pick('pubchem_cid')),
    inchikey: pick('inchikey'),
    smiles: pick('canonical_smiles') || pick('smiles'),
    molecular_formula: pick('molecular_formula'),
    molecular_weight: parseNum(pick('molecular_weight')),
    resolution_status: pick('resolution_status'),
    resolution_source: pick('resolution_source'),
  };
}

export function resolveDrug(query: string) {
  const store = getDataStore();
  const trimmed = String(query ?? '').trim();
  const base = {
    query: trimmed,
    matched: false,
    ambiguous: false,
    drug_id: null as string | null,
    drug_name: null as string | null,
    candidates: [] as Array<Record<string, unknown>>,
    identity: identityFor(null, null),
    warnings: [] as string[],
  };

  if (!trimmed) {
    base.warnings.push('Enter a drug name or a supported identifier.');
    return base;
  }
  if (store.mapping.length === 0) {
    base.warnings.push('The drug mapping is unavailable.');
    return base;
  }

  const matches = mappingMatches(store.mapping, trimmed);
  if (matches.length === 0) {
    const fallback = evidenceIndexMatches(trimmed);
    if (fallback.length > 0) {
      const distinct = new Set(fallback.map((c) => c.drug_id));
      base.candidates = fallback;
      base.ambiguous = distinct.size > 1;
      if (base.ambiguous) {
        base.warnings.push(`'${trimmed}' matches ${fallback.length} evidence-index entries.`);
        return base;
      }
      const chosen = fallback[0];
      return {
        ...base,
        matched: true,
        drug_id: chosen.drug_id,
        drug_name: chosen.name,
        identity: {
          drug_id: chosen.drug_id,
          name: chosen.name,
          active_ingredient: stripSalts(chosen.name) || chosen.name,
          pubchem_cid: chosen.pubchem_cid,
          inchikey: chosen.inchikey,
          smiles: chosen.smiles,
          molecular_formula: null,
          molecular_weight: null,
          resolution_status: chosen.resolution_status,
          resolution_source: chosen.resolution_source,
        },
        warnings: [...chosen.warnings],
      };
    }
    base.warnings.push(
      `No local mapping entry matches '${trimmed}'. The drug may not be covered by the acquired sources.`
    );
    return base;
  }

  const candidates: Array<{
    drug_id: string;
    name: string;
    resolution_status: string | null;
    pubchem_cid: number | null;
    inchikey: string | null;
  }> = [];
  const seen = new Set<string>();
  for (const row of matches) {
    const did = cleanStr(row.drug_id) ?? '';
    const name = cleanStr(row.original_name) ?? did;
    const sig = `${did}|${normalise(name)}`;
    if (seen.has(sig)) continue;
    seen.add(sig);
    candidates.push({
      drug_id: did,
      name,
      resolution_status: cleanStr(row.resolution_status),
      pubchem_cid: parseCid(row.pubchem_cid),
      inchikey: cleanStr(row.inchikey),
    });
  }

  const distinctIds = new Set(candidates.map((c) => c.drug_id).filter(Boolean));
  base.candidates = candidates;
  base.ambiguous = distinctIds.size > 1;
  if (base.ambiguous) {
    base.warnings.push(
      `'${trimmed}' matches ${distinctIds.size} distinct drug identifiers. Select one explicitly before interpreting evidence.`
    );
    return base;
  }

  const chosen = candidates[0];
  const mappingRow =
    store.mapping.find((r) => String(r.drug_id ?? '') === chosen.drug_id) ?? null;
  const identity = identityFor(chosen.drug_id, mappingRow);
  const warnings: string[] = [];
  const status = normalise(identity.resolution_status);
  if (status && status !== 'resolved') {
    warnings.push(
      `Identifier resolution status is '${identity.resolution_status}'. No PubChem structure is available, so molecular analysis uses fallback evidence.`
    );
  }
  if (chosen.drug_id.startsWith('NAME:')) {
    warnings.push(
      'This drug is keyed by name (NAME:...) rather than a PubChem CID. Evidence recorded under a CID key is not joined to it automatically.'
    );
  }

  return {
    ...base,
    matched: true,
    drug_id: chosen.drug_id,
    drug_name: identity.name || chosen.name,
    identity,
    warnings,
  };
}

function adrEvidence(drugId: string | null, nameKey: string | null) {
  const store = getDataStore();
  const csvRows = drugRows(store.adr, drugId, nameKey);
  const siderRows = drugId ? store.siderByDrugId.get(drugId) ?? [] : [];
  const faersAdrRow = drugId
    ? store.faersAdr.find((r) => String(r.drug_id ?? '') === String(drugId))
    : undefined;

  const byTerm = new Map<
    string,
    { meddra_id: string | null; records: number; sources: Set<string>; pregnancyRelated: boolean }
  >();
  const allSources = new Set<string>();

  for (const row of csvRows) {
    const term = cleanStr(row.adverse_event_term);
    if (!term) continue;
    const ds = cleanStr(row.source_dataset) ?? 'OnSIDES / local ADR';
    allSources.add(ds);
    const lower = term.toLowerCase();
    const isPreg = PREGNANCY_ADR_KEYWORDS.some((kw) => lower.includes(kw));
    const existing = byTerm.get(term) ?? {
      meddra_id: cleanStr(row.meddra_id),
      records: 0,
      sources: new Set<string>(),
      pregnancyRelated: isPreg,
    };
    existing.records += 1;
    existing.sources.add(ds);
    byTerm.set(term, existing);
  }

  for (const row of siderRows) {
    const term = row.term;
    if (!term) continue;
    allSources.add(row.source_dataset);
    const lower = term.toLowerCase();
    const isPreg = PREGNANCY_ADR_KEYWORDS.some((kw) => lower.includes(kw));
    const existing = byTerm.get(term) ?? {
      meddra_id: row.meddra_id || null,
      records: 0,
      sources: new Set<string>(),
      pregnancyRelated: isPreg,
    };
    existing.records += 1;
    existing.sources.add(row.source_dataset);
    byTerm.set(term, existing);
  }

  let faersVectorCount = 0;
  let faersQuality = 0;
  if (faersAdrRow && faersAdrRow.has_adr_data === '1') {
    allSources.add('FAERS ADR Feature Matrix');
    faersQuality = parseNum(faersAdrRow.quality_adr) ?? 0;
    for (let i = 0; i < 20; i++) {
      if (Number(faersAdrRow[`ADR_${i}`] ?? 0) > 0) {
        faersVectorCount++;
      }
    }
  }

  const allTerms = Array.from(byTerm.entries())
    .map(([term, info]) => ({
      term,
      meddra_id: info.meddra_id,
      records: info.records,
      pregnancy_related: info.pregnancyRelated,
      sources: Array.from(info.sources).sort().join(', '),
    }))
    .sort(
      (a, b) =>
        Number(b.pregnancy_related) - Number(a.pregnancy_related) ||
        b.records - a.records ||
        a.term.localeCompare(b.term)
    );

  const pregnancyTerms = allTerms.filter((t) => t.pregnancy_related);
  const available = allTerms.length > 0 || faersVectorCount > 0;

  return {
    available,
    term_count: allTerms.length,
    pregnancy_related_term_count: pregnancyTerms.length,
    pregnancy_related_terms: pregnancyTerms.slice(0, 30),
    record_count: csvRows.length + siderRows.length,
    faers_adr_vector_hits: faersVectorCount,
    faers_adr_quality: faersQuality,
    source_datasets: Array.from(allSources).sort(),
    terms: allTerms.slice(0, 50),
    reason: available ? null : 'No general ADR (SIDER/FAERS term) records for this drug.',
  };
}

function reproductiveToxicityEvidence(drugId: string | null) {
  const store = getDataStore();
  const rows = drugRows(store.toxicity, drugId);
  const flagRow = drugId
    ? store.toxicityFlags.find((r) => String(r.drug_id ?? '') === String(drugId))
    : undefined;

  const studyFlags: string[] = [];
  if (flagRow && flagRow.has_tox_data === '1') {
    for (const key of ['Study_DEV', 'Study_CHR', 'Study_SUB', 'Study_MGR', 'Study_SAC']) {
      if (Number(flagRow[key] ?? 0) > 0) {
        studyFlags.push(key.replace('Study_', ''));
      }
    }
  }

  if (rows.length === 0 && studyFlags.length === 0) {
    return {
      available: false,
      record_count: 0,
      species: [] as string[],
      study_types: [] as string[],
      study_flags: [] as string[],
      records: [] as Array<Record<string, unknown>>,
      sources: [] as string[],
      reason: 'No nonclinical toxicity study records for this drug.',
      is_human_evidence: false,
    };
  }

  const species = Array.from(
    new Set(rows.map((r) => cleanStr(r.species)).filter(Boolean) as string[])
  ).sort();
  const studyTypes = Array.from(
    new Set([
      ...(rows.map((r) => cleanStr(r.study_type)).filter(Boolean) as string[]),
      ...studyFlags,
    ])
  ).sort();
  const sources = Array.from(
    new Set([
      ...(rows.map((r) => cleanStr(r.source_dataset)).filter(Boolean) as string[]),
      ...(studyFlags.length > 0 ? ['EPA ToxRefDB summary flags'] : []),
    ])
  ).sort();
  const displayCols = [
    'species',
    'strain',
    'study_type',
    'evidence_type',
    'study_year',
    'admin_route',
    'admin_method',
    'dose_start',
    'dose_start_unit',
    'dose_end',
    'dose_end_unit',
    'study_citation',
    'source_dataset',
  ];
  const records = rows.map((r) => {
    const out: Record<string, unknown> = {};
    for (const col of displayCols) {
      if (col in r) {
        out[col] = cleanStr(r[col]);
      }
    }
    return out;
  });
  return {
    available: true,
    record_count: Math.max(rows.length, studyFlags.length),
    species,
    study_types: studyTypes,
    study_flags: studyFlags,
    records,
    sources,
    reason: null,
    is_human_evidence: false,
  };
}

function pregnancyEvidence(drugId: string | null, nameKey: string | null) {
  const store = getDataStore();
  const rows = drugRows(store.pregnancy, drugId, nameKey);
  const featRow = drugId
    ? store.pregnancyFeatures.find((r) => String(r.drug_id ?? '') === String(drugId))
    : undefined;

  if (rows.length === 0 && !featRow) {
    return {
      available: false,
      record_count: 0,
      reported_count_total: null as number | null,
      categories: { maternal: [], fetal: [], neonatal: [] } as Record<
        string,
        Array<Record<string, unknown>>
      >,
      category_counts: { maternal: 0, fetal: 0, neonatal: 0 },
      outcome_totals: [] as Array<{
        category: string;
        outcome: string;
        records: number;
        reported_count: number | null;
      }>,
      sources: [] as string[],
      evidence_types: [] as string[],
      trimesters: [] as string[],
      unknown_trimester: true,
      records: [] as Array<Record<string, unknown>>,
      reason: 'No human pregnancy-outcome evidence records for this drug.',
      is_human_evidence: true,
    };
  }

  const categories: Record<string, Array<Record<string, unknown>>> = {
    maternal: [],
    fetal: [],
    neonatal: [],
  };
  const totalsMap = new Map<
    string,
    { category: string; outcome: string; records: number; reported_count: number | null }
  >();
  const sourceSet = new Set<string>();
  const evidenceTypeSet = new Set<string>();
  let totalReported: number | null = null;
  const formattedRecords: Array<Record<string, unknown>> = [];

  for (const r of rows) {
    const outcome = cleanStr(r.outcome) ?? '';
    const group = cleanStr(r.outcome_group) ?? '';
    const category = pregnancyOutcomeCategory(outcome, group);
    const count = parseNum(r.reported_count);
    if (count !== null) {
      totalReported = (totalReported ?? 0) + count;
    }
    const sf = cleanStr(r.source_file);
    if (sf) sourceSet.add(sf);
    const et = cleanStr(r.evidence_type);
    if (et) evidenceTypeSet.add(et);

    const recordObj = {
      category,
      outcome_group: cleanStr(r.outcome_group),
      outcome: cleanStr(r.outcome),
      reported_term: cleanStr(r.reported_term),
      reported_count: count,
      source_file: sf,
      evidence_type: et,
    };
    formattedRecords.push(recordObj);
    if (category in categories) {
      categories[category].push(recordObj);
    }

    const tKey = `${category}::${outcome}`;
    const existing = totalsMap.get(tKey) ?? {
      category,
      outcome,
      records: 0,
      reported_count: null,
    };
    existing.records += 1;
    if (count !== null) {
      existing.reported_count = (existing.reported_count ?? 0) + count;
    }
    totalsMap.set(tKey, existing);
  }

  // Fallback to real_pregnancy_features.csv if detailed rows are absent
  if (rows.length === 0 && featRow) {
    const featureOutcomes: Array<[string, string, string]> = [
      ['reports_fetal_death', 'fetal_death', 'fetal'],
      ['reports_preterm_birth', 'preterm_birth', 'neonatal'],
      ['reports_fetal_growth_restriction', 'fetal_growth_restriction', 'fetal'],
      ['reports_congenital_anomaly', 'congenital_anomaly', 'fetal'],
    ];
    for (const [col, outcome, category] of featureOutcomes) {
      const cnt = parseNum(featRow[col]);
      if (cnt !== null && cnt > 0) {
        totalReported = (totalReported ?? 0) + cnt;
        sourceSet.add('real_pregnancy_features.csv');
        evidenceTypeSet.add('aggregate spontaneous-report summary');
        const rec = {
          category,
          outcome_group: outcome.replace(/_/g, ' '),
          outcome,
          reported_term: outcome.replace(/_/g, ' '),
          reported_count: cnt,
          source_file: 'real_pregnancy_features.csv',
          evidence_type: 'aggregate spontaneous-report summary',
        };
        formattedRecords.push(rec);
        categories[category].push(rec);
        totalsMap.set(`${category}::${outcome}`, {
          category,
          outcome,
          records: 1,
          reported_count: cnt,
        });
      }
    }
  }

  const masterRows = drugRows(store.master, drugId, nameKey);
  const trimesterSet = new Set<string>();
  for (const mr of masterRows) {
    const t = cleanStr(mr.trimester);
    if (t) trimesterSet.add(t);
  }
  const trimesters = Array.from(trimesterSet).sort();
  const unknownTrimester =
    trimesters.length === 0 || trimesters.every((v) => normalise(v) === 'unknown');

  const outcomeTotals = Array.from(totalsMap.values()).sort(
    (a, b) => a.category.localeCompare(b.category) || a.outcome.localeCompare(b.outcome)
  );

  return {
    available: formattedRecords.length > 0,
    record_count: formattedRecords.length,
    reported_count_total: totalReported,
    categories,
    category_counts: {
      maternal: categories.maternal.length,
      fetal: categories.fetal.length,
      neonatal: categories.neonatal.length,
    },
    outcome_totals: outcomeTotals,
    sources: Array.from(sourceSet).sort(),
    evidence_types: Array.from(evidenceTypeSet).sort(),
    trimesters,
    unknown_trimester: unknownTrimester,
    records: formattedRecords,
    reason: formattedRecords.length > 0 ? null : 'No human pregnancy-outcome evidence records for this drug.',
    is_human_evidence: true,
  };
}

function tgaClassificationEvidence(drugId: string | null, nameKey: string | null) {
  const store = getDataStore();
  const rows = drugRows(store.classifications, drugId, nameKey);
  if (rows.length === 0) {
    return {
      available: false,
      category: null as string | null,
      safety_statement: null as string | null,
      therapeutic_classes: [] as string[],
      records: [] as Array<Record<string, unknown>>,
    };
  }
  const first = rows[0];
  const classes = Array.from(
    new Set(
      rows
        .flatMap((r) => [
          cleanStr(r['Classification 1']),
          cleanStr(r['Classification 2']),
          cleanStr(r['Classification 3']),
        ])
        .filter(Boolean) as string[]
    )
  );
  return {
    available: true,
    category: cleanStr(first.Category),
    safety_statement: cleanStr(first['Safety statement']),
    therapeutic_classes: classes,
    records: rows.map((r) => ({
      name: cleanStr(r.Name),
      category: cleanStr(r.Category),
      safety_statement: cleanStr(r['Safety statement']),
      classification_1: cleanStr(r['Classification 1']),
      classification_2: cleanStr(r['Classification 2']),
      classification_3: cleanStr(r['Classification 3']),
    })),
  };
}

function targetInteractionEvidence(drugName: string | null, nameKey: string | null) {
  const store = getDataStore();
  const candidates = new Set<string>();
  if (drugName) {
    candidates.add(drugName.toLowerCase().trim());
    candidates.add(stripSalts(drugName));
  }
  if (nameKey) {
    candidates.add(nameKey.toLowerCase().trim());
    candidates.add(stripSalts(nameKey));
  }
  for (const key of candidates) {
    if (!key) continue;
    const hits = store.targetsByDrugName.get(key);
    if (hits && hits.length > 0) {
      return {
        available: true,
        count: hits.length,
        targets: hits.slice(0, 25),
      };
    }
  }
  return {
    available: false,
    count: 0,
    targets: [] as Array<Record<string, unknown>>,
  };
}

function faersEvidence(drugId: string | null) {
  const store = getDataStore();
  const rows = drugRows(store.faers, drugId);
  if (rows.length === 0) {
    return {
      available: false,
      report_count: null as number | null,
      outcome_count: null as number | null,
      source_name: null as string | null,
      reason: 'No FAERS summary row is linked.',
    };
  }
  const row = rows[0];
  return {
    available: true,
    report_count: parseCid(row.faers_report_count),
    outcome_count: parseNum(row.faers_outcome_count),
    source_name: cleanStr(row.faers_source_name),
    reason: null,
  };
}

function availabilityFor(drugId: string | null) {
  const store = getDataStore();
  const rows = drugRows(store.availability, drugId);
  if (rows.length === 0) {
    return {
      available: false,
      sources: {} as Record<string, unknown>,
      missing_sources: [] as string[],
    };
  }
  const row = rows[0];
  const sources: Record<string, { status: string | null; metadata_completeness: number | null }> =
    {};
  const missing: string[] = [];
  for (const col of Object.keys(row)) {
    if (col === 'drug_id' || !col.endsWith('_status')) continue;
    const src = col.slice(0, -'_status'.length);
    const status = cleanStr(row[col]);
    const completeness = parseNum(row[`${src}_metadata_completeness`]);
    sources[src] = { status, metadata_completeness: completeness };
    if (!status || status.toLowerCase() !== 'available') {
      missing.push(src);
    }
  }
  return { available: true, sources, missing_sources: missing.sort() };
}

function researchPriorityIndex(profile: {
  matched: boolean;
  ambiguous: boolean;
  pregnancy: { available: boolean; unknown_trimester?: boolean };
  reproductive_toxicity: { available: boolean };
}) {
  if (!profile.matched || profile.ambiguous) {
    return {
      available: false,
      score: null,
      max_score: 3,
      components: [],
      reason: 'Resolve one drug identity before assessing evidence gaps.',
    };
  }
  const domains = [
    {
      domain: 'Drug-specific human pregnancy outcome records',
      covered: Boolean(profile.pregnancy.available),
    },
    {
      domain: 'Explicit gestational timing',
      covered:
        Boolean(profile.pregnancy.available) &&
        !Boolean(profile.pregnancy.unknown_trimester ?? true),
    },
    {
      domain: 'Drug-specific developmental/reproductive toxicity records',
      covered: Boolean(profile.reproductive_toxicity.available),
    },
  ];
  const components = domains.map((d) => ({
    domain: d.domain,
    covered: d.covered,
    gap_points: d.covered ? 0 : 1,
  }));
  return {
    available: true,
    score: components.reduce((acc, c) => acc + c.gap_points, 0),
    max_score: components.length,
    components,
    reason: null,
  };
}

export function buildDrugProfile(query: string) {
  const store = getDataStore();
  const resolution = resolveDrug(query);
  const molecular = analyseMolecule(resolution.identity.smiles);

  if (!resolution.matched || resolution.ambiguous) {
    return {
      ...resolution,
      molecular,
      adr: {
        available: false,
        term_count: 0,
        pregnancy_related_term_count: 0,
        pregnancy_related_terms: [],
        record_count: 0,
        faers_adr_vector_hits: 0,
        faers_adr_quality: 0,
        source_datasets: [] as string[],
        terms: [],
        reason: null as string | null,
      },
      reproductive_toxicity: {
        available: false,
        record_count: 0,
        species: [] as string[],
        study_types: [] as string[],
        study_flags: [] as string[],
        records: [] as Array<Record<string, unknown>>,
        sources: [] as string[],
        reason: null as string | null,
        is_human_evidence: false,
      },
      pregnancy: {
        available: false,
        record_count: 0,
        reported_count_total: null as number | null,
        categories: { maternal: [], fetal: [], neonatal: [] } as Record<
          string,
          Array<Record<string, unknown>>
        >,
        category_counts: { maternal: 0, fetal: 0, neonatal: 0 },
        outcome_totals: [] as Array<{
          category: string;
          outcome: string;
          records: number;
          reported_count: number | null;
        }>,
        sources: [] as string[],
        evidence_types: [] as string[],
        trimesters: [] as string[],
        unknown_trimester: true,
        records: [] as Array<Record<string, unknown>>,
        reason: null as string | null,
        is_human_evidence: true,
      },
      tga_classification: {
        available: false,
        category: null as string | null,
        safety_statement: null as string | null,
        therapeutic_classes: [] as string[],
        records: [] as Array<Record<string, unknown>>,
      },
      targets: {
        available: false,
        count: 0,
        targets: [] as Array<{
          target_name: string;
          target_class: string;
          gene: string;
          action_type: string;
          organism: string;
          source: string;
        }>,
      },
      related_pregnancy_evidence: {
        available: false,
        threshold: 0.2,
        reason: null as string | null,
        similarity_method: undefined as string | undefined,
        candidates: [] as Array<{
          drug_id: string;
          drug_name: string;
          similarity: number;
          source_record_count: number;
          tga_category: string | null;
          toxicity_record_count: number;
          estimated_concern_signal: number;
          outcomes: Array<{ outcome: string; source_rows: number; reported_count: number | null }>;
          sources: string[];
          timing: string;
        }>,
      },
      faers: {
        available: false,
        report_count: null as number | null,
        outcome_count: null as number | null,
        source_name: null as string | null,
        reason: null as string | null,
      },
      availability: {
        available: false,
        sources: {} as Record<string, unknown>,
        missing_sources: [] as string[],
      },
      name_key: null as string | null,
      evidence_counts: {
        adr_terms: 0,
        pregnancy_records: 0,
        nonclinical_toxicity_records: 0,
        faers_reports: null as number | null,
      },
      research_priority: {
        available: false,
        score: null as number | null,
        max_score: 3,
        components: [] as Array<{ domain: string; covered: boolean; gap_points: number }>,
        reason: null as string | null,
      },
    };
  }

  const drugId = resolution.drug_id;
  const mappingRow = store.mapping.find((r) => String(r.drug_id ?? '') === String(drugId));
  let nameKey = mappingRow ? cleanStr(mappingRow.name_key) : null;
  if (!nameKey && drugId && drugId.startsWith('NAME:')) {
    nameKey = normalise(decodeURIComponent(drugId.slice(5)));
  }

  const adr = adrEvidence(drugId, nameKey);
  const reproductiveToxicity = reproductiveToxicityEvidence(drugId);
  const pregnancy = pregnancyEvidence(drugId, nameKey);
  const tgaClassification = tgaClassificationEvidence(drugId, nameKey);
  const targets = targetInteractionEvidence(resolution.drug_name, nameKey);
  const relatedPregnancyEvidence = structuralAnalogueEvidence(
    drugId,
    resolution.identity.smiles
  );
  const faers = faersEvidence(drugId);
  const availability = availabilityFor(drugId);

  const identity = { ...resolution.identity };
  if (!identity.molecular_formula && molecular.available) {
    identity.molecular_formula = molecular.molecular_formula;
  }
  if (identity.molecular_weight === null && molecular.available) {
    identity.molecular_weight = molecular.descriptors.molecular_weight ?? null;
  }

  const warnings = [...resolution.warnings];
  if (!pregnancy.available) {
    warnings.push(
      'Direct human pregnancy-outcome records are absent; trimester fallback hierarchy is active using regulatory/ADR/analogue/nonclinical/molecular evidence.'
    );
  }
  if (!reproductiveToxicity.available) {
    warnings.push('No nonclinical developmental/reproductive toxicity assay records for this drug.');
  }

  return {
    ...resolution,
    identity,
    molecular,
    adr,
    reproductive_toxicity: reproductiveToxicity,
    pregnancy,
    tga_classification: tgaClassification,
    targets,
    related_pregnancy_evidence: relatedPregnancyEvidence,
    faers,
    availability,
    name_key: nameKey,
    evidence_counts: {
      adr_terms: adr.term_count,
      pregnancy_records: pregnancy.record_count,
      nonclinical_toxicity_records: reproductiveToxicity.record_count,
      faers_reports: faers.report_count,
    },
    research_priority: researchPriorityIndex({
      matched: resolution.matched,
      ambiguous: resolution.ambiguous,
      pregnancy,
      reproductive_toxicity: reproductiveToxicity,
    }),
    warnings,
  };
}
