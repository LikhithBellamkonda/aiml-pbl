import OCL from 'openchemlib';
import { getDataStore } from './dataLoader.js';

export interface FormulaPart {
  element: string;
  count: number;
}

export interface MolecularAnalysis {
  available: boolean;
  reason: string | null;
  canonical_smiles?: string;
  molecular_formula: string | null;
  formula_parts: FormulaPart[];
  scaffold?: string | null;
  structure_png: string | null;
  descriptors: Record<string, number>;
  fingerprint: {
    type?: string;
    on_bits?: number;
    density?: number;
  };
  svg: string | null;
  drawing_error: string | null;
}

export function parseFormulaParts(formula: string | null): FormulaPart[] {
  if (!formula) return [];
  const parts: FormulaPart[] = [];
  const regex = /([A-Z][a-z]?)(\d*)/g;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(formula)) !== null) {
    if (!match[1]) continue;
    parts.push({
      element: match[1],
      count: match[2] ? Number.parseInt(match[2], 10) : 1,
    });
  }
  return parts;
}

function countBits(arr: number[]): number {
  let total = 0;
  for (const word of arr) {
    let v = word | 0;
    v = v - ((v >>> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
    total += (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
  }
  return total;
}

function tanimotoFromIndices(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  let intersection = 0;
  let union = 0;
  for (let i = 0; i < len; i++) {
    const x = a[i] | 0;
    const y = b[i] | 0;
    const andVal = x & y;
    const orVal = x | y;
    if (andVal !== 0) {
      let v = andVal;
      v = v - ((v >>> 1) & 0x55555555);
      v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
      intersection += (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
    }
    if (orVal !== 0) {
      let v = orVal;
      v = v - ((v >>> 1) & 0x55555555);
      v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
      union += (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
    }
  }
  return union === 0 ? 0 : intersection / union;
}

export function analyseMolecule(smiles: string | null): MolecularAnalysis {
  if (!smiles || typeof smiles !== 'string' || !smiles.trim()) {
    return {
      available: false,
      reason: 'No SMILES is recorded for this drug.',
      descriptors: {},
      fingerprint: {},
      svg: null,
      structure_png: null,
      drawing_error: null,
      molecular_formula: null,
      formula_parts: [],
    };
  }

  try {
    const mol = OCL.Molecule.fromSmiles(smiles.trim());
    if (!mol || mol.getAllAtoms() === 0) {
      return {
        available: false,
        reason: 'The recorded SMILES could not be parsed.',
        descriptors: {},
        fingerprint: {},
        svg: null,
        structure_png: null,
        drawing_error: null,
        molecular_formula: null,
        formula_parts: [],
      };
    }

    const props = new OCL.MoleculeProperties(mol);
    const mf = mol.getMolecularFormula();
    const formula = mf.formula || null;
    const ringSet = mol.getRingSet();
    const ringCount = ringSet ? ringSet.getSize() : 0;
    const heavyAtoms = mol.getAllAtoms();
    let aromaticAtoms = 0;
    for (let i = 0; i < heavyAtoms; i++) {
      if (mol.isAromaticAtom(i)) aromaticAtoms++;
    }

    const idx = mol.getIndex();
    const onBits = countBits(idx);
    const totalBits = idx.length * 32;

    let svg: string | null = null;
    let drawingError: string | null = null;
    try {
      svg = mol.toSVG(360, 260, undefined, {
        autoCrop: true,
        autoCropMargin: 12,
        suppressChiralText: true,
        suppressCIPParity: true,
        suppressESR: true,
        noImplicitAtomLabelColors: false,
      });
    } catch (err) {
      drawingError = err instanceof Error ? err.message : String(err);
    }

    const round3 = (n: number) => Math.round(n * 1000) / 1000;

    return {
      available: true,
      reason: null,
      canonical_smiles: mol.toSmiles() || smiles.trim(),
      molecular_formula: formula,
      formula_parts: parseFormulaParts(formula),
      scaffold:
        aromaticAtoms > 0
          ? 'Aromatic ring system'
          : ringCount > 0
            ? 'Alicyclic ring system'
            : 'Acyclic',
      structure_png: null,
      descriptors: {
        molecular_weight: round3(mf.relativeWeight),
        clogp: round3(props.logP),
        tpsa: round3(props.polarSurfaceArea),
        h_bond_donors: props.donorCount,
        h_bond_acceptors: props.acceptorCount,
        rotatable_bonds: props.rotatableBondCount,
        ring_count: ringCount,
        heavy_atoms: heavyAtoms,
      },
      fingerprint: {
        type: `OpenChemLib ${totalBits}-bit substructure descriptor`,
        on_bits: onBits,
        density: totalBits > 0 ? Math.round((onBits / totalBits) * 10000) / 10000 : 0,
      },
      svg,
      drawing_error: drawingError,
    };
  } catch (err) {
    return {
      available: false,
      reason: `Could not parse SMILES: ${err instanceof Error ? err.message : String(err)}`,
      descriptors: {},
      fingerprint: {},
      svg: null,
      structure_png: null,
      drawing_error: null,
      molecular_formula: null,
      formula_parts: [],
    };
  }
}

export interface AnalogueCandidate {
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
}

interface IndexedAnalogue extends Omit<AnalogueCandidate, 'similarity'> {
  fingerprint: number[];
}

const TGA_WEIGHTS: Record<string, number> = {
  X: 90,
  D: 78,
  C: 54,
  B3: 46,
  B2: 32,
  B1: 24,
  A: 14,
};

let cachedPool: IndexedAnalogue[] | null = null;

function normalise(value: string | null | undefined): string {
  if (!value) return '';
  return String(value).toLowerCase().trim().replace(/\s+/g, ' ');
}

function getIndexedAnaloguePool(): IndexedAnalogue[] {
  if (cachedPool) return cachedPool;
  const store = getDataStore();

  const pregById = new Map<string, typeof store.pregnancy>();
  const pregByName = new Map<string, typeof store.pregnancy>();
  for (const row of store.pregnancy) {
    const did = (row.drug_id ?? '').trim();
    const nk = normalise(row.name_key);
    if (did) {
      const list = pregById.get(did) ?? [];
      list.push(row);
      pregById.set(did, list);
    } else if (nk) {
      const list = pregByName.get(nk) ?? [];
      list.push(row);
      pregByName.set(nk, list);
    }
  }

  const tgaById = new Map<string, string>();
  for (const row of store.classifications) {
    const did = (row.drug_id ?? '').trim();
    const cat = (row.Category ?? '').trim().toUpperCase();
    if (did && cat && !tgaById.has(did)) {
      tgaById.set(did, cat);
    }
  }

  const toxCountById = new Map<string, number>();
  for (const row of store.toxicity) {
    const did = (row.drug_id ?? '').trim();
    if (did) toxCountById.set(did, (toxCountById.get(did) ?? 0) + 1);
  }
  for (const row of store.toxicityFlags) {
    const did = (row.drug_id ?? '').trim();
    if (did && row.has_tox_data === '1' && !toxCountById.has(did)) {
      toxCountById.set(did, 1);
    }
  }

  const seenIds = new Set<string>();
  const results: IndexedAnalogue[] = [];

  for (const row of store.mapping) {
    const cid = (row.drug_id ?? '').trim();
    const smiles = (row.canonical_smiles ?? '').trim();
    if (!cid || !smiles || seenIds.has(cid)) continue;
    seenIds.add(cid);

    const evidenceRows = [...(pregById.get(cid) ?? [])];
    const nk = normalise(row.name_key);
    if (nk && pregByName.has(nk)) {
      evidenceRows.push(...(pregByName.get(nk) ?? []));
    }
    const tgaCat = tgaById.get(cid) ?? null;
    const toxCnt = toxCountById.get(cid) ?? 0;

    if (evidenceRows.length === 0 && !tgaCat && toxCnt === 0) continue;

    try {
      const mol = OCL.Molecule.fromSmiles(smiles);
      if (!mol || mol.getAllAtoms() === 0) continue;
      const fp = mol.getIndex();

      const outcomeMap = new Map<string, { rows: number; count: number | null }>();
      const sourceSet = new Set<string>();
      let totalReported = 0;
      for (const ev of evidenceRows) {
        const sf = (ev.source_file ?? '').trim();
        if (sf) sourceSet.add(sf);
        const outcome = (ev.outcome ?? '').trim();
        if (!outcome) continue;
        const entry = outcomeMap.get(outcome) ?? { rows: 0, count: null };
        entry.rows += 1;
        const num = Number.parseFloat(ev.reported_count ?? '');
        if (!Number.isNaN(num)) {
          entry.count = (entry.count ?? 0) + num;
          totalReported += num;
        }
        outcomeMap.set(outcome, entry);
      }

      if (tgaCat) sourceSet.add(`TGA Category ${tgaCat}`);
      if (toxCnt > 0) sourceSet.add('EPA ToxRefDB / Nonclinical');

      const outcomes = Array.from(outcomeMap.entries())
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([outcome, stats]) => ({
          outcome,
          source_rows: stats.rows,
          reported_count: stats.count,
        }));

      const tgaSignal = tgaCat ? (TGA_WEIGHTS[tgaCat] ?? 40) : null;
      const pregSignal =
        evidenceRows.length > 0
          ? Math.min(92, 32 + 52 * (Math.log1p(totalReported || evidenceRows.length) / Math.log1p(500)))
          : null;
      const toxSignal = toxCnt > 0 ? Math.min(72, 34 + 8 * Math.log1p(toxCnt)) : null;

      const signals = [tgaSignal, pregSignal, toxSignal].filter((v): v is number => v !== null);
      const estimatedConcernSignal =
        signals.length > 0
          ? Math.round((signals.reduce((a, b) => a + b, 0) / signals.length) * 10) / 10
          : 25;

      results.push({
        drug_id: cid,
        drug_name: (row.original_name ?? '').trim() || cid,
        fingerprint: fp,
        source_record_count: evidenceRows.length,
        tga_category: tgaCat,
        toxicity_record_count: toxCnt,
        estimated_concern_signal: estimatedConcernSignal,
        outcomes,
        sources: Array.from(sourceSet).sort(),
        timing:
          evidenceRows.length > 0
            ? 'Untimed human source (projected)'
            : tgaCat
              ? 'Regulatory classification'
              : 'Nonclinical animal assay',
      });
    } catch {
      // Skip invalid SMILES
    }
  }

  cachedPool = results;
  return results;
}

export const SIMILARITY_THRESHOLD = 0.2;

export function structuralAnalogueEvidence(
  drugId: string | null,
  smiles: string | null,
  limit = 5,
  threshold = SIMILARITY_THRESHOLD
) {
  if (!smiles || !smiles.trim()) {
    return {
      available: false,
      threshold,
      candidates: [] as AnalogueCandidate[],
      reason: 'No mapped structure is available for a structural comparison.',
    };
  }

  try {
    const queryMol = OCL.Molecule.fromSmiles(smiles.trim());
    if (!queryMol || queryMol.getAllAtoms() === 0) {
      return {
        available: false,
        threshold,
        candidates: [] as AnalogueCandidate[],
        reason: 'The recorded SMILES could not be parsed for comparison.',
      };
    }

    const queryFp = queryMol.getIndex();
    const pool = getIndexedAnaloguePool();
    const allScored: AnalogueCandidate[] = pool
      .filter((c) => c.drug_id !== String(drugId ?? ''))
      .map((c) => ({
        drug_id: c.drug_id,
        drug_name: c.drug_name,
        similarity: tanimotoFromIndices(queryFp, c.fingerprint),
        source_record_count: c.source_record_count,
        tga_category: c.tga_category,
        toxicity_record_count: c.toxicity_record_count,
        estimated_concern_signal: c.estimated_concern_signal,
        outcomes: c.outcomes,
        sources: c.sources,
        timing: c.timing,
      }))
      .filter((c) => c.similarity >= threshold)
      .sort((a, b) => b.similarity - a.similarity || a.drug_name.localeCompare(b.drug_name))
      .slice(0, Math.max(0, limit));

    return {
      available: allScored.length > 0,
      threshold,
      reason:
        allScored.length > 0
          ? null
          : `No mapped structurally related compounds met the Tanimoto similarity threshold (>= ${threshold.toFixed(2)}).`,
      similarity_method: `OpenChemLib 512-bit substructure fingerprint; Tanimoto similarity (threshold >= ${threshold.toFixed(2)})`,
      candidates: allScored,
    };
  } catch {
    return {
      available: false,
      threshold,
      candidates: [] as AnalogueCandidate[],
      reason: 'The recorded SMILES could not be parsed for comparison.',
    };
  }
}
