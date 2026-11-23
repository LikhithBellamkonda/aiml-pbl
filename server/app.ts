import express from 'express';
import { getDataStore, getMedicationNames } from './dataLoader.js';
import { buildDrugProfile } from './drugService.js';
import { evaluateDrugSafety } from './scoringService.js';

function computeHasEvaluatedModel() {
  const store = getDataStore();
  const decisions =
    (store.feasibility.target_decisions as Array<{ outcome?: string; decision?: string }>) ?? [];
  const expectedOutcomes = new Set([
    'fetal_death',
    'preterm_birth',
    'fetal_growth_restriction',
    'congenital_anomaly',
  ]);
  const trainableOutcomes = new Set(
    decisions.filter((d) => d.decision === 'trainable').map((d) => d.outcome ?? '')
  );
  const metadataOutcomes = new Set((store.metadata.outcomes as string[]) ?? []);
  const expectedMatchesMetadata =
    expectedOutcomes.size === metadataOutcomes.size &&
    Array.from(expectedOutcomes).every((o) => metadataOutcomes.has(o));
  const expectedInTrainable = Array.from(expectedOutcomes).every((o) =>
    trainableOutcomes.has(o)
  );
  const modelMetrics = (store.metrics.model as Record<string, unknown>) ?? null;

  return Boolean(
    Object.keys(store.metadata).length > 0 &&
      expectedMatchesMetadata &&
      expectedInTrainable &&
      store.filesPresent.checkpoint &&
      modelMetrics &&
      Object.keys(modelMetrics).length > 0
  );
}

export function createApiApp() {
  const app = express();
  app.use(express.json());

  app.get('/api/health', (_req, res) => {
    res.json({ ok: true });
  });

  app.get('/api/bootstrap', (_req, res) => {
    const store = getDataStore();
    const medicationNames = getMedicationNames();
    const evaluatedModel = computeHasEvaluatedModel();
    res.json({
      medicationNames,
      feasibility: store.feasibility,
      reproductiveToxMetrics: store.reproductiveToxMetrics,
      labelsPresent: store.filesPresent.labels,
      evaluatedModel,
      filesPresent: store.filesPresent,
    });
  });

  app.get('/api/drug', (req, res) => {
    const query = String(req.query.query ?? '').trim();
    const selectedName = String(req.query.selectedName ?? '').trim();
    const explicitDrugId = String(req.query.drugId ?? '').trim();
    const timing = String(req.query.timing ?? 'First Trimester (T1)').trim();
    const target = String(req.query.target ?? 'Fetal Outcomes').trim();

    const store = getDataStore();
    const medicationNames = getMedicationNames();
    const effectiveQuery = explicitDrugId || query || selectedName;

    if (!effectiveQuery) {
      return res.status(400).json({ error: 'Query or medication name is required.' });
    }

    let profile = buildDrugProfile(effectiveQuery);

    if (!explicitDrugId && !query && selectedName) {
      const drugRows = store.mapping.filter((r) => (r.original_name ?? '') === selectedName);
      const rowId = drugRows[0]?.drug_id ?? '';
      profile = buildDrugProfile(rowId || selectedName);
    }

    if (profile.ambiguous) {
      return res.json({
        mode: 'ambiguous',
        query: effectiveQuery,
        profile,
        evaluation: null,
        suggestions: [],
      });
    }

    if (!profile.matched) {
      const needle = effectiveQuery.toLowerCase();
      const suggestions = medicationNames
        .filter((name) => name.toLowerCase().includes(needle))
        .slice(0, 12);
      return res.json({
        mode: 'unmatched',
        query: effectiveQuery,
        profile,
        evaluation: null,
        suggestions,
      });
    }

    const resolvedName = String(profile.identity?.name || effectiveQuery);
    const drugId = profile.drug_id ? String(profile.drug_id) : null;
    const timingFilter = timing === 'All Trimesters' ? 'All' : timing;
    const targetFilter =
      target === 'Fetal Outcomes' ? 'Fetal Outcomes Only' : 'Maternal Outcomes Only';

    const evaluation = evaluateDrugSafety(resolvedName, timingFilter, targetFilter, drugId, {
      profileOverride: profile,
    });

    return res.json({
      mode: 'matched',
      query: effectiveQuery,
      selectedName: resolvedName,
      drugId,
      profile,
      evaluation,
      suggestions: [],
    });
  });

  return app;
}
