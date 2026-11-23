import assert from 'node:assert/strict';
import test from 'node:test';
import { getDataStore } from './dataLoader.js';
import { buildDrugProfile } from './drugService.js';
import { evaluateDrugSafety } from './scoringService.js';

function assertAllThreeTrimestersValid(ev: ReturnType<typeof evaluateDrugSafety>, contextLabel: string) {
  const t1 = ev.trimester_results.first_trimester;
  const t2 = ev.trimester_results.second_trimester;
  const t3 = ev.trimester_results.third_trimester;

  for (const [label, tr] of [
    ['T1', t1],
    ['T2', t2],
    ['T3', t3],
  ] as const) {
    assert.ok(tr, `${contextLabel} missing ${label}`);
    assert.equal(typeof tr.concern_score, 'number', `${contextLabel} ${label} concern_score must be numeric`);
    assert.ok(tr.concern_score > 0, `${contextLabel} ${label} concern_score must NEVER be 0 (got ${tr.concern_score})`);
    assert.ok(tr.maternal_score > 0, `${contextLabel} ${label} maternal_score must NEVER be 0`);
    assert.ok(tr.fetal_score > 0, `${contextLabel} ${label} fetal_score must NEVER be 0`);
    assert.ok(!tr.priority_band.toLowerCase().includes('abstain'), `${contextLabel} ${label} must not Abstain`);
    assert.ok(!tr.evidence_category.toLowerCase().includes('abstain'), `${contextLabel} ${label} category must not Abstain`);
    assert.ok(tr.confidence_score > 0 && tr.confidence_score <= 100, `${contextLabel} ${label} confidence out of range`);
    assert.ok(tr.adverse_outcomes.length > 0, `${contextLabel} ${label} must report outcomes/signals`);
    assert.ok(tr.supporting_evidence.length > 0, `${contextLabel} ${label} must report supporting evidence`);
    assert.ok(!tr.score_label.toLowerCase().includes('validated clinical'), `${contextLabel} must not claim clinical probability`);
  }

  assert.ok(ev.risk_score !== null && ev.risk_score > 0, `${contextLabel} risk_score must be > 0`);
}

test('1. Drug with strong human pregnancy evidence (tretinoin) resolves & scores all 3 trimesters > 0', () => {
  const profile = buildDrugProfile('tretinoin');
  assert.equal(profile.matched, true);
  assert.equal(profile.pregnancy.available, true);
  const ev = evaluateDrugSafety(profile.drug_name!, 'All', 'Fetal Outcomes Only', profile.drug_id, {
    profileOverride: profile,
  });
  assertAllThreeTrimestersValid(ev, 'tretinoin');
  const tierH = ev.tier_breakdown.find((t) => t.tier === 'H');
  assert.equal(tierH?.available, true);
  assert.ok(ev.trimester_results.first_trimester.concern_score >= 50, 'Tretinoin T1 should be high concern');
});

test('2. Drug with general SIDER/FAERS ADRs and limited/no direct human pregnancy outcomes (abacavir)', () => {
  const profile = buildDrugProfile('abacavir');
  assert.equal(profile.matched, true);
  assert.equal(profile.adr.available, true);
  assert.ok(profile.adr.term_count > 0, 'SIDER MedDRA terms should be populated');
  const ev = evaluateDrugSafety(profile.drug_name!, 'All', 'Fetal Outcomes Only', profile.drug_id, {
    profileOverride: profile,
  });
  assertAllThreeTrimestersValid(ev, 'abacavir');
  const tierA = ev.tier_breakdown.find((t) => t.tier === 'A');
  assert.equal(tierA?.available, true);
});

test('3. Drug with nonclinical animal toxicity / veterinary ADR records keeps animal evidence separated', () => {
  const store = getDataStore();
  const toxRow = store.toxicityFlags.find((r) => r.has_tox_data === '1' && r.Study_DEV === '1');
  assert.ok(toxRow, 'Expected at least one Study_DEV drug in real_toxicity.csv');
  const profile = buildDrugProfile(toxRow.drug_id);
  assert.equal(profile.matched, true);
  assert.equal(profile.reproductive_toxicity.available, true);
  const ev = evaluateDrugSafety(profile.drug_name!, 'All', 'Fetal Outcomes Only', profile.drug_id, {
    profileOverride: profile,
  });
  assertAllThreeTrimestersValid(ev, `animal-tox:${profile.drug_id}`);
  const tierT = ev.tier_breakdown.find((t) => t.tier === 'T');
  assert.equal(tierT?.available, true);
  assert.equal(tierT?.is_human_evidence, false, 'Animal toxicity must never be marked as human evidence');
  assert.ok(tierT?.summary.toLowerCase().includes('nonclinical'), 'Must explicitly label as nonclinical');
});

test('4. Drug with chemical similarity analogues (Tanimoto >= 0.20) uses Tier 4 safely', () => {
  const profile = buildDrugProfile('abacavir');
  assert.equal(profile.matched, true);
  assert.equal(profile.related_pregnancy_evidence.available, true);
  assert.ok(profile.related_pregnancy_evidence.candidates.length > 0);
  const topSim = profile.related_pregnancy_evidence.candidates[0].similarity;
  assert.ok(topSim >= 0.2, `Expected Tanimoto >= 0.20, got ${topSim}`);
  const ev = evaluateDrugSafety(profile.drug_name!, 'All', 'Fetal Outcomes Only', profile.drug_id, {
    profileOverride: profile,
  });
  const tierS = ev.tier_breakdown.find((t) => t.tier === 'S');
  assert.equal(tierS?.available, true);
  assert.equal(tierS?.is_human_evidence, false, 'Chemical analogue matches are not direct human outcomes');
});

test('5. Drug with missing direct pregnancy records never returns 0% or Abstain across trimesters', () => {
  const store = getDataStore();
  const pregIds = new Set(store.pregnancy.map((r) => r.drug_id).filter(Boolean));
  const candidate = store.mapping.find((r) => r.drug_id && !pregIds.has(r.drug_id) && r.canonical_smiles);
  assert.ok(candidate, 'Expected mapped drug without direct pregnancy rows');
  const profile = buildDrugProfile(candidate.drug_id);
  assert.equal(profile.matched, true);
  assert.equal(profile.pregnancy.available, false);
  const ev = evaluateDrugSafety(profile.drug_name!, 'All', 'Fetal Outcomes Only', profile.drug_id, {
    profileOverride: profile,
  });
  assertAllThreeTrimestersValid(ev, `no-preg:${profile.drug_name}`);
  assert.ok(
    ev.trimester_results.first_trimester.missing_evidence_notes.some((n) =>
      n.toLowerCase().includes('not zero risk')
    )
  );
});

test('6. Drug with incomplete molecular data (no SMILES biologic e.g. adalimumab) runs fallback without crash', () => {
  const profile = buildDrugProfile('adalimumab');
  assert.equal(profile.matched, true);
  assert.equal(profile.molecular.available, false);
  const ev = evaluateDrugSafety(profile.drug_name!, 'All', 'Fetal Outcomes Only', profile.drug_id, {
    profileOverride: profile,
  });
  assertAllThreeTrimestersValid(ev, 'adalimumab');
  assert.ok(ev.ml_assessment.ml_score < 20, 'Biologic without SMILES should yield <20% ML score');
  assert.equal(ev.ai_reasoning.triggered, true, 'Missing SMILES (<20% ML) should trigger AI reasoning');
  assert.ok(ev.ai_reasoning.ai_supplementary_score > 0);
});

test('7. Unknown or unmatched drug handles cleanly without crashing or fabricating 0% safety', () => {
  const profile = buildDrugProfile('notarealmedicationxyz_999');
  assert.equal(profile.matched, false);
  assert.equal(profile.ambiguous, false);
  assert.ok(profile.warnings.length > 0);
});

test('8. Conflicting evidence across tiers triggers explicit conflict detection', () => {
  // Find or construct a high-TGA/human drug paired with a low-animal or untimed projection
  const baseProfile = buildDrugProfile('tretinoin');
  const conflictingProfile = {
    ...baseProfile,
    reproductive_toxicity: {
      available: true,
      record_count: 1,
      species: ['rat'],
      study_types: ['SUB'],
      study_flags: ['SUB'],
      sources: ['synthetic_test_fixture'],
      records: [{ species: 'rat', study_type: 'SUB' }],
      reason: null,
      is_human_evidence: false,
    },
    molecular: {
      ...baseProfile.molecular,
      available: false,
      descriptors: {},
    },
  } as unknown as ReturnType<typeof buildDrugProfile>;

  const ev = evaluateDrugSafety('tretinoin', 'All', 'Fetal Outcomes Only', baseProfile.drug_id, {
    profileOverride: conflictingProfile,
  });
  assertAllThreeTrimestersValid(ev, 'conflicting-profile');
  assert.ok(ev.conflicts.length > 0, 'Expected explicit conflict flags when tiers disagree');
});

test('9. Missing ML artifacts degrades cleanly to deterministic fallback without error or 0%', () => {
  const profile = buildDrugProfile('amoxicillin');
  assert.equal(profile.matched, true);
  const ev = evaluateDrugSafety(profile.drug_name!, 'All', 'Fetal Outcomes Only', profile.drug_id, {
    profileOverride: profile,
    forceMissingArtifacts: true,
  });
  assertAllThreeTrimestersValid(ev, 'missing-artifacts');
  assert.equal(ev.ml_assessment.human_checkpoint_present, false);
  assert.equal(ev.ml_assessment.nonclinical_benchmark_present, false);
  assert.ok(ev.ml_assessment.artifact_note.length > 0);
});

test('10. <20% ML fallback triggers supplementary AI reasoning and keeps ML, AI, concern score, and confidence distinct', () => {
  // Salt-stripped query test as well
  const profile = buildDrugProfile('adalimumab');
  const ev = evaluateDrugSafety(profile.drug_name!, 'All', 'Fetal Outcomes Only', profile.drug_id, {
    profileOverride: profile,
  });
  assert.ok(ev.ml_assessment.ml_score < 20, `Expected ml_score < 20, got ${ev.ml_assessment.ml_score}`);
  assert.equal(ev.ai_reasoning.triggered, true);
  assert.ok(ev.ai_reasoning.ai_supplementary_score !== ev.ml_assessment.ml_score);
  assert.ok(ev.trimester_results.first_trimester.concern_score > 0);
  assert.ok(ev.confidence.overall_confidence_score > 0);
  assert.ok(
    ev.ai_reasoning.key_signals.length > 0,
    'Supplementary AI reasoning must cite concrete signals from prior tiers/structure gaps'
  );

  // Also test salt-stripped resolution (e.g., abacavir sulfate -> abacavir)
  const saltProfile = buildDrugProfile('abacavir sulfate');
  assert.equal(saltProfile.matched, true, 'Salt stripping should resolve abacavir sulfate -> abacavir');
  assert.equal(saltProfile.drug_id, 'CID000441300');
});

test('11. Vercel serverless entry (api/index.ts) serves /api/health, /api/bootstrap, and /api/drug cleanly', async () => {
  const { default: vercelApp } = await import('../api/index.js');
  const server = vercelApp.listen(0);
  const address = server.address();
  const port = typeof address === 'object' && address ? address.port : 0;
  assert.ok(port > 0, 'Expected ephemeral port');

  try {
    const health = await fetch(`http://127.0.0.1:${port}/api/health`).then((r) => r.json());
    assert.equal(health.ok, true);

    const bootstrap = await fetch(`http://127.0.0.1:${port}/api/bootstrap`).then((r) => r.json());
    assert.ok(Array.isArray(bootstrap.medicationNames) && bootstrap.medicationNames.length > 1000);

    const drug = await fetch(`http://127.0.0.1:${port}/api/drug?query=tretinoin`).then((r) => r.json());
    assert.equal(drug.mode, 'matched');
    assert.ok(drug.evaluation.trimester_results.first_trimester.concern_score > 0);
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});
