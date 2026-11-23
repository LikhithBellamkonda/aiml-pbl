import React from 'react';
import { DrugEvaluation, DrugProfile, TrimesterResult } from '../types';
import { BarChart } from './BarChart';
import { DataTable } from './DataTable';
import { Expander } from './Expander';

function bandBadgeClass(score: number): string {
  if (score >= 65) return 'status-badge badge-high';
  if (score >= 45) return 'status-badge badge-modhigh';
  if (score >= 25) return 'status-badge badge-moderate';
  return 'status-badge badge-low';
}

function confBadgeClass(level: string): string {
  if (level === 'High') return 'status-badge badge-low';
  if (level === 'Moderate') return 'status-badge badge-moderate';
  return 'status-badge badge-modhigh';
}

export const EvidenceTab: React.FC<{
  evaluation: DrugEvaluation;
  selectedName: string;
  targetLabel: string;
  profile: DrugProfile;
}> = ({ evaluation, selectedName, targetLabel, profile }) => {
  const trimesterResults = evaluation.trimester_results;
  const tierBreakdown = evaluation.tier_breakdown || [];
  const ml = evaluation.ml_assessment;
  const ai = evaluation.ai_reasoning;
  const conf = evaluation.confidence;
  const conflicts = evaluation.conflicts || [];
  const tga = profile.tga_classification;
  const related = profile.related_pregnancy_evidence;
  const pregTerms = profile.adr?.pregnancy_related_terms || [];

  const cards: TrimesterResult[] = trimesterResults
    ? [
        trimesterResults.first_trimester,
        trimesterResults.second_trimester,
        trimesterResults.third_trimester,
      ]
    : [];

  return (
    <div>
      <div className="card-container">
        <div className="trimester-card-header" style={{ marginBottom: '0.5rem' }}>
          <div>
            <h3 style={{ margin: 0 }}>
              {selectedName} · Trimester-Wise Screening Results ({targetLabel})
            </h3>
            <p className="caption" style={{ marginBottom: 0 }}>
              {evaluation.risk_scoring_status}
            </p>
          </div>
          {conf && (
            <div style={{ textAlign: 'right' }}>
              <span className={confBadgeClass(conf.overall_confidence_level)}>
                Evidence Confidence: {conf.overall_confidence_level} ({conf.overall_confidence_score}%)
              </span>
              <div className="caption" style={{ margin: '0.2rem 0 0 0' }}>
                Uncertainty: {conf.uncertainty_percent}%
              </div>
            </div>
          )}
        </div>

        {tga?.available && (
          <div className="alert alert-info">
            <strong>Regulatory Pregnancy Classification (TGA): Category {tga.category || 'Listed'}</strong>
            {tga.safety_statement ? ` — ${tga.safety_statement}` : ''}
          </div>
        )}

        <div className="grid-3" style={{ marginTop: '0.75rem' }}>
          {cards.map((tr) => (
            <div className="trimester-card" key={tr.trimester}>
              <div className="trimester-card-header">
                <div>
                  <strong style={{ fontSize: '1.05rem' }}>{tr.trimester_label}</strong>
                  <div className="caption" style={{ margin: 0 }}>
                    {tr.gestational_window}
                  </div>
                </div>
                <span className={bandBadgeClass(tr.concern_score)}>{tr.priority_band}</span>
              </div>

              <div>
                <div className="trimester-score-big">{tr.concern_score.toFixed(1)} / 100</div>
                <div className="caption" style={{ margin: '0.15rem 0 0 0' }}>
                  Maternal: {tr.maternal_score.toFixed(1)} · Fetal: {tr.fetal_score.toFixed(1)}
                </div>
              </div>

              <div>
                <span className={confBadgeClass(tr.confidence_level)}>
                  Confidence: {tr.confidence_level} ({tr.confidence_score}%) · Uncertainty: {tr.uncertainty_percent}%
                </span>
              </div>

              <div style={{ fontSize: '0.84rem' }}>
                <strong>Primary Evidence Category:</strong> {tr.evidence_category}
              </div>

              <div style={{ fontSize: '0.84rem' }}>
                <strong>Key Outcomes / Signals:</strong>
                <ul className="detail-list">
                  {tr.adverse_outcomes.slice(0, 4).map((o, idx) => (
                    <li key={idx}>{o}</li>
                  ))}
                </ul>
              </div>

              <div style={{ fontSize: '0.82rem' }}>
                <strong>Supporting Evidence ({tr.active_tiers.map((t) => `T${t}`).join(', ')}):</strong>
                <ul className="detail-list">
                  {tr.supporting_evidence.slice(0, 3).map((s, idx) => (
                    <li key={idx}>{s}</li>
                  ))}
                </ul>
              </div>

              {tr.missing_evidence_notes.length > 0 && (
                <div className="caption" style={{ marginTop: 'auto', fontSize: '0.78rem' }}>
                  <strong>Gaps:</strong> {tr.missing_evidence_notes.slice(0, 2).join(' ')}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {conflicts.length > 0 && (
        <div className="card-container">
          <h3>Evidence Conflicts &amp; Limitations ({conflicts.length})</h3>
          {conflicts.map((c, idx) => (
            <div
              key={idx}
              className={`alert ${c.severity === 'High' ? 'alert-error' : c.severity === 'Moderate' ? 'alert-warning' : 'alert-info'}`}
            >
              <strong>
                [{c.severity}] {c.title}
              </strong>{' '}
              (Tiers: {c.tiers_involved.join(', ')}) — {c.detail}
            </div>
          ))}
        </div>
      )}

      {ml && ai && (
        <div className="card-container">
          <h3>Model Estimation &amp; &lt;20% AI Reasoning Layer</h3>
          <p className="caption">
            ML prediction score, &lt;20% supplementary AI reasoning score, final trimester concern
            scores, and evidence confidence are reported separately for scientific transparency.
          </p>
          <div className="grid-4" style={{ marginBottom: '0.75rem' }}>
            <div className="metric-card">
              <div className="metric-label">ML Physicochemical Score</div>
              <div className="metric-value">{ml.ml_score.toFixed(1)} / 100</div>
              <div className="metric-delta">Placental transfer: {ml.placental_transfer_propensity}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">&lt;20% AI Fallback Status</div>
              <div className="metric-value">{ai.triggered ? 'TRIGGERED' : 'Standby (>=20%)'}</div>
              <div className="metric-delta">{ai.trigger_reason}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">AI Supplementary Estimate</div>
              <div className="metric-value">{ai.ai_supplementary_score.toFixed(1)} / 100</div>
              <div className="metric-delta">Tier 6 blended: {ai.blended_tier6_score.toFixed(1)}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Overall Evidence Confidence</div>
              <div className="metric-value">
                {conf?.overall_confidence_level} ({conf?.overall_confidence_score}%)
              </div>
              <div className="metric-delta">Uncertainty: {conf?.uncertainty_percent}%</div>
            </div>
          </div>
          <div className={`alert ${ai.triggered ? 'alert-warning' : 'alert-info'}`}>
            <strong>Reasoning Summary:</strong> {ai.explanation} <em>{ai.gap_interpretation}</em>
          </div>
          <p className="caption">
            <strong>Model artifact note:</strong> {ml.artifact_note}
          </p>
          {(ml.structural_alerts.length > 0 || ml.target_alerts.length > 0) && (
            <p className="caption">
              <strong>Features used:</strong> {[...ml.structural_alerts, ...ml.target_alerts].join(' · ')}
            </p>
          )}
        </div>
      )}

      <div className="card-container">
        <h3>6-Tier Fallback Hierarchy Breakdown</h3>
        <p className="caption">
          Missing tiers receive 0 effective weight (excluded from denominators rather than diluting scores toward 0%).
        </p>
        <DataTable
          columns={[
            'Tier',
            'Source Category',
            'Status',
            'Records',
            'T1 Score',
            'T2 Score',
            'T3 Score',
            'Renorm. Weight',
            'Summary',
          ]}
          rows={tierBreakdown.map((t) => ({
            Tier: `Tier ${t.tier}`,
            'Source Category': t.name,
            Status: t.available ? (t.is_human_evidence ? 'Available (Human)' : 'Available (Nonclinical/Est.)') : 'Missing (Excluded)',
            Records: t.evidence_count,
            'T1 Score': t.scores_by_trimester.first_trimester?.toFixed(1) ?? 'Excluded',
            'T2 Score': t.scores_by_trimester.second_trimester?.toFixed(1) ?? 'Excluded',
            'T3 Score': t.scores_by_trimester.third_trimester?.toFixed(1) ?? 'Excluded',
            'Renorm. Weight': t.available ? `${(t.effective_weight * 100).toFixed(1)}%` : '0%',
            Summary: t.summary,
          }))}
        />
      </div>

      <Expander title={`Supporting Details: Pregnancy Outcomes, ADRs (${pregTerms.length} preg-tagged), & Analogues`}>
        {profile.pregnancy?.outcome_totals && profile.pregnancy.outcome_totals.length > 0 ? (
          <>
            <h4>Human pregnancy outcomes ({profile.pregnancy.record_count} rows)</h4>
            <DataTable
              columns={['category', 'outcome', 'records', 'reported_count']}
              rows={profile.pregnancy.outcome_totals}
            />
            <BarChart
              items={profile.pregnancy.outcome_totals
                .filter((r) => r.reported_count !== null)
                .map((r) => ({
                  label: r.outcome.replace(/_/g, ' '),
                  value: Number(r.reported_count ?? r.records),
                }))}
            />
          </>
        ) : (
          <div className="alert alert-info">
            No direct human pregnancy-outcome rows recorded; fallback tiers 2–6 provide the active trimester estimates.
          </div>
        )}

        {pregTerms.length > 0 && (
          <>
            <h4>Pregnancy-tagged ADR terms from SIDER</h4>
            <DataTable
              columns={['term', 'meddra_id', 'records', 'sources']}
              rows={pregTerms}
            />
          </>
        )}

        {related?.available && related.candidates.length > 0 && (
          <>
            <h4>Structurally Similar Chemical Analogues (Tanimoto &gt;= 0.20)</h4>
            <DataTable
              columns={['drug_name', 'similarity', 'tga_category', 'source_record_count', 'timing']}
              rows={related.candidates.map((c) => ({
                drug_name: c.drug_name,
                similarity: c.similarity.toFixed(3),
                tga_category: c.tga_category || 'Not listed',
                source_record_count: c.source_record_count,
                timing: c.timing,
              }))}
            />
          </>
        )}
      </Expander>
    </div>
  );
};
