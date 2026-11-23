import React from 'react';
import { BootstrapResponse } from '../types';
import { DataTable } from './DataTable';

export const ReadinessTab: React.FC<{ bootstrap: BootstrapResponse }> = ({ bootstrap }) => {
  const feasibility = bootstrap.feasibility || {};
  const decisions = feasibility.target_decisions || [];
  let readiness = feasibility.training_readiness || {};

  if (!readiness.status && decisions.length > 0) {
    const blocked = !decisions.some((d) => d.decision === 'trainable');
    readiness = {
      status: blocked ? 'blocked' : 'not_verified',
      reason: blocked ? decisions[0].reason || '' : '',
    };
  }

  const labelsPresent = bootstrap.labelsPresent;
  const evaluatedModel = bootstrap.evaluatedModel;
  const statusStr = String(readiness.status || '');
  const trainingStatus = evaluatedModel
    ? 'Trained'
    : labelsPresent
      ? 'Labels to validate'
      : statusStr.startsWith('pregnancy_model_blocked') || statusStr === 'blocked'
        ? 'Blocked (Uses 6-Tier Fallback + AI)'
        : 'Exploratory Nonclinical Only';

  const toxEndpoints = Object.entries(bootstrap.reproductiveToxMetrics?.endpoints ?? {}).map(
    ([name, ep]) => {
      const rf = ep.models?.random_forest;
      const lr = ep.models?.logistic_regression;
      return {
        Endpoint: name,
        Records: ep.records,
        'Positive %': `${ep.positive_percent.toFixed(1)}%`,
        Scaffolds: ep.scaffold_groups,
        'RF AUROC': rf ? rf.auroc.toFixed(3) : 'Not measured',
        'RF AUPRC': rf ? rf.auprc.toFixed(3) : 'Not measured',
        'LR AUROC': lr ? lr.auroc.toFixed(3) : 'Not measured',
      };
    }
  );

  return (
    <div>
      <div className="card-container">
        <h3>Clinical Pregnancy Model Readiness</h3>
        <div className="grid-3" style={{ marginBottom: '0.75rem' }}>
          <div className="metric-card">
            <div className="metric-label">Human Clinical Training Status</div>
            <div className="metric-value" style={{ fontSize: '1.15rem' }}>{trainingStatus}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Validated Human Cohort Labels</div>
            <div className="metric-value">{labelsPresent ? 'Present' : 'Absent'}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Runtime Scoring Mode</div>
            <div className="metric-value" style={{ fontSize: '1.1rem' }}>6-Tier + &lt;20% AI Fallback</div>
          </div>
        </div>

        {readiness.reason && <p className="caption">{readiness.reason}</p>}

        {decisions.length > 0 && (
          <DataTable
            columns={['outcome', 'decision', 'reason']}
            rows={decisions.map((item) => ({
              outcome: String(item.outcome || '').replace(/_/g, ' '),
              decision: item.decision || 'unknown',
              reason: item.reason || '',
            }))}
          />
        )}
      </div>

      {toxEndpoints.length > 0 && (
        <div className="card-container">
          <h3>Nonclinical OECD Guideline Benchmark Metrics (Exploratory Only)</h3>
          <p className="caption">
            Out-of-fold 5-fold GroupKFold by Bemis-Murcko scaffold from{' '}
            <code>{bootstrap.reproductiveToxMetrics?.source_file || 'OECD nonclinical dataset'}</code>.
            These evaluate nonclinical animal endpoints—never human clinical pregnancy outcomes.
          </p>
          <DataTable
            columns={['Endpoint', 'Records', 'Positive %', 'Scaffolds', 'RF AUROC', 'RF AUPRC', 'LR AUROC']}
            rows={toxEndpoints}
          />
          {(bootstrap.reproductiveToxMetrics?.important_limitations ?? []).map((lim, i) => (
            <p className="caption" key={i}>
              • {lim}
            </p>
          ))}
        </div>
      )}
    </div>
  );
};
