import React from 'react';
import { DrugEvaluation } from '../types';
import { DataTable } from './DataTable';
import { Expander } from './Expander';

export const AnimalAdrTab: React.FC<{ evaluation: DrugEvaluation }> = ({ evaluation }) => {
  const ev = evaluation.animal_adr_evidence;
  return (
    <div className="card-container">
      <h3>Animal Adverse Drug Reaction Evidence</h3>
      <p className="caption">
        Veterinary spontaneous reports (FDA CVM ADAE, VeDDRA terms). Supplementary evidence only:
        not a human pregnancy risk conclusion, not incidence, and no causality is implied.
      </p>
      {!ev.available ? (
        <div className="alert alert-info">
          {ev.message || 'No matching animal ADR records in the local snapshot.'}
        </div>
      ) : (
        <>
          <div className="grid-3" style={{ marginBottom: '0.75rem' }}>
            <div className="metric-card">
              <div className="metric-label">Animal ADR reports</div>
              <div className="metric-value">{ev.report_count ?? 0}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Species represented</div>
              <div className="metric-value">{ev.species_count ?? 0}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Dominant species</div>
              <div className="metric-value">{ev.top_species || 'Unknown'}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Reported reaction terms</div>
              <div className="metric-value">{ev.reaction_count ?? 0}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Serious reports</div>
              <div className="metric-value">{ev.serious_report_count ?? 0}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Dominant reaction category</div>
              <div className="metric-value">{ev.top_reaction_category || 'Unclassified'}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Record completeness</div>
              <div className="metric-value">{(ev.record_completeness ?? 0).toFixed(2)} / 1</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Drug mapping status</div>
              <div className="metric-value">{ev.mapping_status || 'Not mapped'}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Reporting period</div>
              <div className="metric-value">
                {ev.report_date_min || '?'} to {ev.report_date_max || '?'}
              </div>
            </div>
          </div>
          <p className="caption">
            Source: {ev.sources || 'FDA CVM ADAE'} · mapping: {ev.mapping_status || 'unknown'} ·
            deaths/euthanasia reports: {ev.death_or_euthanasia_report_count ?? 0}
          </p>
          {ev.source_url && (
            <p>
              <a href={ev.source_url} target="_blank" rel="noreferrer">
                Original dataset documentation
              </a>
            </p>
          )}
          <p className="caption">
            Limitations: spontaneous reporting is voluntary and has no exposure denominator, so
            counts are neither incidence nor severity. Reports describe product-level exposure and
            may involve combination products.
          </p>
        </>
      )}

      <Expander title="Evidence categories kept separate">
        <div style={{ marginTop: '0.5rem' }}>
          <DataTable
            columns={['evidence_type', 'label', 'meaning']}
            rows={evaluation.evidence_distinction}
          />
        </div>
      </Expander>
    </div>
  );
};
