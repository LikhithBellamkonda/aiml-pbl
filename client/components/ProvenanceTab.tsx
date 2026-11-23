import React from 'react';
import { DrugEvaluation, DrugProfile } from '../types';
import { DataTable } from './DataTable';

export const ProvenanceTab: React.FC<{
  evaluation: DrugEvaluation;
  profile: DrugProfile;
}> = ({ evaluation, profile }) => {
  const pregRecords = profile.pregnancy?.records ?? [];
  const toxRecords = profile.reproductive_toxicity?.records ?? [];
  const tgaRecords = profile.tga_classification?.records ?? [];
  const adrTerms = profile.adr?.terms ?? [];

  return (
    <div>
      <div className="card-container">
        <h3>Active Evidence Provenance</h3>
        {evaluation.provenance_badges.length === 0 ? (
          <div className="alert alert-info">Baseline computational fallback active.</div>
        ) : (
          evaluation.provenance_badges.map((prov, idx) => (
            <p className="caption" key={idx}>
              ● <strong>{prov.source_dataset}</strong> — {prov.evidence_granularity} ({prov.evidence_status})
            </p>
          ))
        )}
      </div>

      {tgaRecords.length > 0 && (
        <div className="card-container">
          <h3>TGA Pregnancy Classification Records ({tgaRecords.length})</h3>
          <DataTable columns={Object.keys(tgaRecords[0])} rows={tgaRecords} />
        </div>
      )}

      {pregRecords.length > 0 && (
        <div className="card-container">
          <h3>Human Pregnancy Source Records ({pregRecords.length})</h3>
          <DataTable columns={Object.keys(pregRecords[0])} rows={pregRecords} />
        </div>
      )}

      {toxRecords.length > 0 && (
        <div className="card-container">
          <h3>Nonclinical Animal Toxicity Records ({toxRecords.length})</h3>
          <DataTable columns={Object.keys(toxRecords[0])} rows={toxRecords} />
        </div>
      )}

      {adrTerms.length > 0 && (
        <div className="card-container">
          <h3>Top SIDER / ADR Terms ({adrTerms.length} shown)</h3>
          <DataTable columns={Object.keys(adrTerms[0])} rows={adrTerms.slice(0, 30)} />
        </div>
      )}
    </div>
  );
};
