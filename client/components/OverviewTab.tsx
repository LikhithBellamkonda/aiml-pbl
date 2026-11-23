import React from 'react';
import { DrugProfile } from '../types';
import { DataTable } from './DataTable';
import { Expander } from './Expander';

function renderFormula(formula: string | null | undefined) {
  if (!formula) return null;
  const parts: Array<{ element: string; count: string }> = [];
  const regex = /([A-Z][a-z]?)(\d*)/g;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(formula)) !== null) {
    if (!match[1]) continue;
    parts.push({ element: match[1], count: match[2] || '' });
  }
  return (
    <div className="formula-display">
      {parts.map((part, idx) => (
        <React.Fragment key={idx}>
          {part.element}
          {part.count && part.count !== '1' ? <sub>{part.count}</sub> : null}
        </React.Fragment>
      ))}
    </div>
  );
}

export const OverviewTab: React.FC<{ profile: DrugProfile }> = ({ profile }) => {
  const identity = profile.identity;
  const molecular = profile.molecular;
  const counts = profile.evidence_counts;
  const researchPriority = profile.research_priority;
  const adr = profile.adr;
  const toxicity = profile.reproductive_toxicity;
  const pregnancy = profile.pregnancy;
  const related = profile.related_pregnancy_evidence;
  const faers = profile.faers;
  const tga = profile.tga_classification;
  const targets = profile.targets;

  const formula = identity.molecular_formula || molecular.molecular_formula;
  const smiles = identity.smiles || molecular.canonical_smiles;
  const descriptors = molecular.descriptors || {};

  const availabilityRows = [
    {
      'Evidence Tier / Source': 'Tier 1: Human pregnancy outcomes',
      category: 'human',
      records: counts.pregnancy_records > 0 ? counts.pregnancy_records : 'Unavailable (fallback active)',
    },
    {
      'Evidence Tier / Source': 'Tier 2: TGA regulatory classification',
      category: 'regulatory',
      records: tga?.available ? `Category ${tga.category || 'Listed'}` : 'Unavailable',
    },
    {
      'Evidence Tier / Source': 'Tier 2: FAERS reporting summary',
      category: 'human',
      records: counts.faers_reports ?? 'Unavailable',
    },
    {
      'Evidence Tier / Source': 'Tier 3: General ADR terms (SIDER / FAERS)',
      category: 'human',
      records: counts.adr_terms > 0 ? counts.adr_terms : 'Unavailable',
    },
    {
      'Evidence Tier / Source': 'Tier 4: Chemical structural analogues (>=0.20)',
      category: 'structural similarity',
      records: related.available ? `${related.candidates.length} analogues` : 'No match >= 0.20',
    },
    {
      'Evidence Tier / Source': 'Tier 5: Developmental / reproductive toxicity studies',
      category: 'animal (nonclinical)',
      records: counts.nonclinical_toxicity_records > 0 ? counts.nonclinical_toxicity_records : 'Unavailable',
    },
  ];

  return (
    <div>
      <div className="card-container">
        <h3>Standardized drug identity</h3>
        <div className="grid-4">
          <div className="metric-card">
            <div className="metric-label">Drug / Active Ingredient</div>
            <div className="metric-value" style={{ fontSize: '1.15rem' }}>
              {identity.name || profile.drug_name || 'Unmapped'}
            </div>
            <div className="metric-delta">
              Ingredient: {identity.active_ingredient || 'Not parsed'}
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">PubChem CID</div>
            <div className="metric-value">{identity.pubchem_cid ?? 'No CID mapped'}</div>
            <div className="metric-delta">ID: {identity.drug_id || profile.drug_id || 'Unassigned'}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">InChIKey</div>
            <div className="metric-value" style={{ fontSize: '0.95rem' }}>
              {identity.inchikey || 'No InChIKey mapped'}
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">TGA Pregnancy Category</div>
            <div className="metric-value">
              {tga?.available ? (tga.category || 'Listed') : 'Not listed'}
            </div>
            <div className="metric-delta">Status: {identity.resolution_status || 'unknown'}</div>
          </div>
        </div>
        {tga?.available && tga.safety_statement && (
          <div className="alert alert-info">
            <strong>TGA Statement:</strong> {tga.safety_statement}
          </div>
        )}
        {profile.warnings.map((w, i) => (
          <div className="alert alert-warning" key={i}>
            {w}
          </div>
        ))}
      </div>

      <div className="card-container">
        <h3>Chemical structure &amp; targets</h3>
        <div className="grid-2">
          <div>
            <p className="caption">
              Molecular formula (computed locally from SMILES)
            </p>
            {formula ? (
              renderFormula(formula)
            ) : (
              <div className="alert alert-info">Formula is not recorded for this compound.</div>
            )}
            {identity.molecular_weight !== null && identity.molecular_weight !== undefined && (
              <div className="metric-card" style={{ marginBottom: '0.75rem' }}>
                <div className="metric-label">Molecular weight (g/mol)</div>
                <div className="metric-value">
                  {Number(identity.molecular_weight).toFixed(2)}
                </div>
              </div>
            )}
            {smiles ? (
              <>
                <p className="caption">Canonical SMILES</p>
                <div className="code-block">{smiles}</div>
              </>
            ) : (
              <p className="caption">SMILES not available (macromolecule, combination, or unmapped compound).</p>
            )}
          </div>
          <div className="structure-box">
            {molecular.svg ? (
              <>
                <img
                  src={`data:image/svg+xml;utf8,${encodeURIComponent(molecular.svg)}`}
                  alt="2D chemical structure"
                  style={{
                    maxWidth: '340px',
                    width: '100%',
                    background: '#ffffff',
                    border: '1px solid rgba(128,128,128,0.28)',
                    borderRadius: '6px',
                  }}
                />
                <p className="caption">
                  Scaffold: {molecular.scaffold || 'not classified'} · On-bits:{' '}
                  {molecular.fingerprint?.on_bits ?? 0}
                </p>
              </>
            ) : (
              <div className="alert alert-info">
                {molecular.reason
                  ? `Structure not drawn: ${molecular.reason}`
                  : 'No 2D structure is mapped to this drug identity.'}
              </div>
            )}
          </div>
        </div>
      </div>

      {targets && targets.available && targets.targets.length > 0 && (
        <Expander title={`Pharmacological target interactions (${targets.count})`}>
          <DataTable
            columns={['target_name', 'gene', 'target_class', 'action_type', 'organism', 'source']}
            rows={targets.targets}
          />
        </Expander>
      )}

      <div className="card-container">
        <h3>Evidence availability across 6 tiers</h3>
        <DataTable
          columns={['Evidence Tier / Source', 'category', 'records']}
          rows={availabilityRows}
        />
        <p className="caption">
          Unavailable sources never default to zero risk. Missing tiers are excluded from weight
          denominators while remaining tiers and molecular/AI estimation populate T1/T2/T3 scores.
        </p>
      </div>

      {researchPriority.available && (
        <Expander title={`Evidence-gap checklist (${researchPriority.score} / ${researchPriority.max_score} gaps)`}>
          <DataTable
            columns={['Evidence domain', 'Status', 'Gap points']}
            rows={researchPriority.components.map((c) => ({
              'Evidence domain': c.domain,
              Status: c.covered ? 'Available' : 'Missing',
              'Gap points': c.gap_points,
            }))}
          />
        </Expander>
      )}

      <Expander title={`General & pregnancy-related ADR terms (${adr.term_count})`}>
        {!adr.available ? (
          <div className="alert alert-info">{adr.reason || 'No general ADR evidence recorded.'}</div>
        ) : (
          <>
            <p className="caption">
              {adr.term_count} distinct terms ({adr.pregnancy_related_term_count ?? 0} pregnancy-related)
              from {adr.source_datasets.join(', ') || 'SIDER/FAERS'}.
            </p>
            {adr.terms.length > 0 && (
              <DataTable
                columns={['term', 'pregnancy_related', 'meddra_id', 'records', 'sources']}
                rows={adr.terms}
              />
            )}
          </>
        )}
      </Expander>

      <Expander title="Molecular descriptors">
        {!molecular.available ? (
          <div className="alert alert-info">
            {molecular.reason || 'No molecular descriptors for this identity.'}
          </div>
        ) : (
          <DataTable
            columns={['property', 'value']}
            rows={Object.entries(descriptors).map(([property, value]) => ({
              property,
              value,
            }))}
          />
        )}
      </Expander>

      <Expander title={`Nonclinical toxicity studies (${toxicity.record_count})`}>
        {!toxicity.available ? (
          <div className="alert alert-info">
            {toxicity.reason || 'No linked developmental/reproductive toxicity rows.'}
          </div>
        ) : (
          <>
            <p className="caption">
              {toxicity.record_count} nonclinical records · species:{' '}
              {toxicity.species.join(', ') || 'not recorded'} · types:{' '}
              {toxicity.study_types.join(', ') || 'listed'}
            </p>
            {toxicity.records.length > 0 && (
              <DataTable columns={Object.keys(toxicity.records[0])} rows={toxicity.records} />
            )}
          </>
        )}
      </Expander>

      <Expander title={`Human pregnancy-outcome records (${pregnancy.record_count})`}>
        {!pregnancy.available ? (
          <div className="alert alert-info">
            {pregnancy.reason || 'No linked human pregnancy rows.'}
          </div>
        ) : (
          <DataTable columns={Object.keys(pregnancy.records[0] || {})} rows={pregnancy.records} />
        )}
      </Expander>

      <Expander title="Chemical structural analogues (Tanimoto >= 0.20)">
        {related.available ? (
          <DataTable
            columns={[
              'Nearest compound',
              'Tanimoto similarity',
              'TGA',
              'Pregnancy rows',
              'Toxicity rows',
              'Observed outcomes',
            ]}
            rows={related.candidates.map((c) => ({
              'Nearest compound': c.drug_name,
              'Tanimoto similarity': c.similarity.toFixed(3),
              TGA: c.tga_category || 'Not listed',
              'Pregnancy rows': c.source_record_count,
              'Toxicity rows': c.toxicity_record_count ?? 0,
              'Observed outcomes':
                c.outcomes.map((o) => o.outcome.replace(/_/g, ' ')).join(', ') || 'Classification/Tox signal only',
            }))}
          />
        ) : (
          <div className="alert alert-info">
            {related.reason || 'No structurally similar compounds exceeded 0.20 Tanimoto threshold.'}
          </div>
        )}
      </Expander>

      <Expander title="FAERS aggregate reporting summary">
        {!faers.available ? (
          <div className="alert alert-info">{faers.reason || 'No FAERS summary row linked.'}</div>
        ) : (
          <p className="caption">
            Aggregate reports: {faers.report_count ?? 'Not recorded'} · outcome rows:{' '}
            {faers.outcome_count ?? 'Not recorded'} · source: {faers.source_name || 'FAERS'}.
          </p>
        )}
      </Expander>
    </div>
  );
};
