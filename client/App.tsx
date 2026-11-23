import React, { useEffect, useState } from 'react';
import { AnimalAdrTab } from './components/AnimalAdrTab';
import { DataTable } from './components/DataTable';
import { EvidenceTab } from './components/EvidenceTab';
import { OverviewTab } from './components/OverviewTab';
import { ProvenanceTab } from './components/ProvenanceTab';
import { ReadinessTab } from './components/ReadinessTab';
import { BootstrapResponse, DrugApiResponse } from './types';

const TIMING_OPTIONS = [
  'All Trimesters',
  'First Trimester (T1)',
  'Second Trimester (T2)',
  'Third Trimester (T3)',
  'Entire Pregnancy',
] as const;

const TABS = [
  'Trimester Results & Evidence',
  'Overview',
  'Animal ADR',
  'Provenance & Records',
  'Data & Model Readiness',
] as const;

type TabKey = (typeof TABS)[number];

export function App() {
  const [bootstrap, setBootstrap] = useState<BootstrapResponse | null>(null);
  const [bootstrapError, setBootstrapError] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [selectedName, setSelectedName] = useState('');
  const [explicitDrugId, setExplicitDrugId] = useState('');
  const [selectedTiming, setSelectedTiming] = useState<string>(TIMING_OPTIONS[0]);
  const [targetLabel, setTargetLabel] = useState<'Fetal Outcomes' | 'Maternal Outcomes'>(
    'Fetal Outcomes'
  );
  const [activeTab, setActiveTab] = useState<TabKey>('Trimester Results & Evidence');

  const [drugData, setDrugData] = useState<DrugApiResponse | null>(null);
  const [loadingDrug, setLoadingDrug] = useState(false);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDebouncedQuery(searchQuery);
    }, 250);
    return () => window.clearTimeout(timer);
  }, [searchQuery]);

  useEffect(() => {
    let cancelled = false;
    fetch('/api/bootstrap')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<BootstrapResponse>;
      })
      .then((data) => {
        if (cancelled) return;
        setBootstrap(data);
        if (data.medicationNames.length > 0) {
          const preferred =
            data.medicationNames.find((n) => n.toLowerCase() === 'tretinoin') ||
            data.medicationNames[0];
          setSelectedName(preferred);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setBootstrapError(err instanceof Error ? err.message : 'Failed to load dataset index');
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!bootstrap) return;
    const effectiveQuery = debouncedQuery.trim();
    if (!explicitDrugId && !effectiveQuery && !selectedName) return;

    let cancelled = false;
    setLoadingDrug(true);
    const params = new URLSearchParams({
      query: effectiveQuery,
      selectedName,
      drugId: explicitDrugId,
      timing: selectedTiming,
      target: targetLabel,
    });

    fetch(`/api/drug?${params.toString()}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<DrugApiResponse>;
      })
      .then((data) => {
        if (!cancelled) {
          setDrugData(data);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setDrugData(null);
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingDrug(false);
      });

    return () => {
      cancelled = true;
    };
  }, [bootstrap, debouncedQuery, selectedName, explicitDrugId, selectedTiming, targetLabel]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h2>Search</h2>

        <div className="field-group">
          <label className="field-label" htmlFor="drug-search-input">
            Drug name or identifier
          </label>
          <input
            id="drug-search-input"
            className="text-input"
            type="text"
            value={searchQuery}
            onChange={(e) => {
              setExplicitDrugId('');
              setSearchQuery(e.target.value);
            }}
            placeholder="e.g. tretinoin, abacavir, adalimumab, 444795"
          />
          <div className="field-help">
            Matches standardized/brand names, salts, combination ingredients, drug_id, PubChem CID, or InChIKey.
          </div>
        </div>

        <div className="field-group">
          <label className="field-label" htmlFor="medication-select">
            Select Indexed Medication
          </label>
          <select
            id="medication-select"
            className="select-input"
            value={selectedName}
            onChange={(e) => {
              setExplicitDrugId('');
              setSearchQuery('');
              setSelectedName(e.target.value);
            }}
            disabled={!bootstrap || bootstrap.medicationNames.length === 0}
          >
            {(bootstrap?.medicationNames ?? []).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>

        <div className="field-group">
          <label className="field-label" htmlFor="timing-select">
            Focus Timing View
          </label>
          <select
            id="timing-select"
            className="select-input"
            value={selectedTiming}
            onChange={(e) => setSelectedTiming(e.target.value)}
          >
            {TIMING_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </div>

        <div className="field-group">
          <span className="field-label">Outcome Focus</span>
          <div className="radio-group">
            {(['Fetal Outcomes', 'Maternal Outcomes'] as const).map((opt) => (
              <label className="radio-option" key={opt}>
                <input
                  type="radio"
                  name="view-target"
                  checked={targetLabel === opt}
                  onChange={() => setTargetLabel(opt)}
                />
                <span>{opt}</span>
              </label>
            ))}
          </div>
        </div>

        <hr className="sidebar-divider" />
        <h2>6-Tier Screening Hierarchy</h2>
        <p className="caption">
          1. Direct human pregnancy → 2. Pregnancy ADR &amp; TGA → 3. General SIDER/FAERS ADRs → 4.
          Tanimoto chemical analogues → 5. Animal nonclinical/CVM tox → 6. Molecular ML + &lt;20% AI
          fallback.
        </p>
      </aside>

      <main className="main-content">
        <div className="main-container">
          <h1 className="page-title">MEPRank Trimester Evidence &amp; Screening Explorer</h1>
          <p className="caption">
            Trimester-wise exploratory screening priority scores across all 3 trimesters using a
            6-tier evidence fallback hierarchy. Missing data is never treated as 0% risk or proof of safety.
          </p>
          <div className="alert alert-warning">
            Research-support prototype only. Numerical concern scores (8.0–98.0) are exploratory
            screening priority indices—not clinically validated human risk probabilities or prescribing
            guidance.
          </div>

          {bootstrapError && (
            <div className="alert alert-error">
              Processed drug data could not be loaded: {bootstrapError}
            </div>
          )}

          {!bootstrap && !bootstrapError && (
            <div className="alert alert-info">Loading local drug index &amp; evidence tables...</div>
          )}

          {bootstrap && drugData && drugData.mode === 'ambiguous' && (
            <div className="card-container">
              <div className="alert alert-warning">
                &apos;{drugData.query}&apos; matches multiple drug identifiers. Click a candidate below to
                inspect its trimester evaluation:
              </div>
              <div className="chip-row">
                {drugData.profile.candidates.map((cand) => (
                  <button
                    key={cand.drug_id}
                    type="button"
                    className="chip-btn"
                    onClick={() => setExplicitDrugId(cand.drug_id)}
                  >
                    {cand.name} ({cand.drug_id})
                  </button>
                ))}
              </div>
              {drugData.profile.candidates.length > 0 && (
                <DataTable
                  columns={Object.keys(drugData.profile.candidates[0])}
                  rows={drugData.profile.candidates as unknown as Array<Record<string, unknown>>}
                />
              )}
            </div>
          )}

          {bootstrap && drugData && drugData.mode === 'unmatched' && (
            <div className="card-container">
              <div className="alert alert-info">
                No indexed match found for &apos;{drugData.query}&apos;. Absence from local datasets does
                not imply pregnancy safety (no fabricated zero-risk score is shown).
              </div>
              {drugData.suggestions.length > 0 && (
                <>
                  <p>
                    <strong>Similar indexed drug names (click to load):</strong>
                  </p>
                  <div className="chip-row">
                    {drugData.suggestions.map((name) => (
                      <button
                        key={name}
                        type="button"
                        className="chip-btn"
                        onClick={() => {
                          setExplicitDrugId('');
                          setSearchQuery('');
                          setSelectedName(name);
                        }}
                      >
                        {name}
                      </button>
                    ))}
                  </div>
                </>
              )}
              {drugData.profile.warnings.map((w, i) => (
                <p className="caption" key={i}>
                  {w}
                </p>
              ))}
            </div>
          )}

          {bootstrap && drugData && drugData.mode === 'matched' && drugData.evaluation && (
            <>
              <div className="tabs-bar" role="tablist">
                {TABS.map((tab) => (
                  <button
                    key={tab}
                    type="button"
                    role="tab"
                    aria-selected={activeTab === tab}
                    className={`tab-btn${activeTab === tab ? ' active' : ''}`}
                    onClick={() => setActiveTab(tab)}
                  >
                    {tab}
                  </button>
                ))}
              </div>

              {loadingDrug && (
                <p className="caption">Computing 6-tier trimester evaluation...</p>
              )}

              {activeTab === 'Trimester Results & Evidence' && (
                <EvidenceTab
                  evaluation={drugData.evaluation}
                  selectedName={drugData.selectedName || selectedName}
                  targetLabel={targetLabel}
                  profile={drugData.profile}
                />
              )}
              {activeTab === 'Overview' && <OverviewTab profile={drugData.profile} />}
              {activeTab === 'Animal ADR' && <AnimalAdrTab evaluation={drugData.evaluation} />}
              {activeTab === 'Provenance & Records' && (
                <ProvenanceTab evaluation={drugData.evaluation} profile={drugData.profile} />
              )}
              {activeTab === 'Data & Model Readiness' && <ReadinessTab bootstrap={bootstrap} />}
            </>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
