# MEPRank: Evidence-Aware Trimester-Specific Pregnancy Drug-Safety Prioritization

**Evidence-aware multimodal prioritization of potential pregnancy-stage-specific drug-safety signals.**

> **Research & Screening Disclaimer:** MEPRank is an exploratory evidence-synthesis and signal-prioritization prototype. It is **not** a clinical decision-support or prescribing tool. Screening priority scores (`8.0–98.0`), confidence estimates, and nonclinical/analogue fallbacks are **not** calibrated clinical risk probabilities, incidence rates, or proof of safety or causality.

---

## Reframed Problem Statement

Clinical pregnancy datasets rarely contain large, patient-level, trimester-timed outcome cohorts with exposed/unexposed denominators. Rather than returning empty `Abstain` states or misrepresenting missing evidence as `0.0%` risk, MEPRank implements a **6-tier trimester-aware fallback and evidence-provenance framework**:

1. **Always scores all 3 core trimesters** (`T1` Weeks 0–13 organogenesis, `T2` Weeks 14–27 growth/maturation, `T3` Weeks 28+ late physiology & neonatal transition) for every resolved drug, emitting strictly non-zero screening priority scores (`8.0–98.0`) alongside separate maternal and fetal scores.
2. **Never treats missing evidence as zero risk**: absent tiers are excluded from weighted denominators (`effective_weight = 0`) and explicitly documented in per-trimester missing-evidence notes.
3. **Separates signal magnitude from evidence confidence**: reports `confidence_score` / `uncertainty_percent` and `Q_cov` evidence quality coverage independently of screening concern scores.
4. **Preserves strict evidentiary boundaries**: human pregnancy outcomes, regulatory pregnancy classifications, general human ADRs, structural chemical analogues, nonclinical animal assays/veterinary reports, and molecular ML/AI estimates are kept separated and explicitly labeled.
5. **Detects cross-tier conflicts**: flags disagreements (e.g., benign regulatory category vs. teratogenic ADR signals or animal toxicity vs. human records).

---

## 6-Tier Trimester Fallback Hierarchy (`server/scoringService.ts`)

Every resolved drug is evaluated across six explicit evidence tiers, weighted by source availability and stage relevance (`data/scoring_config.json`):

| Tier | Code | Evidence Family | Primary Local Sources | Separation & Safeguards |
|------|------|-----------------|------------------------|-------------------------|
| **Tier 1** | `H` | Human Pregnancy-Specific Outcomes | `data/real_pregnancy_outcomes.csv`, `data/real_pregnancy_features.csv` | Aggregate spontaneous-report / study counts; gestational timing preserved when present, never silently fabricated as timed when absent |
| **Tier 2** | `P` | Pregnancy ADR & Regulatory Classifications | `data/pregnancy_classifications.csv` (TGA A/B1/B2/B3/C/D/X), `data/real_faers_summary.csv`, pregnancy-linked MedDRA terms | Regulatory categories and FAERS volume summaries kept distinct from patient-level incidence |
| **Tier 3** | `A` | General Human ADRs | `datasets/meddra_all_se.tsv.gz` (SIDER), `data/real_faers_adr.csv` | Systemic human ADR terms screened via keyword & organ-system mapping; clearly marked non-pregnancy-specific |
| **Tier 4** | `S` | Chemical Structural Analogues | OpenChemLib 512-bit fingerprints (`data/analogue_fingerprints.json`) | Tanimoto similarity `>= 0.20` to drugs with pregnancy/toxicity records; explicitly marked analogue-only |
| **Tier 5** | `T` | Animal Developmental/Reproductive Toxicity & Veterinary ADRs | `data/real_toxicity_evidence.csv` (EPA ToxRefDB), `data/real_toxicity.csv`, `data/animal_adr_features.csv` (FDA CVM ADAE) | Nonclinical signal only (`is_human_evidence = false`); never promoted to human outcome labels |
| **Tier 6** | `M` | Molecular Placental-Transfer ML & AI Gap Reasoning | OpenChemLib descriptors (MW, cLogP, TPSA, HBD, HBA, rotatable bonds, rings) + `datasets/drug.target.interaction.tsv.gz` | Triggers **supplementary AI gap reasoning** (`ai_supplementary_score`) whenever ML screening score `< 20%` or SMILES/target data is incomplete |

### Missingness & Confidence Rules
- **No `Abstain` / No `0.0%` Missingness**: Every matched query produces `T1`, `T2`, and `T3` concern scores (`> 0`), maternal/fetal scores, priority bands, evidence categories, supporting evidence lists, and missing-evidence notes.
- **Evidence Quality (`Q_cov`)**: Computed per tracked source family as `0.5 × completeness + 0.5 × volume` (`log1p(count) / log1p(reference_volume)`, capped at `1`), averaged over observed sources.
- **Unmatched & Ambiguous Queries**: Unresolved or multi-candidate queries surface explicit candidate suggestions or ambiguity notices without inventing drug identifiers or scores.

---

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| TypeScript/Express REST API & serverless handler | ✅ Implemented | `server/app.ts`, `server.ts`, `api/index.ts` |
| React 19 + Vite 6 interactive dashboard | ✅ Implemented | `client/App.tsx` and tabbed views (`Overview`, `Evidence`, `Animal ADR`, `Provenance`, `Readiness`) |
| Multi-identifier & salt/combination drug matching | ✅ Implemented | Name, `drug_id`, PubChem CID, InChIKey, canonical SMILES, salt-stripped active ingredients, combination splitting (`server/drugService.ts`) |
| Pure-JS cheminformatics & analogue cache | ✅ Implemented | OpenChemLib 2D SVG rendering, physicochemical descriptors, 512-bit Tanimoto similarity (`server/chemistry.ts`, `data/analogue_fingerprints.json`) |
| 6-tier trimester fallback + `< 20%` ML fallback AI reasoning | ✅ Implemented | `server/scoringService.ts` |
| Cross-tier conflict detection & provenance tracking | ✅ Implemented | `detectConflicts()`, per-tier availability/weight/human-evidence flags |
| Exploratory OECD nonclinical guideline CV benchmarks | ⚠️ Exploratory Only | 5-fold Bemis–Murcko scaffold GroupKFold on ToxRefDB `TG414`–`TG443t` (`data/reproductive_tox_training_metrics.json`, AUROC `0.514–0.707`) — not human pregnancy metrics |
| Automated Node integration & serverless tests | ✅ Implemented | `npm test` (`server/pipeline.test.ts`, 11 test cases) |
| Supervised clinical pregnancy model (Dirichlet evidential DL, multi-task neural heads, human AUROC/AUPRC/ECE) | ❌ Blocked / Future Work | Requires patient-level, trimester-timed human pregnancy cohort data (`data/pregnancy_labels.csv` absent; see `NEXT_STEPS_RESEARCH_AUDIT.md`) |
| ROR/PRR disproportionality ratios | ❌ Blocked | Acquired aggregate FAERS/Mendeley tables lack case/non-case exposed-pregnancy denominators |

---

## Project Structure

```text
.
├── api/
│   └── index.ts                  # Vercel serverless Express entry point
├── client/
│   ├── App.tsx                   # Main React application shell & drug query controls
│   ├── main.tsx                  # React DOM entry point
│   ├── styles.css                # Dashboard styling
│   ├── types.ts                  # Shared TypeScript API/UI interfaces
│   └── components/
│       ├── OverviewTab.tsx       # 2D structure, descriptors, T1/T2/T3 scores, tier breakdown, ML/AI notes
│       ├── EvidenceTab.tsx       # Human pregnancy, regulatory, SIDER/FAERS ADRs, chemical analogues
│       ├── AnimalAdrTab.tsx      # Separated nonclinical toxicity & FDA CVM veterinary ADR panel
│       ├── ProvenanceTab.tsx     # Source completeness, Q_cov breakdown, conflict flags, missing-source notes
│       ├── ReadinessTab.tsx      # Dataset feasibility audit & exploratory OECD nonclinical benchmark metrics
│       ├── BarChart.tsx
│       ├── DataTable.tsx
│       └── Expander.tsx
├── server/
│   ├── app.ts                    # Express API routes (/api/health, /api/bootstrap, /api/drug)
│   ├── dataLoader.ts             # In-memory loader for CSV/JSON/gzipped TSV datasets + fingerprint cache
│   ├── drugService.ts            # Identifier matching, salt/combo normalization, drug profile assembly
│   ├── chemistry.ts              # OpenChemLib SVG structure, descriptors, 512-bit fingerprints, Tanimoto
│   ├── scoringService.ts         # 6-tier trimester scoring, <20% ML fallback, confidence, conflicts
│   ├── buildAnalogueCache.ts     # Script to precompute data/analogue_fingerprints.json
│   └── pipeline.test.ts          # 11 integration & serverless verification tests
├── server.ts                     # Local Node HTTP server (Vite middleware in dev, dist/ static in prod)
├── data/                         # Processed evidence tables, scoring config, feasibility JSON, fingerprint cache
├── datasets/                     # Raw/compressed reference tables (SIDER meddra_all_se.tsv.gz, drug-target TSV, etc.)
├── docs/                         # ARCHITECTURE.md, CASE_STUDIES.md
├── reports/                      # Feasibility, evidence inventory, animal ADR, and trimester evaluation reports
├── vercel.json                   # Vercel build, static output, serverless includeFiles & rewrites
├── vite.config.ts                # Vite build configuration (outputs to dist/)
└── package.json
```

---

## Local Setup & Commands

**Prerequisites:** Node.js `>= 22` and `npm` (or `bun`).

```bash
# 1. Install dependencies
npm install

# 2. Start local full-stack development server (Express API + Vite frontend on http://localhost:3000)
npm run dev
```

### Verification, Build & Maintenance Scripts

```bash
# Type-check all client, server, and API TypeScript code
npm run lint

# Run all 11 end-to-end pipeline & serverless handler tests
npm test

# Build production frontend bundle to dist/
npm run build

# Start production Express server serving dist/ and /api/*
npm start

# Regenerate data/analogue_fingerprints.json (1,173 OpenChemLib 512-bit fingerprints) if drug_mapping.csv changes
npm run build:cache
```

### API Endpoints

- `GET /api/health` — Returns `{ "ok": true }`.
- `GET /api/bootstrap` — Returns searchable medication list (`medicationNames`), dataset feasibility summary (`feasibility`), exploratory OECD nonclinical metrics (`reproductiveToxMetrics`), and file availability metadata.
- `GET /api/drug?query=<term>&timing=<window>&target=<fetal|maternal>` — Resolves by drug name, `drug_id`, PubChem CID, InChIKey, SMILES, salt-stripped ingredient, or combination component and returns `{ mode, profile, evaluation, suggestions }`. Supports `timing` (`First Trimester (T1)`, `Second Trimester (T2)`, `Third Trimester (T3)`, `Entire Pregnancy`, `All Trimesters`) and `target` (`Fetal Outcomes`, `Maternal Outcomes`).

---

## Deploying to Vercel

This repository is pre-configured for zero-config Vercel deployment:

- **`vercel.json`** builds the Vite client (`npm run build` -> `dist/`), routes `/api/(.*)` to the serverless Express function at `api/index.ts`, serves SPA routes via `/index.html`, and bundles required datasets via `includeFiles`:
  - `data/**` (processed CSV/JSON evidence tables and precomputed `data/analogue_fingerprints.json` for fast cold starts)
  - `datasets/meddra_all_se.tsv.gz`
  - `datasets/drug.target.interaction.tsv.gz`
- **Deployment steps:**
  1. Push the repository to GitHub (ensuring `data/` and `datasets/meddra_all_se.tsv.gz` / `datasets/drug.target.interaction.tsv.gz` are committed).
  2. Import the repository into Vercel and deploy with default settings (`vercel.json` handles build, rewrites, and function bundling).

---

## Data Sources & Scientific Limitations

- **Processed & Compressed Sources**:
  - `data/drug_mapping.csv` (1,726 medication records; 1,173 with PubChem CID & canonical SMILES)
  - `data/real_pregnancy_outcomes.csv` (2,083 rows) & `data/real_pregnancy_features.csv` (67 drugs)
  - `data/pregnancy_classifications.csv` (1,778 TGA classification/safety rows)
  - `data/real_faers_summary.csv` (75 drugs) & `data/real_faers_adr.csv` (685 drugs)
  - `data/real_toxicity_evidence.csv` (184 EPA ToxRefDB rows), `data/real_toxicity.csv`, & `data/animal_adr_features.csv` (158 drugs derived from FDA CVM ADAE)
  - `datasets/meddra_all_se.tsv.gz` (SIDER side-effect terms) & `datasets/drug.target.interaction.tsv.gz` (19,378 drug-target rows)
- **Key Limitations**:
  - **Spontaneous Reporting Bias**: FAERS, OnSIDES/SIDER, Mendeley supplements, and FDA CVM ADAE reflect voluntary/label reporting with stimulated reporting, under-reporting, and confounding by indication. Counts are not incidence.
  - **No Clinical Denominators**: Valid 2×2 tables for ROR/PRR cannot be computed from these aggregate counts without exposed/unexposed pregnancy denominators.
  - **Timing Granularity**: Most pharmacovigilance and regulatory sources omit trimester labels; `T1`/`T2`/`T3` profiles reflect endpoint stage relevance (`data/scoring_config.json`) unless direct gestational timing is recorded.
  - **Nonclinical & Analogue Boundaries**: Animal assays (`Tier 5`) and structural Tanimoto analogues (`Tier 4`) support hypothesis generation only and never confirm human fetal or maternal safety or harm.
  - **Research Roadmap**: See `NEXT_STEPS_RESEARCH_AUDIT.md`, `docs/CASE_STUDIES.md`, and `reports/` for detailed audits and gap analyses.
