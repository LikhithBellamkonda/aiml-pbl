import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import json

def generate_mock_data():
    np.random.seed(42)
    os.makedirs('data', exist_ok=True)

    num_drugs = 200

    # Generate some realistic-looking SMILES
    base_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O", # Aspirin
        "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C", # Penicillin G
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
        "CC(C)(C)NCC(O)C1=CC(=C(O)C=C1)CO", # Albuterol
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C" # Testosterone
    ]

    # Expand to num_drugs by slightly mutating or repeating
    smiles_list = []
    for i in range(num_drugs):
        smiles = base_smiles[i % len(base_smiles)]
        # Add a random simple group to make them unique
        if i >= len(base_smiles):
            smiles = smiles + "C" * (i % 3)
        smiles_list.append(smiles)

    drug_ids = [f"DRUG_{i:04d}" for i in range(num_drugs)]

    # 1. FAERS / General ADR Data
    # 20 general side effects
    adr_features = np.random.rand(num_drugs, 20)
    # Simulate some drugs missing ADR data (e.g. 10%)
    missing_adr_mask = np.random.rand(num_drugs) < 0.1
    adr_features[missing_adr_mask] = 0

    adr_df = pd.DataFrame(adr_features, columns=[f"ADR_{i}" for i in range(20)])
    adr_df['drug_id'] = drug_ids
    adr_df['has_adr_data'] = (~missing_adr_mask).astype(int)

    # Add pregnancy reports (missingness indicator)
    # 0 reports is common
    pregnancy_reports = np.random.poisson(lam=0.5, size=num_drugs)
    adr_df['n_pregnancy_reports'] = pregnancy_reports

    adr_df.to_csv('data/mock_faers_adr.csv', index=False)

    # 2. PubChem (SMILES)
    pubchem_df = pd.DataFrame({
        'drug_id': drug_ids,
        'smiles': smiles_list
    })
    # Simulate some invalid/missing SMILES
    missing_smiles_mask = np.random.rand(num_drugs) < 0.05
    pubchem_df.loc[missing_smiles_mask, 'smiles'] = None
    pubchem_df.to_csv('data/mock_pubchem.csv', index=False)

    # 3. ToxCast/Tox21 (Developmental Toxicity Labels)
    # Binary labels for 3 toxicity endpoints
    tox_labels = np.random.randint(0, 2, size=(num_drugs, 3)).astype(float)
    # Simulate missing labels (e.g., 50% missing)
    missing_tox_mask = np.random.rand(num_drugs, 3) < 0.5
    tox_labels[missing_tox_mask] = np.nan

    tox_df = pd.DataFrame(tox_labels, columns=['Tox_Dev', 'Tox_Repro', 'Tox_Teratogen'])
    tox_df['drug_id'] = drug_ids

    # Create mask column for whether any tox data exists
    tox_df['has_tox_data'] = (~missing_tox_mask.all(axis=1)).astype(int)
    tox_df.to_csv('data/mock_toxicity.csv', index=False)

    print("Mock data generated successfully in data/")

if __name__ == "__main__":
    generate_mock_data()
