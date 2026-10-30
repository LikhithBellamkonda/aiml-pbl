import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import os

class PregnancyDataset(Dataset):
    def __init__(self, adr_path, pubchem_path, tox_path):
        # Load datasets
        self.adr_df = pd.read_csv(adr_path).set_index('drug_id')
        self.pubchem_df = pd.read_csv(pubchem_path).set_index('drug_id')
        self.tox_df = pd.read_csv(tox_path).set_index('drug_id')

        # Merge all on drug_id
        self.merged_df = self.adr_df.join(self.pubchem_df).join(self.tox_df)
        self.drug_ids = self.merged_df.index.tolist()

        # Define target outcomes (using toxicity endpoints as proxies for pregnancy outcomes)
        self.target_cols = ['Tox_Dev', 'Tox_Repro', 'Tox_Teratogen']

    def _smiles_to_graph(self, smiles):
        if pd.isna(smiles) or not isinstance(smiles, str):
            return None, None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None

        # 1. 2D Descriptors
        desc_list = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol)
        ]

        # 2. Molecular Graph for GNN
        node_features = []
        for atom in mol.GetAtoms():
            # Basic atom features: atomic num, degree, formal charge
            feat = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                float(atom.GetIsAromatic())
            ]
            node_features.append(feat)

        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Directed edges for PyG (i->j and j->i)
            edge_index += [[i, j], [j, i]]

            # Bond features: bond type as float
            b_type = float(bond.GetBondTypeAsDouble())
            edge_attr += [[b_type], [b_type]]

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.empty((0, 1), dtype=torch.float)

        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return graph_data, torch.tensor(desc_list, dtype=torch.float)

    def __len__(self):
        return len(self.drug_ids)

    def __getitem__(self, idx):
        row = self.merged_df.iloc[idx]

        # 1. ADR Features
        adr_cols = [c for c in row.index if c.startswith('ADR_')]
        adr_features = torch.tensor(row[adr_cols].values.astype(float), dtype=torch.float)

        # 2. Molecular Features
        smiles = row['smiles']
        mol_graph, mol_desc = self._smiles_to_graph(smiles)

        # If molecular processing failed or SMILES missing, create dummy tensors
        if mol_graph is None:
            mol_graph = Data(x=torch.zeros((1, 4)), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, 1), dtype=torch.float))
            mol_desc = torch.zeros(6, dtype=torch.float)
            has_mol_data = 0.0
        else:
            has_mol_data = 1.0

        # 3. Toxicity Data (Evidence source)
        tox_features = row[self.target_cols].values.astype(float)
        # Handle NaNs
        tox_mask_arr = ~np.isnan(tox_features)
        tox_features = np.nan_to_num(tox_features)

        tox_feat_tensor = torch.tensor(tox_features, dtype=torch.float)

        # 4. Target Label (Pregnancy ADR Signal)
        # We now have real target labels from the Pregnancy database via Actual_Pregnancy_Risk
        actual_risk = row.get('Actual_Pregnancy_Risk', np.nan)
        if pd.isna(actual_risk):
            y = torch.tensor([[0.0]], dtype=torch.float)
            y_mask = torch.tensor([[0.0]], dtype=torch.float) # Not available
        else:
            y = torch.tensor([[float(actual_risk)]], dtype=torch.float)
            y_mask = torch.tensor([[1.0]], dtype=torch.float) # Available

        # 5. Availability Masks [ADR, Mol, Tox]
        has_adr = float(row['has_adr_data'])
        has_tox = float(row['has_tox_data'])

        availability_mask = torch.tensor([has_adr, has_mol_data, has_tox], dtype=torch.float)

        # Pack everything into a dictionary to return along with graph
        mol_graph.adr_features = adr_features.unsqueeze(0)
        mol_graph.mol_desc = mol_desc.unsqueeze(0)
        mol_graph.tox_features = tox_feat_tensor.unsqueeze(0)
        mol_graph.y = y
        mol_graph.y_mask = y_mask
        mol_graph.availability_mask = availability_mask.unsqueeze(0)

        return mol_graph

def get_dataloader(adr_path, pubchem_path, tox_path, batch_size=32, shuffle=True):
    from torch_geometric.loader import DataLoader as GeometricDataLoader
    dataset = PregnancyDataset(adr_path, pubchem_path, tox_path)
    loader = GeometricDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
