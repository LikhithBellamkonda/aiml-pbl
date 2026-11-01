from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


ATOM_FEATURES = 29
BOND_FEATURES = 6
DESCRIPTOR_FEATURES = 6
PREGNANCY_FEATURES = 4
FAERS_FEATURES = 1
OUTCOME_NAMES = (
    "fetal_death",
    "preterm_birth",
    "fetal_growth_restriction",
    "congenital_anomaly",
)
LABEL_PROVENANCE_COLUMNS = (
    "source_dataset",
    "source_reference",
    "outcome_definition",
    "evidence_type",
)


@dataclass
class DrugSample:
    drug_id: str
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    mol_desc: torch.Tensor
    adr_features: torch.Tensor
    pregnancy_features: torch.Tensor
    faers_features: torch.Tensor
    availability_mask: torch.Tensor
    evidence_quality: torch.Tensor
    y: torch.Tensor
    y_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "DrugSample":
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                setattr(self, field, value.to(device))
        return self


@dataclass
class DrugBatch:
    drug_ids: list[str]
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor
    mol_desc: torch.Tensor
    adr_features: torch.Tensor
    pregnancy_features: torch.Tensor
    faers_features: torch.Tensor
    availability_mask: torch.Tensor
    evidence_quality: torch.Tensor
    y: torch.Tensor
    y_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "DrugBatch":
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                setattr(self, field, value.to(device))
        return self


def _read_indexed(path: str | Path, source: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "drug_id" not in frame.columns:
        raise ValueError(f"{source} dataset must contain a 'drug_id' column: {path}")
    frame["drug_id"] = frame["drug_id"].astype(str)
    if frame["drug_id"].duplicated().any():
        duplicates = frame.loc[frame["drug_id"].duplicated(), "drug_id"].head(5).tolist()
        raise ValueError(f"{source} dataset has duplicate drug_id values: {duplicates}")
    return frame.set_index("drug_id")


def _flag(row: pd.Series, column: str) -> bool:
    value = row.get(column, False)
    return False if pd.isna(value) else bool(value)


def _smiles_to_graph(smiles: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        return None

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None

    atom_numbers = (1, 6, 7, 8, 9, 15, 16, 17, 35, 53)
    atom_rows: list[list[float]] = []
    for atom in molecule.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        features = [float(atomic_number == number) for number in atom_numbers]
        features.append(float(atomic_number not in atom_numbers))
        features.extend(float(atom.GetDegree() == degree) for degree in range(6))
        features.extend(float(atom.GetFormalCharge() == charge) for charge in range(-2, 3))
        features.append(float(atom.GetIsAromatic()))
        features.append(float(atom.IsInRing()))
        features.extend(
            float(atom.GetHybridization() == hybridization)
            for hybridization in (
                Chem.HybridizationType.SP,
                Chem.HybridizationType.SP2,
                Chem.HybridizationType.SP3,
                Chem.HybridizationType.SP3D,
                Chem.HybridizationType.SP3D2,
            )
        )
        atom_rows.append(features)

    bond_rows: list[list[float]] = []
    edge_pairs: list[list[int]] = []
    bond_types = (
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    )
    for bond in molecule.GetBonds():
        bond_features = [float(bond.GetBondType() == kind) for kind in bond_types]
        bond_features.extend((float(bond.GetIsConjugated()), float(bond.IsInRing())))
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_pairs.extend(([start, end], [end, start]))
        bond_rows.extend((bond_features, bond_features))

    descriptors = torch.tensor(
        [
            Descriptors.MolWt(molecule) / 500.0,
            Descriptors.MolLogP(molecule) / 10.0,
            Descriptors.TPSA(molecule) / 200.0,
            Descriptors.NumHDonors(molecule) / 10.0,
            Descriptors.NumHAcceptors(molecule) / 15.0,
            Descriptors.NumRotatableBonds(molecule) / 20.0,
        ],
        dtype=torch.float32,
    )
    x = torch.tensor(atom_rows, dtype=torch.float32)
    edge_index = (
        torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        if edge_pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_attr = (
        torch.tensor(bond_rows, dtype=torch.float32)
        if bond_rows
        else torch.empty((0, BOND_FEATURES), dtype=torch.float32)
    )
    return x, edge_index, edge_attr, descriptors


class PregnancyDataset(Dataset[DrugSample]):
    """Loads source evidence without treating missing outcomes as negative labels."""

    def __init__(
        self,
        adr_path: str | Path,
        pubchem_path: str | Path,
        pregnancy_path: str | Path | None = None,
        labels_path: str | Path | None = None,
        faers_path: str | Path | None = None,
    ) -> None:
        self.adr_df = _read_indexed(adr_path, "ADR")
        self.pubchem_df = _read_indexed(pubchem_path, "PubChem")
        self.pregnancy_df = (
            _read_indexed(pregnancy_path, "pregnancy evidence")
            if pregnancy_path and Path(pregnancy_path).exists()
            else pd.DataFrame()
        )
        self.labels_df = (
            _read_indexed(labels_path, "pregnancy labels")
            if labels_path and Path(labels_path).exists()
            else pd.DataFrame()
        )
        self.faers_df = (
            _read_indexed(faers_path, "FAERS evidence")
            if faers_path and Path(faers_path).exists()
            else pd.DataFrame()
        )
        self.merged_df = self.adr_df.join(self.pubchem_df, how="outer", rsuffix="_pubchem")
        if not self.pregnancy_df.empty:
            self.merged_df = self.merged_df.join(
                self.pregnancy_df, how="outer", rsuffix="_pregnancy"
            )
        if not self.faers_df.empty:
            self.merged_df = self.merged_df.join(self.faers_df, how="outer", rsuffix="_faers")
        if not self.labels_df.empty:
            self.merged_df = self.merged_df.join(self.labels_df, how="outer", rsuffix="_labels")
        self.drug_ids = self.merged_df.index.astype(str).tolist()
        self.adr_cols = sorted(
            (column for column in self.merged_df if column.startswith("ADR_")),
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )[:20]
        self.pregnancy_cols = [f"reports_{name}" for name in OUTCOME_NAMES]
        self.label_cols = [f"label_{name}" for name in OUTCOME_NAMES]
        self.mask_cols = [f"mask_{name}" for name in OUTCOME_NAMES]

    def __len__(self) -> int:
        return len(self.drug_ids)

    @staticmethod
    def _number(row: pd.Series, column: str, default: float = 0.0) -> float:
        value = row.get(column, default)
        return default if pd.isna(value) else float(value)

    def __getitem__(self, idx: int) -> DrugSample:
        row = self.merged_df.iloc[idx]
        graph = _smiles_to_graph(row.get("smiles"))
        if graph is None:
            x = torch.empty((0, ATOM_FEATURES), dtype=torch.float32)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, BOND_FEATURES), dtype=torch.float32)
            mol_desc = torch.zeros(DESCRIPTOR_FEATURES, dtype=torch.float32)
            has_molecule = 0.0
        else:
            x, edge_index, edge_attr, mol_desc = graph
            has_molecule = 1.0

        adr_values = [self._number(row, column) for column in self.adr_cols]
        adr_values.extend([0.0] * (20 - len(adr_values)))
        adr_available = _flag(row, "has_adr_data")
        pregnancy_values = [self._number(row, column) for column in self.pregnancy_cols]
        pregnancy_available = _flag(row, "has_pregnancy_outcome_data")
        faers_value = self._number(row, "faers_report_count")
        faers_available = _flag(row, "has_faers_data")

        # Source quality is intentionally unknown unless a dataset provides a measured value.
        quality_values = [
            self._number(row, f"quality_{name}")
            for name in ("adr", "descriptors", "graph", "pregnancy", "faers")
        ]

        labels: list[float] = []
        label_masks: list[float] = []
        for name, label_col, mask_col in zip(OUTCOME_NAMES, self.label_cols, self.mask_cols):
            label = row.get(label_col, np.nan)
            supplied_mask = row.get(mask_col, np.nan)
            is_observed = (
                not pd.isna(label)
                and not pd.isna(supplied_mask)
                and bool(supplied_mask)
                and float(label) in (0.0, 1.0)
            )
            labels.append(float(label) if is_observed else 0.0)
            label_masks.append(float(is_observed))

        return DrugSample(
            drug_id=self.drug_ids[idx],
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mol_desc=mol_desc,
            adr_features=torch.tensor(adr_values, dtype=torch.float32),
            pregnancy_features=torch.tensor(pregnancy_values, dtype=torch.float32),
            faers_features=torch.tensor([faers_value], dtype=torch.float32),
            availability_mask=torch.tensor(
                [
                    float(adr_available),
                    has_molecule,
                    has_molecule,
                    float(pregnancy_available),
                    float(faers_available),
                ],
                dtype=torch.float32,
            ),
            evidence_quality=torch.tensor(quality_values, dtype=torch.float32),
            y=torch.tensor(labels, dtype=torch.float32),
            y_mask=torch.tensor(label_masks, dtype=torch.float32),
        )


def collate_drugs(samples: list[DrugSample]) -> DrugBatch:
    if not samples:
        raise ValueError("Cannot create a batch from an empty sample list")
    node_features = []
    edge_indices = []
    edge_features = []
    node_batches = []
    offset = 0
    for batch_idx, sample in enumerate(samples):
        node_features.append(sample.x)
        node_batches.append(torch.full((sample.x.shape[0],), batch_idx, dtype=torch.long))
        if sample.edge_index.numel():
            edge_indices.append(sample.edge_index + offset)
            edge_features.append(sample.edge_attr)
        offset += sample.x.shape[0]

    return DrugBatch(
        drug_ids=[sample.drug_id for sample in samples],
        x=torch.cat(node_features, dim=0),
        edge_index=(
            torch.cat(edge_indices, dim=1)
            if edge_indices
            else torch.empty((2, 0), dtype=torch.long)
        ),
        edge_attr=(
            torch.cat(edge_features, dim=0)
            if edge_features
            else torch.empty((0, BOND_FEATURES), dtype=torch.float32)
        ),
        batch=torch.cat(node_batches),
        mol_desc=torch.stack([sample.mol_desc for sample in samples]),
        adr_features=torch.stack([sample.adr_features for sample in samples]),
        pregnancy_features=torch.stack([sample.pregnancy_features for sample in samples]),
        faers_features=torch.stack([sample.faers_features for sample in samples]),
        availability_mask=torch.stack([sample.availability_mask for sample in samples]),
        evidence_quality=torch.stack([sample.evidence_quality for sample in samples]),
        y=torch.stack([sample.y for sample in samples]),
        y_mask=torch.stack([sample.y_mask for sample in samples]),
    )


def get_dataloader(
    adr_path: str | Path,
    pubchem_path: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    pregnancy_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    faers_path: str | Path | None = None,
) -> DataLoader[DrugBatch]:
    dataset = PregnancyDataset(
        adr_path,
        pubchem_path,
        pregnancy_path=pregnancy_path,
        labels_path=labels_path,
        faers_path=faers_path,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_drugs,
    )
