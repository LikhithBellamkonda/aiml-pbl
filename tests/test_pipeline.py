import pytest
import torch
import pandas as pd
import numpy as np
from src.data_preprocessing import get_dataloader, PregnancyDataset
from src.models.framework import MultimodalPregnancyADRModel
from src.models.fusion import SourceGatedFusion

@pytest.fixture
def dummy_data():
    return 'data/mock_faers_adr.csv', 'data/mock_pubchem.csv', 'data/mock_toxicity.csv'

def test_dataset_loading(dummy_data):
    adr, pub, tox = dummy_data
    dataset = PregnancyDataset(adr, pub, tox)

    assert len(dataset) == 200

    item = dataset[0]
    # Check that item is a PyTorch Geometric Data object with custom fields
    assert hasattr(item, 'x')
    assert hasattr(item, 'edge_index')
    assert hasattr(item, 'adr_features')
    assert hasattr(item, 'y')
    assert hasattr(item, 'y_mask')
    assert hasattr(item, 'availability_mask')

def test_dataloader(dummy_data):
    adr, pub, tox = dummy_data
    loader = get_dataloader(adr, pub, tox, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    assert batch.adr_features.shape == (4, 20)
    assert batch.y.shape == (4, 1)
    assert hasattr(batch, 'tox_features')
    assert batch.tox_features.shape == (4, 3)
    assert batch.availability_mask.shape == (4, 3)

def test_source_gated_fusion():
    fusion = SourceGatedFusion(rep_dim=128)

    batch_size = 2
    h_adr = torch.randn(batch_size, 128)
    h_mol = torch.randn(batch_size, 128)
    h_tox = torch.randn(batch_size, 128)

    # Simulate mask where first item has all, second item missing ADR
    mask = torch.tensor([[1., 1., 1.],
                         [0., 1., 1.]])

    out = fusion(h_adr, h_mol, h_tox, mask)

    assert out.shape == (batch_size, 128)
    # the second item's output shouldn't have nan
    assert not torch.isnan(out).any()

def test_full_model_forward(dummy_data):
    adr, pub, tox = dummy_data
    loader = get_dataloader(adr, pub, tox, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    model = MultimodalPregnancyADRModel(num_outcomes=1)

    out = model(batch)
    assert out.shape == (4, 1)
    assert (out >= 0).all() and (out <= 1).all() # Sigmoid output

def test_uncertainty_estimation(dummy_data):
    adr, pub, tox = dummy_data
    loader = get_dataloader(adr, pub, tox, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    model = MultimodalPregnancyADRModel(num_outcomes=1)

    mean_pred, var_pred = model.predict_with_uncertainty(batch, num_samples=5)

    assert mean_pred.shape == (2, 1)
    assert var_pred.shape == (2, 1)
    assert (var_pred >= 0).all() # Variance should be positive
