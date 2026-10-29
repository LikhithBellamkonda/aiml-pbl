import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import os
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from src.data_preprocessing import get_dataloader, PregnancyDataset
from src.models.framework import MultimodalPregnancyADRModel
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)
            loss = criterion(preds, batch.y)
            masked_loss = loss * batch.y_mask

            num_valid = batch.y_mask.sum()
            if num_valid > 0:
                final_loss = masked_loss.sum() / num_valid
                total_loss += final_loss.item()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
            all_masks.append(batch.y_mask.cpu().numpy())

    if len(all_preds) == 0:
        return 0.0, 0.0

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_masks = np.vstack(all_masks)

    auc_scores = []
    for i in range(1):
        valid_idx = all_masks[:, i] == 1
        if valid_idx.sum() > 0 and len(np.unique(all_targets[valid_idx, i])) > 1:
            auc = roc_auc_score(all_targets[valid_idx, i], all_preds[valid_idx, i])
            auc_scores.append(auc)

    mean_auc = np.mean(auc_scores) if auc_scores else 0.0
    mean_loss = total_loss / len(loader) if len(loader) > 0 else 0.0

    return mean_loss, mean_auc

def train_model():
    # 1. Setup Data
    dataset = PregnancyDataset(
        'data/real_faers_adr.csv',
        'data/real_pubchem.csv',
        'data/real_toxicity.csv'
    )

    # Check dataset size to handle tiny test datasets (like CSV.csv which has only 1 row)
    dataset_size = len(dataset)
    if dataset_size < 3:
        print("Warning: Dataset is extremely small. Using entire dataset for train/val/test to prevent errors.")
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    else:
        train_size = int(0.7 * dataset_size)
        val_size = int(0.15 * dataset_size)
        test_size = dataset_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 2. Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalPregnancyADRModel(num_outcomes=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Using Binary Cross Entropy.
    criterion = nn.BCELoss(reduction='none')

    epochs = 5
    best_val_auc = -1

    print(f"Starting training on {device} with {dataset_size} samples...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward
            preds = model(batch)

            # Masked Loss Calculation
            loss = criterion(preds, batch.y)
            masked_loss = loss * batch.y_mask

            num_valid = batch.y_mask.sum()
            if num_valid > 0:
                final_loss = masked_loss.sum() / num_valid
                final_loss.backward()
                optimizer.step()
                total_loss += final_loss.item()

        # Evaluate
        train_loss, train_auc = evaluate(model, train_loader, criterion, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

        # Save best model
        if val_auc >= best_val_auc or epoch == epochs - 1: # save last epoch if AUC is uncomputable
            best_val_auc = val_auc
            os.makedirs('src/models', exist_ok=True)
            torch.save(model.state_dict(), 'src/models/pregnancy_adr_model.pt')

    print("Training complete.")

    # Test Evaluation
    model.load_state_dict(torch.load('src/models/pregnancy_adr_model.pt', weights_only=True))
    test_loss, test_auc = evaluate(model, test_loader, criterion, device)
    print(f"Test Set Evaluation | Loss: {test_loss:.4f} | AUC: {test_auc:.4f}")

if __name__ == "__main__":
    train_model()
