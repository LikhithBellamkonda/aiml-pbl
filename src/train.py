import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from src.data_preprocessing import get_dataloader
from src.models.framework import MultimodalPregnancyADRModel
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def train_model():
    # 1. Setup Data
    loader = get_dataloader(
        'data/mock_faers_adr.csv',
        'data/mock_pubchem.csv',
        'data/mock_toxicity.csv',
        batch_size=16,
        shuffle=True
    )

    # 2. Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalPregnancyADRModel(num_outcomes=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Using Binary Cross Entropy.
    # Notice we don't use reduction='mean' directly because we need to apply the mask first.
    criterion = nn.BCELoss(reduction='none')

    epochs = 5

    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_masks = []

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward
            preds = model(batch)

            # Masked Loss Calculation
            # The mask specifies if a label is available (1) or missing (0).
            loss = criterion(preds, batch.y)
            masked_loss = loss * batch.y_mask

            # Average loss only over available labels
            num_valid = batch.y_mask.sum()
            if num_valid > 0:
                final_loss = masked_loss.sum() / num_valid
                final_loss.backward()
                optimizer.step()
                total_loss += final_loss.item()

            # For metrics
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
            all_masks.append(batch.y_mask.cpu().numpy())

        # Calculate AUROC
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        all_masks = np.vstack(all_masks)

        auc_scores = []
        for i in range(1): # 1 target
            valid_idx = all_masks[:, i] == 1
            if valid_idx.sum() > 0 and len(np.unique(all_targets[valid_idx, i])) > 1:
                auc = roc_auc_score(all_targets[valid_idx, i], all_preds[valid_idx, i])
                auc_scores.append(auc)

        mean_auc = np.mean(auc_scores) if auc_scores else 0.0

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f} | Mean AUROC: {mean_auc:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    train_model()
