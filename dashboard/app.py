import streamlit as st
import pandas as pd
import numpy as np
import torch
import sys
import os

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import get_dataloader
from src.models.framework import MultimodalPregnancyADRModel

@st.cache_resource
def load_data_and_model():
    # Load Model
    model = MultimodalPregnancyADRModel(num_outcomes=1)
    model.eval()

    # Load trained weights
    model_path = 'src/models/pregnancy_adr_model.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))

    # Load Dataloader
    loader = get_dataloader(
        'data/real_faers_adr.csv',
        'data/real_pubchem.csv',
        'data/real_toxicity.csv',
        batch_size=1, # One by one for dashboard
        shuffle=False
    )

    # Get metadata for the dashboard
    pubchem_df = pd.read_csv('data/real_pubchem.csv').set_index('drug_id')
    adr_df = pd.read_csv('data/real_faers_adr.csv').set_index('drug_id')

    return model, loader, pubchem_df, adr_df

st.title("Pregnancy ADR Research Dashboard")
st.markdown("Predicting potential pregnancy-related safety concerns and estimating uncertainty under sparse-evidence conditions.")

try:
    model, loader, pubchem_df, adr_df = load_data_and_model()

    drug_ids = list(pubchem_df.index)
    selected_drug = st.selectbox("Select a Drug ID", drug_ids)

    if selected_drug:
        st.subheader(f"Drug Analysis: {selected_drug}")

        col1, col2 = st.columns(2)
        with col1:
            smiles = pubchem_df.loc[selected_drug, 'smiles']
            st.markdown("**Canonical SMILES:**")
            st.code(smiles if pd.notna(smiles) else "N/A")

        with col2:
            n_reports = adr_df.loc[selected_drug, 'n_pregnancy_reports']
            st.markdown("**Available Evidence:**")
            st.write(f"- Pregnancy-Specific Reports: {int(n_reports)}")
            has_mol = "Yes" if pd.notna(smiles) else "No"
            st.write(f"- Molecular Features: {has_mol}")

        st.markdown("---")
        st.subheader("Model Predictions")

        # Get data for this specific drug by iterating until we find it
        # Since PregnancyDataset merges the DFs based on adr_df's index, we need to find
        # the true integer location of the selected_drug in that specific merged dataframe.
        # We can look up the index in the dataset directly.
        dataset_drug_ids = loader.dataset.drug_ids
        try:
            true_idx = dataset_drug_ids.index(selected_drug)
            for i, batch in enumerate(loader):
                if i == true_idx:
                    selected_batch = batch
                    break
        except ValueError:
            st.error("Drug ID not found in the processed dataset.")
            st.stop()

        if st.button("Run Prediction"):
            with st.spinner("Running Monte Carlo Dropout for Uncertainty Estimation..."):
                # Set a fixed random seed so that Monte Carlo Dropout produces the EXACT same
                # probabilistic outputs every single time the user clicks the button for a specific drug.
                torch.manual_seed(42)
                mean_pred, var_pred = model.predict_with_uncertainty(selected_batch, num_samples=20)

                mean_pred = mean_pred.flatten().numpy()
                var_pred = var_pred.flatten().numpy()

                outcomes = ['Pregnancy ADR Signal']

                res_df = pd.DataFrame({
                    'Outcome': outcomes,
                    'Probability': [f"{p:.2%}" for p in mean_pred],
                    'Uncertainty (Variance)': [f"{v:.4f}" for v in var_pred]
                })

                st.table(res_df)

                # Flag Insufficient Evidence
                if int(n_reports) == 0 and pd.isna(smiles):
                    st.warning("⚠️ **Low-Confidence Prediction**: This drug has no pregnancy-specific reports and missing molecular features. The prediction relies heavily on imputed averages or general ADRs.")
                elif int(n_reports) == 0:
                    st.info("ℹ️ **Note**: This drug has no direct pregnancy reports. Prediction relies on molecular structure and general ADR features.")

except Exception as e:
    st.error(f"Error loading dashboard: {e}")
