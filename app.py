# SubcellularLocalisation App
# author: Rithwik Nambiar
# Date: 2025_09_17

import streamlit as st
import torch
import pandas as pd
from io import StringIO

# Import model and embeddings
from embeddings.esm_embeddings import generate_esm_embeddings
from model import load_trained_model

# -----CONFIGURATION-----
MODEL_PATH = 'results/protein_classifier.pt'
LABELS = ['Nucleus', 'Cytoplasm', 'Secreted']
INPUT_DIM = 320
HIDDEN_DIM = 256

# -----LOAD MODEL-----
@st.cache_resource
def load_model():
    return load_trained_model(
        MODEL_PATH,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=len(LABELS),
        device='cpu'
    )

model = load_model()

# -----STREAMLIT APP-----
st.set_page_config(page_title="Protein Subcellular Localisation", layout="wide")
st.title("🧬 Protein Subcellular Localisation Predictor")

uploaded_file = st.file_uploader("Upload a FASTA file", type=["fasta", "fa", "txt"])

if uploaded_file is not None:
    fasta_text = uploaded_file.read().decode("utf-8")

    # Extract sequences and names from FASTA
    sequences, names = [], []
    current_seq = []
    current_name = None

    for line in StringIO(fasta_text):
        line = line.strip()
        if line.startswith(">"):
            if current_name and current_seq:
                sequences.append("".join(current_seq))
                current_seq = []
            current_name = line[1:].strip()

            # If UniProt-style header, extract short name (e.g. TAF13_HUMAN)
            if "|" in current_name:
                parts = current_name.split("|")
                if len(parts) >= 3:
                    current_name = parts[2].split()[0]

            names.append(current_name)
        elif line:
            current_seq.append(line)

    if current_name and current_seq:
        sequences.append("".join(current_seq))

    # Prediciton
    if st.button(" Run Prediction"):
        if not sequences:
            st.error("No valid sequences found in the uploaded FASTA file.")
        else:
            with st.spinner('Generating embeddings...'):
                embeddings = generate_esm_embeddings(sequences, names)

            with st.spinner('Predicting subcellular localisation...'):
                model.eval()
                with torch.no_grad():
                    inputs = torch.tensor(embeddings, dtype=torch.float32)
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs).numpy()
                    binary_preds = (probs > 0.5).astype(int)

            # Build results DataFrame
            results_df = pd.DataFrame(probs, columns=LABELS, index=names)
            binary_df = pd.DataFrame(binary_preds, columns=LABELS, index=names)

            st.success("✅ Prediction complete!")

            st.write("### 📊 Probabilities")
            st.dataframe(results_df.style.format("{:.2f}"))

            st.write("### ✅ Binary Predictions (threshold = 0.5)")
            st.dataframe(binary_df)

            # Downloadable results
            csv = binary_df.reset_index().rename(columns={"index": "Protein Name"}).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name='subcellular_localisation_predictions.csv',
                mime='text/csv',
            )
