# Protein Subcellular Localisation Predictor

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-1.30-orange.svg)](https://streamlit.io/)

---

## 🧬 Project Overview

The **Protein Subcellular Localisation Predictor** is a deep learning-based web application that predicts the subcellular compartments of proteins based on their amino acid sequences.

Understanding protein localisation is crucial for studying protein function, cellular pathways, and designing molecular biology experiments.

This tool uses **pretrained ESM2 protein language models** to generate embeddings and a **multi-label neural network** to classify proteins into compartments such as:

* Nucleus
* Cytoplasm
* Secreted

---

## ⚡ Features

* Upload protein sequences in **FASTA format**.
* Generate embeddings **on the fly** using **ESM2**.
* Multi-label prediction for **multiple compartments per protein**.
* Display **probabilities and binary predictions** interactively.
* Download results as **CSV**.

---

## 🛠 Technology Stack

* **Deep Learning:** PyTorch
* **Protein Language Model:** ESM2 (Facebook AI / FairSeq)
* **Web App Framework:** Streamlit
* **Data Handling:** NumPy, Pandas
* **Evaluation Metrics:** F1-score, AUROC, Precision-Recall curves

---

## 💾 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/protein_localisation_app.git
cd protein_localisation_app
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**

```bash
streamlit run app.py
```

---

## 📂 Folder Structure

```
protein_localisation_app/
│
├── app.py                  # Streamlit web app
├── model.py                # Model loader
├── requirements.txt        # Python dependencies
├── embeddings/
│   └── esm_embeddings.py   # ESM2 embedding generator
├── results/
│   └── protein_classifier.pt  # Trained model
└── Test.txt                # Sample FASTA file for testing
```

---

## 📝 Usage

1. Open the app in a browser.
2. Upload a **FASTA file** with one or more protein sequences.
3. Click **Run Prediction**.
4. View **probabilities** and **binary predictions** for each protein.
5. Download results as a **CSV file**.

---

## 🧪 Demo

| Protein      | Nucleus | Cytoplasm | Secreted |
| ------------ | ------- | --------- | -------- |
| TAF13\_HUMAN | ✅       | ❌         | ❌        |
| INS\_HUMAN   | ❌       | ❌         | ✅        |
| SYUB\_HUMAN  | ❌       | ✅         | ❌        |

---

## 🔗 References

* [ESM Protein Language Models](https://github.com/facebookresearch/esm)
* [UniProt Protein Database](https://www.uniprot.org)

---

## 📚 Future Work

* Add support for **more subcellular compartments**.
* Batch FASTA upload with **progress bar**.
* Integrate **3D structure-based features** for more accurate predictions.
