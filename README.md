# 🧬 Subcellular Localisation Predictor

**Author**: Rithwik Nambiar

This project predicts the **subcellular localisation** of human proteins using deep learning-based protein sequence embeddings. It uses **UniProt-reviewed human protein data** and compares the performance of multiple embedding techniques. The final classifier is trained using the **ESM2** model.

---

## 📁 Contents

├── Data.tsv # UniProt data with protein names, sequences, and subcellular locations
├── 1_Modifying_uniprot_data.ipynb # Preprocesses the raw UniProt data
├── 2_Testing_Embedders.ipynb # Benchmarks three different protein embedders
├── 3_Protein_Classifier.ipynb # Trains a classifier using ESM2 embeddings
├── Report.pdf # Project report (methods, results, discussion)

---

## 🎯 Objective

1. **Benchmark** different protein embedders for speed and memory usage.
2. **Develop a classifier** to predict subcellular localisation from amino acid sequences.

---

## 🧪 Data Source

- [UniProt](https://www.uniprot.org/) - Manually reviewed human protein dataset (Swiss-Prot section)

---

## 🚀 How to Use

### Step-by-step

1. **Run `1_modifying_uniprot_data.py`**
   - Input: `Data.tsv`
   - Output: `modified_data.csv` — cleaned and formatted dataset.

2. **Run `2_Testing_Embedders.py`**
   - Tests three different protein embedders on a stratified sample.
   - Compares their speed and memory usage.

3. **Run `3_protein_classifier.py`**
   - Trains a classification model on the entire dataset using **ESM2** embeddings.
   - Outputs performance metrics and evaluation plots.

---

## 📈 Results

- Benchmarked 3 popular embedders on a sample dataset.
- ESM2 selected for best performance.
- Final model performance metrics include:
  - Accuracy
  - Confusion Matrix
  - ROC-AUC scores

Refer to `Report.pdf` for detailed methodology and results.

---

## 🛠 Dependencies

- Python 3.8+
- pandas, numpy
- scikit-learn
- torch, transformers
- tqdm, matplotlib, seaborn

> It is recommended to use a virtual environment or Conda for dependency management.

---

## 📜 License

This project is open-source and licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- **UniProt** for providing high-quality curated protein datasets
- **Meta AI (FAIR)** for the ESM protein language models
- **Hugging Face Transformers** for easy model access and inference

---

## 💡 Future Work

- Extend to **multi-label classification** for proteins localised in multiple compartments
- Include **post-translational modifications** and **motif analysis**
- Develop a **web interface** for interactive predictions by users
