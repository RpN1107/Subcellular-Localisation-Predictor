# -----------------------------DEEP LEARNING PROJECT----------------------------
#Rithwik Nambiar 20244013
#Sameera Miraj 20244016
#Abhilasha Singh 20244001
#Yash Kalyan Sakharkar 20244023
#Rohan Ankit 20244014

# ------------------------------------------------------------------------------
# Objective: 
# 1. To test different embedders for their speed and memory
# 2. Create a classifier to predict protein subcellular localisation from its amino
#    acid sequence

# Data Used: Uniprot data for reviewed human proteins


#CONTENTS#
 
1. Data.tsv is the data downloaded from uniprot containing protein name, sequence and subcellular localisation

2. 1_Modifying_uniprot_data.ipynb is used to create the input data in the required format

3. 2_Testing_Embedders.ipynb uses the data to test three embedders on a stratified dataset sample

4. 3_Protein_Classifier.ipynb trains a classifier with ESM2 embedder on the entire input data

5. Report.pdf

6. "Scripts" contains the .py versions of all the scripts



#STEPS TO FOLLOW#

1. Set your file path and run 1_Modifying_uniprot_data.ipynb to generate modified_data.csv

2. Use 2_Testing_Embedders.ipynb as test the performance of the embedders

3. 3_Protein_Classifier.ipynb runs the classifier on the entire data
