# esm_embeddings.py
# author: Rithwik Nambiar
# date: 2025_09_17

import torch 
import esm
import numpy as np


def generate_esm_embeddings(sequences, names, mdoel_name="esm2_t6_8M_UR50D"):
    """
    Generate ESM embeddings for a list of protein sequences.

    Parameters:
    sequences (list of str): List of protein sequences.
    names (list of str): List of sequence names corresponding to the sequences.
    model_name (str): Name of the ESM model to use.

    Returns:
    np.ndarray: Array of shape (num_sequences, embedding_dim) containing the embeddings.
    """
    # Load the pre-trained ESM model and alphabet
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_embeddings = []
    batch_size = 8

    for i in range(0, len(sequences), batch_size):
        batch_data = list(zip(names[i:i + batch_size], sequences[i:i + batch_size]))
        _, _, batch_tokens = batch_converter(batch_data)
        bacth_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers = [6])
            token_representations = results["representations"][6]
        
        for j, (_, seq) in enumerate(batch_data):
            seq_len = len(seq)
            embedding = token_representations[j, 1:seq_len + 1].mean(0)
            all_embeddings.append(embedding.cpu().numpy())
    
    return np.stack(all_embeddings)