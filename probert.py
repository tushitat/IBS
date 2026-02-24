#import
import torch
import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#configuration check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#load the probert model (for protein and peptides)
model_name = "Rostlab/prot_bert"

tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
model = BertModel.from_pretrained(model_name)

model = model.to(device)
model.eval()

print("ProtBERT loaded successfully.")

#load and read the dataset
df = pd.read_csv("nonredundant_final.csv")

print("Dataset shape:", df.shape)
assert "sequence" in df.columns, "Column 'sequence' not found in dataset."

sequences = df["sequence"].tolist()

#prerequisite 
def preprocess_sequence(sequence):
    sequence = sequence.replace(" ", "")
    sequence = " ".join(list(sequence))
    return sequence

#embedding function
def get_batch_embeddings(sequences, batch_size=16):
    
    all_embeddings = []
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        batch = [preprocess_sequence(seq) for seq in batch]
        
        inputs = tokenizer(batch,
                           return_tensors="pt",
                           padding=True,
                           truncation=True)
        
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        #mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = embeddings.cpu().numpy()
        
        all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)

#extracting the features
print("Extracting ProtBERT embeddings...")
bert_features = get_batch_embeddings(sequences, batch_size=16)

print("Embedding matrix shape:", bert_features.shape)

#standardization
scaler = StandardScaler()
bert_scaled = scaler.fit_transform(bert_features)

#pca reduction of 1024 features
pca = PCA(n_components=100)
bert_reduced = pca.fit_transform(bert_scaled)

print("Reduced feature shape:", bert_reduced.shape)
print("Total explained variance:", sum(pca.explained_variance_ratio_))

np.save("bert_embeddings_1024.npy", bert_features)
np.save("bert_embeddings_pca_100.npy", bert_reduced)

pd.DataFrame(bert_reduced).to_csv("bert_pca_100.csv", index=False)

print("Feature extraction completed successfully.")
