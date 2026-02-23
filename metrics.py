import pandas as pd
import numpy as np
from Bio.Align import PairwiseAligner
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

df = pd.read_csv("nonredundant_final.csv")

print("Final dataset size:", len(df))
aligner = PairwiseAligner()
aligner.mode = "global"

#Needleman–Wunsch
aligner.match_score = 1
aligner.mismatch_score = -1
aligner.open_gap_score = -2
aligner.extend_gap_score = -2   #Linear gap model

def global_identity(s1, s2):
    score = aligner.score(s1, s2)
    max_possible = min(len(s1), len(s2)) * aligner.match_score
    return score / max_possible

sample_size = min(200, len(df))
sample_sequences = random.sample(df["sequence"].tolist(), sample_size)

print("Computing similarity matrix on", sample_size, "sequences")

#similarity matrix
similarity_matrix = np.zeros((sample_size, sample_size))

for i in tqdm(range(sample_size)):
    for j in range(i, sample_size):
        sim = global_identity(sample_sequences[i], sample_sequences[j])
        similarity_matrix[i][j] = sim
        similarity_matrix[j][i] = sim

upper_triangle = similarity_matrix[np.triu_indices(sample_size, k=1)]

mean_sim = np.mean(upper_triangle)
median_sim = np.median(upper_triangle)
max_sim = np.max(upper_triangle)
min_sim = np.min(upper_triangle)
std_sim = np.std(upper_triangle)

pairs_above_80 = np.sum(upper_triangle >= 0.80)

print("\n===== NON-REDUNDANCY METRICS =====")
print("Mean similarity:", round(mean_sim, 4))
print("Median similarity:", round(median_sim, 4))
print("Max similarity:", round(max_sim, 4))
print("Min similarity:", round(min_sim, 4))
print("Std deviation:", round(std_sim, 4))
print("Pairs ≥ 80% similarity:", pairs_above_80)

plt.figure(figsize=(8,5))
plt.hist(upper_triangle, bins=40)
plt.axvline(0.80)
plt.title("Pairwise Similarity Distribution (Non-redundant Dataset)")
plt.xlabel("Global Sequence Identity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("similarity_histogram.png")
plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(similarity_matrix, cmap="viridis")
plt.title("Similarity Matrix Heatmap (Sampled Sequences)")
plt.tight_layout()
plt.savefig("similarity_heatmap.png")
plt.show()

print("\nSimilarity histogram saved as similarity_histogram.png")
print("Similarity heatmap saved as similarity_heatmap.png")