import pandas as pd
import numpy as np
from Bio.Align import PairwiseAligner
from tqdm import tqdm

df = pd.read_csv("final_final.csv")

print("Original dataset size:", len(df))

#to ensure column names
df = df.rename(columns={
    "Sequence": "sequence",
    "ID": "id"
})

df["sequence"] = df["sequence"].astype(str)
df["sequence"] = df["sequence"].str.strip()
df["sequence"] = df["sequence"].str.upper()
df["sequence"] = df["sequence"].str.replace(r"[^A-Z]", "", regex=True)

df = df[df["sequence"] != ""]
df = df.dropna()

print("After cleaning:", len(df))

#remove exact replicas
df = df.drop_duplicates(subset="sequence")
print("After removing exact duplicates:", len(df))

#remove length less than 5 and greater than 10
df["length"] = df["sequence"].apply(len)
df = df[(df["length"] >= 5) & (df["length"] <= 100)]

print("After length filtering:", len(df))

aligner = PairwiseAligner()
aligner.mode = "global"
aligner.match_score = 1
aligner.mismatch_score = -1
aligner.open_gap_score = -2
aligner.extend_gap_score = -2

def global_identity(s1, s2):
    score = aligner.score(s1, s2)
    max_possible = min(len(s1), len(s2)) * aligner.match_score
    return score / max_possible


non_redundant_sequences = []
non_redundant_rows = []

sequences = df["sequence"].tolist()

for idx, seq in tqdm(list(enumerate(sequences))):
    keep = True
    for existing in non_redundant_sequences:
        # quick length filter to speed up
        if abs(len(seq) - len(existing)) / max(len(seq), len(existing)) > 0.3:
            continue

        if global_identity(seq, existing) > 0.80:
            keep = False
            break

    if keep:
        non_redundant_sequences.append(seq)
        non_redundant_rows.append(df.iloc[idx])

# Create final dataframe
df_nonredundant = pd.DataFrame(non_redundant_rows)

print("After removing >80% similar sequences:", len(df_nonredundant))

df_nonredundant = df_nonredundant.drop(columns=["length"])
output_file = "nonredundant_final.csv"
df_nonredundant.to_csv(output_file, index=False)

print("Output saved as:", output_file)

print("\nRedundancy removal complete.")
