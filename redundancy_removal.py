import pandas as pd
from Bio import pairwise2


INPUT_FILE = "binary_therapeutic_dataset_balanced.csv"
OUTPUT_FILE = "final_clean_dataset.csv"

MATCH_SCORE = 1
MISMATCH_PENALTY = -1
GAP_PENALTY = -2

THRESHOLD = 0.7  # similarity threshold

# ---------------------------------------
# LOAD DATASET
# ---------------------------------------
df = pd.read_csv(INPUT_FILE)

print("Original dataset size:", len(df))

df = df.drop_duplicates(subset='Sequence')
print("After exact duplicate removal:", len(df))

sequences = df['Sequence'].tolist()
labels = df['Label'].tolist()


def calculate_similarity(seq1, seq2):

    alignment = pairwise2.align.globalms(
        seq1, seq2,
        MATCH_SCORE,
        MISMATCH_PENALTY,
        GAP_PENALTY,
        GAP_PENALTY
    )[0]

    aligned_seq1 = alignment[0]
    aligned_seq2 = alignment[1]

    matches = 0
    mismatches = 0
    gaps = 0

    for a, b in zip(aligned_seq1, aligned_seq2):
        if a == '-' or b == '-':
            gaps += 1
        elif a == b:
            matches += 1
        else:
            mismatches += 1

    # Apply your formula
    score = (
        matches * MATCH_SCORE +
        mismatches * MISMATCH_PENALTY +
        gaps * GAP_PENALTY
    )

    
    max_possible_score = min(len(seq1), len(seq2)) * MATCH_SCORE

    similarity = score / max_possible_score

    return similarity


non_redundant_sequences = []
non_redundant_labels = []

print("\nRemoving redundant sequences...\n")

for i, seq in enumerate(sequences):

    if i % 20 == 0:
        print(f"Processing {i+1}/{len(sequences)}")

    redundant = False

    for kept_seq in non_redundant_sequences:
        similarity = calculate_similarity(seq, kept_seq)

        if similarity >= THRESHOLD:
            redundant = True
            break

    if not redundant:
        non_redundant_sequences.append(seq)
        non_redundant_labels.append(labels[i])


df_clean = pd.DataFrame({
    "Sequence": non_redundant_sequences,
    "Label": non_redundant_labels
})

print("\nFinal non-redundant size:", len(df_clean))

df_clean.to_csv(OUTPUT_FILE, index=False)


print("\nValidating dataset...")

max_similarity = 0

for i in range(len(non_redundant_sequences)):
    for j in range(i+1, len(non_redundant_sequences)):
        similarity = calculate_similarity(
            non_redundant_sequences[i],
            non_redundant_sequences[j]
        )
        if similarity > max_similarity:
            max_similarity = similarity

print("Maximum pairwise similarity remaining:", max_similarity)

reduction_percent = (
    (len(df) - len(df_clean)) / len(df)
) * 100

print("Reduction percentage:", round(reduction_percent, 2), "%")

print("\nRedundancy removal completed using formula-based scoring.")
