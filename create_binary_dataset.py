import pandas as pd
import random



INPUT_FILE = "thpdb.csv"
OUTPUT_FILE = "binary_therapeutic_dataset_balanced.csv"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


df = pd.read_csv(INPUT_FILE, encoding="latin1")

print("Original dataset shape:", df.shape)

therapeutic_sequences = df['Peptide Sequence'].dropna().tolist()

therapeutic_sequences = list(set(therapeutic_sequences))

num_therapeutic = len(therapeutic_sequences)
print("Therapeutic peptides after cleaning:", num_therapeutic)


def generate_random_peptide(length):
    return ''.join(random.choices(AMINO_ACIDS, k=length))

def shuffle_sequence(seq):
    seq_list = list(seq)
    random.shuffle(seq_list)
    return ''.join(seq_list)

half_n = num_therapeutic // 2

# Half random decoys
random_decoys = [
    generate_random_peptide(len(seq))
    for seq in therapeutic_sequences[:half_n]
]

# Half shuffled decoys
shuffled_decoys = [
    shuffle_sequence(seq)
    for seq in therapeutic_sequences[half_n:]
]

decoys = random_decoys + shuffled_decoys

print("Decoys generated:", len(decoys))



decoys = list(set(decoys) - set(therapeutic_sequences))

while len(decoys) < num_therapeutic:
    seq = random.choice(therapeutic_sequences)
    new_decoy = generate_random_peptide(len(seq))
    
    if new_decoy not in therapeutic_sequences:
        decoys.append(new_decoy)

decoys = decoys[:num_therapeutic]

print("Final decoy count:", len(decoys))


therapeutic_df = pd.DataFrame({
    "Sequence": therapeutic_sequences,
    "Label": 1
})

decoy_df = pd.DataFrame({
    "Sequence": decoys,
    "Label": 0
})

final_df = pd.concat([therapeutic_df, decoy_df], ignore_index=True)

final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

final_df.to_csv(OUTPUT_FILE, index=False)

print("Final dataset shape:", final_df.shape)
print("Saved as:", OUTPUT_FILE)
