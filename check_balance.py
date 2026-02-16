import pandas as pd

df = pd.read_csv("non_redundant_binary_dataset.csv")

print("Total samples:", len(df))
print("\nClass distribution:")
print(df['Label'].value_counts())

counts = df['Label'].value_counts()

therapeutic = counts.get(1, 0)
decoy = counts.get(0, 0)

print("\nTherapeutic:", therapeutic)
print("Decoys:", decoy)

if decoy != 0:
    ratio = therapeutic / decoy
    print("\nTherapeutic : Decoy ratio =", round(ratio, 2), ": 1")
