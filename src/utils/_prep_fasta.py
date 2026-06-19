import pandas as pd

test = pd.read_csv(r"D:\Research_AI_Bio\03_Datasets\Processed\amp_test_real.csv")
pos = test[test["label_amp"]==1]
neg = test[test["label_amp"]==0]

with open(r"D:\Research_AI_Bio\03_Datasets\Final\test_pos.fasta", "w") as f:
    for idx, row in pos.iterrows():
        f.write(">pos_{}\n{}\n".format(idx, row["sequence"]))

with open(r"D:\Research_AI_Bio\03_Datasets\Final\test_neg.fasta", "w") as f:
    for idx, row in neg.iterrows():
        f.write(">neg_{}\n{}\n".format(idx, row["sequence"]))

train = pd.read_csv(r"D:\Research_AI_Bio\03_Datasets\Processed\amp_train_real.csv")
train_pos = train[train["label_amp"]==1]
train_neg = train[train["label_amp"]==0]

with open(r"D:\Research_AI_Bio\03_Datasets\Final\train_pos.fasta", "w") as f:
    for idx, row in train_pos.iterrows():
        f.write(">pos_{}\n{}\n".format(idx, row["sequence"]))

with open(r"D:\Research_AI_Bio\03_Datasets\Final\train_neg.fasta", "w") as f:
    for idx, row in train_neg.iterrows():
        f.write(">neg_{}\n{}\n".format(idx, row["sequence"]))

print("Done. Train pos:", len(train_pos), "neg:", len(train_neg))
print("Test pos:", len(pos), "neg:", len(neg))
