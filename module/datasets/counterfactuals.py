import pandas as pd
import numpy as np
import random


def add_counterfactuals(df, bins, label_col="Outcome", r=2, m=10, eps=0.3, seed=0):
    rng = random.Random(seed)
    binned = pd.DataFrame(index=df.index)
    for col, (edges, labels) in bins.items():
        binned[col] = pd.cut(df[col], bins=edges, labels=labels)

    feature_cols = list(bins.keys())
    binned_arr = binned[feature_cols].to_numpy()

    n = len(df)
    hamming = np.zeros((n, n), dtype=int)
    for j in range(len(feature_cols)):
        col_vals = binned_arr[:, j]
        diff = col_vals[:, None] != col_vals[None, :]
        hamming += diff

    labels = df[label_col].to_numpy()
    counterfactuals = [[] for _ in range(n)]

    for i in range(n):
        neighbor_idx = np.where((hamming[i] <= r) & (hamming[i] > 0))[0]
        if len(neighbor_idx) < m:
            continue

        same = [j for j in neighbor_idx if labels[j] == labels[i]]
        diff = [j for j in neighbor_idx if labels[j] != labels[i]]
        rng.shuffle(same)
        rng.shuffle(diff)

        subset = [i]
        use_same = True
        si, di = 0, 0
        while len(subset) < m:
            if use_same and si < len(same):
                subset.append(same[si])
                si += 1
            elif not use_same and di < len(diff):
                subset.append(diff[di])
                di += 1
            elif si < len(same):
                subset.append(same[si])
                si += 1
            elif di < len(diff):
                subset.append(diff[di])
                di += 1
            else:
                break
            use_same = not use_same

        subset_labels = labels[subset]
        pos = (subset_labels == 1).sum()
        neg = (subset_labels == 0).sum()
        balance = abs(pos - neg) / len(subset)
        if balance > eps:
            continue

        for j in subset:
            if j == i:
                continue
            diff_cols = [c for c in feature_cols if binned_arr[i, feature_cols.index(c)] != binned_arr[j, feature_cols.index(c)]]
            counterfactuals[i].append((j, diff_cols))

    df = df.copy()
    df["counterfactuals"] = counterfactuals
    return df

if __name__ == "__main__":
    from diabetes import DiabetesDataset
    dataset = DiabetesDataset("/home/piyush/Desktop/Code/SandbarASR/Faithfulness/module/datasets/data/diabetes_with_split.csv", split="train")
    dataset.df = add_counterfactuals(dataset.df, dataset.bins, label_col="Outcome", r=2, m=10, eps=0.3, seed=0)
    breakpoint()