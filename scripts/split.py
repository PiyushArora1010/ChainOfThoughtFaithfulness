"""
Reads a CSV, adds a "split" column with a random 90/10 train/test split,
and writes the result to a new CSV.

Usage:
    python add_split_column.py input.csv output.csv --train-frac 0.9 --seed 42
"""
import argparse
import numpy as np
import pandas as pd


def add_split_column(input_path: str, output_path: str, train_frac: float = 0.9, seed: int = 42):
    df = pd.read_csv(input_path)

    rng = np.random.default_rng(seed)
    n = len(df)
    n_train = int(round(n * train_frac))

    # shuffle indices, first n_train go to train, rest to test
    indices = rng.permutation(n)
    split = np.full(n, "test", dtype=object)
    split[indices[:n_train]] = "train"

    df["split"] = split

    df.to_csv(output_path, index=False)

    actual_train = (df["split"] == "train").sum()
    actual_test = (df["split"] == "test").sum()
    print(f"Total rows: {n}")
    print(f"Train: {actual_train} ({actual_train / n:.1%})")
    print(f"Test:  {actual_test} ({actual_test / n:.1%})")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a 90/10 train/test split column to a CSV.")
    parser.add_argument("--input_csv", help="Path to input CSV file")
    parser.add_argument("--output_csv", help="Path to write output CSV file")
    parser.add_argument("--train-frac", type=float, default=0.9, help="Fraction of rows assigned to train (default: 0.9)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    add_split_column(args.input_csv, args.output_csv, args.train_frac, args.seed)