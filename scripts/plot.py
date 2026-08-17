# """
# Reads a CSV, creates a stratified train/test split using explicit row counts,
# and writes only the selected train + test rows to a new CSV.

# Rows are sampled without replacement, so train and test are guaranteed
# to have no overlap.

# Usage:
# python add_split_column.py \
#     --input_csv input.csv \
#     --output_csv output.csv \
#     --train-size 90000 \
#     --test-size 10000 \
#     --stratify-col label \
#     --seed 42
# """

# import argparse
# import numpy as np
# import pandas as pd


# def add_split_column(
#     input_path: str,
#     output_path: str,
#     train_size: int,
#     test_size: int,
#     seed: int = 42,
#     stratify_col: str = None,
# ):
#     df = pd.read_csv(input_path)
#     n = len(df)

#     # ------------------------------------------------------------------
#     # Validate arguments
#     # ------------------------------------------------------------------
#     if train_size < 0 or test_size < 0:
#         raise ValueError("train_size and test_size must be non-negative.")

#     total_requested = train_size + test_size

#     if total_requested > n:
#         raise ValueError(
#             f"Requested {total_requested} rows "
#             f"(train={train_size}, test={test_size}), "
#             f"but dataset only contains {n} rows."
#         )

#     if train_size == 0 and test_size == 0:
#         raise ValueError("At least one of train_size or test_size must be > 0.")

#     if stratify_col is not None and stratify_col not in df.columns:
#         raise ValueError(
#             f"Stratification column '{stratify_col}' not found. "
#             f"Available columns: {list(df.columns)}"
#         )

#     rng = np.random.default_rng(seed)

#     # ------------------------------------------------------------------
#     # Non-stratified split
#     # ------------------------------------------------------------------
#     if stratify_col is None:
#         indices = rng.permutation(n)

#         train_indices = indices[:train_size]

#         # IMPORTANT:
#         # Test starts AFTER train, so there can be no overlap.
#         test_indices = indices[
#             train_size:train_size + test_size
#         ]

#         split = np.full(n, "unused", dtype=object)

#         split[train_indices] = "train"
#         split[test_indices] = "test"

#     # ------------------------------------------------------------------
#     # Stratified split
#     # ------------------------------------------------------------------
#     else:
#         groups = df.groupby(
#             stratify_col,
#             dropna=False,
#             sort=False,
#         ).groups

#         group_items = list(groups.items())

#         group_sizes = np.array(
#             [len(indices) for _, indices in group_items],
#             dtype=int,
#         )

#         # --------------------------------------------------------------
#         # Allocate train rows proportionally across strata.
#         # --------------------------------------------------------------
#         raw_train = (group_sizes / n) * train_size
#         train_counts = np.floor(raw_train).astype(int)

#         train_remainder = train_size - train_counts.sum()

#         if train_remainder > 0:
#             fractional = raw_train - train_counts

#             # Largest fractional parts get the remaining rows.
#             order = np.argsort(-fractional)

#             for i in order[:train_remainder]:
#                 train_counts[i] += 1

#         # --------------------------------------------------------------
#         # Allocate test rows proportionally across strata.
#         #
#         # IMPORTANT:
#         # We account for train_counts so test can never overlap
#         # with train.
#         # --------------------------------------------------------------
#         raw_test = (group_sizes / n) * test_size
#         test_counts = np.floor(raw_test).astype(int)

#         # Maximum number of rows still available in each group.
#         available_for_test = group_sizes - train_counts

#         # Make sure initial allocation does not exceed available rows.
#         test_counts = np.minimum(
#             test_counts,
#             available_for_test,
#         )

#         test_remainder = test_size - test_counts.sum()

#         if test_remainder > 0:
#             fractional = raw_test - test_counts

#             # Only groups with remaining capacity can receive more.
#             fractional = np.where(
#                 available_for_test > test_counts,
#                 fractional,
#                 -np.inf,
#             )

#             order = np.argsort(-fractional)

#             allocated = 0

#             for i in order:
#                 if allocated >= test_remainder:
#                     break

#                 if test_counts[i] < available_for_test[i]:
#                     test_counts[i] += 1
#                     allocated += 1

#             if allocated != test_remainder:
#                 raise ValueError(
#                     "Could not allocate the requested test size while "
#                     "maintaining a stratified split. This can happen when "
#                     "some strata are too small."
#                 )

#         # Final sanity checks.
#         if train_counts.sum() != train_size:
#             raise RuntimeError(
#                 f"Internal error: allocated {train_counts.sum()} train rows "
#                 f"but requested {train_size}."
#             )

#         if test_counts.sum() != test_size:
#             raise RuntimeError(
#                 f"Internal error: allocated {test_counts.sum()} test rows "
#                 f"but requested {test_size}."
#             )

#         # --------------------------------------------------------------
#         # Sample each stratum without replacement.
#         # --------------------------------------------------------------
#         split = np.full(n, "unused", dtype=object)

#         for i, (_, group_indices) in enumerate(group_items):
#             group_indices = np.asarray(group_indices)

#             shuffled = rng.permutation(group_indices)

#             n_train_group = train_counts[i]
#             n_test_group = test_counts[i]

#             # First portion goes to train.
#             train_indices = shuffled[:n_train_group]

#             # Immediately following portion goes to test.
#             #
#             # Since these come from the same shuffled array but from
#             # non-overlapping slices, train/test cannot overlap.
#             test_indices = shuffled[
#                 n_train_group:n_train_group + n_test_group
#             ]

#             split[train_indices] = "train"
#             split[test_indices] = "test"

#     # ------------------------------------------------------------------
#     # Add split column.
#     # ------------------------------------------------------------------
#     df["split"] = split

#     # ------------------------------------------------------------------
#     # Keep ONLY train + test.
#     #
#     # Any rows marked "unused" are discarded.
#     # ------------------------------------------------------------------
#     df = df[
#         df["split"].isin(["train", "test"])
#     ].reset_index(drop=True)

#     # ------------------------------------------------------------------
#     # Final validation.
#     # ------------------------------------------------------------------
#     actual_train = (df["split"] == "train").sum()
#     actual_test = (df["split"] == "test").sum()

#     if actual_train != train_size:
#         raise RuntimeError(
#             f"Expected {train_size} train rows, "
#             f"got {actual_train}."
#         )

#     if actual_test != test_size:
#         raise RuntimeError(
#             f"Expected {test_size} test rows, "
#             f"got {actual_test}."
#         )

#     if len(df) != train_size + test_size:
#         raise RuntimeError(
#             f"Expected {train_size + test_size} total rows, "
#             f"got {len(df)}."
#         )

#     # Verify that there is no overlap.
#     #
#     # At this point each source row has only one split value, so overlap
#     # is structurally impossible. This assertion provides an additional
#     # sanity check.
#     if (df["split"] == "train").sum() + (
#         df["split"] == "test"
#     ).sum() != len(df):
#         raise RuntimeError("Train/test overlap detected.")

#     # ------------------------------------------------------------------
#     # Save.
#     # ------------------------------------------------------------------
#     df.to_csv(output_path, index=False)

#     print(f"Original rows: {n}")
#     print(f"Train rows:    {actual_train}")
#     print(f"Test rows:     {actual_test}")
#     print(f"Output rows:   {len(df)}")

#     if stratify_col is not None:
#         print(f"Stratified by: {stratify_col}")

#         print("\nStratification distribution:")
#         print(
#             pd.crosstab(
#                 df[stratify_col],
#                 df["split"],
#                 normalize="columns",
#                 dropna=False,
#             )
#         )

#     print(f"\nSaved to: {output_path}")


# def main():
#     parser = argparse.ArgumentParser(
#         description=(
#             "Create a stratified train/test split with explicit row counts."
#         )
#     )

#     parser.add_argument(
#         "--input_csv",
#         required=True,
#         help="Path to input CSV file.",
#     )

#     parser.add_argument(
#         "--output_csv",
#         required=True,
#         help="Path to output CSV file.",
#     )

#     parser.add_argument(
#         "--train-size",
#         type=int,
#         required=True,
#         help="Exact number of rows to use for training.",
#     )

#     parser.add_argument(
#         "--test-size",
#         type=int,
#         required=True,
#         help="Exact number of rows to use for testing.",
#     )

#     parser.add_argument(
#         "--stratify-col",
#         type=str,
#         default=None,
#         help="Column to use for stratification.",
#     )

#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="Random seed for reproducibility. Default: 42.",
#     )

#     args = parser.parse_args()

#     add_split_column(
#         input_path=args.input_csv,
#         output_path=args.output_csv,
#         train_size=args.train_size,
#         test_size=args.test_size,
#         stratify_col=args.stratify_col,
#         seed=args.seed,
#     )


# if __name__ == "__main__":
#     main()

import sys
import argparse
import shutil

# --------------------------------------------------
# CLI arguments
# --------------------------------------------------
parser = argparse.ArgumentParser(
    description="Plot values for a given TAG from a log file as an ASCII line chart."
)
parser.add_argument("tag", help="Tag string to search for in the log file")
parser.add_argument(
    "--log",
    default="logs/Qwen3_4B_all.log",
    help="Path to the log file (default: logs/Qwen3_4B_all.log)",
)
parser.add_argument(
    "--mode",
    choices=["raw", "ema"],
    default="raw",
    help="Plot raw values or an exponential moving average (default: raw)",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.01,
    help="Smoothing factor for EMA, in (0, 1] (default: 0.1). Only used with --mode ema",
)

args = parser.parse_args()

log_file = args.log
TAG = args.tag

values = []

# --------------------------------------------------
# Read values
# --------------------------------------------------
with open(log_file, "r") as f:
    for line in f:
        if TAG in line:
            parts = line.strip().split(TAG)
            value = parts[-1].split(",")[0].strip()

            try:
                values.append(float(value))
            except ValueError:
                print(f"Could not convert: {value}")

if not values:
    print(f"No values found for {TAG}")
    sys.exit(1)

original_values = values
total_iterations = len(original_values)

# --------------------------------------------------
# Optional EMA smoothing (applied on the full series
# before downsampling, so it uses all data points)
# --------------------------------------------------
if args.mode == "ema":
    alpha = args.alpha
    if not (0 < alpha <= 1):
        print("--alpha must be in (0, 1]")
        sys.exit(1)

    ema_values = [original_values[0]]
    for v in original_values[1:]:
        ema_values.append(alpha * v + (1 - alpha) * ema_values[-1])

    series = ema_values
    series_label = f"{TAG} (EMA, alpha={alpha})"
else:
    series = original_values
    series_label = f"{TAG} (raw)"

# --------------------------------------------------
# Terminal dimensions (auto-detect, with fallback)
# --------------------------------------------------
term_size = shutil.get_terminal_size(fallback=(150, 24))

Y_LABEL_WIDTH = 10  # width reserved for "  123.45 |" prefix
NON_PLOT_ROWS = 8  # header + axis + labels + margins

WIDTH = max(20, term_size.columns - Y_LABEL_WIDTH - 1)
HEIGHT = max(5, term_size.lines - NON_PLOT_ROWS)

# --------------------------------------------------
# Downsample to fit terminal width
# --------------------------------------------------
if total_iterations > WIDTH:
    step = total_iterations / WIDTH
    indices = [
        min(int(i * step), total_iterations - 1)
        for i in range(WIDTH)
    ]
    plot_values = [series[i] for i in indices]
else:
    indices = list(range(total_iterations))
    plot_values = series

# --------------------------------------------------
# Y-axis range
# --------------------------------------------------
vmin = min(plot_values)
vmax = max(plot_values)

if vmax == vmin:
    vmax += 1

# --------------------------------------------------
# Header
# --------------------------------------------------
print()
print(f"{series_label} values")
print(f"min={vmin:.3f}, max={vmax:.3f}")
print(f"iterations={total_iterations}   (terminal: {term_size.columns}x{term_size.lines})")
print()

# --------------------------------------------------
# Compute row index (0 = top) for each plotted value
# --------------------------------------------------
def value_to_row(value):
    normalized = (value - vmin) / (vmax - vmin)
    return HEIGHT - 1 - int(normalized * (HEIGHT - 1))

rows_for_values = [value_to_row(v) for v in plot_values]

# --------------------------------------------------
# Build grid and draw a connected line plot:
# for each column, fill between the row of the
# previous point and the row of the current point
# --------------------------------------------------
grid = [[" " for _ in range(len(plot_values))] for _ in range(HEIGHT)]

for col, row in enumerate(rows_for_values):
    if col == 0:
        grid[row][col] = "●"
        continue

    prev_row = rows_for_values[col - 1]
    lo, hi = min(prev_row, row), max(prev_row, row)

    for r in range(lo, hi + 1):
        grid[r][col] = "│" if r != row else "●"

# --------------------------------------------------
# Plot
# --------------------------------------------------
for row in range(HEIGHT):
    threshold = vmax - (row / (HEIGHT - 1)) * (vmax - vmin)
    line = "".join(grid[row])
    print(f"{threshold:8.2f} |{line}")

# --------------------------------------------------
# X-axis
# --------------------------------------------------
print("         +" + "-" * len(plot_values))

# --------------------------------------------------
# X-axis labels every 100 iterations
# --------------------------------------------------
label_line = [" "] * len(plot_values)

for iteration in range(0, total_iterations, 100):
    if total_iterations > 1:
        pos = round(iteration / (total_iterations - 1) * (len(plot_values) - 1))
    else:
        pos = 0

    label = str(iteration)
    start = pos - len(label) // 2

    if start < 0:
        start = 0
    if start + len(label) > len(label_line):
        start = len(label_line) - len(label)

    for j, char in enumerate(label):
        if 0 <= start + j < len(label_line):
            label_line[start + j] = char

# Also show the final iteration if it isn't a multiple of 100
last_iteration = total_iterations - 1

if last_iteration % 100 != 0:
    if total_iterations > 1:
        pos = len(plot_values) - 1
    else:
        pos = 0

    label = str(last_iteration)
    start = pos - len(label) // 2

    if start < 0:
        start = 0
    if start + len(label) > len(label_line):
        start = len(label_line) - len(label)

    for j, char in enumerate(label):
        if 0 <= start + j < len(label_line):
            label_line[start + j] = char

print("         " + "".join(label_line))
print()