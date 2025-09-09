import csv
import os
from collections import defaultdict
import pandas as pd
import time
from parafac_fun import run_sparse_parafac
"""
MOOC network

https://snap.stanford.edu/data/act-mooc.html

"""


file_path = 'mooc_actions.tsv'
# file_path = 'structural_anomalies5/normal/1_temporal_graph_1-5000-2000_.csv'

with open(file_path, 'r') as f:
    lines = f.readlines()

# Parse each line into (int, int, float)
# Parse each line, skip the first column
data = [list(map(float, line.strip().split()[1:])) for line in lines[1:]]

# Transpose to get columns
cols = list(zip(*data))

# Number of rows
num_rows = len(data)

# Min and max per column
min_values = [min(col) for col in cols]
max_values = [max(col) for col in cols]

# Display results
print(f"Number of rows: {num_rows}")
for i, (min_val, max_val) in enumerate(zip(min_values, max_values), 1):
    print(f"Column {i} -> Min: {min_val}, Max: {max_val}")


# Separate columns
node1_list, node2_list, time_list = zip(*data)

# Get unique node IDs and build mapping
unique_nodes = sorted(set(node1_list + node2_list))
id_map = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}

# Map node IDs to 0-based
node1_new = [id_map[n1] for n1 in node1_list]
node2_new = [id_map[n2] for n2 in node2_list]

# Normalize timestamps
min_time = min(time_list)
time_new = [t - min_time for t in time_list]

# Build normalized data
normalized_data = list(zip(node1_new, node2_new, time_new))

# Transpose to get columns
cols = list(zip(*normalized_data))

# Stats
num_rows = len(normalized_data)
min_values = [min(col) for col in cols]
max_values = [max(col) for col in cols]

# Output
print(f"Number of rows: {num_rows}")
for i, (min_val, max_val) in enumerate(zip(min_values, max_values), 1):
    print(f"Column {i} -> Min: {min_val}, Max: {max_val}")


# Save to CSV
# output_csv_path = 'mooc_actions.csv'
#
# # Create the directory if it doesn't exist
# # os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
# with open(output_csv_path, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     # writer.writerow(['node1', 'node2', 'timestamp'])  # header
#     writer.writerows(normalized_data)


def resample_mooc(input_file, output_csv, bin_seconds=60):
    """
    Resample MOOC edge timestamps into bins (seconds/minutes/hours) and save to CSV.

    Parameters:
        input_file (str): CSV with columns 'node1','node2','timestamp'
        output_csv (str): Path to save resampled CSV
        bin_seconds (int): Size of each time bin in seconds
    """
    # Read CSV with headers, cast timestamp to numeric
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='raise')

    # Compute integer bin
    df['bin'] = (df['timestamp'] // bin_seconds).astype(int)

    # Keep original nodes and new bin
    df_resampled = df[['node1', 'node2', 'bin']]

    # Save
    df_resampled.to_csv(output_csv, index=False, header=False)
    print(f"Saved resampled data to {output_csv}")
    print(f"Number of rows: {len(df_resampled)}")
    print(df_resampled.head())

# resample_mooc("mooc_actions.csv", "mooc_1min.csv", bin_seconds=60)
# resample_mooc("mooc_actions.csv", "mooc_1hour.csv", bin_seconds=3600)



########## PARAFAC
overlapping = str(0)
window_size = 48
step_size = window_size # no overlap
Time = 720
rank = 3
nNodes = 7047

csv_path = "mooc_1hour.csv"
start_point =0

for time_start in range(start_point, Time - window_size + 1, step_size):
    time_end = time_start + window_size

    # Save factors to a unique file in same folder
    # filename = "structural_anomalies5/anomaly9" + f"/factors{overlapping}%-{time_start}-{time_end}-WS:{window_size}.pkl"
    filename = "factors/" + f"factors{overlapping}%-{time_start}-{time_end}-WS:{window_size}.pkl"

    print(f"⏳ Decomposing: [{time_start}-{time_end}]")

    start_time = time.time()

    sorted_weights, sorted_factors = run_sparse_parafac(
    csv_path, time_start, time_end, nNodes, filename,
    rank=rank, n_iter_max=100, tol=1e-5 )
    print(f"\n✅ PARAFAC decompositions weights, {sorted_weights}.")
    elapsed_time = time.time() - start_time
    print(f"✅ Saved to: {filename} (⏱ {elapsed_time:.2f} sec)")