import csv
import os
from collections import defaultdict

file_path = 'RealData/SFHH/SFHH-conf-sensor.edges'
# file_path = 'structural_anomalies5/normal/1_temporal_graph_1-5000-2000_.csv'

with open(file_path, 'r') as f:
    lines = f.readlines()

# Parse each line into (int, int, float)
data = [list(map(float, line.strip().split(','))) for line in lines]

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



# Count how many timestamps fall into each 1000-unit time window
bin_counts = defaultdict(int)

for t in time_new:
    bin_index = int(t) // 57150
    bin_counts[bin_index] += 1

# Print row counts per -unit timestamp window
print("\nRow counts per -unit timestamp window:")
for bin_index in sorted(bin_counts):
    start = bin_index * 57150
    end = start + 57149
    count = bin_counts[bin_index]
    print(f"{start:6.0f} – {end:6.0f}: {count} rows")

# # Save to CSV
# output_csv_path = 'RealData/SFHH_sensor_normalized.csv'
#
# # Create the directory if it doesn't exist
# os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
# with open(output_csv_path, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['node1', 'node2', 'timestamp'])  # header
#     writer.writerows(normalized_data)

number = 114300
divisors = [d for d in range(3001, number + 1) if number % d == 0]
print(divisors)