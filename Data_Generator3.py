import csv
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import random


def generate_stochastic_graph_with_communities_and_cycles(
    nNodes=1000,
    Time=10,
    Features=1,
    num_comm=2,
    num_cycle=2,
    num_samples=1,
    high_prob=0.9,
    low_prob=0.1,
    save_path_prefix="output",
    feat_means=None,
    feat_ranges=None,
    device=None,
    visualising=False,
    num_snapshots=4,
    node_i=2,
    node_j=5
):
    assert nNodes % num_comm == 0, "nNodes must be divisible by num_comm"
    if num_cycle > 0:
        assert Time % num_cycle == 0, "Time must be divisible by num_cycle"

    idx_split = nNodes // num_comm
    block_size = Time // num_cycle if num_cycle > 0 else Time
    node_communities = torch.div(torch.arange(nNodes), idx_split, rounding_mode='floor')

    # Generate feature distributions
    if num_cycle == 0:
        feat_means = feat_means or np.random.uniform(1, 5, num_comm)
        feat_ranges = feat_ranges or np.random.uniform(0.1, 0.4, num_comm)
        means = torch.tensor(feat_means, dtype=torch.float16)
        ranges = torch.tensor(feat_ranges, dtype=torch.float16)
    else:
        feat_means = feat_means or np.random.uniform(1, 5, num_comm * num_cycle)
        feat_ranges = feat_ranges or np.random.uniform(0.1, 0.4, num_comm * num_cycle)
        means, ranges = generate_means_ranges(feat_means, feat_ranges, num_comm, num_cycle)
        means = torch.tensor(means, dtype=torch.float16)
        ranges = torch.tensor(ranges, dtype=torch.float16)

    # Create and fill feature tensor
    X = torch.zeros((nNodes, Time, Features), dtype=torch.float16, device=device)
    for node in range(nNodes):
        comm = node_communities[node]
        for t in range(Time):
            if num_cycle == 0:
                mean = means[comm]
                spread = ranges[comm]
            else:
                cycle = (t // block_size) % num_cycle
                mean = means[comm, cycle]
                spread = ranges[comm, cycle]
            X[node, t, 0] = torch.normal(mean, spread, size=(1,))

    # Save features
    feature_path = f"{save_path_prefix}_features_comm{num_comm}_cycle{num_cycle}, time:{Time}.pt"
    torch.save({'features': X}, feature_path)

    # Save edges (safe overwrite)
    edge_path = f"{save_path_prefix}_edges_comm{num_comm}_cycle{num_cycle}, time:{Time}.csv"
    if os.path.exists(edge_path):
        os.remove(edge_path)

    comm_ranges = [(i * idx_split, (i + 1) * idx_split) for i in range(num_comm)]

    with open(edge_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        for t in range(Time):
            block_is_high = (t // block_size) % 2 == 0 if num_cycle > 0 else True
            intra_prob = high_prob if block_is_high else low_prob
            inter_prob = low_prob / 10

            # Intra-community edges
            for start, end in comm_ranges:
                num_edges = int(intra_prob * ((end - start) ** 2) // 2)
                edges = sample_unique_edges(num_edges, (start, end), (start, end), t, nNodes, Time)
                writer.writerows(edges)

            # Inter-community edges
            for i in range(num_comm):
                for j in range(i + 1, num_comm):
                    start_i, end_i = comm_ranges[i]
                    start_j, end_j = comm_ranges[j]
                    num_edges = int(inter_prob * ((end_i - start_i) * (end_j - start_j)) // 2)
                    edges = sample_unique_edges(num_edges, (start_i, end_i), (start_j, end_j), t, nNodes, Time)
                    writer.writerows(edges)

            # print(f"[comm={num_comm}, cycle={num_cycle}] Time {t} processed.")

    print(f"Graph saved to: {edge_path} and {feature_path}")



    if visualising:
        A_tensor = load_edges_to_adjacency_tensor(edge_path, nNodes, Time)
        visualize_temporal_graph(A_tensor, Time, node_i, node_j, num_snapshots=num_snapshots)

    return edge_path


def sample_unique_edges(n, node_range_a, node_range_b, t, nNodes, Time, self_loop=False):
    """
    Returns a set of unique (u,v,t) edges with time-step safety.
    """
    edges = set()
    for _ in range(n * 2):  # attempt more times to ensure uniqueness
        u = random.randint(*node_range_a)
        v = random.randint(*node_range_b)
        if not self_loop and u == v:
            continue
        if 0 <= u < nNodes and 0 <= v < nNodes and 0 <= t < Time:
            edges.add((str(u), str(v), str(t)))
            edges.add((str(v), str(u), str(t)))  # undirected
        if len(edges) >= n * 2:
            break
    return edges


def generate_means_ranges(means, ranges, num_comm, num_cycle=2):
    assert len(means) == num_comm * num_cycle
    assert len(ranges) == num_comm * num_cycle
    return np.array(means).reshape(num_comm, num_cycle), np.array(ranges).reshape(num_comm, num_cycle)


def load_edges_to_adjacency_tensor(edge_csv_path, nNodes, Time):
    A_tensor = torch.zeros((nNodes, nNodes, Time), dtype=torch.float32)
    with open(edge_csv_path, 'r') as f:
        for line in f:
            try:
                u, v, t = map(int, line.strip().split(','))
                if u < nNodes and v < nNodes and t < Time:
                    A_tensor[u, v, t] = 1.0
                    A_tensor[v, u, t] = 1.0
            except ValueError:
                print(f"⚠️ Skipping invalid line: {line.strip()}")
    return A_tensor


def visualize_temporal_graph(A_tensor, Time, node_i, node_j, num_snapshots=4):
    selected_t = np.linspace(0, Time - 1, num_snapshots, dtype=int).tolist()
    fig, axes = plt.subplots(1, len(selected_t), figsize=(len(selected_t) * 4, 3))

    if len(selected_t) == 1:
        axes = [axes]

    for idx, t in enumerate(selected_t):
        adj_matrix = A_tensor[:, :, t].detach().cpu().numpy()
        G = nx.from_numpy_array(adj_matrix)
        pos = nx.spring_layout(G, seed=42)
        ax = axes[idx]
        nx.draw(G, pos, ax=ax, node_size=200, with_labels=True, font_size=9)
        ax.set_title(f'Time {t}')
    plt.tight_layout()

    # Plot edge time series
    edge_series = A_tensor[node_i, node_j, :].detach().cpu().numpy()
    plt.figure(figsize=(8, 3.5))
    plt.plot(range(Time), edge_series, marker='o')
    plt.title(f"Edge presence between Node {node_i} and {node_j}")
    plt.xlabel("Time")
    plt.ylabel("Edge Weight")
    plt.grid(True)
    plt.show()

# #
# # # ✅ Run test
# if __name__ == "__main__":
#
#
#     generate_stochastic_graph_with_communities_and_cycles(
#         nNodes=100,
#         Time=1000,
#         Features=1,
#         num_comm=10,
#         num_cycle=2,
#         high_prob=0.5,
#         low_prob=0.05,
#         save_path_prefix="structural_anomalies3/anomaly5/",
#         visualising=False,
#         num_snapshots=10,
#         node_i=2,
#         node_j=5
#     )


def generate_random_temporal_graph(
    nNodes=2000,
    Time=1000,
    Features=1,
    edge_prob=0.001,
    save_path_edges="random_edges.csv",
    save_path_features="random_features.pt",
    seed=42,
):
    random.seed(seed)
    torch.manual_seed(seed)

    # Generate random features
    X = torch.randn((nNodes, Time, Features), dtype=torch.float16)
    torch.save({'features': X}, save_path_features)

    # Remove old edge file
    if os.path.exists(save_path_edges):
        os.remove(save_path_edges)

    with open(save_path_edges, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        for t in range(Time):
            num_edges = int(edge_prob * nNodes**2)
            sampled = set()
            while len(sampled) < num_edges:
                u = random.randint(0, nNodes - 1)
                v = random.randint(0, nNodes - 1)
                if u != v and (u, v) not in sampled:
                    sampled.add((u, v))
                    sampled.add((v, u))  # undirected
                    writer.writerow([u, v, t])

        print(f"✅ Edges saved to: {save_path_edges}")
        print(f"✅ Features saved to: {save_path_features}")



# if __name__ == "__main__":
#     generate_random_temporal_graph(
#         nNodes=100,
#         Time=100,
#         Features=1,
#         edge_prob=0.1,
#         save_path_edges="structural_anomalies2/anomaly10/random_edges_0.1.csv",
#         save_path_features="structural_anomalies2/anomaly10/random_features_0.1.pt"
#     )

