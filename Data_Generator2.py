import torch
import networkx as nx
import matplotlib.pyplot as plt
import math
from itertools import product
import numpy as np
import random
import pickle
import csv
from tensorly.contrib.sparse import tensor as sparse_tensor


def generate_temporal_graph_dataset_new(
    nNodes=5,
    Time=10,
    Features=1,
    num_cycle=2,
    num_comm=2,
    num_samples=1,
    high_prob=0.9,
    low_prob=0.1,
    save_path_features="features.pt",
    save_path_edges="edges.csv",
    visualize=False,
    node_i=0,
    node_j=1,
    num_snapshots=4,
    feat_means=[1.2, 2.5, 3.1, 4.0],
    feat_ranges=[0.2, 0.3, 0.1, 0.4],
    device=None
):
    assert Time % num_cycle == 0, "Time must be divisible by num_cycle"
    block_size = Time // num_cycle

    # Assign nodes to communities
    assert nNodes % num_comm == 0, "nNodes must be divisible by num_communities"
    node_communities = torch.div(torch.arange(nNodes), nNodes // num_comm, rounding_mode='floor')

    # # Build probability tensor P[a, b, t_block]
    # P = torch.full((num_comm, num_comm, num_cycle), low_prob / 10)
    # for c in range(num_comm):
    #     for t_block in range(num_cycle):
    #         if t_block % 2 == 0:
    #             P[c, c, t_block] = high_prob
    #         else:
    #             P[c, c, t_block] = low_prob



    means, ranges = generate_means_ranges(feat_means, feat_ranges, num_comm, num_cycle=2)
    means = torch.tensor(means, dtype=torch.float16)
    ranges = torch.tensor(ranges, dtype=torch.float16)

    # Preallocate feature tensor X: shape (nNodes, Time, Features)
    X = torch.zeros((nNodes, Time, Features), dtype=torch.float16, device=device)

    # Generate features per node over time
    for node in range(nNodes):
        comm = node_communities[node]
        for t in range(Time):
            cycle = (t // block_size) % 2
            mean = means[comm, cycle]
            spread = ranges[comm, cycle]
            X[node, t, 0] = torch.normal(mean, spread, size=(1,))


    torch.save({'features': X}, save_path_features)


    idx_split = nNodes // num_comm  # Assuming num_comm is defined

    with open(save_path_edges, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

    for sample_i in range(num_samples):


        for t in range(Time):
            t_block = t // block_size
            block_is_high = (t_block % 2 == 0)

            intra_prob = high_prob if block_is_high else low_prob
            inter_prob = low_prob / 10

            num_edges_comm = int(intra_prob * (idx_split ** 2)) // 2
            num_edges_comm12 = int(inter_prob * (idx_split ** 2)) // 2

            # Community 1 edges
            edges1 = sample_unique_edges(num_edges_comm, (0, idx_split), (0, idx_split), t, self_loop='False')
            # Community 2 edges
            edges2 = sample_unique_edges(num_edges_comm, (idx_split, nNodes), (idx_split, nNodes), t, self_loop='False')
            # Inter-community edges
            edges3 = sample_unique_edges(num_edges_comm12, (0, idx_split), (idx_split, nNodes), t, self_loop='False')

            with open(save_path_edges, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(list(edges1))
                writer.writerows(list(edges2))
                writer.writerows(list(edges3))

            print(f"Time {t} processed.")


    # if visualize:
    #     _visualize_temporal_graph(all_adj_EdgeSet[0], Time, node_i, node_j, num_snapshots)

def generate_means_ranges(means, ranges, num_comm, num_cycle=2):
    # Ensure the inputs have the correct shape
    assert len(means) == num_comm * num_cycle, "Length of means must match num_comm * num_cycle"
    assert len(ranges) == num_comm * num_cycle, "Length of ranges must match num_comm * num_cycle"

    community_cycle_means = np.array(means).reshape(num_comm, num_cycle)
    community_cycle_ranges = np.array(ranges).reshape(num_comm, num_cycle)

    return community_cycle_means, community_cycle_ranges

def sample_unique_edges(n, node_range_a, node_range_b, t, self_loop='False'):
    edges = set()
    # attempts = 0
    # max_attempts = n * 5 and attempts < max_attempts

    for i in range(n):
        # Vectorized sampling n edges at once
        u = random.randint(node_range_a[0], node_range_a[1]-1)
        v = random.randint(node_range_b[0], node_range_b[1]-1)
        if self_loop and u==v:
            continue
        else:
            u = str(u)
            v = str(v)
            t = str(t)
            edges.add((u, v, t))
            edges.add((v, u, t))
        # attempts += 1

    # if len(edges) < n:
    #     print(f"Warning: only got {len(edges)} edges out of {n} after {attempts} attempts")

    return edges

############## visualisation ###################
def _visualize_temporal_graph(A_tensor, Time, node_i, node_j, num_snapshots):
    """
    Visualize specific time points of a temporal adjacency tensor and edge time series.

    Args:
        A_tensor (torch.Tensor): Adjacency tensor of shape (n, n, T)
        Time (int): Total number of time steps
        node_i (int): Source node index
        node_j (int): Target node index
        selected_t (list[int], optional): List of time indices to visualize. If None, show all.
    """
    # Default to first 4 time steps


    selected_t = np.linspace(0, Time - 1, num_snapshots, dtype=int).tolist()  # e.g., [0, 6, 13, 19] for T=20

    fig, axes = plt.subplots(1, len(selected_t), figsize=(len(selected_t) * 3, 3))

    if len(selected_t) == 1:
        axes = [axes]

    for idx, t in enumerate(selected_t):
        adj_matrix = A_tensor.to_dense()[:, :, t].detach().cpu().numpy()

        G = nx.from_numpy_array(adj_matrix)

        pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistency
        ax = axes[idx]

        nx.draw(G, pos, ax=ax, node_size=300, with_labels=True,
                labels={i: str(i) for i in G.nodes}, font_size=10)
        ax.set_title(f'Time {t}')

    plt.tight_layout()

    # Plot edge time series for full timeline
    edge_timeseries = A_tensor.to_dense()[node_i, node_j, :].detach().cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.plot(range(Time), edge_timeseries, marker='o', linestyle='-')
    plt.title(f'Edge presence between node {node_i} and node {node_j} over time')
    plt.xlabel('Time step')
    plt.xticks(range(Time))
    plt.ylabel('Edge weight')
    plt.ylim(edge_timeseries.min() - 0.1, edge_timeseries.max() + 0.1)
    plt.grid(True)
    plt.show()


def split_sum_into_3d(num_units):
    if num_units < 3:
        raise ValueError("Cannot split values less than 3 into 3 positive parts")

    # Start by dividing equally
    base = num_units // 3
    remainder = num_units % 3

    # Distribute the remainder to keep values close
    parts = [base] * 3
    for i in range(remainder):
        parts[i] += 1

    return tuple(sorted(parts))
