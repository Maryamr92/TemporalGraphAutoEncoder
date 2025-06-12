# temporal_graph_generator.py

import torch
import networkx as nx
import matplotlib.pyplot as plt
import math
from itertools import product
import numpy as np

from tensorly.contrib.sparse import tensor as sparse_tensor



def dense_to_sparse_tensor(A):
    """
    Convert a dense 3D tensor to sparse COO format (indices, values).
    """
    # A = A.numpy()
    indices = np.array(np.nonzero(A))
    values = A[indices[0], indices[1], indices[2]]
    shape = A.shape

    return torch.tensor(indices, dtype=torch.long), torch.tensor(values, dtype=torch.float32), shape



def generate_temporal_graph_dataset(
        nNodes=5,
        Time=10,
        Features=1,
        num_cycle=2,
        num_comm=2,
        num_samples=1,
        high_prob=0.9,
        low_prob=0.1,
        save_path="temporal_graph_dataset_samples.pt",
        visualize=False,
        node_i=0,
        node_j=1,
        num_snapshots = 4,
        feat_means = [1.2, 2.5, 3.1, 4.0],
        feat_ranges = [0.2, 0.3, 0.1, 0.4]

):
    """
    Generate synthetic temporal graph dataset with node features and save it.

    Args:
        n (int): Number of nodes.
        T (int): Number of time steps.
        F (int): Number of node features.
        num_blocks (int): Number of distinct time blocks for probability changes.
        num_comm (int): Number of communities into which nodes are divided.
        num_samples (int): Number of temporal graph samples to generate.
        high_prob (float): High intra-group interaction probability.
        low_prob (float): Low intra-group interaction probability.
        save_path (str): Path to save the generated dataset.
        visualize (bool): Whether to visualize the generated graphs and edges.
        node_i (int): First node to track edge presence (only if visualize=True).
        node_j (int): Second node to track edge presence (only if visualize=True).

    Returns:
        dict: Dictionary containing adjacency and feature tensors.
    """

    # Assert T is divisible by num_blocks
    assert Time % num_cycle == 0, "T must be evenly divisible by num_blocks"
    block_size = Time // num_cycle

    # Assign nodes to communities
    node_communities = np.array([i * num_comm // nNodes for i in range(nNodes)])

    P = np.zeros((num_comm, num_comm, num_cycle))

    for a in range(num_comm):
        for b in range(num_comm):
            for t_block in range(num_cycle):
                if a == b:
                    # Intra-group interaction probabilities
                    prob = high_prob if t_block % 2 == 0 else low_prob
                else:
                    # Inter-group interaction probabilities (lower)
                    prob = (high_prob / 10) if t_block % 2 == 0 else (low_prob / 10)
                    # prob = low_prob

                P[a, b, t_block] = prob

    # Initialize storage for all samples
    all_adj_tensors = []
    all_feat_tensors = []

    # Generate random node features (shared across samples)
    # X = torch.randn((nNodes, Time, Features))

    means, ranges = generate_means_ranges(feat_means, feat_ranges, num_comm, num_cycle=2)

    print("Means:\n", means)
    print("Ranges:\n", ranges)

    # Initialize X with zeros
    X = torch.zeros((nNodes, Time, Features))

    for node in range(nNodes):
        comm = node_communities[node]
        for t in range(Time):
            t_block = t // block_size
            # Repeat only first two cycles if more
            effective_cycle = t_block % 2

            mean = means[comm, effective_cycle]
            spread = ranges[comm, effective_cycle]

            # Draw from normal distribution (can be changed to uniform if needed)
            value = np.random.normal(loc=mean, scale=spread)
            X[node, t, 0] = value  # Features assumed to be 1  **** if changes here we need to change the code ****



    ## Generate samples
    for _ in range(num_samples):
        A = np.zeros((nNodes, nNodes, Time))  # Adjacency tensor (n, n, T)

        for t in range(Time):
            # t = Time // priode #  in baraye dashtane chanta pride zamani hast
            t_block = t // block_size
            # print(f't_block', {t_block})
            for i in range(nNodes):
                for j in range(i +1 , nNodes):  # Only upper triangle needed (symmetry)
                    # group_i = 0 if i < split_idx else 1
                    # group_j = 0 if j < split_idx else 1
                    # if np.random.rand() <= P[group_i, group_j, t_block]:
                    #     A[i, j, t] = A[j, i, t] = 1  # Symmetric graph
                    group_i = node_communities[i]
                    group_j = node_communities[j]
                    prob = P[group_i, group_j, t_block]
                    if np.random.rand() <= prob:
                        A[i, j, t] = A[j, i, t] = 1  # Symmetric graph

        # Store tensors
        all_adj_tensors.append(torch.tensor(A, dtype=torch.float32))
        all_feat_tensors.append(X.clone())

        # Define means and ranges for each community and cycle

        # means, ranges = generate_means_ranges(num_comm, num_cycle=2)
        # means, ranges = generate_means_ranges2(feat_means, feat_ranges, num_comm, num_cycle=2)
        #
        # print("Means:\n", means)
        # print("Ranges:\n", ranges)
        #
        # # Initialize X with zeros
        # X = torch.zeros((nNodes, Time, Features))
        #
        # for node in range(nNodes):
        #     comm = node_communities[node]
        #     for t in range(Time):
        #         t_block = t // block_size
        #         # Repeat only first two cycles if more
        #         effective_cycle = t_block % 2
        #
        #         mean = means[comm, effective_cycle]
        #         spread = ranges[comm, effective_cycle]
        #
        #         # Draw from normal distribution (can be changed to uniform if needed)
        #         value = np.random.normal(loc=mean, scale=spread)
        #         X[node, t, 0] = value  # Features assumed to be 1  **** if changes here we need to change the code ****
        #
        # all_feat_tensors.append(X.clone())


    # Pack dataset
    dataset = {
        "adj": all_adj_tensors,  # List of adjacency tensors
        "feat": all_feat_tensors  # List of feature tensors
    }

    # Save dataset
    torch.save(dataset, save_path)

    # Optionally visualize
    if visualize and num_samples > 0:
        _visualize_temporal_graph(all_adj_tensors[0], Time, node_i, node_j, num_snapshots)

    return dataset


def generate_means_ranges(means, ranges, num_comm, num_cycle=2):
    # Ensure the inputs have the correct shape
    assert len(means) == num_comm * num_cycle, "Length of means must match num_comm * num_cycle"
    assert len(ranges) == num_comm * num_cycle, "Length of ranges must match num_comm * num_cycle"

    community_cycle_means = np.array(means).reshape(num_comm, num_cycle)
    community_cycle_ranges = np.array(ranges).reshape(num_comm, num_cycle)

    return community_cycle_means, community_cycle_ranges


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
        adj_matrix = A_tensor[:, :, t].detach().cpu().numpy()
        G = nx.from_numpy_array(adj_matrix)

        pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistency
        ax = axes[idx]

        nx.draw(G, pos, ax=ax, node_size=300, with_labels=True,
                labels={i: str(i) for i in G.nodes}, font_size=10)
        ax.set_title(f'Time {t}')

    plt.tight_layout()

    # Plot edge time series for full timeline
    edge_timeseries = A_tensor[node_i, node_j, :].detach().cpu().numpy()
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


