# temporal_graph_generator.py

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt


def generate_temporal_graph_dataset(
        nNodes=5,
        Time=10,
        Features=1,
        num_blocks=2,
        num_comm=2,
        num_samples=1,
        high_prob=0.8,
        low_prob=0.1,
        save_path="temporal_graph_dataset_samples.pt",
        visualize=False,
        node_i=0,
        node_j=1
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
    assert Time % num_blocks == 0, "T must be evenly divisible by num_blocks"
    block_size = Time // num_blocks

    # Divide nodes into two groups
    # num_comm = 1
    split_idx = nNodes // num_comm
    # V1 = list(range(split_idx))
    # V2 = list(range(split_idx, nNodes))
    # Assign nodes to communities
    node_communities = np.array([i * num_comm // nNodes for i in range(nNodes)])

    P = np.zeros((num_comm, num_comm, num_blocks))

    for a in range(num_comm):
        for b in range(num_comm):
            for t_block in range(num_blocks):
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
    X = torch.randn((nNodes, Time, Features))

    # Generate samples
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
    # Pack dataset
    dataset = {
        "adj": all_adj_tensors,  # List of adjacency tensors
        "feat": all_feat_tensors  # List of feature tensors
    }
    # print(all_adj_tensors[0].shape)

    # Save dataset
    torch.save(dataset, save_path)

    # Optionally visualize
    if visualize and num_samples > 0:
        _visualize_temporal_graph(all_adj_tensors[0], Time, node_i, node_j, num_blocks)

    return dataset


def _visualize_temporal_graph(A_tensor, Time, node_i, node_j, num_blocks):
    """
    Internal helper to visualize temporal graphs and a specific edge over time.

    Args:
        A_tensor (torch.Tensor): Adjacency tensor (n, n, T)
        T (int): Number of time steps
        node_i (int): First node to track
        node_j (int): Second node to track
    """
    fig, axes = plt.subplots(1, Time, figsize=(Time * 3, 3))

    for t in range(int(Time)):
        adj_matrix = A_tensor[:, :, t].numpy()
        G = nx.from_numpy_array(adj_matrix)

        pos = nx.spring_layout(G, seed=42)  # Consistent layout across time
        ax = axes[t]

        nx.draw(G, pos, ax=ax, node_size=300, with_labels=True,
                labels={i: str(i) for i in G.nodes}, font_size=10)

        ax.set_title(f'Time {t}')

    plt.tight_layout()

    # Visualize edge presence over time between selected nodes
    edge_timeseries = A_tensor[node_i, node_j, :].numpy()

    plt.figure(figsize=(8, 4))
    plt.plot(range(Time), edge_timeseries, marker='o', linestyle='-')
    plt.title(f'Edge presence between node {node_i} and node {node_j} over time')
    plt.xlabel('Time step')
    plt.xticks(range(Time))
    plt.ylabel('Edge (1 = exists, 0 = not)')
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.show()


