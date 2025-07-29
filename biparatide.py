import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import os
import csv

def visualize_bipartite_temporal_graph(
    edge_csv_path,
    nU,
    nV,
    Time,
    node_u=0,
    node_v=0,
    num_snapshots=4,
    seed=42
):
    """
    Visualizes selected time steps of a bipartite temporal graph and the presence time series of one U-V edge.

    Parameters:
        edge_csv_path: Path to CSV with edges in format (u, v, t)
        nU, nV: Number of nodes in sets U and V
        Time: Total number of time steps
        node_u, node_v: Specific edge (U-node, V-node) to track presence over time
        num_snapshots: Number of time snapshots to plot
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    # Read edges and organize by time
    edge_dict = defaultdict(list)
    edge_timeseries = np.zeros(Time)

    with open(edge_csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            u, v, t = map(int, row)
            edge_dict[t].append((u, v))
            if u == node_u and v == nU + node_v:
                edge_timeseries[t] = 1

    # Select time steps to visualize evenly
    selected_t = np.linspace(0, Time - 1, num_snapshots, dtype=int)

    fig, axes = plt.subplots(1, len(selected_t), figsize=(len(selected_t) * 4, 4), squeeze=False)

    for i, t in enumerate(selected_t):
        G = nx.Graph()
        G.add_nodes_from(range(nU), bipartite=0)
        G.add_nodes_from(range(nU, nU + nV), bipartite=1)
        G.add_edges_from(edge_dict[t])

        # Create bipartite layout: U nodes at x=0, V nodes at x=1, y spread evenly
        pos = {u: (0, u) for u in range(nU)}
        pos.update({v: (1, v - nU) for v in range(nU, nU + nV)})

        ax = axes[0, i]

        # Draw nodes with different shapes/colors
        nx.draw_networkx_nodes(G, pos, nodelist=range(nU), node_color='skyblue',
                               node_shape='o', node_size=100, ax=ax, label='U nodes')
        nx.draw_networkx_nodes(G, pos, nodelist=range(nU, nU + nV), node_color='salmon',
                               node_shape='s', node_size=100, ax=ax, label='V nodes')
        nx.draw_networkx_edges(G, pos, ax=ax)

        ax.set_title(f"Time {t}")
        ax.axis('off')

        # Add legend only on first subplot
        if i == 0:
            ax.legend(scatterpoints=1, loc='upper right')

    plt.tight_layout()

    # Plot edge presence over time for the specific edge (u,v)
    plt.figure(figsize=(8, 3.5))
    plt.plot(range(Time), edge_timeseries, marker='o', color='purple')
    plt.title(f"Presence of edge (U={node_u}, V={node_v}) over time")
    plt.xlabel("Time")
    plt.ylabel("Presence (1=present, 0=absent)")
    plt.xticks(range(0, Time, max(1, Time // 10)))
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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
            str_u = str(u)
            str_v = str(v)
            str_t = str(t)
            edges.add((str_u, str_v, str_t))
            # edges.add(v, u, t)  # undirected
        if len(edges) >= n * 2:
            break
    return edges

def generate_bipartite_graph_with_cycles(
    nU=100,
    nV=100,
    Time=100,
    density_high=0.3,
    density_low=0.003,
    num_cycles=5,
    save_path_edges="bipartite_edges_cyclic.csv",
    seed=42,
    visualisation = False,
    vis_node_u=10,
    vis_node_v=20,
    vis_snapshots=5
):

    random.seed(seed)
    os.makedirs(os.path.dirname(save_path_edges), exist_ok=True)

    if os.path.exists(save_path_edges):
        os.remove(save_path_edges)

    nNodes = nU + nV
    node_range_u = (0, nU - 1)
    node_range_v = (nU, nU + nV - 1)
    max_possible_edges = nU * nV

    block_size = Time // (num_cycles * 2)  # half-on, half-off per cycle

    with open(save_path_edges, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        for t in range(Time):
            # Determine whether current time is in high or low activity phase
            cycle_position = (t // block_size) % 2
            # edges_per_t = edges_high if cycle_position == 0 else edges_low
            density = density_high if cycle_position == 0 else density_low
            edges_per_t = int(density * max_possible_edges)


            edges = sample_unique_edges(
                n=edges_per_t,
                node_range_a=node_range_u,
                node_range_b=node_range_v,
                t=t,
                nNodes=nNodes,
                Time=Time,
                self_loop=False
            )
            writer.writerows(list(edges))


    if visualisation:
        visualize_bipartite_temporal_graph(
            save_path_edges,
            nU,
            nV,
            Time,
            node_u=vis_node_u,
            node_v=vis_node_v,
            num_snapshots=vis_snapshots,
            seed=42
        )


    print(f"✅ Cyclic bipartite edges saved to: {save_path_edges}")

# # Example usage:
# generate_bipartite_graph_with_cycles(
#     nU=2500,
#     nV=2500,
#     Time=2000,
#     edges_high=0.005,
#     edges_low=0.0003,
#     num_cycles=400,
#     save_path_edges="structural_anomalies5/anomaly7/bipartite_features_cycle400_time2000.csv",
#     seed=42,
#     visualisation = False
# )