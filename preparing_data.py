

import torch

def split_dataset(adj_list_full, feat_list_full, train_ratio=0.8, seed=42):
    """
    Splits the dataset into training and validation sets.

    Parameters:
        adj_list_full (list): Full list of adjacency matrices.
        feat_list_full (list): Full list of feature matrices.
        train_ratio (float): Ratio of data to use for training.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        adj_list_train, feat_list_train, adj_list_val, feat_list_val
    """
    if seed is not None:
        torch.manual_seed(seed)

    num_samples = len(adj_list_full)
    split_idx = int(train_ratio * num_samples)
    indices = torch.randperm(num_samples)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    adj_list_train = [adj_list_full[i] for i in train_indices]
    feat_list_train = [feat_list_full[i] for i in train_indices]

    adj_list_val = [adj_list_full[i] for i in val_indices]
    feat_list_val = [feat_list_full[i] for i in val_indices]

    return adj_list_train, feat_list_train, adj_list_val, feat_list_val


def split_dataset_sparse(feat_list_full, sparse_adj_list, num_samples, train_ratio=0.8, seed=42):
    """
    Splits the dataset into training and validation sets for sparse representation.

    Parameters:
        adj_list_dense (list): Full list of dense adjacency matrices (for alignment only).
        feat_list_full (list): Full list of feature matrices.
        sparse_adj_list (list of dict): Sparse representations of adjacency matrices.
        train_ratio (float): Ratio of data to use for training.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        sparse_adj_train, feat_list_train, sparse_adj_val, feat_list_val
    """
    if seed is not None:
        torch.manual_seed(seed)

    split_idx = int(train_ratio * num_samples)
    indices = torch.randperm(num_samples)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    sparse_adj_train = [sparse_adj_list[i] for i in train_indices]
    feat_list_train = [feat_list_full[i] for i in train_indices]

    sparse_adj_val = [sparse_adj_list[i] for i in val_indices]
    feat_list_val = [feat_list_full[i] for i in val_indices]

    return sparse_adj_train, feat_list_train, sparse_adj_val, feat_list_val
