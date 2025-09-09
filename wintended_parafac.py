# wintended_sparse_parafac.py
# Complete WINTENDED pipeline using sparse PARAFAC
# Requirements: pip install numpy scipy sparse tensorly

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import sparse
import pickle
import tensorly as tl
from tensorly.contrib.sparse.decomposition import parafac as sparse_parafac
from tensorly import cp_to_tensor

from scipy.signal import find_peaks

tl.set_backend("numpy")

@dataclass
class WintendedParams:
    rank: int = 3
    window: int = 20
    max_iter: int = 200
    tol: float = 1e-6
    peak_prominence: float = 0.5
    peak_distance: int = 5
    random_state: int = 42

@dataclass
class WintendedResult:
    score: np.ndarray
    peaks_idx: np.ndarray
    window_scores: List[Dict[str, np.ndarray]]
    factors_per_window: List[Dict[str, np.ndarray]]
    params: WintendedParams

# ---------- Build sparse tensor ----------
def build_sparse_tensor_from_edge_lists(edges_per_t: List[np.ndarray], num_nodes: int, directed: bool = True) -> sparse.COO:
    coords = []
    data = []
    T = len(edges_per_t)
    for t, E in enumerate(edges_per_t):
        if E.size == 0:
            continue
        E = np.asarray(E)
        w = np.ones(E.shape[0])
        u = E[:,0].astype(int)
        v = E[:,1].astype(int)
        for i,j,wi in zip(u,v,w):
            coords.append([i,j,t])
            data.append(wi)
            if not directed:
                coords.append([j,i,t])
                data.append(wi)
    coords = np.array(coords).T
    data = np.array(data)
    return sparse.COO(coords, data, shape=(num_nodes, num_nodes, T))

# ---------- Window density & reconstruction ----------
def _window_density(Xw: sparse.COO) -> float:
    N = Xw.shape[0]
    mask = np.ones((N,N), dtype=bool)
    np.fill_diagonal(mask, False)
    dens = Xw.todense()[mask,:].sum() / (N*(N-1)*Xw.shape[2] + 1e-12)
    return float(dens)

def _reconstruction_error(Xw: sparse.COO, Xw_hat: np.ndarray) -> float:
    num = np.linalg.norm(Xw.todense() - Xw_hat)
    den = np.linalg.norm(Xw.todense()) + 1e-12
    return float(num / den)

# ---------- WINTENDED sparse sliding-window ----------
def wintended_detect_sparse_parafac(X_sparse: sparse.COO, params: WintendedParams) -> WintendedResult:
    N, N2, T = X_sparse.shape
    assert N == N2, "Adjacency slices must be square"
    w = params.window
    score = np.zeros(T, dtype=float)
    window_scores: List[Dict[str,np.ndarray]] = []
    factors_per_window: List[Dict[str,np.ndarray]] = []

    for t_end in range(w, T+1):
        Xw_sparse = X_sparse[:, :, t_end-w:t_end]
        try:
            weights, factors = sparse_parafac(
                Xw_sparse,
                rank=params.rank,
                n_iter_max=params.max_iter,
                tol=params.tol,
                init='random',
                normalize_factors=True,
                random_state=params.random_state
            )
            # Correct reconstruction

            Xw_hat = cp_to_tensor((weights, factors))

        except np.linalg.LinAlgError:
            weights = np.ones(params.rank)
            factors = [np.zeros((N, params.rank)), np.zeros((N, params.rank)), np.zeros((w, params.rank))]
            Xw_hat = np.zeros((N, N, w))

        # Density and reconstruction error
        dens = _window_density(Xw_sparse)
        rec_err = _reconstruction_error(Xw_sparse, Xw_hat)
        s = float(dens * (1.0 + rec_err))
        score[t_end-1] = s

        window_scores.append({
            "t_start": np.array([t_end-w]),
            "t_end": np.array([t_end-1]),
            "density": np.array([dens]),
            "reconstruction_error": np.array([rec_err]),
            "score": np.array([s]),
        })
        print(f"window_scores, {window_scores[-1]}")

        factors_per_window.append({
            "weights": weights,
            "factors": factors
        })

    # Normalize scores
    mu = np.nanmedian(score)
    mad = np.nanmedian(np.abs(score - mu)) + 1e-12
    z = (score - mu) / (1.4826 * mad)

    # Peak detection
    peaks, _ = find_peaks(z, prominence=params.peak_prominence, distance=params.peak_distance)

    return WintendedResult(score=z, peaks_idx=peaks, window_scores=window_scores, factors_per_window=factors_per_window, params=params)

# ---------- CSV loader ----------
def load_csv_to_sparse_tensor(csv_path: str, time_start: int, time_end: int, nNodes: int) -> sparse.COO:
    data = np.genfromtxt(csv_path, delimiter=',', dtype=np.int32)
    if data.ndim == 1:
        data = data.reshape(1,-1)
    mask = (data[:,2] >= time_start) & (data[:,2] < time_end)
    data = data[mask]
    t_adjusted = data[:,2] - time_start
    T = time_end - time_start
    edges_per_t = []
    for t in range(T):
        edges_at_t = data[t_adjusted == t][:,:2]
        edges_per_t.append(edges_at_t)
    return build_sparse_tensor_from_edge_lists(edges_per_t, num_nodes=nNodes, directed=True)

# ---------- Example usage ----------
if __name__ == "__main__":
    import numpy.random as rnd
    N, T = 50, 100
    rng = rnd.default_rng(0)
    edges = []
    for t in range(T):
        p = 0.01 if t not in range(38,45) else 0.05
        m = int(p*N*(N-1))
        uv = rng.integers(0,N,size=(m,2))
        uv = uv[uv[:,0]!=uv[:,1]]
        edges.append(uv)

    X_sparse = build_sparse_tensor_from_edge_lists(edges, num_nodes=N)
    params = WintendedParams(rank=3, window=15, peak_prominence=2.0, peak_distance=8)
    result = wintended_detect_sparse_parafac(X_sparse, params)

    print("Detected event times:", result.peaks_idx.tolist())
