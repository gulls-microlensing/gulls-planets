import os
import pickle
import numpy as np

try:
    from sklearn.mixture import GaussianMixture
except Exception as e:
    GaussianMixture = None


def _transform_targets(inc_deg, phi_deg, period_years):
    # i: use mu = cos(i) with i in radians
    inc_rad = np.deg2rad(inc_deg)
    mu = np.cos(inc_rad)
    # phi: use sin/cos representation to handle circularity
    phi_rad = np.deg2rad(phi_deg)
    sin_phi = np.sin(phi_rad)
    cos_phi = np.cos(phi_rad)
    # P: work in log10 days for numerical stability
    period_days = np.asarray(period_years) * 365.25
    log10P = np.log10(np.clip(period_days, 1e-6, None))
    return np.vstack([mu, sin_phi, cos_phi, log10P]).T


def _inverse_transform_targets(y):
    # y = [mu, sin_phi, cos_phi, log10P]
    mu = y[:, 0]
    sin_phi = y[:, 1]
    cos_phi = y[:, 2]
    log10P = y[:, 3]

    # i from mu
    inc_rad = np.arccos(np.clip(mu, -1.0, 1.0))
    inc_deg = np.rad2deg(inc_rad)

    # phi from atan2
    phi_rad = np.arctan2(sin_phi, cos_phi)
    phi_deg = (np.rad2deg(phi_rad)) % 360.0

    # P from log10 days
    period_days = 10.0 ** log10P
    period_years = period_days / 365.25

    return inc_deg, phi_deg, period_years


def build_bins(xs, xq, nx=20, nq=20, quantile=True):
    if quantile:
        xs_edges = np.quantile(xs, np.linspace(0, 1, nx + 1))
        xq_edges = np.quantile(xq, np.linspace(0, 1, nq + 1))
        # De-duplicate potential equal edges in heavy tails
        xs_edges = np.unique(xs_edges)
        xq_edges = np.unique(xq_edges)
    else:
        xs_edges = np.linspace(xs.min(), xs.max(), nx + 1)
        xq_edges = np.linspace(xq.min(), xq.max(), nq + 1)
    return xs_edges, xq_edges


def fit_binned_gmms(xs, xq, y, xs_edges, xq_edges, min_samples=200, max_components=3, random_state=42):
    if GaussianMixture is None:
        raise RuntimeError("scikit-learn is required (GaussianMixture). Install sklearn in your environment.")

    nx = len(xs_edges) - 1
    nq = len(xq_edges) - 1
    models = {}
    for ix in range(nx):
        xs_lo, xs_hi = xs_edges[ix], xs_edges[ix + 1]
        in_xs = (xs >= xs_lo) & (xs < xs_hi if ix < nx - 1 else xs <= xs_hi)
        for iq in range(nq):
            xq_lo, xq_hi = xq_edges[iq], xq_edges[iq + 1]
            in_xq = (xq >= xq_lo) & (xq < xq_hi if iq < nq - 1 else xq <= xq_hi)
            mask = in_xs & in_xq
            n = int(mask.sum())
            if n < min_samples:
                continue
            data = y[mask]
            # Try 1..max_components and pick by BIC
            best = None
            best_bic = np.inf
            for k in range(1, max_components + 1):
                gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=random_state)
                gmm.fit(data)
                bic = gmm.bic(data)
                if bic < best_bic:
                    best_bic = bic
                    best = gmm
            models[(ix, iq)] = {
                'gmm': best,
                'count': n,
                'xs_range': (xs_lo, xs_hi),
                'xq_range': (xq_lo, xq_hi),
            }
    return models


def save_artifact(path, meta):
    with open(path, 'wb') as f:
        pickle.dump(meta, f)


def load_artifact(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def find_bin(xs_val, xq_val, xs_edges, xq_edges):
    ix = np.searchsorted(xs_edges, xs_val, side='right') - 1
    iq = np.searchsorted(xq_edges, xq_val, side='right') - 1
    ix = np.clip(ix, 0, len(xs_edges) - 2)
    iq = np.clip(iq, 0, len(xq_edges) - 2)
    return int(ix), int(iq)


def sample_from_artifact(artifact, s, q, n_samples=1, return_degrees=True):
    xs = np.log10(s)
    xq = np.log10(q)
    xs_edges = artifact['xs_edges']
    xq_edges = artifact['xq_edges']
    models = artifact['models']
    ix, iq = find_bin(xs, xq, xs_edges, xq_edges)
    key = (ix, iq)
    if key not in models:
        raise KeyError(f"No model for bin {key}; try neighbors or retrain with more data/lower min_samples.")
    gmm = models[key]['gmm']
    samples = gmm.sample(n_samples=n_samples)[0]
    inc_deg, phi_deg, period_years = _inverse_transform_targets(samples)
    if return_degrees:
        return inc_deg, phi_deg, period_years
    else:
        # Return radians for i, phi if requested
        return np.deg2rad(inc_deg), np.deg2rad(phi_deg), period_years
