
import os
import glob
import argparse
import numpy as np
import pandas as pd
from gmm.binned_gmm import (
	_transform_targets,
	build_bins,
	fit_binned_gmms,
	save_artifact,
)

def iter_master_rows(paths, chunksize=200_000):
	for path in paths:
		try:
			for chunk in pd.read_hdf(path, chunksize=chunksize):
				yield chunk
		except Exception as e:
			print(f"Warning: failed to read {path}: {e}")

def make_dataset(master_glob):
	files = sorted(glob.glob(master_glob))
	if not files:
		raise FileNotFoundError(f"No master files found for pattern: {master_glob}")

	xs_list, xq_list, y_list = [], [], []
	n_rows = 0
	for chunk in iter_master_rows(files):
		cols = chunk.columns
		needed = ['Planet_s', 'Planet_q', 'Planet_inclination', 'Planet_orbphase', 'Planet_period']
		missing = [c for c in needed if c not in cols]
		if missing:
			print(f"Skipping chunk missing columns: {missing}")
			continue

		s = chunk['Planet_s'].to_numpy()
		q = chunk['Planet_q'].to_numpy()
		inc = chunk['Planet_inclination'].to_numpy()    # degrees
		phi = chunk['Planet_orbphase'].to_numpy()       # degrees (0..360)
		P = chunk['Planet_period'].to_numpy()           # years

		# Transform inputs and targets
		xs = np.log10(np.clip(s, 1e-12, None))
		xq = np.log10(np.clip(q, 1e-12, None))
		y = _transform_targets(inc, phi, P)

		xs_list.append(xs)
		xq_list.append(xq)
		y_list.append(y)
		n_rows += len(xs)

	xs_all = np.concatenate(xs_list, axis=0)
	xq_all = np.concatenate(xq_list, axis=0)
	y_all = np.concatenate(y_list, axis=0)
	print(f"Loaded {n_rows} rows total for modeling.")
	return xs_all, xq_all, y_all

def main():
	parser = argparse.ArgumentParser(description="Train binned GMMs for (i,phi,P) | (s,q)")
	parser.add_argument('--glob', required=False, default='filter_selection/6f_overguide_m*/analysis/6f_overguide_m*.out.hdf5',
						help='Glob pattern for master HDF5 files across mass bins')
	parser.add_argument('--nx', type=int, default=20)
	parser.add_argument('--nq', type=int, default=20)
	parser.add_argument('--min-samples', type=int, default=500)
	parser.add_argument('--max-components', type=int, default=3)
	parser.add_argument('--artifact', default='binned_gmm_artifact.pkl')
	args = parser.parse_args()

	xs, xq, y = make_dataset(args.glob)
	xs_edges, xq_edges = build_bins(xs, xq, nx=args.nx, nq=args.nq, quantile=True)
	models = fit_binned_gmms(xs, xq, y, xs_edges, xq_edges,
							 min_samples=args.min_samples,
							 max_components=args.max_components,
							 random_state=42)

	meta = {
		'xs_edges': xs_edges,
		'xq_edges': xq_edges,
		'models': models,
		'transforms': {
			'inputs': 'log10(s), log10(q)',
			'targets': '[mu=cos(i), sin(phi), cos(phi), log10(P_days)]',
		}
	}
	save_artifact(args.artifact, meta)
	print(f"Saved artifact to {args.artifact} with {len(models)} fitted bins.")

if __name__ == '__main__':
	main()
