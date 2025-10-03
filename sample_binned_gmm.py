import argparse
from binned_gmm import load_artifact, sample_from_artifact


def main():
    p = argparse.ArgumentParser(description='Sample (i,phi,P) from binned GMM given (s,q)')
    p.add_argument('--artifact', required=True)
    p.add_argument('--s', type=float, required=True)
    p.add_argument('--q', type=float, required=True)
    p.add_argument('-n', '--n-samples', type=int, default=1)
    args = p.parse_args()

    art = load_artifact(args.artifact)
    inc_deg, phi_deg, period_years = sample_from_artifact(art, args.s, args.q, n_samples=args.n_samples)
    for i, (idg, phdg, py) in enumerate(zip(inc_deg, phi_deg, period_years)):
        print(f"{i}: i={idg:.3f} deg, phi={phdg:.3f} deg, P={py:.6f} yr")


if __name__ == '__main__':
    main()
