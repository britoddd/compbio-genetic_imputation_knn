#!/usr/bin/env python3
"""
Demo & integration test for KNN genetic imputation.

Generates synthetic SNP data, introduces random missingness, runs imputation,
and prints accuracy metrics. No external files needed.

Run:
    python demo.py
    python demo.py --samples 200 --snps 500 --missing 0.2 --k 7
"""

import argparse
import numpy as np
import pandas as pd
from knn_imputer import KNNGenotypeImputer, evaluate, MISSING


def simulate_genotypes(
    n_samples: int,
    n_snps: int,
    missing_rate: float,
    maf_range: tuple = (0.05, 0.5),
    ld_blocks: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """
    Simulate a genotype matrix with LD block structure.

    Each LD block shares a latent haplotype; SNPs within a block are
    correlated, giving KNN genuine signal to exploit. Without LD, all
    SNPs are independent and KNN reduces to frequency-based imputation.

    Missing values are introduced completely at random (MCAR).
    """
    rng = np.random.default_rng(seed)
    mafs = rng.uniform(*maf_range, size=n_snps)
    block_size = n_snps // ld_blocks

    X = np.zeros((n_samples, n_snps), dtype=np.int8)
    for b in range(ld_blocks):
        start = b * block_size
        end = start + block_size if b < ld_blocks - 1 else n_snps
        block_snps = end - start

        # Latent haplotype pair per sample (0 or 1 for each chromosome)
        hap1 = rng.binomial(1, mafs[start:end], size=(n_samples, block_snps))
        hap2 = rng.binomial(1, mafs[start:end], size=(n_samples, block_snps))
        # Add correlated noise within block (shared ancestry proxy)
        shared = rng.binomial(1, mafs[start:end], size=(n_samples, block_snps))
        hap1 = np.where(rng.random((n_samples, block_snps)) < 0.7, shared, hap1)
        hap2 = np.where(rng.random((n_samples, block_snps)) < 0.7, shared, hap2)
        X[:, start:end] = (hap1 + hap2).clip(0, 2).astype(np.int8)

    mask = rng.random(X.shape) < missing_rate
    X[mask] = MISSING
    return X


def run_demo(n_samples: int, n_snps: int, missing_rate: float, k: int, seed: int) -> None:
    print(f"\n{'='*55}")
    print(f"  KNN Genetic Imputation Demo")
    print(f"  Samples: {n_samples}  |  SNPs: {n_snps}  |  Missing: {missing_rate:.0%}  |  k: {k}")
    print(f"{'='*55}\n")

    X = simulate_genotypes(n_samples, n_snps, missing_rate, ld_blocks=10, seed=seed)
    actual_miss = (X == MISSING).mean()
    print(f"[Data]  {n_samples} samples × {n_snps} SNPs  |  missing rate: {actual_miss:.2%}")

    # ---- Evaluation (mask-and-impute) ----
    imputer = KNNGenotypeImputer(k=k, weight="distance", min_shared_snps=5)
    metrics = evaluate(imputer, X, mask_fraction=0.10, seed=seed + 1)

    print("\n[Evaluation — masked 10% of observed genotypes]")
    print(f"  Concordance (hard call accuracy):  {metrics['concordance']:.4f}  ({metrics['concordance']*100:.2f}%)")
    print(f"  Imputation R²  (dosage vs truth):  {metrics['r2']:.4f}")
    print(f"  Genotypes masked / total:          {metrics['n_masked']} / {metrics['n_total_observed']}")

    # ---- Full imputation ----
    imputer.fit(X)
    hard_calls, dosage, confidence = imputer.impute(X)

    remaining_miss = (hard_calls == MISSING).mean()
    print(f"\n[Imputation]  Remaining MISSING after imputation: {remaining_miss:.4%}")

    # ---- Spot-check: show 5 samples × 10 SNPs ----
    print("\n[Spot check]  Hard calls — first 5 samples, first 10 SNPs")
    df = pd.DataFrame(hard_calls[:5, :10])
    df.index = [f"S{i+1}" for i in range(5)]
    df.columns = [f"SNP{j+1}" for j in range(10)]
    print(df.to_string())

    print("\n[Spot check]  Dosage (weighted mean) — same region")
    df_dos = pd.DataFrame(dosage[:5, :10].round(3))
    df_dos.index = df.index
    df_dos.columns = df.columns
    print(df_dos.to_string())

    # ---- k sensitivity ----
    print("\n[k sensitivity]  Concordance at different k values:")
    print(f"  {'k':>4}  {'Concordance':>12}  {'R²':>8}")
    for ki in [1, 3, 5, 10, 20]:
        if ki > n_samples - 1:
            continue
        imp = KNNGenotypeImputer(k=ki, weight="distance", min_shared_snps=5)
        m = evaluate(imp, X, mask_fraction=0.10, seed=seed + 1)
        print(f"  {ki:>4}  {m['concordance']:>12.4f}  {m['r2']:>8.4f}")

    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KNN imputation demo")
    p.add_argument("--samples",  type=int,   default=100,  help="Number of samples")
    p.add_argument("--snps",     type=int,   default=300,  help="Number of SNPs")
    p.add_argument("--missing",  type=float, default=0.15, help="Missing rate (0–1)")
    p.add_argument("--k",        type=int,   default=5,    help="Number of neighbors")
    p.add_argument("--seed",     type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(args.samples, args.snps, args.missing, args.k, args.seed)
