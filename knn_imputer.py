#!/usr/bin/env python3
"""
KNN-based Genetic Imputation for Computational Biology

Imputes missing SNP genotypes using K-Nearest Neighbors with masked Hamming
distance. Only positions where BOTH samples have observed genotypes contribute
to the distance, making it robust to arbitrary missingness patterns.

Supports:
  - Input:  CSV/TSV matrix | VCF | PLINK binary (.bed/.bim/.fam)
  - Output: hard calls (0/1/2) + dosage (weighted mean) + confidence score
  - Eval:   random masking of known genotypes → concordance & R²

Usage (CLI):
  python knn_imputer.py data.csv --evaluate --k 7
  python knn_imputer.py data.vcf --reference panel.vcf --output-prefix out
  python knn_imputer.py plink_prefix --k 10 --weight uniform
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MISSING: int = -1  # sentinel for missing genotype (int8 safe)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_csv(path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load samples × SNPs CSV or TSV matrix.
    Row index = sample IDs, column headers = SNP IDs.
    Valid genotype values: 0, 1, 2. Anything else is treated as missing.
    """
    p = Path(path)
    sep = "\t" if p.suffix in (".tsv", ".txt") else ","
    df = pd.read_csv(p, sep=sep, index_col=0)
    sample_ids: List[str] = list(df.index.astype(str))
    snp_ids: List[str] = list(df.columns.astype(str))
    # Coerce to numeric; non-numeric → NaN
    df = df.apply(pd.to_numeric, errors="coerce")
    matrix = df.to_numpy(dtype=float)
    # Round dosage floats, mark NaN as MISSING
    result = np.where(np.isnan(matrix), MISSING, np.round(matrix)).astype(np.int8)
    return result, sample_ids, snp_ids


def load_vcf(path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Parse a VCF file (GT field only).
    Diploid: 0/0 → 0, 0/1 or 1/0 → 1, 1/1 → 2, ./. or . → MISSING.
    Multi-allelic sites are encoded as alt allele dosage (sum of allele indices).
    """
    sample_ids: List[str] = []
    snp_ids: List[str] = []
    rows: List[List[int]] = []

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.strip().split("\t")
                sample_ids = header[9:]
                continue
            parts = line.strip().split("\t")
            if len(parts) < 10:
                continue
            chrom, pos, _, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
            snp_ids.append(f"{chrom}:{pos}:{ref}:{alt}")
            fmt_fields = parts[8].split(":")
            try:
                gt_idx = fmt_fields.index("GT")
            except ValueError:
                gt_idx = 0
            genotypes: List[int] = []
            for sample_field in parts[9:]:
                gt_raw = sample_field.split(":")[gt_idx]
                alleles = gt_raw.replace("|", "/").split("/")
                if "." in alleles:
                    genotypes.append(MISSING)
                else:
                    try:
                        genotypes.append(sum(int(a) for a in alleles))
                    except ValueError:
                        genotypes.append(MISSING)
            rows.append(genotypes)

    if not rows:
        raise ValueError(f"No variant records found in {path}")

    matrix = np.array(rows, dtype=np.int8).T  # (samples, SNPs)
    return matrix, sample_ids, snp_ids


def load_plink(prefix: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load PLINK binary format (.bed / .bim / .fam).
    Requires pandas-plink: pip install pandas-plink
    """
    try:
        from pandas_plink import read_plink1_bin
    except ImportError:
        raise ImportError(
            "PLINK format requires pandas-plink.\n"
            "Install it with:  pip install pandas-plink"
        )
    G = read_plink1_bin(f"{prefix}.bed", f"{prefix}.bim", f"{prefix}.fam", verbose=False)
    matrix = G.values  # (samples, SNPs), 0/1/2/NaN
    sample_ids = [str(s) for s in G.sample.values]
    snp_ids = [str(v) for v in G.variant.values]
    result = np.where(np.isnan(matrix), MISSING, np.round(matrix)).astype(np.int8)
    return result, sample_ids, snp_ids


def load_auto(path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """Auto-detect format from file extension and load accordingly."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".vcf":
        log.info("Loading VCF: %s", path)
        return load_vcf(path)
    if ext in (".csv", ".tsv", ".txt"):
        log.info("Loading CSV/TSV: %s", path)
        return load_csv(path)
    if p.with_suffix(".bed").exists():
        log.info("Loading PLINK binary: %s.*", path)
        return load_plink(str(p.with_suffix("")))
    # Fallback
    log.info("Unknown extension — attempting CSV load: %s", path)
    return load_csv(path)


# ---------------------------------------------------------------------------
# KNN Imputer
# ---------------------------------------------------------------------------

class KNNGenotypeImputer:
    """
    K-Nearest-Neighbor imputer for SNP genotype data.

    Distance is Hamming distance computed only on positions where BOTH the
    query sample and the candidate neighbor have observed (non-missing)
    genotypes. This avoids biasing distances toward neighbors that happen to
    be observed at the same missing positions.

    Parameters
    ----------
    k : int
        Number of nearest neighbors to use.
    weight : {'distance', 'uniform'}
        'distance'  — inverse-distance weighting (closer neighbors count more).
        'uniform'   — all k neighbors count equally (majority vote / mean).
    min_shared_snps : int
        Minimum number of shared observed positions required for a valid
        distance computation. Pairs below this threshold are skipped (distance
        set to infinity, i.e., excluded from imputation).
    """

    def __init__(
        self,
        k: int = 5,
        weight: str = "distance",
        min_shared_snps: int = 10,
    ) -> None:
        if weight not in ("distance", "uniform"):
            raise ValueError("weight must be 'distance' or 'uniform'")
        self.k = k
        self.weight = weight
        self.min_shared_snps = min_shared_snps
        self._reference: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, reference: np.ndarray) -> "KNNGenotypeImputer":
        """Store the reference panel used to find neighbors. Missing = -1."""
        self._reference = np.asarray(reference, dtype=np.int8)
        n, p = self._reference.shape
        miss_pct = 100.0 * (self._reference == MISSING).mean()
        log.info("Reference panel: %d samples × %d SNPs  (%.2f%% missing)", n, p, miss_pct)
        return self

    # ------------------------------------------------------------------
    def _hamming_distances(self, query: np.ndarray) -> np.ndarray:
        """
        Vectorised masked Hamming distances from one query to all reference rows.

        For each reference sample r:
            shared  = positions where both query and r are observed
            dist(r) = mismatches(shared) / |shared|  if |shared| >= min_shared_snps
                    = inf                             otherwise
        """
        ref = self._reference
        q_obs = (query != MISSING)  # (n_snps,) bool

        # Positions where reference is observed: (n_ref, n_snps) bool
        r_obs = (ref != MISSING)

        # shared[i] = number of positions observed in both query and ref[i]
        shared_counts = r_obs[:, q_obs].sum(axis=1)  # (n_ref,)

        # mismatches: only at positions observed in query
        q_vals = query[q_obs]  # (n_obs_query,)
        r_at_q_obs = ref[:, q_obs]  # (n_ref, n_obs_query)

        # Mismatch: either ref is missing here, or alleles differ
        ref_missing_at_q = (r_at_q_obs == MISSING)
        allele_mismatch = (r_at_q_obs != q_vals) & ~ref_missing_at_q
        mismatches = allele_mismatch.sum(axis=1).astype(float)  # (n_ref,)

        valid = shared_counts >= self.min_shared_snps
        distances = np.full(len(ref), np.inf)
        distances[valid] = mismatches[valid] / shared_counts[valid]
        return distances

    # ------------------------------------------------------------------
    def _impute_one(
        self, query: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Impute missing positions in a single query sample.

        Returns
        -------
        hard_calls  : int8 (n_snps,)  — majority-vote genotype (0/1/2) or MISSING
        dosage      : float32 (n_snps,) — weighted mean allele count
        confidence  : float32 (n_snps,) — weight-normalised vote share of hard call
        """
        ref = self._reference
        missing_pos = np.where(query == MISSING)[0]

        hard_calls = query.copy()
        dosage = np.where(query == MISSING, np.nan, query.astype(float)).astype(np.float32)
        confidence = np.where(query == MISSING, 0.0, 1.0).astype(np.float32)

        if len(missing_pos) == 0:
            return hard_calls, dosage, confidence

        dists = self._hamming_distances(query)
        finite_mask = np.isfinite(dists)
        if not finite_mask.any():
            log.debug("No valid neighbors found; leaving positions as MISSING.")
            return hard_calls, dosage, confidence

        finite_idx = np.where(finite_mask)[0]
        k_actual = min(self.k, len(finite_idx))
        nn_order = np.argsort(dists[finite_idx])[:k_actual]
        neighbor_idx = finite_idx[nn_order]
        neighbor_dist = dists[neighbor_idx]

        if self.weight == "distance":
            eps = 1e-10
            weights = 1.0 / (neighbor_dist + eps)
        else:
            weights = np.ones(k_actual, dtype=float)
        weights /= weights.sum()

        neighbor_geno = ref[neighbor_idx]  # (k_actual, n_snps)

        for pos in missing_pos:
            col = neighbor_geno[:, pos]
            obs_mask = col != MISSING
            if not obs_mask.any():
                continue
            w = weights[obs_mask]
            w = w / w.sum()
            vals = col[obs_mask].astype(float)

            # Dosage: weighted mean
            dosage[pos] = float(np.dot(w, vals))

            # Hard call: weighted vote across {0, 1, 2}
            votes = np.array([
                w[vals == g].sum() if np.any(vals == g) else 0.0
                for g in (0.0, 1.0, 2.0)
            ])
            best_g = int(np.argmax(votes))
            hard_calls[pos] = best_g
            confidence[pos] = float(votes[best_g])

        return hard_calls, dosage, confidence

    # ------------------------------------------------------------------
    def impute(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Impute all samples in X against the fitted reference panel.

        Parameters
        ----------
        X : (n_samples, n_snps) int8 array with missing coded as -1

        Returns
        -------
        hard_calls  : (n_samples, n_snps) int8
        dosage      : (n_samples, n_snps) float32
        confidence  : (n_samples, n_snps) float32
        """
        if self._reference is None:
            raise RuntimeError("Call fit() before impute().")

        n, p = X.shape
        hard_calls = np.empty((n, p), dtype=np.int8)
        dosage = np.full((n, p), np.nan, dtype=np.float32)
        confidence = np.zeros((n, p), dtype=np.float32)

        t0 = time.time()
        log_every = max(1, n // 10)

        for i in range(n):
            hc, dos, conf = self._impute_one(X[i])
            hard_calls[i] = hc
            dosage[i] = dos
            confidence[i] = conf
            if (i + 1) % log_every == 0:
                log.info("  Imputed %d / %d samples  (%.1fs)", i + 1, n, time.time() - t0)

        log.info("Imputation complete in %.1fs", time.time() - t0)
        return hard_calls, dosage, confidence


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    imputer: KNNGenotypeImputer,
    X: np.ndarray,
    mask_fraction: float = 0.10,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Mask a random fraction of known genotypes, impute, then measure accuracy.

    Metrics
    -------
    concordance   Fraction of masked positions where hard call == truth genotype.
    r2            Pearson R² between imputed dosage and true dosage values.
    allele_r2     R² computed on 0/1 allele level (each diploid site = 2 alleles).
    n_masked      Number of genotypes masked.
    """
    rng = np.random.default_rng(seed)
    observed = np.argwhere(X != MISSING)  # (n_observed, 2)

    n_mask = max(1, int(len(observed) * mask_fraction))
    mask_idx = rng.choice(len(observed), size=n_mask, replace=False)
    mask_coords = observed[mask_idx]  # rows/cols to mask

    X_masked = X.copy()
    truth = X[mask_coords[:, 0], mask_coords[:, 1]].copy()
    X_masked[mask_coords[:, 0], mask_coords[:, 1]] = MISSING

    log.info(
        "Masking %d genotypes (%.1f%% of observed) for evaluation",
        n_mask,
        100 * mask_fraction,
    )

    imputer.fit(X_masked)
    hard_calls, dosage, _ = imputer.impute(X_masked)

    pred_hc = hard_calls[mask_coords[:, 0], mask_coords[:, 1]]
    pred_dos = dosage[mask_coords[:, 0], mask_coords[:, 1]]
    truth_f = truth.astype(float)

    concordance = float((pred_hc == truth).mean())

    # R²: only on positions where imputation produced a valid dosage (not NaN)
    valid = ~np.isnan(pred_dos)
    if valid.sum() < 2:
        r2 = float("nan")
    else:
        t = truth_f[valid]
        p = pred_dos[valid]
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        r2 = float("nan") if ss_tot == 0 else 1.0 - float(np.sum((t - p) ** 2)) / ss_tot

    return {
        "concordance": concordance,
        "r2": r2,
        "n_masked": int(n_mask),
        "n_imputed": int(valid.sum()),
        "n_total_observed": int(len(observed)),
        "missing_rate_input": float((X == MISSING).mean()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="KNN genetic imputation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Genotype file: CSV/TSV, VCF, or PLINK prefix")
    p.add_argument(
        "--reference",
        default=None,
        help="Separate reference panel (same format). Omit to use input as self-reference.",
    )
    p.add_argument("-k", type=int, default=5, help="Number of nearest neighbors")
    p.add_argument(
        "--weight",
        choices=["distance", "uniform"],
        default="distance",
        help="Neighbor weighting scheme",
    )
    p.add_argument(
        "--min-shared-snps",
        type=int,
        default=10,
        help="Min shared observed SNPs to compute a valid distance",
    )
    p.add_argument("--output-prefix", default="imputed", help="Prefix for output CSV files")
    p.add_argument("--evaluate", action="store_true", help="Run masked accuracy evaluation")
    p.add_argument("--mask-fraction", type=float, default=0.10, help="Fraction of known genotypes to mask")
    p.add_argument("--seed", type=int, default=42, help="Random seed for masking")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    X, sample_ids, snp_ids = load_auto(args.input)
    log.info("Dataset: %d samples × %d SNPs  (%.2f%% missing)", *X.shape, 100.0 * (X == MISSING).mean())

    ref = load_auto(args.reference)[0] if args.reference else X

    imputer = KNNGenotypeImputer(
        k=args.k,
        weight=args.weight,
        min_shared_snps=args.min_shared_snps,
    )

    if args.evaluate:
        metrics = evaluate(imputer, X, mask_fraction=args.mask_fraction, seed=args.seed)
        print("\n=== Evaluation Results ===")
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"  {key:<28} {val:.4f}")
            else:
                print(f"  {key:<28} {val}")
        print()

    imputer.fit(ref)
    hard_calls, dosage, confidence = imputer.impute(X)

    px = args.output_prefix
    pd.DataFrame(hard_calls, index=sample_ids, columns=snp_ids).to_csv(f"{px}_hard_calls.csv")
    pd.DataFrame(dosage,     index=sample_ids, columns=snp_ids).to_csv(f"{px}_dosage.csv")
    pd.DataFrame(confidence, index=sample_ids, columns=snp_ids).to_csv(f"{px}_confidence.csv")
    log.info("Saved outputs: %s_hard_calls.csv | %s_dosage.csv | %s_confidence.csv", px, px)


if __name__ == "__main__":
    main()
