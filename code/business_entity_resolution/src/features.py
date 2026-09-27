"""Reads candidate_pairs.tsv plus cleaned data, computes pairwise similarity features
(string similarity, address similarity, context features), outputs a flat pair-level
feature table.

Step 1: Data loading scaffold and candidate pair explosion.
Step 2: Name-based similarity features (token sort ratio, token set ratio, char 3-gram Jaccard).
Step 3: Address and postal code features (token Jaccard, char 3-gram Jaccard, tri-state postal match).
Step 4: Context features (country match, legal suffix match, domain flag, non-Latin script flag).
Step 5: Full pipeline assembly and persistence to pair_features.parquet.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

try:
    import psutil
except ImportError:
    psutil = None



def get_repo_root() -> Path:
    """Find repository root by looking for known markers or directory depth."""
    candidate = Path(__file__).resolve().parent.parent.parent.parent
    if (candidate / "student_resource").exists() or (candidate / "data").exists():
        return candidate
    cwd = Path.cwd()
    if (cwd / "student_resource").exists() or (cwd / "data").exists():
        return cwd
    if (cwd.parent.parent / "student_resource").exists() or (cwd.parent.parent / "data").exists():
        return cwd.parent.parent
    return candidate


# ==============================================================================
# LOUD WARNING / PLACEHOLDER STUB:
# The function below creates a temporary stub candidate_pairs.tsv for local testing
# when output/sample/candidate_pairs.tsv does not exist yet.
# THIS STUB IS A PLACEHOLDER ONLY AND MUST BE DELETED ONCE AADITYA'S REAL
# candidate_pairs.tsv EXISTS! DO NOT SILENTLY PREFER THE STUB IF THE REAL FILE
# IS PRESENT.
# ==============================================================================
def _create_sample_candidate_pairs_stub(
    output_path: Path,
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
) -> pd.DataFrame:
    """Create a temporary stub candidate_pairs.tsv for sample testing.

    Takes the first 10 S2 and first 10 S3 entity_ids for each S1 entity.
    """
    print("=" * 80)
    print("! [LOUD WARNING] output/sample/candidate_pairs.tsv does not exist yet.")
    print("! Generating a temporary STUB candidate_pairs.tsv for testing.")
    print("! THIS STUB IS A PLACEHOLDER ONLY AND MUST BE DELETED ONCE AADITYA'S")
    print("! REAL candidate_pairs.tsv EXISTS! DO NOT SILENTLY PREFER THIS STUB.")
    print("=" * 80)

    if not s1_path.exists():
        raise FileNotFoundError(f"Source 1 sample file not found: {s1_path}")
    if not s2_path.exists():
        raise FileNotFoundError(f"Source 2 sample file not found: {s2_path}")
    if not s3_path.exists():
        raise FileNotFoundError(f"Source 3 sample file not found: {s3_path}")

    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    s2_df = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False)
    s3_df = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False)

    s2_cands = s2_df["entity_id"].dropna().unique()[:10].tolist()
    s3_cands = s3_df["entity_id"].dropna().unique()[:10].tolist()
    fake_candidates = s2_cands + s3_cands
    fake_candidates_str = ",".join(fake_candidates)

    stub_df = pd.DataFrame(
        {
            "source1_entity_id": s1_df["entity_id"].astype(str).str.strip(),
            "candidate_entity_ids": fake_candidates_str,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stub_df.to_csv(output_path, sep="\t", index=False)
    print(
        f"[INFO] Wrote stub candidate pairs ({len(stub_df):,} S1 rows, {len(fake_candidates)} candidates each) to: {output_path}"
    )
    return stub_df


def _explode_candidate_pairs_df(df: pd.DataFrame) -> pd.DataFrame:
    """Explode candidate_entity_ids column into one row per (source1_entity_id, candidate_entity_id).

    Handles empty candidate_entity_ids gracefully: empty candidate lists contribute
    zero rows (no error and no NaN row).
    """
    if "source1_entity_id" not in df.columns or "candidate_entity_ids" not in df.columns:
        raise ValueError(
            f"Expected columns 'source1_entity_id' and 'candidate_entity_ids' in candidate pairs file, got {list(df.columns)}"
        )

    s1_ids = df["source1_entity_id"].astype(str).str.strip()
    cand_col = df["candidate_entity_ids"].astype(str).str.strip()

    def parse_candidates(c_str: str) -> list[str]:
        if not c_str or c_str == "nan":
            return []
        return [cid.strip() for cid in c_str.split(",") if cid.strip()]

    candidate_lists = cand_col.apply(parse_candidates)

    temp_df = pd.DataFrame(
        {
            "source1_entity_id": s1_ids,
            "candidate_entity_id": candidate_lists,
        }
    )

    # Explode: an empty list [] results in a NaN row in candidate_entity_id
    exploded = temp_df.explode("candidate_entity_id")

    # Drop NaN or empty rows so S1 entities with empty candidate lists contribute 0 rows
    exploded = exploded.dropna(subset=["candidate_entity_id"])
    exploded = exploded[exploded["candidate_entity_id"] != ""]
    exploded = exploded.reset_index(drop=True)

    return exploded


def load_candidate_pairs(sample: bool) -> pd.DataFrame:
    """Read candidate pairs TSV and explode into one row per (source1_entity_id, candidate_entity_id).

    Parameters
    ----------
    sample : bool
        If True, reads output/sample/candidate_pairs.tsv. If this file does not exist,
        generates a temporary testing stub.
        If False, reads output/candidate_pairs.tsv.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['source1_entity_id', 'candidate_entity_id'].
    """
    repo_root = get_repo_root()

    if sample:
        pairs_path = repo_root / "output" / "sample" / "candidate_pairs.tsv"
        # If real candidate_pairs.tsv is present, do not silently prefer or regenerate stub
        if not pairs_path.exists():
            s1_path = repo_root / "data" / "sample" / "sample_source1.tsv"
            s2_path = repo_root / "data" / "sample" / "sample_source2.tsv"
            s3_path = repo_root / "data" / "sample" / "sample_source3.tsv"
            df = _create_sample_candidate_pairs_stub(pairs_path, s1_path, s2_path, s3_path)
        else:
            df = pd.read_csv(pairs_path, sep="\t", dtype=str, keep_default_na=False)
    else:
        pairs_path = repo_root / "output" / "candidate_pairs.tsv"
        if not pairs_path.exists():
            raise FileNotFoundError(
                f"Candidate pairs file not found: {pairs_path}. "
                "Please run blocking.py first to generate the full candidate pairs."
            )
        df = pd.read_csv(pairs_path, sep="\t", dtype=str, keep_default_na=False)

    return _explode_candidate_pairs_df(df)


def load_clean_data(sample: bool) -> dict[str, pd.DataFrame]:
    """Read cleaned parquet files from data/processed/.

    Parameters
    ----------
    sample : bool
        If True, reads cleaned parquet files from data/processed/sample/.
        If False, reads cleaned parquet files from data/processed/.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping source names (e.g. 'source1', 'source2', 'source3')
        and file stems to their corresponding cleaned DataFrames.

    Raises
    ------
    FileNotFoundError
        If processed directory or any required cleaned parquet file is missing.
    """
    repo_root = get_repo_root()

    if sample:
        clean_dir = repo_root / "data" / "processed" / "sample"
    else:
        clean_dir = repo_root / "data" / "processed"

    if not clean_dir.exists():
        raise FileNotFoundError(
            f"Cleaned data directory not found: {clean_dir}. "
            "Please run normalize.py first to generate cleaned parquet files."
        )

    # Match *_clean.parquet first, or fallback to *.parquet if naming differs
    parquet_files = sorted(clean_dir.glob("*_clean.parquet"))
    if not parquet_files:
        parquet_files = sorted(p for p in clean_dir.glob("*.parquet") if p.is_file())

    if not parquet_files:
        raise FileNotFoundError(
            f"No cleaned parquet files found in {clean_dir}. "
            "Please run normalize.py first to generate cleaned data."
        )

    clean_data: dict[str, pd.DataFrame] = {}
    for p in parquet_files:
        df = pd.read_parquet(p)
        stem = p.stem
        clean_data[stem] = df
        clean_data[stem.replace("_clean", "")] = df

        lower = stem.lower()
        if "source1" in lower or "s1" in lower:
            clean_data.setdefault("source1", df)
            clean_data.setdefault("s1", df)
        elif "source2" in lower or "s2" in lower:
            clean_data.setdefault("source2", df)
            clean_data.setdefault("s2", df)
        elif "source3" in lower or "s3" in lower:
            clean_data.setdefault("source3", df)
            clean_data.setdefault("s3", df)

    # Ensure all three sources are present
    required = ["source1", "source2", "source3"]
    missing = [src for src in required if src not in clean_data]
    if missing:
        raise FileNotFoundError(
            f"Missing cleaned parquet data for {missing} in {clean_dir}. "
            "Expected cleaned parquet files for all three sources. Please run normalize.py first."
        )

    return clean_data


def attach_labels(
    pairs_df: pd.DataFrame,
    sample: bool = False,
    matched_pairs: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Attach boolean is_match column from ground truth dataset.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        DataFrame with columns 'source1_entity_id' and 'candidate_entity_id'.
    sample : bool, default False
        If True, reads data/sample/sample_ground_truth.tsv.
        If False, reads student_resource/dataset/train/train_ground_truth.tsv
        (or data/train_ground_truth.tsv).
    matched_pairs : set[tuple[str, str]] | None, optional
        Pre-built set of (source1_entity_id, candidate_id) positive pairs.
        If None, loaded and parsed from the appropriate ground truth TSV.

    Returns
    -------
    pd.DataFrame
        Copy of pairs_df with an added boolean column 'is_match' (True if the pair
        appears in the ground truth matches for that S1 entity).
    """
    if matched_pairs is None:
        repo_root = get_repo_root()

        if sample:
            gt_path = repo_root / "data" / "sample" / "sample_ground_truth.tsv"
        else:
            candidate_paths = [
                repo_root / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv",
                repo_root / "data" / "train_ground_truth.tsv",
            ]
            gt_path = next((p for p in candidate_paths if p.exists()), None)

        if gt_path is None or not gt_path.exists():
            raise FileNotFoundError(
                f"Ground truth file not found at {gt_path}. Please check data path."
            )

        gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
        if "source1_entity_id" not in gt_df.columns or "matched_entity_ids" not in gt_df.columns:
            raise ValueError(
                f"Expected columns 'source1_entity_id' and 'matched_entity_ids' in {gt_path}, found {list(gt_df.columns)}"
            )

        # Build set of all positive (source1_entity_id, matched_candidate_id) pairs
        matched_pairs = set()
        for s1, mids in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
            s1_clean = str(s1).strip()
            if mids:
                for mid in str(mids).split(","):
                    mid_clean = mid.strip()
                    if mid_clean:
                        matched_pairs.add((s1_clean, mid_clean))

    out_df = pairs_df.copy()
    s1_series = out_df["source1_entity_id"].astype(str).str.strip()
    cand_series = out_df["candidate_entity_id"].astype(str).str.strip()

    out_df["is_match"] = np.fromiter(
        ((s1, cand) in matched_pairs for s1, cand in zip(s1_series, cand_series)),
        dtype=bool,
        count=len(out_df),
    )

    return out_df


def _char_3gram_jaccard(name1: str, name2: str) -> float:
    """Compute Jaccard similarity of character 3-gram sets between two names.

    Returns 0.0 if either name is empty.
    """
    if not name1 or not name2:
        return 0.0
    g1 = {name1[i : i + 3] for i in range(len(name1) - 2)} if len(name1) >= 3 else {name1}
    g2 = {name2[i : i + 3] for i in range(len(name2) - 2)} if len(name2) >= 3 else {name2}
    union = g1 | g2
    if not union:
        return 0.0
    return len(g1 & g2) / len(union)


def compute_name_features(
    pairs_df: pd.DataFrame,
    clean_data: dict[str, pd.DataFrame],
    sample: bool = False,
) -> pd.DataFrame:
    """Compute name-based pairwise similarity features.

    Joins pairs_df with clean_data["source1"] for Source 1's name_clean, and with
    the appropriate Source 2 or Source 3 table for candidate's name_clean (inferred
    via candidate_entity_id prefix convention: 'S2-' for source2, 'S3-' for source3).

    Features computed:
    - name_token_sort_ratio: rapidfuzz token_sort_ratio normalized to [0, 1]
    - name_token_set_ratio: rapidfuzz token_set_ratio normalized to [0, 1]
    - name_char_ngram_jaccard: character 3-gram set Jaccard similarity in [0, 1]

    Missing or empty name_clean on either side outputs 0.0 for all three features.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        DataFrame with columns 'source1_entity_id' and 'candidate_entity_id'.
    clean_data : dict[str, pd.DataFrame]
        Dictionary of cleaned DataFrames for 'source1', 'source2', and 'source3'.
    sample : bool, default False
        Whether running in sample mode.

    Returns
    -------
    pd.DataFrame
        DataFrame with pair metadata and the three computed name similarity features.
    """
    out_df = pairs_df.copy()

    # Pre-index entity_id -> name_clean for quick lookup
    s1_table = clean_data["source1"]
    s2_table = clean_data["source2"]
    s3_table = clean_data["source3"]

    s1_name_map = dict(zip(s1_table["entity_id"].astype(str), s1_table["name_clean"].fillna("")))
    s2_name_map = dict(zip(s2_table["entity_id"].astype(str), s2_table["name_clean"].fillna("")))
    s3_name_map = dict(zip(s3_table["entity_id"].astype(str), s3_table["name_clean"].fillna("")))

    # Retrieve Source 1 names
    s1_ids = out_df["source1_entity_id"].astype(str).str.strip()
    s1_names = s1_ids.map(s1_name_map).fillna("").tolist()

    # Route candidate entity IDs to source2 or source3 using ID prefix convention
    cand_ids = out_df["candidate_entity_id"].astype(str).str.strip()
    s2_mask = cand_ids.str.startswith("S2-")
    s3_mask = cand_ids.str.startswith("S3-")

    cand_names_series = pd.Series("", index=out_df.index, dtype=str)
    cand_names_series.loc[s2_mask] = cand_ids.loc[s2_mask].map(s2_name_map).fillna("")
    cand_names_series.loc[s3_mask] = cand_ids.loc[s3_mask].map(s3_name_map).fillna("")

    # Fallback for any records without expected prefix: look in combined map
    unknown_mask = ~s2_mask & ~s3_mask
    if unknown_mask.any():
        print(
            f"[WARNING] {unknown_mask.sum()} candidate_entity_id values did not match "
            f"the S2-/S3- prefix convention: {cand_ids[unknown_mask].unique()[:5]}"
        )
        combined_fallback = {**s2_name_map, **s3_name_map}
        cand_names_series.loc[unknown_mask] = (
            cand_ids.loc[unknown_mask].map(combined_fallback).fillna("")
        )

    cand_names = cand_names_series.tolist()

    # Compute features for each pair using rapidfuzz.process.cpdist (vectorized C++ batch scoring)
    n = len(s1_names)
    if n == 0:
        token_sort_scores = np.empty(0, dtype=np.float32)
        token_set_scores = np.empty(0, dtype=np.float32)
        char_ngram_scores = np.empty(0, dtype=np.float32)
    else:
        # RapidFuzz cpdist returns an (N, 1) array evaluated across all CPU cores
        token_sort_scores = (
            process.cpdist(
                s1_names,
                cand_names,
                scorer=fuzz.token_sort_ratio,
                workers=-1,
                dtype=np.float32,
            ).ravel()
            / 100.0
        ).astype(np.float32)

        token_set_scores = (
            process.cpdist(
                s1_names,
                cand_names,
                scorer=fuzz.token_set_ratio,
                workers=-1,
                dtype=np.float32,
            ).ravel()
            / 100.0
        ).astype(np.float32)

        # Pre-allocated np.float32 array for character 3-gram Jaccard
        char_ngram_scores = np.zeros(n, dtype=np.float32)
        for i, (name1, name2) in enumerate(zip(s1_names, cand_names)):
            n1 = str(name1).strip() if name1 is not None else ""
            n2 = str(name2).strip() if name2 is not None else ""

            # Handle missing/empty name on either side: output 0.0 for all features
            if not n1 or not n2:
                token_sort_scores[i] = 0.0
                token_set_scores[i] = 0.0
            else:
                char_ngram_scores[i] = _char_3gram_jaccard(n1, n2)

    out_df["name_token_sort_ratio"] = token_sort_scores
    out_df["name_token_set_ratio"] = token_set_scores
    out_df["name_char_ngram_jaccard"] = char_ngram_scores

    return out_df


def _compute_address_chunk_worker(
    s1_slice: list[str],
    cand_slice: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute address token Jaccard and char 3-gram Jaccard on a batch slice."""
    n = len(s1_slice)
    tok_res = np.zeros(n, dtype=np.float32)
    char_res = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a1 = s1_slice[i]
        a2 = cand_slice[i]
        if a1 and a2:
            toks1 = set(a1.split())
            toks2 = set(a2.split())
            u = toks1 | toks2
            if u:
                tok_res[i] = len(toks1 & toks2) / len(u)
            char_res[i] = _char_3gram_jaccard(a1, a2)
    return tok_res, char_res


def compute_address_features(
    pairs_df: pd.DataFrame,
    clean_data: dict[str, pd.DataFrame],
    executor: ProcessPoolExecutor | None = None,
) -> pd.DataFrame:
    """Compute address-based pairwise similarity features using address_clean.

    Joins pairs_df with clean_data["source1"] for Source 1's address_clean, and with
    the appropriate Source 2 or Source 3 table for candidate's address_clean (inferred
    via candidate_entity_id prefix convention: 'S2-' for source2, 'S3-' for source3).

    Features computed:
    - address_token_jaccard: word-token Jaccard similarity (|intersection| / |union|)
    - address_char_ngram_jaccard: character 3-gram set Jaccard similarity (reusing _char_3gram_jaccard)

    Missing or empty address_clean on either side outputs 0.0 for both features.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        DataFrame with columns 'source1_entity_id' and 'candidate_entity_id'.
    clean_data : dict[str, pd.DataFrame]
        Dictionary of cleaned DataFrames for 'source1', 'source2', and 'source3'.
    executor : ProcessPoolExecutor | None, optional
        Reused process pool executor for parallel chunk computation across CPU cores.

    Returns
    -------
    pd.DataFrame
        DataFrame with pairs and the two computed address features.
    """
    out_df = pairs_df.copy()

    # Pre-index entity_id -> address_clean for quick lookup
    s1_table = clean_data["source1"]
    s2_table = clean_data["source2"]
    s3_table = clean_data["source3"]

    s1_addr_map = dict(zip(s1_table["entity_id"].astype(str), s1_table["address_clean"].fillna("")))
    s2_addr_map = dict(zip(s2_table["entity_id"].astype(str), s2_table["address_clean"].fillna("")))
    s3_addr_map = dict(zip(s3_table["entity_id"].astype(str), s3_table["address_clean"].fillna("")))

    # Retrieve Source 1 addresses
    s1_ids = out_df["source1_entity_id"].astype(str).str.strip()
    s1_addrs = s1_ids.map(s1_addr_map).fillna("").tolist()

    # Route candidate entity IDs to source2 or source3 using ID prefix convention
    cand_ids = out_df["candidate_entity_id"].astype(str).str.strip()
    s2_mask = cand_ids.str.startswith("S2-")
    s3_mask = cand_ids.str.startswith("S3-")

    cand_addrs_series = pd.Series("", index=out_df.index, dtype=str)
    cand_addrs_series.loc[s2_mask] = cand_ids.loc[s2_mask].map(s2_addr_map).fillna("")
    cand_addrs_series.loc[s3_mask] = cand_ids.loc[s3_mask].map(s3_addr_map).fillna("")

    # Fallback for any records without expected prefix: look in combined map
    unknown_mask = ~s2_mask & ~s3_mask
    if unknown_mask.any():
        print(
            f"[WARNING] {unknown_mask.sum()} candidate_entity_id values did not match "
            f"the S2-/S3- prefix convention: {cand_ids[unknown_mask].unique()[:5]}"
        )
        combined_fallback = {**s2_addr_map, **s3_addr_map}
        cand_addrs_series.loc[unknown_mask] = (
            cand_ids.loc[unknown_mask].map(combined_fallback).fillna("")
        )

    cand_addrs = cand_addrs_series.tolist()

    # Compute features for each pair using parallel worker processes across CPU cores
    n = len(s1_addrs)
    if n == 0:
        out_df["address_token_jaccard"] = np.empty(0, dtype=np.float32)
        out_df["address_char_ngram_jaccard"] = np.empty(0, dtype=np.float32)
        return out_df

    if n < 10_000 or (os.cpu_count() or 1) <= 1:
        token_jaccard_scores, char_ngram_scores = _compute_address_chunk_worker(s1_addrs, cand_addrs)
    elif executor is not None:
        num_workers = getattr(executor, "_max_workers", os.cpu_count() or 4)
        chunk_len = (n + num_workers - 1) // num_workers
        slices = [
            (
                s1_addrs[i * chunk_len : min(n, (i + 1) * chunk_len)],
                cand_addrs[i * chunk_len : min(n, (i + 1) * chunk_len)],
            )
            for i in range(num_workers)
        ]
        futures = [
            executor.submit(_compute_address_chunk_worker, s1_s, cand_s)
            for s1_s, cand_s in slices
        ]
        results = [f.result() for f in futures]
        token_jaccard_scores = np.concatenate([r[0] for r in results])
        char_ngram_scores = np.concatenate([r[1] for r in results])
    else:
        num_workers = min(12, os.cpu_count() or 4)
        chunk_len = (n + num_workers - 1) // num_workers
        slices = [
            (
                s1_addrs[i * chunk_len : min(n, (i + 1) * chunk_len)],
                cand_addrs[i * chunk_len : min(n, (i + 1) * chunk_len)],
            )
            for i in range(num_workers)
        ]
        with ProcessPoolExecutor(max_workers=num_workers) as internal_executor:
            futures = [
                internal_executor.submit(_compute_address_chunk_worker, s1_s, cand_s)
                for s1_s, cand_s in slices
            ]
            results = [f.result() for f in futures]
        token_jaccard_scores = np.concatenate([r[0] for r in results])
        char_ngram_scores = np.concatenate([r[1] for r in results])

    out_df["address_token_jaccard"] = token_jaccard_scores
    out_df["address_char_ngram_jaccard"] = char_ngram_scores

    return out_df


def compute_postal_code_features(
    pairs_df: pd.DataFrame,
    clean_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute postal_code_match feature between source1 and candidate entities.

    Values:
    - -1.0 if either side's postal_code is empty or missing (uninformative/no signal)
    -  0.0 if both sides have non-empty postal_code but they differ
    -  1.0 if both sides have non-empty postal_code and match exactly

    Parameters
    ----------
    pairs_df : pd.DataFrame
        DataFrame with columns 'source1_entity_id' and 'candidate_entity_id'.
    clean_data : dict[str, pd.DataFrame]
        Dictionary of cleaned DataFrames for 'source1', 'source2', and 'source3'.

    Returns
    -------
    pd.DataFrame
        DataFrame with pairs and the computed postal_code_match feature.
    """
    out_df = pairs_df.copy()

    # Pre-index entity_id -> postal_code for quick lookup
    s1_table = clean_data["source1"]
    s2_table = clean_data["source2"]
    s3_table = clean_data["source3"]

    s1_postal_map = dict(zip(s1_table["entity_id"].astype(str), s1_table["postal_code"].fillna("")))
    s2_postal_map = dict(zip(s2_table["entity_id"].astype(str), s2_table["postal_code"].fillna("")))
    s3_postal_map = dict(zip(s3_table["entity_id"].astype(str), s3_table["postal_code"].fillna("")))

    # Retrieve Source 1 postal codes
    s1_ids = out_df["source1_entity_id"].astype(str).str.strip()
    s1_postals = s1_ids.map(s1_postal_map).fillna("").tolist()

    # Route candidate entity IDs to source2 or source3 using ID prefix convention
    cand_ids = out_df["candidate_entity_id"].astype(str).str.strip()
    s2_mask = cand_ids.str.startswith("S2-")
    s3_mask = cand_ids.str.startswith("S3-")

    cand_postals_series = pd.Series("", index=out_df.index, dtype=str)
    cand_postals_series.loc[s2_mask] = cand_ids.loc[s2_mask].map(s2_postal_map).fillna("")
    cand_postals_series.loc[s3_mask] = cand_ids.loc[s3_mask].map(s3_postal_map).fillna("")

    # Fallback for any records without expected prefix: look in combined map
    unknown_mask = ~s2_mask & ~s3_mask
    if unknown_mask.any():
        print(
            f"[WARNING] {unknown_mask.sum()} candidate_entity_id values did not match "
            f"the S2-/S3- prefix convention: {cand_ids[unknown_mask].unique()[:5]}"
        )
        combined_fallback = {**s2_postal_map, **s3_postal_map}
        cand_postals_series.loc[unknown_mask] = (
            cand_ids.loc[unknown_mask].map(combined_fallback).fillna("")
        )

    cand_postals = cand_postals_series.tolist()

    # Compute tri-state match for each pair using pre-allocated np.float32 array
    n = len(s1_postals)
    postal_matches = np.full(n, -1.0, dtype=np.float32)
    for i, (p1_raw, p2_raw) in enumerate(zip(s1_postals, cand_postals)):
        p1 = str(p1_raw).strip() if p1_raw is not None else ""
        p2 = str(p2_raw).strip() if p2_raw is not None else ""

        if p1 and p2:
            postal_matches[i] = 1.0 if p1.lower() == p2.lower() else 0.0

    out_df["postal_code_match"] = postal_matches

    # Sanity-check: print count and percentage where feature is not -1.0
    comparable_count = int((out_df["postal_code_match"] != -1.0).sum())
    total_pairs = len(out_df)
    pct_comparable = (comparable_count / total_pairs * 100.0) if total_pairs > 0 else 0.0
    print(
        f"[INFO] postal_code_match comparable pairs (not -1.0): "
        f"{comparable_count:,d} / {total_pairs:,d} ({pct_comparable:.4f}%)"
    )

    return out_df


def _is_truthy(val) -> bool:
    """Helper to convert boolean/string/numeric flag into a strict boolean."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val == 1
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "t", "yes")
    return bool(val)


def compute_context_features(
    pairs_df: pd.DataFrame,
    clean_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute context features using columns produced by normalize.py.

    Features computed:
    - country_match:
        1.0 if both sides' country are non-empty and equal (case-insensitive)
        0.0 if both non-empty and different
       -1.0 if either side is empty/missing
    - legal_suffix_match:
        1.0 if both sides' name_legal_suffix are non-empty and equal
        0.0 if both non-empty and different
       -1.0 if either side is empty/missing
    - either_is_domain:
        1.0 if either side's name_is_domain flag is true, else 0.0
    - either_has_non_latin_script:
        1.0 if either side's has_non_latin_script flag is true, else 0.0

    Parameters
    ----------
    pairs_df : pd.DataFrame
        DataFrame with columns 'source1_entity_id' and 'candidate_entity_id'.
    clean_data : dict[str, pd.DataFrame]
        Dictionary of cleaned DataFrames for 'source1', 'source2', and 'source3'.

    Returns
    -------
    pd.DataFrame
        DataFrame with pairs and the four computed context features.
    """
    out_df = pairs_df.copy()

    s1_table = clean_data["source1"]
    s2_table = clean_data["source2"]
    s3_table = clean_data["source3"]

    s1_ids_clean = s1_table["entity_id"].astype(str)
    s2_ids_clean = s2_table["entity_id"].astype(str)
    s3_ids_clean = s3_table["entity_id"].astype(str)

    # Pre-index mappings for the 4 attributes
    s1_country_map = dict(zip(s1_ids_clean, s1_table["country"].fillna("")))
    s2_country_map = dict(zip(s2_ids_clean, s2_table["country"].fillna("")))
    s3_country_map = dict(zip(s3_ids_clean, s3_table["country"].fillna("")))

    s1_suffix_map = dict(zip(s1_ids_clean, s1_table["name_legal_suffix"].fillna("")))
    s2_suffix_map = dict(zip(s2_ids_clean, s2_table["name_legal_suffix"].fillna("")))
    s3_suffix_map = dict(zip(s3_ids_clean, s3_table["name_legal_suffix"].fillna("")))

    s1_domain_map = dict(zip(s1_ids_clean, s1_table["name_is_domain"].fillna(False)))
    s2_domain_map = dict(zip(s2_ids_clean, s2_table["name_is_domain"].fillna(False)))
    s3_domain_map = dict(zip(s3_ids_clean, s3_table["name_is_domain"].fillna(False)))

    s1_script_map = dict(zip(s1_ids_clean, s1_table["has_non_latin_script"].fillna(False)))
    s2_script_map = dict(zip(s2_ids_clean, s2_table["has_non_latin_script"].fillna(False)))
    s3_script_map = dict(zip(s3_ids_clean, s3_table["has_non_latin_script"].fillna(False)))

    # Source 1 lookups
    s1_ids = out_df["source1_entity_id"].astype(str).str.strip()
    s1_countries = s1_ids.map(s1_country_map).fillna("").tolist()
    s1_suffixes = s1_ids.map(s1_suffix_map).fillna("").tolist()
    s1_domains = s1_ids.map(s1_domain_map).fillna(False).tolist()
    s1_scripts = s1_ids.map(s1_script_map).fillna(False).tolist()

    # Candidate routing via S2-/S3- prefix convention
    cand_ids = out_df["candidate_entity_id"].astype(str).str.strip()
    s2_mask = cand_ids.str.startswith("S2-")
    s3_mask = cand_ids.str.startswith("S3-")

    cand_country_series = pd.Series("", index=out_df.index, dtype=str)
    cand_country_series.loc[s2_mask] = cand_ids.loc[s2_mask].map(s2_country_map).fillna("")
    cand_country_series.loc[s3_mask] = cand_ids.loc[s3_mask].map(s3_country_map).fillna("")

    cand_suffix_series = pd.Series("", index=out_df.index, dtype=str)
    cand_suffix_series.loc[s2_mask] = cand_ids.loc[s2_mask].map(s2_suffix_map).fillna("")
    cand_suffix_series.loc[s3_mask] = cand_ids.loc[s3_mask].map(s3_suffix_map).fillna("")

    cand_domain_series = pd.Series(False, index=out_df.index, dtype=object)
    cand_domain_series.loc[s2_mask] = cand_ids.loc[s2_mask].map(s2_domain_map).fillna(False)
    cand_domain_series.loc[s3_mask] = cand_ids.loc[s3_mask].map(s3_domain_map).fillna(False)

    cand_script_series = pd.Series(False, index=out_df.index, dtype=object)
    cand_script_series.loc[s2_mask] = cand_ids.loc[s2_mask].map(s2_script_map).fillna(False)
    cand_script_series.loc[s3_mask] = cand_ids.loc[s3_mask].map(s3_script_map).fillna(False)

    # Fallback for any records without expected prefix: look in combined map
    unknown_mask = ~s2_mask & ~s3_mask
    if unknown_mask.any():
        print(
            f"[WARNING] {unknown_mask.sum()} candidate_entity_id values did not match "
            f"the S2-/S3- prefix convention: {cand_ids[unknown_mask].unique()[:5]}"
        )
        cand_country_series.loc[unknown_mask] = (
            cand_ids.loc[unknown_mask].map({**s2_country_map, **s3_country_map}).fillna("")
        )
        cand_suffix_series.loc[unknown_mask] = (
            cand_ids.loc[unknown_mask].map({**s2_suffix_map, **s3_suffix_map}).fillna("")
        )
        cand_domain_series.loc[unknown_mask] = (
            cand_ids.loc[unknown_mask].map({**s2_domain_map, **s3_domain_map}).fillna(False)
        )
        cand_script_series.loc[unknown_mask] = (
            cand_ids.loc[unknown_mask].map({**s2_script_map, **s3_script_map}).fillna(False)
        )

    cand_countries = cand_country_series.tolist()
    cand_suffixes = cand_suffix_series.tolist()
    cand_domains = cand_domain_series.tolist()
    cand_scripts = cand_script_series.tolist()

    n = len(s1_countries)
    country_matches = np.full(n, -1.0, dtype=np.float32)
    suffix_matches = np.full(n, -1.0, dtype=np.float32)
    either_domains = np.zeros(n, dtype=np.float32)
    either_scripts = np.zeros(n, dtype=np.float32)

    for i, (c1, c2, sf1, sf2, d1, d2, sc1, sc2) in enumerate(zip(
        s1_countries, cand_countries, s1_suffixes, cand_suffixes, s1_domains, cand_domains, s1_scripts, cand_scripts
    )):
        # 1. country_match
        s1_c = str(c1).strip().lower() if c1 is not None and str(c1) != "nan" else ""
        s2_c = str(c2).strip().lower() if c2 is not None and str(c2) != "nan" else ""
        if s1_c and s2_c:
            country_matches[i] = 1.0 if s1_c == s2_c else 0.0

        # 2. legal_suffix_match
        s1_s = str(sf1).strip().lower() if sf1 is not None and str(sf1) != "nan" else ""
        s2_s = str(sf2).strip().lower() if sf2 is not None and str(sf2) != "nan" else ""
        if s1_s and s2_s:
            suffix_matches[i] = 1.0 if s1_s == s2_s else 0.0

        # 3. either_is_domain
        if _is_truthy(d1) or _is_truthy(d2):
            either_domains[i] = 1.0

        # 4. either_has_non_latin_script
        if _is_truthy(sc1) or _is_truthy(sc2):
            either_scripts[i] = 1.0

    out_df["country_match"] = country_matches
    out_df["legal_suffix_match"] = suffix_matches
    out_df["either_is_domain"] = either_domains
    out_df["either_has_non_latin_script"] = either_scripts

    return out_df


def build_pair_features(sample: bool) -> pd.DataFrame:
    """Run the complete pair feature engineering pipeline and write pair_features.parquet.

    Pipeline execution sequence:
    1. load_candidate_pairs
    2. load_clean_data
    3. attach_labels
    4. compute_name_features
    5. compute_address_features
    6. compute_postal_code_features
    7. compute_context_features

    The returned DataFrame strictly satisfies the locked interface contract:
    - source1_entity_id
    - candidate_entity_id
    - all 10 feature columns
    - is_match (label, boolean)

    Output is saved to:
    - data/processed/sample/pair_features.parquet (if sample=True)
    - data/processed/pair_features.parquet (if sample=False)

    Parameters
    ----------
    sample : bool
        Whether to process the sample fixture dataset or the full dataset.

    Returns
    -------
    pd.DataFrame
        Final assembled pair-level feature table.
    """
    repo_root = get_repo_root()

    # 1. Load candidate pairs
    pairs_df = load_candidate_pairs(sample=sample)

    # 2. Load cleaned parquet data
    clean_data = load_clean_data(sample=sample)

    # 3. Attach labels from ground truth
    df = attach_labels(pairs_df, sample=sample)

    # 4. Compute name similarity features
    df = compute_name_features(df, clean_data, sample=sample)

    # 5. Compute address similarity features
    df = compute_address_features(df, clean_data)

    # 6. Compute postal code match feature
    df = compute_postal_code_features(df, clean_data)

    # 7. Compute context features
    df = compute_context_features(df, clean_data)

    # Define locked interface contract columns
    feature_cols = [
        "name_token_sort_ratio",
        "name_token_set_ratio",
        "name_char_ngram_jaccard",
        "address_token_jaccard",
        "address_char_ngram_jaccard",
        "postal_code_match",
        "country_match",
        "legal_suffix_match",
        "either_is_domain",
        "either_has_non_latin_script",
    ]

    expected_cols = ["source1_entity_id", "candidate_entity_id"] + feature_cols

    # Ensure is_match is strict boolean and placed at the end
    if "is_match" in df.columns:
        df["is_match"] = df["is_match"].astype(bool)
        expected_cols.append("is_match")

    final_df = df[expected_cols].copy()

    # Enforce float32 dtype on all feature columns
    for col in feature_cols:
        final_df[col] = final_df[col].astype(np.float32)

    # Determine destination directory
    if sample:
        out_dir = repo_root / "data" / "processed" / "sample"
    else:
        out_dir = repo_root / "data" / "processed"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pair_features.parquet"

    final_df.to_parquet(out_path, index=False)
    print(f"[INFO] Successfully wrote pair features table to: {out_path}")

    return final_df


def build_pair_features_chunked(
    sample: bool = False,
    chunk_size: int = 5_000_000,
) -> Path:
    """Run pair feature extraction in streaming chunks to keep memory bounded.

    Writes directly to parquet via pyarrow.parquet.ParquetWriter.

    Parameters
    ----------
    sample : bool, default False
        Whether running on sample fixture or full dataset.
    chunk_size : int, default 5_000_000
        Number of pair rows per processing chunk.

    Returns
    -------
    Path
        Path to the written parquet file.
    """
    repo_root = get_repo_root()
    if sample:
        out_dir = repo_root / "data" / "processed" / "sample"
    else:
        out_dir = repo_root / "data" / "processed"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pair_features.parquet"

    print("=" * 80)
    print(f"[CHUNKED PIPELINE] Initializing streaming feature generation (chunk_size={chunk_size:,d})")
    print(f"[CHUNKED PIPELINE] Target output: {out_path}")
    print("=" * 80)

    # 1. Load candidate pairs once
    t_load_start = time.time()
    pairs_df = load_candidate_pairs(sample=sample)
    total_pairs = len(pairs_df)
    print(f"[INFO] Loaded candidate pairs: {total_pairs:,d} rows in {time.time() - t_load_start:.2f}s")

    # 2. Load cleaned data once
    clean_data = load_clean_data(sample=sample)
    print("[INFO] Loaded cleaned source tables once.")

    # 3. Pre-load ground truth labels once to build matched_pairs set
    gt_pairs: set[tuple[str, str]] = set()
    if sample:
        gt_path = repo_root / "data" / "sample" / "sample_ground_truth.tsv"
    else:
        candidate_paths = [
            repo_root / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv",
            repo_root / "data" / "train_ground_truth.tsv",
        ]
        gt_path = next((p for p in candidate_paths if p.exists()), None)

    if gt_path is not None and gt_path.exists():
        gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
        for s1, mids in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
            s1_clean = str(s1).strip()
            if mids:
                for mid in str(mids).split(","):
                    mid_clean = mid.strip()
                    if mid_clean:
                        gt_pairs.add((s1_clean, mid_clean))
        print(f"[INFO] Pre-indexed ground truth matches: {len(gt_pairs):,d} positive pairs.")
    else:
        print("[WARNING] Ground truth file not found; skipping label attachment.")

    feature_cols = [
        "name_token_sort_ratio",
        "name_token_set_ratio",
        "name_char_ngram_jaccard",
        "address_token_jaccard",
        "address_char_ngram_jaccard",
        "postal_code_match",
        "country_match",
        "legal_suffix_match",
        "either_is_domain",
        "either_has_non_latin_script",
    ]
    expected_cols = ["source1_entity_id", "candidate_entity_id"] + feature_cols
    has_labels = len(gt_pairs) > 0
    if has_labels:
        expected_cols.append("is_match")

    num_chunks = max(1, (total_pairs + chunk_size - 1) // chunk_size)
    writer: pq.ParquetWriter | None = None
    t0_pipeline = time.time()
    num_workers = min(12, os.cpu_count() or 4)

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for chunk_idx in range(num_chunks):
                t_chunk_start = time.time()
                start_idx = chunk_idx * chunk_size
                end_idx = min(total_pairs, (chunk_idx + 1) * chunk_size)
                chunk_slice = pairs_df.iloc[start_idx:end_idx].copy()

                # Pipeline execution on chunk: features then labels
                chunk_df = compute_name_features(chunk_slice, clean_data, sample=sample)
                chunk_df = compute_address_features(chunk_df, clean_data, executor=executor)
                chunk_df = compute_postal_code_features(chunk_df, clean_data)
                chunk_df = compute_context_features(chunk_df, clean_data)
                if has_labels:
                    chunk_df = attach_labels(chunk_df, sample=sample, matched_pairs=gt_pairs)
                    chunk_df["is_match"] = chunk_df["is_match"].astype(bool)

                for col in feature_cols:
                    chunk_df[col] = chunk_df[col].astype(np.float32)

                out_table_df = chunk_df[expected_cols]
                arrow_table = pa.Table.from_pandas(out_table_df, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(out_path, arrow_table.schema, compression="snappy")

                writer.write_table(arrow_table)

                chunk_time = time.time() - t_chunk_start
                mem_info = ""
                if psutil is not None:
                    rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                    mem_info = f" | Process RSS: {rss_mb:.1f} MB"

                pairs_per_sec = len(out_table_df) / chunk_time if chunk_time > 0 else 0
                print(
                    f"[CHUNK {chunk_idx + 1}/{num_chunks}] Processed {len(out_table_df):,d} pairs "
                    f"(rows {start_idx:,d} to {end_idx:,d}) in {chunk_time:.2f}s "
                    f"({pairs_per_sec:,.0f} pairs/sec){mem_info}"
                )

    finally:
        if writer is not None:
            writer.close()

    total_time = time.time() - t0_pipeline
    print("=" * 80)
    print(f"[INFO] Streaming complete. Wrote {total_pairs:,d} rows to: {out_path}")
    print(f"[INFO] Output file size: {out_path.stat().st_size:,d} bytes in {total_time:.2f}s")
    print("=" * 80)

    return out_path


def load_candidate_pairs_from_path(candidate_pairs_path: str | Path) -> pd.DataFrame:
    """Load candidate pairs from an arbitrary path, exploding candidates if needed.

    Supports TSV/CSV with 'source1_entity_id' and 'candidate_entity_ids' (comma-separated),
    or already exploded files with 'source1_entity_id' and 'candidate_entity_id'.
    """
    path = Path(candidate_pairs_path)
    if not path.is_absolute():
        repo_root = get_repo_root()
        if not path.exists() and (repo_root / path).exists():
            path = repo_root / path

    if not path.exists():
        raise FileNotFoundError(f"Candidate pairs file not found at: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)

    if "candidate_entity_ids" in df.columns:
        return _explode_candidate_pairs_df(df)
    elif "candidate_entity_id" in df.columns and "source1_entity_id" in df.columns:
        res = df[["source1_entity_id", "candidate_entity_id"]].copy()
        res["source1_entity_id"] = res["source1_entity_id"].astype(str).str.strip()
        res["candidate_entity_id"] = res["candidate_entity_id"].astype(str).str.strip()
        res = res.dropna().reset_index(drop=True)
        return res[res["candidate_entity_id"] != ""]
    else:
        raise ValueError(
            f"Candidate pairs file {path} must contain 'source1_entity_id' and either "
            f"'candidate_entity_ids' or 'candidate_entity_id'. Found: {list(df.columns)}"
        )


def build_pair_features_for_scoring(
    candidate_pairs_path: str | Path,
    clean_data_sample: bool,
    output_path: str | Path,
    chunk_size: int = 5_000_000,
) -> Path:
    """Extract pairwise features for unlabelled candidate pairs in streaming chunks.

    Designed for scoring test/unlabelled candidate pairs where ground truth is unavailable.
    - Reads candidate pairs from an explicit candidate_pairs_path.
    - Loads cleaned source tables using load_clean_data(sample=clean_data_sample).
    - Skips attach_labels entirely (no 'is_match' column is created or expected).
    - Computes name, address, postal code, and context features with persistent
      multiprocessing worker pool reuse.
    - Writes directly to output_path via pyarrow.parquet.ParquetWriter in chunks of chunk_size.

    Parameters
    ----------
    candidate_pairs_path : str | Path
        Path to candidate pairs file (TSV/CSV/parquet).
    clean_data_sample : bool
        Whether to load clean source tables from sample directory (True) or full dataset (False).
    output_path : str | Path
        Destination path for output parquet file (e.g. data/processed/test_pair_features.parquet).
    chunk_size : int, default 5_000_000
        Number of pair rows per streaming chunk to maintain bounded memory.

    Returns
    -------
    Path
        Path to the written parquet file.
    """
    out_path = Path(output_path)
    if not out_path.is_absolute():
        repo_root = get_repo_root()
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"[SCORING PIPELINE] Building pair features from: {candidate_pairs_path}")
    print(f"[SCORING PIPELINE] Target output: {out_path}")
    print(f"[SCORING PIPELINE] Clean data source: {'sample' if clean_data_sample else 'full'}")
    print("=" * 80)

    # 1. Load candidate pairs from specified path
    t_load_start = time.time()
    pairs_df = load_candidate_pairs_from_path(candidate_pairs_path)
    total_pairs = len(pairs_df)
    print(f"[INFO] Loaded candidate pairs: {total_pairs:,d} rows in {time.time() - t_load_start:.2f}s")

    # 2. Load cleaned data once
    clean_data = load_clean_data(sample=clean_data_sample)
    print("[INFO] Loaded cleaned source tables once.")

    feature_cols = [
        "name_token_sort_ratio",
        "name_token_set_ratio",
        "name_char_ngram_jaccard",
        "address_token_jaccard",
        "address_char_ngram_jaccard",
        "postal_code_match",
        "country_match",
        "legal_suffix_match",
        "either_is_domain",
        "either_has_non_latin_script",
    ]
    expected_cols = ["source1_entity_id", "candidate_entity_id"] + feature_cols

    num_chunks = max(1, (total_pairs + chunk_size - 1) // chunk_size)
    writer: pq.ParquetWriter | None = None
    t0_pipeline = time.time()
    num_workers = min(12, os.cpu_count() or 4)

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for chunk_idx in range(num_chunks):
                t_chunk_start = time.time()
                start_idx = chunk_idx * chunk_size
                end_idx = min(total_pairs, (chunk_idx + 1) * chunk_size)
                chunk_slice = pairs_df.iloc[start_idx:end_idx].copy()

                # Pipeline execution on chunk without labels
                chunk_df = compute_name_features(chunk_slice, clean_data, sample=clean_data_sample)
                chunk_df = compute_address_features(chunk_df, clean_data, executor=executor)
                chunk_df = compute_postal_code_features(chunk_df, clean_data)
                chunk_df = compute_context_features(chunk_df, clean_data)

                for col in feature_cols:
                    chunk_df[col] = chunk_df[col].astype(np.float32)

                out_table_df = chunk_df[expected_cols]
                arrow_table = pa.Table.from_pandas(out_table_df, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(out_path, arrow_table.schema, compression="snappy")

                writer.write_table(arrow_table)

                chunk_time = time.time() - t_chunk_start
                mem_info = ""
                if psutil is not None:
                    rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                    mem_info = f" | Process RSS: {rss_mb:.1f} MB"

                pairs_per_sec = len(out_table_df) / chunk_time if chunk_time > 0 else 0
                print(
                    f"[CHUNK {chunk_idx + 1}/{num_chunks}] Processed {len(out_table_df):,d} pairs "
                    f"(rows {start_idx:,d} to {end_idx:,d}) in {chunk_time:.2f}s "
                    f"({pairs_per_sec:,.0f} pairs/sec){mem_info}"
                )

    finally:
        if writer is not None:
            writer.close()

    total_time = time.time() - t0_pipeline
    print("=" * 80)
    print(f"[INFO] Scoring feature extraction complete. Wrote {total_pairs:,d} rows to: {out_path}")
    print(f"[INFO] Output file size: {out_path.stat().st_size:,d} bytes in {total_time:.2f}s")
    print("=" * 80)

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature extraction pipeline: assemble and write pair_features.parquet."
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample fixture dataset instead of the full dataset.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Run feature extraction in scoring mode (no ground-truth labels attached).",
    )
    parser.add_argument(
        "--candidates",
        "--candidates-path",
        dest="candidates_path",
        type=str,
        default=None,
        help="Custom path to candidate pairs file for scoring mode (e.g. output/test_candidate_pairs.tsv).",
    )
    parser.add_argument(
        "--output",
        "--output-path",
        dest="output_path",
        type=str,
        default=None,
        help="Custom output parquet path when in scoring mode.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5_000_000,
        help="Chunk size for chunked full-scale feature extraction (default: 5,000,000).",
    )
    args = parser.parse_args()

    repo_root = get_repo_root()
    t0 = time.time()

    if args.candidates_path or args.score_only:
        cand_path = args.candidates_path or (
            repo_root / "output" / "sample" / "candidate_pairs.tsv" if args.sample
            else repo_root / "output" / "test_candidate_pairs.tsv"
        )
        out_p = args.output_path or (
            repo_root / "data" / "processed" / "sample" / "test_pair_features.parquet" if args.sample
            else repo_root / "data" / "processed" / "test_pair_features.parquet"
        )
        out_path = build_pair_features_for_scoring(
            candidate_pairs_path=cand_path,
            clean_data_sample=args.sample,
            output_path=out_p,
            chunk_size=args.chunk_size,
        )
        total_runtime = time.time() - t0
        print("\n" + "=" * 80)
        print("SCORING FEATURE EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"Output path: {out_path}")
        print(f"File exists on disk: {out_path.exists()} ({out_path.stat().st_size:,d} bytes)")
        print(f"Total runtime (wall-clock): {total_runtime:.2f}s")
    else:
        dest_path = (
            repo_root / "data" / "processed" / "sample" / "pair_features.parquet"
            if args.sample
            else repo_root / "data" / "processed" / "pair_features.parquet"
        )

        print("=" * 80)
        print(f"RUNNING FULL PAIR FEATURES PIPELINE (sample={args.sample})")
        print("=" * 80)

        if args.sample:
            features_df = build_pair_features(sample=True)
            total_runtime = time.time() - t0

            # Print confirmation of output path, final shape, and runtime
            print("\n" + "=" * 80)
            print("PIPELINE EXECUTION CONFIRMATION")
            print("=" * 80)
            print(f"Output path: {dest_path}")
            print(f"File exists on disk: {dest_path.exists()} ({dest_path.stat().st_size:,d} bytes)")
            print(f"Final DataFrame shape: {features_df.shape}")
            print(f"Full column list ({len(features_df.columns)} columns): {list(features_df.columns)}")
            print(f"Total runtime (wall-clock): {total_runtime:.2f}s")

            # Feature columns for describe
            feature_cols = [
                "name_token_sort_ratio",
                "name_token_set_ratio",
                "name_char_ngram_jaccard",
                "address_token_jaccard",
                "address_char_ngram_jaccard",
                "postal_code_match",
                "country_match",
                "legal_suffix_match",
                "either_is_domain",
                "either_has_non_latin_script",
            ]

            print("\nFull .describe() across all 10 feature columns:")
            pd.set_option("display.width", 1200)
            pd.set_option("display.max_columns", None)
            print(features_df[feature_cols].describe().to_string())

            if "is_match" in features_df.columns:
                print("\nFinal is_match value counts (sanity check):")
                print(features_df["is_match"].value_counts(dropna=False).to_string())
                print(f"is_match column dtype: {features_df['is_match'].dtype} (confirmed boolean)")

                print("\nMean of each feature split by is_match (True vs False):")
                print(features_df.groupby("is_match")[feature_cols].mean().to_string())
        else:
            out_path = build_pair_features_chunked(sample=False, chunk_size=args.chunk_size)
            total_runtime = time.time() - t0
            print("\n" + "=" * 80)
            print("FULL-SCALE CHUNKED PIPELINE EXECUTION COMPLETE")
            print("=" * 80)
            print(f"Output path: {out_path}")
            print(f"File exists on disk: {out_path.exists()} ({out_path.stat().st_size:,d} bytes)")
            print(f"Total runtime (wall-clock): {total_runtime:.2f}s")

