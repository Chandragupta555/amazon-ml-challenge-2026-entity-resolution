# ============================================================
# PLACEHOLDER SCRIPT — DELETE ONCE SOUMIL'S REAL train_model.py
# OUTPUT (data/processed/pair_scores.parquet) EXISTS.
# This generates FAKE random scores for isolated testing of
# assign_matches.py only. Do not use for anything else.
# ============================================================

import os
import random
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

GROUND_TRUTH_PATH = os.path.join(REPO_ROOT, "data", "sample", "sample_ground_truth.tsv")
SOURCE2_PATH = os.path.join(REPO_ROOT, "data", "sample", "sample_source2.tsv")
SOURCE3_PATH = os.path.join(REPO_ROOT, "data", "sample", "sample_source3.tsv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "pair_scores.parquet")


def main():
    gt_df = pd.read_csv(GROUND_TRUTH_PATH, sep="\t", dtype=str, keep_default_na=False)
    s2_df = pd.read_csv(SOURCE2_PATH, sep="\t", dtype=str, keep_default_na=False)
    s3_df = pd.read_csv(SOURCE3_PATH, sep="\t", dtype=str, keep_default_na=False)

    source1_entity_ids = gt_df["source1_entity_id"].dropna().unique().tolist()

    s2_candidates = set(s2_df["entity_id"].dropna().unique())
    s3_candidates = set(s3_df["entity_id"].dropna().unique())
    candidate_pool = sorted(list(s2_candidates | s3_candidates))

    random.seed(42)

    rows = []
    for s1_id in source1_entity_ids:
        k = random.randint(0, 8)
        sampled_cands = random.sample(candidate_pool, k)
        for cand_id in sampled_cands:
            score = random.random()
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "score": score,
            })

    df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "score"])
    df["score"] = df["score"].astype(float)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    try:
        df.to_parquet(OUTPUT_PATH, index=False)
    except (ImportError, ModuleNotFoundError) as e:
        print(f"Error: Missing parquet engine ({e}). Please install pyarrow or fastparquet: pip install pyarrow")
        sys.exit(1)
    except Exception as e:
        if "parquet" in str(e).lower() or "pyarrow" in str(e).lower() or "fastparquet" in str(e).lower():
            print(f"Error: Missing parquet engine ({e}). Please install pyarrow: pip install pyarrow")
            sys.exit(1)
        raise

    total_rows = len(df)
    unique_s1 = df["source1_entity_id"].nunique()
    min_score = df["score"].min() if total_rows > 0 else 0.0
    max_score = df["score"].max() if total_rows > 0 else 0.0
    mean_score = df["score"].mean() if total_rows > 0 else 0.0

    print(f"Total rows written: {total_rows}")
    print(f"Unique source1 entities covered: {unique_s1}")
    print(f"Min score: {min_score:.4f}, Max score: {max_score:.4f}, Mean score: {mean_score:.4f}")


if __name__ == "__main__":
    main()
