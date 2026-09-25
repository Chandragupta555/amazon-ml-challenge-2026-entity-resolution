"""Generate a small sample dataset fixture for business entity resolution.

This script samples 3,000 Source 1 entities across different match tiers
(0 matches, 1 match, multiple matches) along with all their ground truth
matching S2/S3 records plus 5,000 distractor records each for Source 2 and
Source 3, writing the results to data/sample/.
"""

from pathlib import Path
import pandas as pd


def resolve_paths():
    """Resolve input dataset path and output directory robustly."""
    repo_root = Path(__file__).resolve().parent.parent
    dataset_dir = repo_root / "student_resource" / "dataset" / "train"

    if not dataset_dir.exists():
        # Fallback to current working directory if script executed differently
        cwd_dir = Path.cwd() / "student_resource" / "dataset" / "train"
        if cwd_dir.exists():
            dataset_dir = cwd_dir
            repo_root = Path.cwd()
        else:
            raise FileNotFoundError(f"Dataset train directory not found at {dataset_dir} or {cwd_dir}")

    output_dir = repo_root / "data" / "sample"
    output_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir, output_dir


def main():
    dataset_dir, output_dir = resolve_paths()
    print(f"Reading training datasets from: {dataset_dir}")
    print(f"Output directory: {output_dir}")

    # 1. Read input files with required options
    print("Loading train_ground_truth.tsv...")
    df_gt = pd.read_csv(
        dataset_dir / "train_ground_truth.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print("Loading train_source1.tsv...")
    df_s1 = pd.read_csv(
        dataset_dir / "train_source1.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print("Loading train_source2.tsv...")
    df_s2 = pd.read_csv(
        dataset_dir / "train_source2.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print("Loading train_source3.tsv...")
    df_s3 = pd.read_csv(
        dataset_dir / "train_source3.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    # 2. Sample 3000 Source 1 entities with a mix of 0, 1, and multiple matches
    print("Sampling 3000 Source 1 entities with random_state=42...")
    match_counts = df_gt["matched_entity_ids"].apply(
        lambda x: len([i for i in x.split(",") if i.strip()])
    )

    # 400 zero matches (singletons), 600 single match, 2000 multiple matches
    sample_zero = df_gt[match_counts == 0].sample(n=400, random_state=42)
    sample_single = df_gt[match_counts == 1].sample(n=600, random_state=42)
    sample_multi = df_gt[match_counts >= 2].sample(n=2000, random_state=42)

    sample_gt = (
        pd.concat([sample_zero, sample_single, sample_multi])
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    sampled_s1_ids = set(sample_gt["source1_entity_id"])
    sample_s1 = df_s1[df_s1["entity_id"].isin(sampled_s1_ids)].reset_index(drop=True)

    # 3. Collect ALL true-match S2/S3 records for the sampled S1 entities
    all_matched_ids = set(
        i for ids_str in sample_gt["matched_entity_ids"].str.split(",") for i in ids_str if i.strip()
    )
    true_s2_ids = {i for i in all_matched_ids if i.startswith("S2-")}
    true_s3_ids = {i for i in all_matched_ids if i.startswith("S3-")}

    true_s2_records = df_s2[df_s2["entity_id"].isin(true_s2_ids)]
    true_s3_records = df_s3[df_s3["entity_id"].isin(true_s3_ids)]

    # Distractors: random 5000 S2 and 5000 S3 records not tied to any of the sampled S1 entities
    print("Sampling 5000 S2 and 5000 S3 distractor/noise records...")
    distractor_s2_pool = df_s2[~df_s2["entity_id"].isin(true_s2_ids)]
    distractor_s3_pool = df_s3[~df_s3["entity_id"].isin(true_s3_ids)]

    distractors_s2 = distractor_s2_pool.sample(n=5000, random_state=42)
    distractors_s3 = distractor_s3_pool.sample(n=5000, random_state=42)

    sample_s2 = (
        pd.concat([true_s2_records, distractors_s2])
        .drop_duplicates(subset=["entity_id"])
        .reset_index(drop=True)
    )
    sample_s3 = (
        pd.concat([true_s3_records, distractors_s3])
        .drop_duplicates(subset=["entity_id"])
        .reset_index(drop=True)
    )

    # 4. Write output files to data/sample/
    files = {
        "sample_source1.tsv": sample_s1,
        "sample_source2.tsv": sample_s2,
        "sample_source3.tsv": sample_s3,
        "sample_ground_truth.tsv": sample_gt,
    }

    print("\nWriting sample fixture files...")
    for filename, df in files.items():
        out_path = output_dir / filename
        df.to_csv(out_path, sep="\t", index=False)

    # 5. Print row counts
    print("\n=== SAMPLE FIXTURE ROW COUNTS ===")
    for filename, df in files.items():
        print(f"{filename:25s}: {len(df):,d} rows")


if __name__ == "__main__":
    main()
