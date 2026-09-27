"""Trains a LightGBM classifier on the pair feature table using train_ground_truth.tsv as labels.
Saves the trained model and per-pair match scores.

Step 1: Grouped, singleton-stratified train/validation split by source1_entity_id.
Step 2: Baseline LightGBM classifier training and gain-based feature importance evaluation.
Step 3: Save trained model to models/model.txt and score all candidate pairs to pair_scores.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
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


def get_repo_root() -> Path:
    """Find repository root across execution contexts."""
    candidate = Path(__file__).resolve().parent.parent.parent.parent
    if (candidate / "student_resource").exists() or (candidate / "data").exists():
        return candidate
    cwd = Path.cwd()
    if (cwd / "student_resource").exists() or (cwd / "data").exists():
        return cwd
    if (cwd.parent.parent / "student_resource").exists() or (cwd.parent.parent / "data").exists():
        return cwd.parent.parent
    return candidate


def load_pair_features(sample: bool) -> pd.DataFrame:
    """Load pair feature table from parquet.

    Parameters
    ----------
    sample : bool
        If True, reads data/processed/sample/pair_features.parquet.
        If False, reads data/processed/pair_features.parquet.

    Returns
    -------
    pd.DataFrame
        Pair feature table.

    Raises
    ------
    FileNotFoundError
        If pair_features.parquet does not exist.
    """
    repo_root = get_repo_root()
    if sample:
        file_path = repo_root / "data" / "processed" / "sample" / "pair_features.parquet"
    else:
        file_path = repo_root / "data" / "processed" / "pair_features.parquet"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Pair features file not found at: {file_path}. "
            "Please run features.py first to generate pair_features.parquet."
        )

    return pd.read_parquet(file_path)


def split_train_val(
    pair_features_df: pd.DataFrame,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform grouped, singleton-stratified train/validation split by source1_entity_id.

    No single Source 1 entity's candidate pairs may appear in both train and validation splits.
    Stratification ensures both splits have approximately the same proportion of singleton entities.

    NOTE ON SINGLETON DEFINITION:
    Here, 'singleton' is derived directly from the pair-level data as an S1 entity that has
    zero positive (is_match == True) candidate pairs in pair_features_df. This is distinct from
    ground truth's own global singleton definition (e.g. an entity might have matches in ground
    truth that were not retrieved by blocking, or vice-versa), but reflects the actual label
    signal available to the model during training.

    Parameters
    ----------
    pair_features_df : pd.DataFrame
        Pair feature table containing 'source1_entity_id' and 'is_match'.
    val_fraction : float, default 0.2
        Fraction of entities allocated to the validation split.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, val_df) filtered pair feature DataFrames.
    """
    if "source1_entity_id" not in pair_features_df.columns:
        raise ValueError("pair_features_df must contain 'source1_entity_id' column.")
    if "is_match" not in pair_features_df.columns:
        raise ValueError("pair_features_df must contain 'is_match' label column.")

    # 1. Compute per-entity positive match counts to determine singleton status
    # Group by source1_entity_id and sum is_match (True=1, False=0)
    match_counts = pair_features_df.groupby("source1_entity_id")["is_match"].sum()

    # Entities with zero positive is_match rows are singletons in this dataset
    singleton_ids = match_counts[match_counts == 0].index.tolist()
    non_singleton_ids = match_counts[match_counts > 0].index.tolist()

    # 2. Perform separate random splits on singleton and non-singleton entity IDs
    train_sing_ids, val_sing_ids = train_test_split(
        singleton_ids,
        test_size=val_fraction,
        random_state=random_state,
    )
    train_non_ids, val_non_ids = train_test_split(
        non_singleton_ids,
        test_size=val_fraction,
        random_state=random_state,
    )

    # 3. Combine into final train and val entity sets
    train_entity_ids = set(train_sing_ids + train_non_ids)
    val_entity_ids = set(val_sing_ids + val_non_ids)

    # 4. Filter pair-level rows (boolean indexing creates new slice; reset_index creates contiguous table without extra .copy())
    train_df = pair_features_df[pair_features_df["source1_entity_id"].isin(train_entity_ids)]
    val_df = pair_features_df[pair_features_df["source1_entity_id"].isin(val_entity_ids)]

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def train_baseline_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[lgb.Booster, dict[str, float]]:
    """Train a baseline LightGBM binary classifier on pair features with validation early stopping.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split pair feature table.
    val_df : pd.DataFrame
        Validation split pair feature table.

    Returns
    -------
    tuple[lgb.Booster, dict[str, float]]
        Trained LightGBM booster and a dictionary of validation metrics:
        - auc_roc: Area Under ROC Curve
        - average_precision: PR-AUC (Average Precision)
        - precision_at_0_5: Precision at default 0.5 threshold
        - recall_at_0_5: Recall at default 0.5 threshold
        - f0_5_at_0_5: F0.5 score at default 0.5 threshold
        - best_iteration: Boosting round with lowest validation loss
    """
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["is_match"].astype(int)
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["is_match"].astype(int)

    # Compute scale_pos_weight from train-split negative-to-positive class imbalance (~18x)
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "verbose": -1,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )

    y_val_prob = booster.predict(X_val)

    # NOTE: 0.5 is just a placeholder threshold for this baseline sanity check.
    # Real threshold tuning happens downstream in assign_matches.py, and since
    # F0.5 is precision-weighted 2x (beta=0.5), the optimal threshold is likely
    # higher than 0.5, not something to solve here.
    y_val_pred_05 = (y_val_prob >= 0.5).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(y_val, y_val_prob)),
        "average_precision": float(average_precision_score(y_val, y_val_prob)),
        "precision_at_0_5": float(precision_score(y_val, y_val_pred_05, zero_division=0.0)),
        "recall_at_0_5": float(recall_score(y_val, y_val_pred_05, zero_division=0.0)),
        "f0_5_at_0_5": float(fbeta_score(y_val, y_val_pred_05, beta=0.5, zero_division=0.0)),
        "best_iteration": int(booster.best_iteration),
    }

    return booster, metrics


def print_feature_importance(booster: lgb.Booster) -> None:
    """Print LightGBM feature importance based on total gain, sorted descending."""
    gain_scores = booster.feature_importance(importance_type="gain")
    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "gain": gain_scores,
        }
    ).sort_values("gain", ascending=False).reset_index(drop=True)

    total_gain = importance_df["gain"].sum()
    importance_df["pct_gain"] = (
        (importance_df["gain"] / total_gain * 100.0) if total_gain > 0 else 0.0
    )

    print("\n" + "=" * 80)
    print("LIGHTGBM FEATURE IMPORTANCE (TOTAL GAIN)")
    print("=" * 80)
    for idx, row in importance_df.iterrows():
        print(
            f"  {idx + 1:2d}. {row['feature']:<30} "
            f"Gain: {row['gain']:>14,.2f} ({row['pct_gain']:>5.2f}%)"
        )
    print("=" * 80)


def save_model(booster: lgb.Booster, sample: bool) -> Path:
    """Save trained LightGBM booster model to models/model.txt.

    Parameters
    ----------
    booster : lgb.Booster
        Trained model booster.
    sample : bool
        Flag indicating if running on sample (model is saved to the same
        models/model.txt path per interface contract).

    Returns
    -------
    Path
        Path to saved model file.
    """
    repo_root = get_repo_root()
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.txt"

    booster.save_model(str(model_path))
    print(f"[INFO] Saved trained LightGBM model to: {model_path} ({model_path.stat().st_size:,d} bytes)")
    return model_path


def score_and_save_predictions(
    booster: lgb.Booster,
    pair_features_df: pd.DataFrame,
    sample: bool,
    chunk_size: int = 5_000_000,
) -> pd.DataFrame:
    """Score all candidate pairs and save results to pair_scores.parquet.

    Scores every row in the full pair_features_df (both train and val combined)
    using booster.predict() on FEATURE_COLUMNS.
    Constructs output DataFrame with exactly:
    - source1_entity_id
    - candidate_entity_id
    - score (float 0-1)

    Writes to:
    - data/processed/sample/pair_scores.parquet (if sample=True)
    - data/processed/pair_scores.parquet (if sample=False)

    Parameters
    ----------
    booster : lgb.Booster
        Trained LightGBM booster.
    pair_features_df : pd.DataFrame
        Full candidate pair features DataFrame.
    sample : bool
        Whether running on sample dataset.
    chunk_size : int, default 5_000_000
        Number of pair rows per scoring/writing chunk to keep memory bounded.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['source1_entity_id', 'candidate_entity_id', 'score'].
    """
    repo_root = get_repo_root()

    if sample:
        out_dir = repo_root / "data" / "processed" / "sample"
    else:
        out_dir = repo_root / "data" / "processed"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pair_scores.parquet"

    total_rows = len(pair_features_df)

    # For sample or small datasets, score directly in one pass
    if total_rows <= chunk_size:
        scores = booster.predict(pair_features_df[FEATURE_COLUMNS]).astype(np.float32)

        scores_df = pd.DataFrame(
            {
                "source1_entity_id": pair_features_df["source1_entity_id"].astype(str),
                "candidate_entity_id": pair_features_df["candidate_entity_id"].astype(str),
                "score": scores,
            }
        )
        scores_df.to_parquet(out_path, index=False)
        print(f"[INFO] Saved pair scores ({len(scores_df):,d} rows) to: {out_path} ({out_path.stat().st_size:,d} bytes)")
        return scores_df

    # For full-scale data (> chunk_size), score and write in streaming chunks
    print(f"[INFO] Streaming pair scoring across {total_rows:,d} rows (chunk_size={chunk_size:,d}) to: {out_path}")
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    writer: pq.ParquetWriter | None = None

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(total_rows, (chunk_idx + 1) * chunk_size)
        chunk_slice = pair_features_df.iloc[start_idx:end_idx]

        chunk_scores = booster.predict(chunk_slice[FEATURE_COLUMNS]).astype(np.float32)
        chunk_df = pd.DataFrame(
            {
                "source1_entity_id": chunk_slice["source1_entity_id"].astype(str),
                "candidate_entity_id": chunk_slice["candidate_entity_id"].astype(str),
                "score": chunk_scores,
            }
        )

        table = pa.Table.from_pandas(chunk_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)
        print(f"[INFO] Scored and wrote chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_df):,d} rows)")

    if writer is not None:
        writer.close()

    print(f"[INFO] Finished saving chunked pair scores ({total_rows:,d} rows) to: {out_path} ({out_path.stat().st_size:,d} bytes)")
    return pd.read_parquet(out_path)


def score_existing_model(
    model_path: str | Path,
    pair_features_path: str | Path,
    output_path: str | Path,
    chunk_size: int = 5_000_000,
) -> Path:
    """Score pair features parquet using a previously-trained LightGBM model.

    Loads the booster from model_path via lgb.Booster(model_file=str(model_path)).
    Reads the pair features parquet (no is_match column expected).
    Scores all pairs using FEATURE_COLUMNS and writes ['source1_entity_id', 'candidate_entity_id', 'score']
    to output_path. For large files, reads and writes in streaming batches to keep memory bounded.

    Parameters
    ----------
    model_path : str | Path
        Path to saved LightGBM model file (e.g. models/model.txt).
    pair_features_path : str | Path
        Path to pair features parquet file (e.g. data/processed/test_pair_features.parquet).
    output_path : str | Path
        Destination path for output pair scores parquet (e.g. data/processed/pair_scores.parquet).
    chunk_size : int, default 5_000_000
        Number of pair rows per scoring/writing chunk to keep memory bounded.

    Returns
    -------
    Path
        Path to saved predictions parquet file.
    """
    model_file = Path(model_path)
    if not model_file.is_absolute():
        repo_root = get_repo_root()
        if not model_file.exists() and (repo_root / model_file).exists():
            model_file = repo_root / model_file

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at: {model_file}")

    features_file = Path(pair_features_path)
    if not features_file.is_absolute():
        repo_root = get_repo_root()
        if not features_file.exists() and (repo_root / features_file).exists():
            features_file = repo_root / features_file

    if not features_file.exists():
        raise FileNotFoundError(f"Pair features file not found at: {features_file}")

    dest_path = Path(output_path)
    if not dest_path.is_absolute():
        repo_root = get_repo_root()
        dest_path = repo_root / dest_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"[SCORE-ONLY] Loading trained model from: {model_file}")
    booster = lgb.Booster(model_file=str(model_file))

    print(f"[SCORE-ONLY] Scoring pair features from: {features_file}")
    print(f"[SCORE-ONLY] Target predictions output: {dest_path}")
    print("=" * 80)

    t0 = time.time()
    pq_file = pq.ParquetFile(features_file)
    total_rows = pq_file.metadata.num_rows

    read_columns = ["source1_entity_id", "candidate_entity_id"] + FEATURE_COLUMNS
    writer: pq.ParquetWriter | None = None
    processed_rows = 0
    batch_idx = 0

    try:
        for batch in pq_file.iter_batches(batch_size=chunk_size, columns=read_columns):
            batch_idx += 1
            t_b0 = time.time()
            batch_df = batch.to_pandas()
            n_batch = len(batch_df)

            # Score using booster on FEATURE_COLUMNS
            scores = booster.predict(batch_df[FEATURE_COLUMNS]).astype(np.float32)

            out_chunk = pd.DataFrame(
                {
                    "source1_entity_id": batch_df["source1_entity_id"].astype(str),
                    "candidate_entity_id": batch_df["candidate_entity_id"].astype(str),
                    "score": scores,
                }
            )

            table = pa.Table.from_pandas(out_chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(dest_path, table.schema, compression="snappy")
            writer.write_table(table)

            processed_rows += n_batch
            t_batch = time.time() - t_b0
            rate = n_batch / t_batch if t_batch > 0 else 0
            print(
                f"[SCORE BATCH {batch_idx}] Scored {n_batch:,d} pairs "
                f"({processed_rows:,d}/{total_rows:,d}) in {t_batch:.2f}s ({rate:,.0f} pairs/sec)"
            )
    finally:
        if writer is not None:
            writer.close()

    total_time = time.time() - t0
    print("=" * 80)
    print(f"[SCORE-ONLY] Finished scoring {processed_rows:,d} pairs in {total_time:.2f}s")
    print(f"[SCORE-ONLY] Predictions written to: {dest_path} ({dest_path.stat().st_size:,d} bytes)")
    print("=" * 80)

    return dest_path


if __name__ == "__main__":
    t_start = time.time()

    parser = argparse.ArgumentParser(
        description="Train model pipeline: split, train, save model, and score candidate pairs."
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample fixture dataset instead of the full dataset.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Skip training and score an existing model against candidate pairs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/model.txt",
        help="Path to trained LightGBM model file (default: models/model.txt).",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Path to candidate pairs file (TSV) or pair features file (parquet).",
    )
    parser.add_argument(
        "--pair-features",
        type=str,
        default=None,
        help="Path to precomputed pair features parquet file (alternative to --candidates).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for scored predictions parquet (default: data/processed/pair_scores.parquet).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5_000_000,
        help="Chunk size for streaming scoring (default: 5,000,000).",
    )
    args = parser.parse_args()

    repo_root = get_repo_root()

    if args.score_only:
        model_path = args.model or "models/model.txt"

        features_path = args.pair_features
        if not features_path and args.candidates:
            cand_p = Path(args.candidates)
            if not cand_p.is_absolute() and not cand_p.exists() and (repo_root / cand_p).exists():
                cand_p = repo_root / cand_p
            if cand_p.suffix == ".parquet":
                features_path = str(cand_p)
            else:
                from business_entity_resolution.src.features import build_pair_features_for_scoring

                intermediate_out = (
                    repo_root / "data" / "processed" / "sample" / "test_pair_features.parquet"
                    if args.sample
                    else repo_root / "data" / "processed" / "test_pair_features.parquet"
                )
                print(f"[INFO] Extracting features from candidate pairs: {cand_p}")
                features_path = str(
                    build_pair_features_for_scoring(
                        candidate_pairs_path=cand_p,
                        clean_data_sample=args.sample,
                        output_path=intermediate_out,
                        chunk_size=args.chunk_size,
                    )
                )

        if not features_path:
            features_path = str(
                repo_root / "data" / "processed" / "sample" / "pair_features.parquet"
                if args.sample
                else repo_root / "data" / "processed" / "pair_features.parquet"
            )

        output_path = args.output or str(
            repo_root / "data" / "processed" / "sample" / "pair_scores.parquet"
            if args.sample
            else repo_root / "data" / "processed" / "pair_scores.parquet"
        )

        score_existing_model(
            model_path=model_path,
            pair_features_path=features_path,
            output_path=output_path,
            chunk_size=args.chunk_size,
        )
    else:
        print("=" * 80)
        print(f"RUNNING TRAIN MODEL PIPELINE (sample={args.sample})")
        print("=" * 80)

        # 1. Load full pair features
        pair_features_df = load_pair_features(sample=args.sample)
        print(
            f"[INFO] Loaded pair features table: {pair_features_df.shape[0]:,d} rows, "
            f"{pair_features_df.shape[1]} columns"
        )

        # 2. Compute grouped stratified split
        train_df, val_df = split_train_val(pair_features_df, val_fraction=0.2, random_state=42)

        # Calculate entity-level statistics
        total_entities = pair_features_df["source1_entity_id"].nunique()
        entity_pos_counts = pair_features_df.groupby("source1_entity_id")["is_match"].sum()
        total_singletons = (entity_pos_counts == 0).sum()
        total_non_singletons = (entity_pos_counts > 0).sum()
        total_singleton_pct = total_singletons / total_entities * 100.0

        train_entities = set(train_df["source1_entity_id"])
        val_entities = set(val_df["source1_entity_id"])

        train_entity_pos = train_df.groupby("source1_entity_id")["is_match"].sum()
        train_singletons = (train_entity_pos == 0).sum()
        train_singleton_pct = train_singletons / len(train_entities) * 100.0

        val_entity_pos = val_df.groupby("source1_entity_id")["is_match"].sum()
        val_singletons = (val_entity_pos == 0).sum()
        val_singleton_pct = val_singletons / len(val_entities) * 100.0

        # Explicitly check zero overlap between train and val entity sets
        overlap = train_entities & val_entities
        assert len(overlap) == 0, f"Leakage detected! {len(overlap)} entity IDs overlap between train and val: {list(overlap)[:5]}"

        print("\n" + "=" * 80)
        print("TRAIN / VALIDATION SPLIT DIAGNOSTICS")
        print("=" * 80)
        print("--- Full Dataset Entity Breakdown ---")
        print(f"Total unique S1 entities   : {total_entities:,d}")
        print(f"Singleton entities (0 pos) : {total_singletons:,d} ({total_singleton_pct:.2f}%)")
        print(f"Non-singleton entities     : {total_non_singletons:,d} ({100.0 - total_singleton_pct:.2f}%)")

        print("\n--- Split Sizes & Entity Counts ---")
        print(
            f"Train set: {len(train_df):,d} rows | {len(train_entities):,d} unique entities "
            f"({len(train_entities)/total_entities*100:.2f}%)"
        )
        print(
            f"Val set  : {len(val_df):,d} rows | {len(val_entities):,d} unique entities "
            f"({len(val_entities)/total_entities*100:.2f}%)"
        )

        print("\n--- Stratification Sanity Check (Singleton Rates) ---")
        print(f"Full dataset singleton rate: {total_singleton_pct:.2f}% ({total_singletons:,d}/{total_entities:,d})")
        print(f"Train split singleton rate : {train_singleton_pct:.2f}% ({train_singletons:,d}/{len(train_entities):,d})")
        print(f"Val split singleton rate   : {val_singleton_pct:.2f}% ({val_singletons:,d}/{len(val_entities):,d})")

        print("\n--- Group Isolation Test ---")
        print(f"Train and Val entity ID overlap: {len(overlap)} entities -> PASS")

        # 3. Train baseline LightGBM model
        print("\n" + "=" * 80)
        print("TRAINING BASELINE LIGHTGBM CLASSIFIER")
        print("=" * 80)
        booster, val_metrics = train_baseline_model(train_df, val_df)

        print("\nValidation Metrics (Evaluated on X_val/y_val):")
        for metric_name, val in val_metrics.items():
            if metric_name == "best_iteration":
                print(f"  {metric_name:<20}: {val}")
            else:
                print(f"  {metric_name:<20}: {val:.4f}")

        # 4. Print feature importance
        print_feature_importance(booster)

        # 5. Save trained model
        print("\n" + "=" * 80)
        print("SAVING MODEL")
        print("=" * 80)
        model_path = save_model(booster, sample=args.sample)

        # 6. Score all candidate pairs and save predictions
        print("\n" + "=" * 80)
        print("SCORING CANDIDATE PAIRS & SAVING PREDICTIONS")
        print("=" * 80)
        scores_df = score_and_save_predictions(booster, pair_features_df, sample=args.sample)

        # Verify score row count equals full pair_features row count
        assert len(scores_df) == len(pair_features_df), (
            f"Mismatch in scored rows! Expected {len(pair_features_df)}, got {len(scores_df)}"
        )

        # Sanity checks on pair_scores
        assert not scores_df["score"].isna().any(), "Found NaN / null in pair scores!"
        assert (scores_df["score"] >= 0.0).all() and (scores_df["score"] <= 1.0).all(), (
            "Scores out of [0, 1] range!"
        )

        repo_root = get_repo_root()
        scores_path = (
            repo_root / "data" / "processed" / "sample" / "pair_scores.parquet"
            if args.sample
            else repo_root / "data" / "processed" / "pair_scores.parquet"
        )

        total_time = time.time() - t_start

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETION & SANITY CHECK SUMMARY")
        print("=" * 80)
        print(f"Model file path       : {model_path} ({model_path.stat().st_size:,d} bytes)")
        print(f"Pair scores file path : {scores_path} ({scores_path.stat().st_size:,d} bytes)")
        print(f"Pair scores row count : {len(scores_df):,d} rows (matches full candidate pairs)")
        print(f"Null/NaN scores count : {scores_df['score'].isna().sum()} (0 confirmed)")
        print(f"Total script runtime  : {total_time:.2f}s (wall-clock)")

        print("\nScore Distribution Basics:")
        print(f"  Min   : {scores_df['score'].min():.6f}")
        print(f"  Max   : {scores_df['score'].max():.6f}")
        print(f"  Mean  : {scores_df['score'].mean():.6f}")
        print(f"  Median: {scores_df['score'].median():.6f}")
        print(f"  Pairs with score >= 0.5: {(scores_df['score'] >= 0.5).sum():,d} ({(scores_df['score'] >= 0.5).mean()*100:.2f}%)")
        print("=" * 80)
