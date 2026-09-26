import argparse
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PAIR_SCORES = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "processed", "pair_scores.parquet")
)
DEFAULT_OUTPUT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "output", "matching_results.tsv")
)
DEFAULT_SOURCE1 = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "..", "student_resource", "dataset", "train", "train_source1.tsv")
)
DEFAULT_GROUND_TRUTH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "..", "student_resource", "dataset", "train", "train_ground_truth.tsv")
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign matches from pair scores."
    )
    parser.add_argument(
        "--pair-scores",
        default=DEFAULT_PAIR_SCORES,
        help=f"Path to pair_scores.parquet (default: {DEFAULT_PAIR_SCORES})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for match assignment (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path to output matching results TSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--source1",
        default=DEFAULT_SOURCE1,
        help=f"Path to full source1 reference TSV (default: {DEFAULT_SOURCE1})",
    )
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Sweep thresholds to find best F0.5, using --ground-truth for scoring.",
    )
    parser.add_argument(
        "--ground-truth",
        dest="ground_truth",
        default=DEFAULT_GROUND_TRUTH,
        help=f"Path to ground truth TSV, used only with --tune-threshold (default: {DEFAULT_GROUND_TRUTH})",
    )
    return parser.parse_args()


def assign_one_to_one(df: pd.DataFrame, threshold: float) -> dict:
    """Greedy match assignment enforcing at most one S1 claim per S2/S3 candidate.

    Filters pairs by score >= threshold, sorts descending by score, and assigns
    candidates greedily to Source 1 entities ensuring no candidate is assigned
    more than once across all Source 1 entities.
    """
    filtered_df = df[df["score"] >= threshold].sort_values(by="score", ascending=False)

    used_candidate_ids = set()
    assignments = {}

    s1_ids = filtered_df["source1_entity_id"].tolist()
    cand_ids = filtered_df["candidate_entity_id"].tolist()

    for s1_id, cand_id in zip(s1_ids, cand_ids):
        if cand_id not in used_candidate_ids:
            used_candidate_ids.add(cand_id)
            assignments.setdefault(s1_id, []).append(cand_id)

    return assignments


def write_matching_tsv(assignments: dict, all_source1_ids: list, output_path: str):
    """Write assigned matches to a TSV file in the exact required format."""
    parent_dir = os.path.dirname(os.path.abspath(output_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_source1_ids:
            matched_ids = assignments.get(s1_id, [])
            matched_str = ",".join(matched_ids) if matched_ids else ""
            f.write(f"{s1_id}\t{matched_str}\n")


def parse_id_set(raw_str):
    if not raw_str:
        return set()
    return {item.strip() for item in raw_str.split(",") if item.strip()}


def score_assignments(assignments: dict, ground_truth_df: pd.DataFrame) -> float:
    scores = []
    for s1_id, true_raw in zip(ground_truth_df["source1_entity_id"], ground_truth_df["matched_entity_ids"]):
        true_set = parse_id_set(true_raw)
        predicted_set = set(assignments.get(s1_id, []))
        true_positives = len(true_set & predicted_set)
        if not true_set and not predicted_set:
            precision, recall, f0_5 = 1.0, 1.0, 1.0
        else:
            precision = true_positives / len(predicted_set) if predicted_set else (1.0 if not true_set else 0.0)
            recall = true_positives / len(true_set) if true_set else 1.0
            denom = 0.25 * precision + recall
            f0_5 = (1.25 * precision * recall) / denom if denom > 0 else 0.0
        scores.append(f0_5)
    return sum(scores) / len(scores) if scores else 0.0


def find_best_threshold(df: pd.DataFrame, ground_truth_df: pd.DataFrame, thresholds: list) -> tuple:
    """Sweep thresholds and return (best_threshold, best_score, all_results).

    all_results is a list of (threshold, score) tuples in the order tried.
    """
    if not thresholds:
        raise ValueError("thresholds list must not be empty — provide at least one threshold to evaluate.")

    all_results = []
    best_threshold = thresholds[0]
    best_score = -1.0

    for t in thresholds:
        assignments = assign_one_to_one(df, t)
        score = score_assignments(assignments, ground_truth_df)
        all_results.append((t, score))
        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score, all_results


def main():
    args = parse_args()

    df = pd.read_parquet(args.pair_scores)

    total_rows = len(df)
    unique_s1 = df["source1_entity_id"].nunique()
    unique_candidates = df["candidate_entity_id"].nunique()
    min_score = df["score"].min()
    max_score = df["score"].max()
    mean_score = df["score"].mean()

    print(f"Total rows: {total_rows}")
    print(f"Unique source1 entities: {unique_s1}")
    print(f"Unique candidate entities: {unique_candidates}")
    print(f"Score stats: min={min_score:.4f}, max={max_score:.4f}, mean={mean_score:.4f}")
    print("\nFirst 5 rows:")
    print(df.head())

    # NOTE: In the real pipeline, --source1 should point to the actual test set's
    # source1 file, not train_source1.tsv — this default is only for isolated dev
    # testing before the real test source1 file naming/location is confirmed.
    s1_df = pd.read_csv(args.source1, sep="\t", dtype=str, keep_default_na=False)
    id_col = "entity_id" if "entity_id" in s1_df.columns else "source1_entity_id"
    all_source1_ids = sorted(s1_df[id_col].unique().tolist())

    if args.tune_threshold:
        ground_truth_df = pd.read_csv(args.ground_truth, sep="\t", dtype=str, keep_default_na=False)
        thresholds = [round(x * 0.05, 2) for x in range(1, 20)]
        best_threshold, best_score, all_results = find_best_threshold(df, ground_truth_df, thresholds)

        print("\nThreshold sweep results:")
        for t, s in all_results:
            print(f"  threshold={t:.2f}: F0.5={s:.4f}")
        print(f"\nBest threshold: {best_threshold:.2f} (F0.5={best_score:.4f})")

        effective_threshold = best_threshold
    else:
        effective_threshold = args.threshold

    assignments = assign_one_to_one(df, effective_threshold)
    write_matching_tsv(assignments, all_source1_ids, args.output)

    total_written = len(all_source1_ids)
    non_empty_count = sum(1 for sid in all_source1_ids if sid in assignments and assignments[sid])
    empty_count = total_written - non_empty_count

    print(f"\nOutput written to: {args.output}")
    print(f"Total rows written: {total_written}")
    print(f"Entities with non-empty matches: {non_empty_count}")
    print(f"Entities with empty matches: {empty_count}")


if __name__ == "__main__":
    main()
