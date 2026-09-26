import argparse
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MATCHING = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "output", "matching_results.tsv"))
DEFAULT_GROUND_TRUTH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "..", "student_resource", "dataset", "train", "train_ground_truth.tsv")
)
DEFAULT_SAMPLE_MATCHING = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "output", "sample", "matching_results.tsv"))
DEFAULT_SAMPLE_GROUND_TRUTH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "sample", "sample_ground_truth.tsv")
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate business entity resolution matching results against ground truth."
    )
    parser.add_argument(
        "--matching",
        default=None,
        help="Path to matching_results.tsv (default: output/matching_results.tsv, or output/sample/matching_results.tsv if --sample is set)",
    )
    parser.add_argument(
        "--ground-truth",
        dest="ground_truth",
        default=None,
        help="Path to ground truth tsv (default: ../../../student_resource/dataset/train/train_ground_truth.tsv, or data/sample/sample_ground_truth.tsv if --sample is set)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="When set, override defaults to output/sample/matching_results.tsv and data/sample/sample_ground_truth.tsv",
    )
    args = parser.parse_args()

    if args.sample:
        if args.matching is None:
            args.matching = DEFAULT_SAMPLE_MATCHING
        if args.ground_truth is None:
            args.ground_truth = DEFAULT_SAMPLE_GROUND_TRUTH
    else:
        if args.matching is None:
            args.matching = DEFAULT_MATCHING
        if args.ground_truth is None:
            args.ground_truth = DEFAULT_GROUND_TRUTH

    return args


def parse_id_set(raw_str: str) -> set:
    if not raw_str:
        return set()
    return {item.strip() for item in raw_str.split(",") if item.strip()}


def main():
    args = parse_args()

    gt_df = pd.read_csv(args.ground_truth, sep="\t", dtype=str, keep_default_na=False)
    matching_df = pd.read_csv(args.matching, sep="\t", dtype=str, keep_default_na=False)

    # Check for country column or lookup source1 table
    country_available = False
    country_map = {}

    if "country" in gt_df.columns:
        country_available = True
        country_map = dict(zip(gt_df["source1_entity_id"], gt_df["country"]))
    else:
        gt_dir = os.path.dirname(os.path.abspath(args.ground_truth))
        candidate_paths = [
            os.path.join(gt_dir, "train_source1.tsv"),
            os.path.join(gt_dir, "sample_source1.tsv"),
            os.path.join(SCRIPT_DIR, "..", "..", "..", "student_resource", "dataset", "train", "train_source1.tsv"),
            os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "sample", "sample_source1.tsv"),
            os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "sample", "train_source1.tsv"),
            os.path.join(SCRIPT_DIR, "..", "data", "sample", "sample_source1.tsv"),
        ]

        source1_file = None
        for path in candidate_paths:
            if os.path.isfile(path):
                source1_file = path
                break

        if source1_file is not None:
            try:
                s1_df = pd.read_csv(source1_file, sep="\t", dtype=str, keep_default_na=False)
                if "country" in s1_df.columns:
                    id_col = (
                        "source1_entity_id"
                        if "source1_entity_id" in s1_df.columns
                        else ("entity_id" if "entity_id" in s1_df.columns else None)
                    )
                    if id_col is not None:
                        country_map = dict(zip(s1_df[id_col], s1_df["country"]))
                        country_available = True
            except Exception:
                country_available = False

    matching_map = dict(zip(matching_df["source1_entity_id"], matching_df["matched_entity_ids"]))

    per_entity_records = []
    missing_from_matching = 0

    s1_ids = gt_df["source1_entity_id"].tolist()
    true_raws = gt_df["matched_entity_ids"].tolist()

    for s1_id, true_raw in zip(s1_ids, true_raws):
        true_set = parse_id_set(true_raw)

        if s1_id in matching_map:
            predicted_set = parse_id_set(matching_map[s1_id])
        else:
            predicted_set = set()
            missing_from_matching += 1

        true_positives = len(true_set & predicted_set)
        if not true_set and not predicted_set:
            precision, recall, f0_5 = 1.0, 1.0, 1.0
        else:
            precision = true_positives / len(predicted_set) if predicted_set else (1.0 if not true_set else 0.0)
            recall = true_positives / len(true_set) if true_set else 1.0
            denom = 0.25 * precision + recall
            f0_5 = (1.25 * precision * recall) / denom if denom > 0 else 0.0

        record = {
            "entity_id": s1_id,
            "precision": precision,
            "recall": recall,
            "f0_5": f0_5,
            "is_singleton": len(true_set) == 0,
        }
        if country_available and s1_id in country_map:
            record["country"] = country_map[s1_id]

        per_entity_records.append(record)

    overall_f05 = (
        sum(r["f0_5"] for r in per_entity_records) / len(per_entity_records)
        if per_entity_records
        else 0.0
    )

    singleton_scores = [r["f0_5"] for r in per_entity_records if r["is_singleton"]]
    nonsingleton_scores = [r["f0_5"] for r in per_entity_records if not r["is_singleton"]]

    singleton_f05 = (
        sum(singleton_scores) / len(singleton_scores)
        if singleton_scores
        else 0.0
    )
    nonsingleton_f05 = (
        sum(nonsingleton_scores) / len(nonsingleton_scores)
        if nonsingleton_scores
        else 0.0
    )

    print(f"Overall macro-average F0.5: {overall_f05:.4f} (count: {len(per_entity_records)})")
    print(f"Singleton F0.5: {singleton_f05:.4f} (count: {len(singleton_scores)})")
    print(f"Non-singleton F0.5: {nonsingleton_f05:.4f} (count: {len(nonsingleton_scores)})")

    if country_available:
        print("F0.5 broken down by country:")
        country_groups = {}
        for r in per_entity_records:
            c = r.get("country")
            if c:
                country_groups.setdefault(c, []).append(r["f0_5"])
        for country in sorted(country_groups.keys()):
            scores = country_groups[country]
            mean_c = sum(scores) / len(scores) if scores else 0.0
            print(f"  {country}: {mean_c:.4f} (count: {len(scores)})")
    else:
        print("Note: Country breakdown skipped because no country column was found.")

    print(f"WARNING: {missing_from_matching} entities missing from matching file")


if __name__ == "__main__":
    main()
