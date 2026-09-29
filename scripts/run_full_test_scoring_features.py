import sys
import time
from pathlib import Path

# Add code directory to Python path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((repo_root / "code").resolve()))

from business_entity_resolution.src.features import build_pair_features_for_scoring


def main():
    candidate_pairs_path = repo_root / "output" / "test_candidate_pairs.tsv"
    output_path = repo_root / "data" / "processed" / "test_pair_features.parquet"

    print("=" * 80, flush=True)
    print("STARTING FULL-SCALE TEST PAIR FEATURE GENERATION", flush=True)
    print(f"Input TSV   : {candidate_pairs_path}", flush=True)
    print(f"Output Path : {output_path}", flush=True)
    print(f"Chunk Size  : 5,000,000 pairs", flush=True)
    print("=" * 80, flush=True)

    t0 = time.time()
    out_file = build_pair_features_for_scoring(
        candidate_pairs_path=candidate_pairs_path,
        clean_data_sample=False,
        output_path=output_path,
        chunk_size=5_000_000,
    )
    total_time = time.time() - t0

    print("=" * 80, flush=True)
    print("FULL-SCALE TEST PAIR FEATURE GENERATION COMPLETE", flush=True)
    print(f"Output File : {out_file}", flush=True)
    print(f"File Size   : {out_file.stat().st_size:,d} bytes", flush=True)
    print(f"Total Time  : {total_time:.2f}s ({total_time / 60:.2f} min, {total_time / 3600:.2f} hours)", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
