"""Fast vectorized candidate pair generation (blocking) using Pandas token merges.
Implements Method A (Name token overlap) and Method C (Address token overlap).
Outputs candidate_pairs.tsv in the organizer's exact required format.
"""

import argparse
from collections import Counter
import gc
import io
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import psutil

# Ensure UTF-8 output and line buffering
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
else:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass


def get_repo_root() -> Path:
    """Find repository root across execution contexts."""
    candidate = Path(__file__).resolve().parent.parent.parent.parent
    if (candidate / "student_resource").exists():
        return candidate
    cwd = Path.cwd()
    if (cwd / "student_resource").exists():
        return cwd
    if (cwd.parent.parent / "student_resource").exists():
        return cwd.parent.parent
    return candidate


def print_ram_status(prefix: str = "") -> None:
    """Print current system memory status."""
    mem = psutil.virtual_memory()
    avail_gb = mem.available / (1024**3)
    used_gb = mem.used / (1024**3)
    print(f"[{prefix}] RAM: Avail={avail_gb:.2f} GB | Used={used_gb:.2f} GB ({mem.percent}%)", flush=True)


def build_s23_token_index(
    df: pd.DataFrame,
    col_name: str,
    max_df_pct: float = 0.01,
    max_entities_per_token: int = 1000,
) -> tuple[pd.DataFrame, set]:
    """Explode S2/S3 text column into a vectorized (token -> entity_id) lookup table.
    Filters common tokens and caps posting lists to prevent Cartesian explosion during merges.
    """
    t0 = time.time()
    n_total = len(df)
    sub = df[["entity_id", col_name]].copy()
    sub[col_name] = sub[col_name].fillna("")
    sub["token"] = sub[col_name].str.split()
    exploded = sub.explode("token")[["entity_id", "token"]].dropna()
    exploded = exploded[exploded["token"] != ""]
    del sub

    # Filter common tokens
    max_df = max(10, int(max_df_pct * n_total))
    counts = exploded["token"].value_counts()
    common_tokens = set(counts[counts > max_df].index)

    if common_tokens:
        exploded = exploded[~exploded["token"].isin(common_tokens)]

    # Cap posting list per token to avoid combinatorial explosion
    if max_entities_per_token and len(exploded) > 0:
        exploded = exploded.groupby("token").head(max_entities_per_token)

    exploded = exploded.reset_index(drop=True)
    return exploded, common_tokens


def process_country_vectorized(
    country_name: str,
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    out_file_handle: io.TextIOBase,
    s1_batch_size: int = 2500,
    max_cands_per_source: int = 50,
) -> tuple[int, int, int]:
    """Process candidate blocking for one country using vectorized pandas operations."""
    t0 = time.time()
    print(f"\n{'='*70}", flush=True)
    print(f"--> Processing Country Group: {country_name}", flush=True)
    print_ram_status(f"{country_name} Start")

    # 1. Load S2 and S3 for this country
    print(f"    Loading Source 2 & 3 for country={country_name}...", flush=True)
    t_load = time.time()
    cols = ["entity_id", "name_clean", "address_clean"]
    s2 = pd.read_parquet(s2_path, columns=cols, filters=[("country", "==", country_name)])
    s3 = pd.read_parquet(s3_path, columns=cols, filters=[("country", "==", country_name)])
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    n_s23 = len(s23)
    print(f"    Loaded S2+S3: {n_s23:,d} entities in {time.time() - t_load:.2f}s", flush=True)

    # 2. Build Method A (Name) and Method C (Address) exploded token tables for S2/S3
    print(f"    Building Method A (Name) token index...", flush=True)
    t_idx = time.time()
    # For large datasets, cap max_df at 0.5% and max_entities_per_token at 1000
    max_df_pct_a = 0.005 if n_s23 > 100000 else 0.01
    s23_a, common_a = build_s23_token_index(
        s23, "name_clean", max_df_pct=max_df_pct_a, max_entities_per_token=1000
    )
    print(
        f"    Method A index: {len(s23_a):,d} token rows (skipped {len(common_a):,d} common tokens) in {time.time() - t_idx:.2f}s",
        flush=True,
    )

    print(f"    Building Method C (Address) token index...", flush=True)
    t_idx_c = time.time()
    max_df_pct_c = 0.005 if n_s23 > 100000 else 0.01
    s23_c, common_c = build_s23_token_index(
        s23, "address_clean", max_df_pct=max_df_pct_c, max_entities_per_token=1000
    )
    print(
        f"    Method C index: {len(s23_c):,d} token rows (skipped {len(common_c):,d} common tokens) in {time.time() - t_idx_c:.2f}s",
        flush=True,
    )
    del s23
    gc.collect()
    print_ram_status(f"{country_name} Indices Ready")

    # 3. Load S1 for this country
    print(f"    Loading Source 1 for country={country_name}...", flush=True)
    s1 = pd.read_parquet(s1_path, columns=cols, filters=[("country", "==", country_name)])
    n_s1 = len(s1)
    print(
        f"    Loaded S1: {n_s1:,d} entities. Processing in batches of {s1_batch_size:,d}...",
        flush=True,
    )

    num_batches = (n_s1 + s1_batch_size - 1) // s1_batch_size
    total_pairs = 0
    zero_cands = 0
    processed_count = 0
    t_process_start = time.time()

    for batch_idx in range(num_batches):
        b_start = batch_idx * s1_batch_size
        b_end = min(b_start + s1_batch_size, n_s1)
        s1_batch = s1.iloc[b_start:b_end].copy()
        b_len = len(s1_batch)

        # --- Method A: Name token join ---
        s1_batch_a = s1_batch[["entity_id", "name_clean"]].copy()
        s1_batch_a["name_clean"] = s1_batch_a["name_clean"].fillna("")
        s1_batch_a["token"] = s1_batch_a["name_clean"].str.split()
        s1_toks_a = s1_batch_a.explode("token")[["entity_id", "token"]].dropna()
        s1_toks_a = s1_toks_a[s1_toks_a["token"] != ""]
        if common_a:
            s1_toks_a = s1_toks_a[~s1_toks_a["token"].isin(common_a)]

        m_a = s1_toks_a.merge(s23_a, on="token", suffixes=("_s1", "_s23"))
        del s1_batch_a, s1_toks_a

        if len(m_a) > 0:
            counts_a = (
                m_a.groupby(["entity_id_s1", "entity_id_s23"])
                .size()
                .reset_index(name="shared_tokens")
            )
            counts_a = counts_a.sort_values(["entity_id_s1", "shared_tokens"], ascending=[True, False])
        else:
            counts_a = pd.DataFrame(columns=["entity_id_s1", "entity_id_s23", "shared_tokens"])
        del m_a

        # --- Method C: Address token join (min 3 shared tokens) ---
        s1_batch_c = s1_batch[["entity_id", "address_clean"]].copy()
        s1_batch_c["address_clean"] = s1_batch_c["address_clean"].fillna("")
        s1_batch_c["token"] = s1_batch_c["address_clean"].str.split()
        s1_toks_c = s1_batch_c.explode("token")[["entity_id", "token"]].dropna()
        s1_toks_c = s1_toks_c[s1_toks_c["token"] != ""]
        if common_c:
            s1_toks_c = s1_toks_c[~s1_toks_c["token"].isin(common_c)]

        m_c = s1_toks_c.merge(s23_c, on="token", suffixes=("_s1", "_s23"))
        del s1_batch_c, s1_toks_c

        if len(m_c) > 0:
            counts_c = (
                m_c.groupby(["entity_id_s1", "entity_id_s23"])
                .size()
                .reset_index(name="shared_tokens")
            )
            counts_c = counts_c[counts_c["shared_tokens"] >= 3]
            counts_c = counts_c.sort_values(["entity_id_s1", "shared_tokens"], ascending=[True, False])
        else:
            counts_c = pd.DataFrame(columns=["entity_id_s1", "entity_id_s23", "shared_tokens"])
        del m_c

        # --- Combine A and C candidates per S1 ---
        # Concat candidates preserving highest scores
        all_counts = pd.concat([counts_a, counts_c], ignore_index=True)
        del counts_a, counts_c

        if len(all_counts) > 0:
            all_counts = all_counts.drop_duplicates(subset=["entity_id_s1", "entity_id_s23"])
            # Filter and cap at max_cands_per_source for S2 and S3
            s2_cands = (
                all_counts[all_counts["entity_id_s23"].str.startswith("S2-")]
                .groupby("entity_id_s1")
                .head(max_cands_per_source)
            )
            s3_cands = (
                all_counts[all_counts["entity_id_s23"].str.startswith("S3-")]
                .groupby("entity_id_s1")
                .head(max_cands_per_source)
            )
            top_pairs = pd.concat([s2_cands, s3_cands], ignore_index=True)
            del all_counts, s2_cands, s3_cands

            # Group candidates into comma-separated strings by entity_id_s1
            cand_dict = (
                top_pairs.groupby("entity_id_s1")["entity_id_s23"]
                .apply(lambda ids: ",".join(ids))
                .to_dict()
            )
            del top_pairs
        else:
            cand_dict = {}

        # Write lines in exact S1 order
        lines = []
        batch_pairs = 0
        batch_zero = 0
        for s1_id in s1_batch["entity_id"]:
            cands_str = cand_dict.get(s1_id, "")
            lines.append(f"{s1_id}\t{cands_str}\n")
            if cands_str:
                batch_pairs += cands_str.count(",") + 1
            else:
                batch_zero += 1

        out_file_handle.writelines(lines)
        out_file_handle.flush()
        del lines, cand_dict, s1_batch

        total_pairs += batch_pairs
        zero_cands += batch_zero
        processed_count += b_len

        elapsed = time.time() - t_process_start
        pct = (processed_count / n_s1) * 100
        qps = processed_count / elapsed if elapsed > 0 else 0
        remaining = n_s1 - processed_count
        eta_m = (remaining / qps) / 60 if qps > 0 else 0
        mem = psutil.virtual_memory()

        if (batch_idx + 1) <= 10 or (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
            print(
                f"    Batch {batch_idx + 1:4d}/{num_batches} | Done: {processed_count:9,d}/{n_s1:,d} ({pct:5.1f}%) | "
                f"Speed: {qps:6.1f} ent/s | Elapsed: {elapsed/60:5.1f}m | ETA: {eta_m:5.1f}m | RAM: {mem.available/(1024**3):5.2f}GB",
                flush=True,
            )


    del s1, s23_a, s23_c
    gc.collect()

    country_elapsed = time.time() - t0
    print(
        f"    Completed Country {country_name} in {country_elapsed:.2f}s ({country_elapsed/60:.2f} min)",
        flush=True,
    )
    print_ram_status(f"{country_name} End")
    return n_s1, total_pairs, zero_cands


def generate_candidate_pairs(
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    output_path: Path,
    target_country: str | None = None,
    s1_batch_size: int = 2500,
) -> None:
    """Generate candidate pairs across country groups and stream output to TSV."""
    t_start = time.time()
    print("=" * 70, flush=True)
    print("=== VECTORIZED CANDIDATE GENERATION PIPELINE ===", flush=True)
    print("=" * 70, flush=True)
    print(f"Input Sources:", flush=True)
    print(f"  Source 1: {s1_path}", flush=True)
    print(f"  Source 2: {s2_path}", flush=True)
    print(f"  Source 3: {s3_path}", flush=True)
    print(f"Output TSV: {output_path}", flush=True)
    print_ram_status("Initial")

    # Discover distinct countries
    print("\nScanning country metadata...", flush=True)
    s1_countries = set(pd.read_parquet(s1_path, columns=["country"])["country"].unique())
    s2_countries = set(pd.read_parquet(s2_path, columns=["country"])["country"].unique())
    s3_countries = set(pd.read_parquet(s3_path, columns=["country"])["country"].unique())
    all_countries = sorted(list(s1_countries | s2_countries | s3_countries))
    print(f"Detected {len(all_countries)} country group(s): {', '.join(all_countries)}", flush=True)

    if target_country:
        if target_country not in all_countries:
            raise ValueError(f"Requested country '{target_country}' not found in dataset! Available: {all_countries}")
        countries = [target_country]
        print(f"Filtered execution to single country: {target_country}", flush=True)
    else:
        # Prioritize India, then US, then any remaining countries (e.g. France)
        priority = {"India": 0, "US": 1}
        countries = sorted(all_countries, key=lambda c: priority.get(c, 99))


    output_path.parent.mkdir(parents=True, exist_ok=True)

    grand_s1 = 0
    grand_pairs = 0
    grand_zero = 0

    # Write fresh header
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")
        out_f.flush()

        for country in countries:
            n_s1, pairs, zeros = process_country_vectorized(
                country_name=country,
                s1_path=s1_path,
                s2_path=s2_path,
                s3_path=s3_path,
                out_file_handle=out_f,
                s1_batch_size=s1_batch_size,
            )
            grand_s1 += n_s1
            grand_pairs += pairs
            grand_zero += zeros

    # Final summary statistics
    avg_cands = (grand_pairs / grand_s1) if grand_s1 > 0 else 0.0
    pct_zero = (grand_zero / grand_s1 * 100) if grand_s1 > 0 else 0.0
    total_time = time.time() - t_start

    print("\n" + "=" * 70, flush=True)
    print("=== CANDIDATE GENERATION SUMMARY STATS ===", flush=True)
    print("=" * 70, flush=True)
    print(f"Total S1 entities processed   : {grand_s1:,d}", flush=True)
    print(f"Total candidate pairs generated: {grand_pairs:,d}", flush=True)
    print(f"Average candidates per S1     : {avg_cands:.2f}", flush=True)
    print(f"S1 entities with 0 candidates : {grand_zero:,d} ({pct_zero:.2f}%)", flush=True)
    print(f"Total time elapsed            : {total_time:.2f}s ({total_time / 60:.2f} min)", flush=True)
    print_ram_status("Final")
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate candidate pairs via fast vectorized blocking.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run candidate generation on sample fixture (data/processed/sample/).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run candidate generation on full test set (data/processed/test_source*.parquet).",
    )
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="Process only a specific country group (e.g. India or US).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2500,
        help="Number of S1 entities per processing batch (default: 2500).",
    )
    args = parser.parse_args()

    repo_root = get_repo_root()

    if args.sample:
        print("=== CANDIDATE GENERATION: SAMPLE MODE ===", flush=True)
        s1_file = repo_root / "data" / "processed" / "sample" / "sample_source1.parquet"
        s2_file = repo_root / "data" / "processed" / "sample" / "sample_source2.parquet"
        s3_file = repo_root / "data" / "processed" / "sample" / "sample_source3.parquet"
        out_file = repo_root / "output" / "sample" / "candidate_pairs.tsv"
    elif args.test:
        print("=== CANDIDATE GENERATION: TEST SET MODE ===", flush=True)
        s1_file = repo_root / "data" / "processed" / "test_source1.parquet"
        s2_file = repo_root / "data" / "processed" / "test_source2.parquet"
        s3_file = repo_root / "data" / "processed" / "test_source3.parquet"
        out_file = repo_root / "output" / "test_candidate_pairs.tsv"
    else:
        print("=== CANDIDATE GENERATION: TRAIN SET MODE ===", flush=True)
        s1_file = repo_root / "data" / "processed" / "train_source1.parquet"
        s2_file = repo_root / "data" / "processed" / "train_source2.parquet"
        s3_file = repo_root / "data" / "processed" / "train_source3.parquet"
        out_file = repo_root / "output" / "candidate_pairs.tsv"

    generate_candidate_pairs(
        s1_path=s1_file,
        s2_path=s2_file,
        s3_path=s3_file,
        output_path=out_file,
        target_country=args.country,
        s1_batch_size=args.batch_size,
    )
