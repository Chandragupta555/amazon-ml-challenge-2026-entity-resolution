"""Builds a candidate pair set (source1_entity_id -> candidate_entity_ids from Source 2/3) using blocking/candidate generation. Outputs output/candidate_pairs.tsv in the organizer's required format."""

import argparse
from collections import Counter, defaultdict
import io
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure standard output handles UTF-8 (e.g. non-Latin scripts on Windows terminals)
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


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


def run_blocking_for_country(
    c_s1: pd.DataFrame,
    c_s2: pd.DataFrame,
    c_s3: pd.DataFrame,
    country_name: str,
) -> dict[str, list[str]]:
    """Generate candidate pairs for all S1 entities within a single country group.

    Combines:
    - Method A: Token overlap inverted index on name_clean (excluding words in >1% of names)
    - Method B: Character 3-5 gram TF-IDF sparse matrix cosine similarity on name_clean (top 30)
    - Method C: Token overlap inverted index on address_clean (excluding words in >1% of addresses, min 3 tokens, top 30)
    Capped at 50 candidates from S2 and 50 candidates from S3 per S1 entity.
    """
    t0 = time.time()
    n_s1 = len(c_s1)
    n_s2 = len(c_s2)
    n_s3 = len(c_s3)
    n_s23 = n_s2 + n_s3

    print(f"\n--> Processing country: {country_name}")
    print(f"    Entities: S1={n_s1:,d} | S2={n_s2:,d} | S3={n_s3:,d} (S2+S3={n_s23:,d})")

    if n_s1 == 0 or n_s23 == 0:
        print(f"    Skipping: no entities in S1 or S2/S3 for country {country_name}")
        return {eid: [] for eid in c_s1["entity_id"]}

    c_s23 = pd.concat([c_s2, c_s3], ignore_index=True)
    s23_ids = c_s23["entity_id"].values
    s1_ids = c_s1["entity_id"].values

    # =========================================================================
    # METHOD A: Token Overlap Inverted Index (name_clean)
    # =========================================================================
    t_a = time.time()
    doc_freq = Counter()
    s23_token_sets = []
    for name in c_s23["name_clean"]:
        toks = set(name.split()) if name else set()
        doc_freq.update(toks)
        s23_token_sets.append(toks)

    # Skip tokens appearing in > 1% of all names in this country group
    max_df = max(10, int(0.01 * n_s23))
    common_tokens = {w for w, count in doc_freq.items() if count > max_df}

    inverted_index = defaultdict(list)
    for eid, toks in zip(s23_ids, s23_token_sets):
        for t in toks:
            if t not in common_tokens:
                inverted_index[t].append(eid)

    cand_a_map = {}
    for s1_id, name in zip(s1_ids, c_s1["name_clean"]):
        toks = (set(name.split()) - common_tokens) if name else set()
        counts = Counter()
        for t in toks:
            for eid in inverted_index.get(t, []):
                counts[eid] += 1
        cand_a_map[s1_id] = [eid for eid, _ in counts.most_common()]

    print(
        f"    Method A (Token Inverted Index): {len(common_tokens):,d} common tokens skipped (>1%), "
        f"queried in {time.time() - t_a:.2f}s"
    )

    # =========================================================================
    # METHOD B: Character n-gram TF-IDF Cosine Similarity (name_clean)
    # =========================================================================
    t_b = time.time()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
    )
    m_s23 = vectorizer.fit_transform(c_s23["name_clean"].fillna(""))
    m_s1 = vectorizer.transform(c_s1["name_clean"].fillna(""))

    cand_b_map = {}
    top_k = 30
    batch_size = 250

    for start in range(0, n_s1, batch_size):
        sub_s1 = m_s1[start : start + batch_size]
        # Sparse matrix multiplication: (batch, D) @ (D, N_s23) -> CSR matrix (batch, N_s23)
        sim = sub_s1.dot(m_s23.T)

        for row_idx in range(sim.shape[0]):
            s1_id = s1_ids[start + row_idx]
            row_slice = sim[row_idx]
            data = row_slice.data
            indices = row_slice.indices

            if len(data) == 0:
                cand_b_map[s1_id] = []
                continue

            if len(data) > top_k:
                # Top k highest cosine similarity
                best_local = np.argpartition(data, -top_k)[-top_k:]
                best_local = best_local[np.argsort(-data[best_local])]
            else:
                best_local = np.argsort(-data)

            best_indices = indices[best_local]
            cand_b_map[s1_id] = [s23_ids[idx] for idx in best_indices]

    print(
        f"    Method B (Char 3-5 gram TF-IDF): vocab={len(vectorizer.vocabulary_):,d}, "
        f"queried in {time.time() - t_b:.2f}s"
    )

    # =========================================================================
    # METHOD C: Address Token Overlap Inverted Index (address_clean)
    # =========================================================================
    t_c = time.time()
    doc_freq_c = Counter()
    s23_addr_token_sets = []
    for addr in c_s23["address_clean"]:
        toks = set(addr.split()) if addr else set()
        doc_freq_c.update(toks)
        s23_addr_token_sets.append(toks)

    # Skip tokens appearing in > 1% of all addresses in this country group
    max_df_c = max(10, int(0.01 * n_s23))
    common_tokens_c = {w for w, count in doc_freq_c.items() if count > max_df_c}

    inverted_index_c = defaultdict(list)
    for eid, toks in zip(s23_ids, s23_addr_token_sets):
        for t in toks:
            if t not in common_tokens_c:
                inverted_index_c[t].append(eid)

    min_shared_addr_tokens = 3
    cand_c_map = {}
    for s1_id, addr in zip(s1_ids, c_s1["address_clean"]):
        toks = (set(addr.split()) - common_tokens_c) if addr else set()
        counts = Counter()
        for t in toks:
            for eid in inverted_index_c.get(t, []):
                counts[eid] += 1
        cand_c_map[s1_id] = [eid for eid, count in counts.most_common(30) if count >= min_shared_addr_tokens]

    print(
        f"    Method C (Address Token Inverted Index): {len(common_tokens_c):,d} common tokens skipped (>1%), "
        f"min_tokens={min_shared_addr_tokens}, queried in {time.time() - t_c:.2f}s"
    )

    # =========================================================================
    # COMBINE & CAP: Union of B, C, A, capped at 50 per source (S2 and S3)
    # =========================================================================
    country_candidates = {}
    for s1_id in s1_ids:
        b_cands = cand_b_map.get(s1_id, [])
        c_cands = cand_c_map.get(s1_id, [])
        a_cands = cand_a_map.get(s1_id, [])

        seen_s2 = set()
        seen_s3 = set()
        cands_s2 = []
        cands_s3 = []

        # Union order: Method B (top 30 name tfidf), Method C (top 30 address matches >= 3 tokens),
        # then Method A (name token overlap matches)
        for source_list in [b_cands, c_cands, a_cands]:
            for eid in source_list:
                if eid.startswith("S2-"):
                    if eid not in seen_s2 and len(cands_s2) < 50:
                        seen_s2.add(eid)
                        cands_s2.append(eid)
                elif eid.startswith("S3-"):
                    if eid not in seen_s3 and len(cands_s3) < 50:
                        seen_s3.add(eid)
                        cands_s3.append(eid)

        country_candidates[s1_id] = cands_s2 + cands_s3

    print(f"    Completed country {country_name} in {time.time() - t0:.2f}s")
    return country_candidates


def generate_candidate_pairs(
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    output_path: Path,
) -> None:
    """Generate candidate pairs across all country groups and write output TSV."""
    t_start = time.time()
    print(f"Loading input parquet files:")
    print(f"  Source 1: {s1_path}")
    print(f"  Source 2: {s2_path}")
    print(f"  Source 3: {s3_path}")

    s1 = pd.read_parquet(s1_path)
    s2 = pd.read_parquet(s2_path)
    s3 = pd.read_parquet(s3_path)

    # Collect distinct countries across all sources
    countries = sorted(
        list(set(s1["country"].unique()) | set(s2["country"].unique()) | set(s3["country"].unique()))
    )
    print(f"\nDetected {len(countries)} country group(s): {', '.join(countries)}")

    all_candidates: dict[str, list[str]] = {}

    for country in countries:
        c_s1 = s1[s1["country"] == country].reset_index(drop=True)
        c_s2 = s2[s2["country"] == country].reset_index(drop=True)
        c_s3 = s3[s3["country"] == country].reset_index(drop=True)

        c_cands = run_blocking_for_country(c_s1, c_s2, c_s3, country_name=country)
        all_candidates.update(c_cands)

    # Write output/candidate_pairs.tsv in organizer's exact required format
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting candidate pairs to: {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1["entity_id"]:
            cands = all_candidates.get(s1_id, [])
            cands_str = ",".join(cands)
            f.write(f"{s1_id}\t{cands_str}\n")

    # Compute and print statistics
    n_s1 = len(s1)
    total_pairs = sum(len(all_candidates.get(s1_id, [])) for s1_id in s1["entity_id"])
    zero_cands = sum(1 for s1_id in s1["entity_id"] if len(all_candidates.get(s1_id, [])) == 0)
    avg_cands = (total_pairs / n_s1) if n_s1 > 0 else 0.0
    pct_zero = (zero_cands / n_s1 * 100) if n_s1 > 0 else 0.0

    print("\n" + "=" * 65)
    print("=== CANDIDATE GENERATION SUMMARY STATS ===")
    print("=" * 65)
    print(f"Total S1 entities processed   : {n_s1:,d}")
    print(f"Total candidate pairs generated: {total_pairs:,d}")
    print(f"Average candidates per S1     : {avg_cands:.2f}")
    print(f"S1 entities with 0 candidates : {zero_cands:,d} ({pct_zero:.2f}%)")
    print(f"Total time elapsed            : {time.time() - t_start:.2f}s")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate candidate pairs via blocking.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run candidate generation on sample fixture (data/processed/sample/).",
    )
    args = parser.parse_args()

    repo_root = get_repo_root()

    if args.sample:
        print("=== CANDIDATE GENERATION: SAMPLE MODE ===")
        s1_file = repo_root / "data" / "processed" / "sample" / "sample_source1.parquet"
        s2_file = repo_root / "data" / "processed" / "sample" / "sample_source2.parquet"
        s3_file = repo_root / "data" / "processed" / "sample" / "sample_source3.parquet"
        out_file = repo_root / "output" / "sample" / "candidate_pairs.tsv"
    else:
        print("=== CANDIDATE GENERATION: FULL DATASET MODE (TRAIN) ===")
        s1_file = repo_root / "data" / "processed" / "train_source1.parquet"
        s2_file = repo_root / "data" / "processed" / "train_source2.parquet"
        s3_file = repo_root / "data" / "processed" / "train_source3.parquet"
        out_file = repo_root / "output" / "candidate_pairs.tsv"

    generate_candidate_pairs(s1_file, s2_file, s3_file, out_file)
