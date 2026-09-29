import time
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

def main():
    score_file = 'data/processed/test_pair_scores.parquet'
    cand_file = 'output/test_candidate_pairs.tsv'
    source1_file = 'data/processed/test_source1.parquet'

    print("=== STARTING SANITY CHECKS ===")
    t0 = time.time()

    # 1. Read test_source1 to get all 1,732,544 test source1 entity IDs
    print("\n[Step 1] Loading ground truth test source1 entities...")
    t_s1 = pq.read_table(source1_file, columns=['entity_id'])
    all_s1_entities = set(t_s1['entity_id'].to_pylist())
    total_test_entities = len(all_s1_entities)
    print(f"Total test source1 entities: {total_test_entities:,}")

    # 2. Read test_candidate_pairs.tsv to get:
    #    - entities with >= 1 candidate
    #    - entities with 0 candidates (empty candidate string)
    print("\n[Step 2] Scanning output/test_candidate_pairs.tsv...")
    cand_has_pairs = set()
    cand_zero_pairs = set()
    total_cand_lines = 0

    with open(cand_file, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            total_cand_lines += 1
            parts = line.rstrip('\r\n').split('\t')
            s1_id = parts[0]
            if len(parts) > 1 and parts[1].strip():
                cand_has_pairs.add(s1_id)
            else:
                cand_zero_pairs.add(s1_id)

    print(f"Total candidate TSV lines: {total_cand_lines:,}")
    print(f"Entities with candidates in TSV: {len(cand_has_pairs):,}")
    print(f"Zero-candidate entities in TSV: {len(cand_zero_pairs):,}")

    # 3. Read test_pair_scores.parquet
    print("\n[Step 3] Scanning data/processed/test_pair_scores.parquet...")
    pf = pq.ParquetFile(score_file)
    total_rows = pf.metadata.num_rows
    num_rg = pf.num_row_groups
    print(f"Total rows in metadata: {total_rows:,} across {num_rg} row groups")

    all_scores = np.empty(total_rows, dtype=np.float32)
    s1_in_scores = set()
    total_nulls = 0
    offset = 0

    t_read_start = time.time()
    for rg_idx in range(num_rg):
        rg = pf.read_row_group(rg_idx, columns=['source1_entity_id', 'score'])
        
        # Check nulls
        rg_nulls = rg['score'].null_count
        total_nulls += rg_nulls
        
        # Unique source1 in this RG
        rg_s1_unique = rg['source1_entity_id'].unique().to_pylist()
        s1_in_scores.update(rg_s1_unique)

        # Scores array
        n_rows = len(rg)
        all_scores[offset : offset + n_rows] = rg['score'].to_numpy()
        offset += n_rows

        if (rg_idx + 1) % 32 == 0 or (rg_idx + 1) == num_rg:
            elapsed = time.time() - t_read_start
            print(f"  Processed row group {rg_idx + 1}/{num_rg} ({offset:,} rows) [{elapsed:.1f}s]")

    print(f"\nFinished reading scores in {time.time() - t_read_start:.1f}s")
    print(f"Total rows read: {offset:,}")
    print(f"Total nulls in score column: {total_nulls:,}")

    # 4. Check score range and NaN
    nan_count = int(np.isnan(all_scores).sum())
    min_val = float(np.min(all_scores))
    max_val = float(np.max(all_scores))
    under_0 = int(np.count_nonzero(all_scores < 0.0))
    over_1 = int(np.count_nonzero(all_scores > 1.0))
    within_bounds = (under_0 == 0 and over_1 == 0 and nan_count == 0)

    # 5. Distribution stats
    mean_val = float(np.mean(all_scores))
    # Median via np.partition (in-place selection to minimize memory overhead)
    mid = len(all_scores) // 2
    if len(all_scores) % 2 == 1:
        np.partition(all_scores, mid)
        median_val = float(all_scores[mid])
    else:
        # For huge array, mid element is sufficient or exact average of mid-1 and mid
        np.partition(all_scores, (mid - 1, mid))
        median_val = float((all_scores[mid - 1] + all_scores[mid]) / 2.0)

    ge_085_count = int(np.count_nonzero(all_scores >= 0.85))
    ge_085_pct = (ge_085_count / len(all_scores)) * 100.0

    # 6. Entity set comparisons
    num_s1_in_scores = len(s1_in_scores)
    missing_from_scores = all_s1_entities - s1_in_scores
    num_missing_from_scores = len(missing_from_scores)

    # Are missing_from_scores exactly the zero-candidate entities?
    is_exact_match = (missing_from_scores == cand_zero_pairs)
    # Are scores entity set exactly the ones with candidates?
    is_cand_exact_match = (s1_in_scores == cand_has_pairs)

    print("\n" + "="*50)
    print("=== FINAL SANITY CHECK RESULTS ===")
    print("="*50)
    print(f"1. Null Check: {total_nulls} nulls across all {total_rows:,} rows. Zero nulls: {total_nulls == 0}")
    print(f"   NaN count: {nan_count}")
    print(f"2. Range Check: [0.0, 1.0] -> Min: {min_val:.8f}, Max: {max_val:.8f}")
    print(f"   Values < 0.0: {under_0}, Values > 1.0: {over_1}")
    print(f"   All strictly within [0.0, 1.0]: {within_bounds}")
    print(f"3. Distribution Stats:")
    print(f"   - Min: {min_val:.8f}")
    print(f"   - Max: {max_val:.8f}")
    print(f"   - Mean: {mean_val:.8f}")
    print(f"   - Median: {median_val:.8f}")
    print(f"   - Count >= 0.85: {ge_085_count:,} ({ge_085_pct:.4f}% of all pairs)")
    print(f"4. Missing Entity ID Verification:")
    print(f"   - Total test source1 entities: {total_test_entities:,}")
    print(f"   - Unique source1 entities in scores: {num_s1_in_scores:,}")
    print(f"   - Missing from scores: {num_missing_from_scores:,}")
    print(f"   - Zero-candidate entities in test_candidate_pairs.tsv: {len(cand_zero_pairs):,}")
    print(f"   - Missing from scores == zero-candidate entities in TSV: {is_exact_match}")
    print(f"   - Present in scores == entities with candidates in TSV: {is_cand_exact_match}")
    print(f"Total time elapsed: {time.time() - t0:.1f}s")

if __name__ == '__main__':
    main()
