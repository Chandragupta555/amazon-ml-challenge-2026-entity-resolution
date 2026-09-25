"""
STEP 1 - Look at the data before building anything.
Run from the student_resource/ folder:   python3 step1_eda.py
Then paste the full printed output back to Claude.
"""
import pandas as pd
from collections import Counter

# NOTE: files are TAB separated. Forgetting sep="\t" gives one giant column.
# keep_default_na=False stops pandas turning empty strings / "NA" into NaN.
def load(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

s1 = load("dataset/train/train_source1.tsv")
s2 = load("dataset/train/train_source2.tsv")
s3 = load("dataset/train/train_source3.tsv")
gt = load("dataset/train/train_ground_truth.tsv")

print("=== SHAPES ===")
print("S1", s1.shape, "| S2", s2.shape, "| S3", s3.shape, "| GT", gt.shape)

print("\n=== COUNTRY COUNTS ===")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    print(name, dict(Counter(df["country"])))

print("\n=== SAMPLE ROWS (5 each) ===")
pd.set_option("display.width", 250, "display.max_colwidth", 70)
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    print("\n--", name)
    print(df.sample(5, random_state=0).to_string(index=False))

print("\n=== EMPTY FIELDS (count of blank name / address) ===")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    print(name, "blank name:", (df["business_name"].str.strip() == "").sum(),
          "| blank address:", (df["business_address"].str.strip() == "").sum())

# ---- ground truth structure ----
gt["ids"] = gt["matched_entity_ids"].apply(lambda x: [i for i in x.split(",") if i])
gt["n_total"] = gt["ids"].apply(len)
gt["n_s2"] = gt["ids"].apply(lambda l: sum(i.startswith("S2-") for i in l))
gt["n_s3"] = gt["ids"].apply(lambda l: sum(i.startswith("S3-") for i in l))

print("\n=== GROUND TRUTH ===")
print("S1 entities:", len(gt))
print("Singleton rate (no matches): %.3f" % (gt["n_total"] == 0).mean())
print("Matches per S1 entity distribution:", dict(sorted(Counter(gt["n_total"]).items())))
print("Avg S2 matches per S1: %.2f | Avg S3 matches per S1: %.2f" % (gt["n_s2"].mean(), gt["n_s3"].mean()))

# KEY QUESTION: does one S2/S3 record ever belong to more than one S1 entity?
owner = Counter(i for l in gt["ids"] for i in l)
multi = [i for i, c in owner.items() if c > 1]
print("Records claimed by >1 S1 entity:", len(multi), "(0 means we can force one-to-one assignment)")

# How many S2/S3 records belong to NO S1 entity at all (pure noise records)?
all_s2 = set(s2["entity_id"]); all_s3 = set(s3["entity_id"])
print("S2 records unmatched to any S1: %d of %d" % (len(all_s2 - set(owner)), len(all_s2)))
print("S3 records unmatched to any S1: %d of %d" % (len(all_s3 - set(owner)), len(all_s3)))

# ---- look at real matched examples side by side ----
print("\n=== 8 MATCHED EXAMPLES (this is what the noise looks like) ===")
idx = {}
for df in (s1, s2, s3):
    for r in df.itertuples(index=False):
        idx[r.entity_id] = (r.business_name, r.business_address, r.country)
shown = 0
for r in gt[gt["n_total"] >= 2].sample(8, random_state=1).itertuples():
    print("\nS1:", r.source1_entity_id, idx[r.source1_entity_id])
    for i in r.ids:
        print("   ->", i, idx[i])
