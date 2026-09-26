"""Cleans and normalizes business_name and business_address fields for all three sources (lowercase, strip accents, expand abbreviations, split legal suffixes). Reads from ../../../student_resource/dataset/, outputs cleaned parquet files."""

import argparse
import gc
import io
from pathlib import Path
import random
import re
import sys
import time
import unicodedata
import pandas as pd
import unidecode

# Ensure standard output handles UTF-8 (e.g. non-Latin scripts on Windows terminals)
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Legal suffixes to detect and strip from the end of business names
LEGAL_SUFFIXES = {
    "llc", "inc", "incorporated", "corp", "corporation",
    "ltd", "limited", "pvt", "private", "plc", "pllc",
    "llp", "lp", "co", "company", "gmbh", "sarl", "sas",
}

# Common address abbreviations to expand
ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
}

# Regex to detect if a business name is just a domain or URL
DOMAIN_REGEX = re.compile(
    r"^(https?://)?(www\.)?([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}(/.*)?$",
    re.IGNORECASE,
)


def _has_non_latin_script(text: str) -> bool:
    """Check if the text contains non-Latin script characters (e.g. Devanagari, Odia)."""
    for ch in text:
        if ord(ch) >= 128:
            cat = unicodedata.category(ch)
            if cat.startswith(("L", "M")):
                name = unicodedata.name(ch, "")
                if "LATIN" not in name:
                    return True
    return False


def normalize_text(s: str) -> str:
    """Normalize input text.

    - Lowercase
    - Transliterate accented Latin characters to base ASCII using unidecode
    - Preserve non-Latin scripts (Devanagari, Odia, Bengali, etc.) as-is
    - Replace '&' with ' and '
    - Remove punctuation except keep alphanumerics, spaces, and non-Latin script characters
    - Collapse multiple spaces to one, strip leading/trailing whitespace
    """
    if not s or not isinstance(s, str):
        return ""

    # 1. Lowercase
    s = s.lower()

    # 2. Transliterate accented Latin characters to base ASCII (preserve non-Latin scripts)
    s = unicodedata.normalize("NFC", s)
    transliterated = []
    for ch in s:
        if ord(ch) >= 128 and "LATIN" in unicodedata.name(ch, ""):
            transliterated.append(unidecode.unidecode(ch))
        else:
            transliterated.append(ch)
    s = "".join(transliterated)

    # 3. Replace '&' with ' and '
    s = s.replace("&", " and ")

    # 4. Collapse dotted single-letter acronyms (e.g. l.l.c. -> llc, l.p. -> lp)
    s = re.sub(r"\b([a-z])\.(?=[a-z]\.|\s|$)", r"\1", s)

    # 5. Remove apostrophes directly to keep word contiguous (e.g. orelee's -> orelees)
    s = re.sub(r"['’`]", "", s)

    # 6. Remove punctuation except keep alphanumerics, spaces, and non-Latin script characters
    cleaned = []
    for ch in s:
        if ch.isalnum() or ch.isspace() or unicodedata.category(ch).startswith("M"):
            cleaned.append(ch)
        elif ord(ch) >= 128 and "LATIN" not in unicodedata.name(ch, "") and not unicodedata.category(ch).startswith(("P", "S")):
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    s = "".join(cleaned)

    # 7. Collapse multiple spaces to one, strip leading/trailing whitespace
    return re.sub(r"\s+", " ", s).strip()


def normalize_business_name(name: str) -> dict:
    """Normalize a business name.

    Returns dict with:
      - clean_name: normalized business name with trailing legal suffix stripped
      - legal_suffix: detected trailing legal suffix (or '' if none)
      - name_is_domain: boolean indicating if name looks like just a domain/URL
    """
    if not name or not isinstance(name, str):
        return {
            "clean_name": "",
            "legal_suffix": "",
            "name_is_domain": False,
        }

    # Detect domain/URL before stripping punctuation
    name_is_domain = bool(DOMAIN_REGEX.match(name.strip()))

    # Apply base text normalization
    clean = normalize_text(name)

    # Detect and strip trailing legal suffixes
    tokens = clean.split()
    stripped_suffixes = []
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIXES:
        stripped_suffixes.insert(0, tokens.pop())

    clean_name = " ".join(tokens)
    legal_suffix = " ".join(stripped_suffixes)

    return {
        "clean_name": clean_name,
        "legal_suffix": legal_suffix,
        "name_is_domain": name_is_domain,
    }


# Street-type words to distinguish house numbers from postal codes
STREET_WORDS = {
    "avenue", "street", "road", "drive", "lane", "court", "way", "boulevard",
    "highway", "place", "terrace", "circle", "parkway", "trail", "square",
    "loop", "pike", "alley", "expressway", "turnpike", "route", "path",
    "crossing", "crescent", "run", "walk", "row",
    "ave", "st", "rd", "dr", "ln", "ct", "blvd", "hwy", "pl", "ter",
    "cir", "pkwy", "trl", "sq", "expwy", "tpke", "rt", "rte", "xing", "cres",
    "marg", "rasta", "gali",
}

# Unit and building prefixes that precede house/unit numbers
UNIT_WORDS = {
    "flat", "plot", "unit", "apt", "apartment", "suite", "ste", "room",
    "door", "ward", "sector", "sec", "block", "blk", "hn", "no", "shop",
}


def extract_postal_code(addr: str) -> str:
    """Extract a standalone 5-6 digit postal code from an address.

    Real postal codes in this dataset typically appear near the END of an address
    (e.g. after a state code or as the final token). House numbers (which usually
    appear at the start or are adjacent to street-type words) are filtered out.
    """
    if not addr or not isinstance(addr, str):
        return ""

    # Split into comma-separated parts
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    if not parts:
        return ""

    # 1. Check the LAST 1-2 tokens of the (unnormalized, but comma-split) address
    last_part = parts[-1]
    tokens = last_part.split()
    if not tokens:
        return ""

    # Check candidate tokens from the last 1-2 tokens of the last comma-part (right to left)
    candidate_tokens = list(reversed(tokens[-2:]))

    for token in candidate_tokens:
        # Match standalone 5-6 digit number (allowing edge punctuation like 92521. or 09133,)
        match = re.match(r"^\D*(\d{5,6})\D*$", token)
        if not match:
            continue
        digits = match.group(1)

        # 2. Do NOT match if immediately followed or preceded by a street-type word
        token_idx = tokens.index(token)

        before_words = [re.sub(r"[^a-zA-Z]", "", w).lower() for w in tokens[:token_idx]]
        after_words = [re.sub(r"[^a-zA-Z]", "", w).lower() for w in tokens[token_idx + 1:]]

        adjacent_words = (before_words[-2:] if before_words else []) + (after_words[:2] if after_words else [])
        if any(w in STREET_WORDS for w in adjacent_words if w):
            continue

        # If followed by another numeric token in the same part (e.g. '11851 20' -> house 11851 on road 20)
        remaining_after = tokens[token_idx + 1:]
        if any(re.match(r"^\d+$", re.sub(r"[^\d]", "", w)) for w in remaining_after if re.sub(r"[^\d]", "", w)):
            continue

        # If preceded by unit/flat/plot words in the same part (e.g. 'FLAT NO 00403')
        if any(w in UNIT_WORDS for w in before_words):
            continue

        return digits

    return ""


def normalize_address(addr: str) -> dict:
    """Normalize a business address.

    Returns dict with:
      - clean_address: normalized address with expanded abbreviations
      - postal_code: extracted 5-6 digit PIN/postal code (or '' if none)
      - has_non_latin_script: boolean indicating if original had non-Latin chars
    """
    if not addr or not isinstance(addr, str):
        return {
            "clean_address": "",
            "postal_code": "",
            "has_non_latin_script": False,
        }

    has_non_latin = _has_non_latin_script(addr)

    # Extract 5-6 digit postal / PIN code from unnormalized address
    postal_code = extract_postal_code(addr)

    # Base text normalization
    clean = normalize_text(addr)

    # Expand common abbreviations: rd->road, st->street, ave->avenue, blvd->boulevard
    for abbr, full in ADDRESS_ABBREVIATIONS.items():
        clean = re.sub(rf"\b{abbr}\b", full, clean)

    return {
        "clean_address": clean,
        "postal_code": postal_code,
        "has_non_latin_script": has_non_latin,
    }


def get_repo_root() -> Path:
    """Find repository root across execution contexts."""
    # normalize.py is at code/business_entity_resolution/src/normalize.py (4 levels deep)
    candidate = Path(__file__).resolve().parent.parent.parent.parent
    if (candidate / "student_resource").exists():
        return candidate
    cwd = Path.cwd()
    if (cwd / "student_resource").exists():
        return cwd
    if (cwd.parent.parent / "student_resource").exists():
        return cwd.parent.parent
    return candidate


def process_file(input_path: str, output_path: str) -> dict | None:
    """Process a single TSV file, apply normalization, and save as parquet.

    Parameters
    ----------
    input_path : str
        Path to input .tsv file.
    output_path : str
        Path to output .parquet file.

    Returns
    -------
    dict or None
        Summary statistics dictionary, or None if input file is missing.
    """
    in_p = Path(input_path)
    out_p = Path(output_path)

    if not in_p.exists():
        print(f"[ERROR] Input file not found: {in_p}. Skipping.")
        return None

    t0 = time.time()
    print(f"--> Processing {in_p.name}...")

    df = pd.read_csv(in_p, sep="\t", dtype=str, keep_default_na=False)
    n_rows = len(df)

    # 1. Normalize business names
    names_clean = []
    names_suffix = []
    names_domain = []
    for name in df["business_name"]:
        r = normalize_business_name(name)
        names_clean.append(r["clean_name"])
        names_suffix.append(r["legal_suffix"])
        names_domain.append(r["name_is_domain"])

    df["name_clean"] = names_clean
    df["name_legal_suffix"] = names_suffix
    df["name_is_domain"] = names_domain
    del names_clean, names_suffix, names_domain

    # 2. Normalize business addresses
    addrs_clean = []
    postal_codes = []
    non_latins = []
    for addr in df["business_address"]:
        r = normalize_address(addr)
        addrs_clean.append(r["clean_address"])
        postal_codes.append(r["postal_code"])
        non_latins.append(r["has_non_latin_script"])

    df["address_clean"] = addrs_clean
    df["postal_code"] = postal_codes
    df["has_non_latin_script"] = non_latins
    del addrs_clean, postal_codes, non_latins

    # 3. Ensure destination directory exists and write parquet
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_p, index=False)

    elapsed = time.time() - t0
    print(f"    Saved: {out_p.name} ({n_rows:,d} rows in {elapsed:.2f}s)")

    # Compute summary stats
    pct_domain = (df["name_is_domain"].sum() / n_rows * 100) if n_rows > 0 else 0.0
    pct_non_latin = (df["has_non_latin_script"].sum() / n_rows * 100) if n_rows > 0 else 0.0
    pct_postal = ((df["postal_code"] != "").sum() / n_rows * 100) if n_rows > 0 else 0.0

    stats = {
        "file": in_p.name,
        "output_path": str(out_p),
        "rows": n_rows,
        "pct_domain": pct_domain,
        "pct_non_latin": pct_non_latin,
        "pct_postal": pct_postal,
        "elapsed": elapsed,
    }

    # Free memory immediately (laptop memory constraint)
    del df
    gc.collect()

    return stats


def print_summary_table(summary_stats: list[dict]) -> None:
    """Print a clean formatted summary table of all processed files."""
    if not summary_stats:
        print("\nNo files were processed.")
        return

    header_file = "File"
    header_rows = "Rows"
    header_domain = "% Domain"
    header_non_latin = "% Non-Latin"
    header_postal = "% Postal Code"
    header_time = "Time (s)"

    col_widths = (28, 12, 12, 14, 16, 10)
    sep_line = "-" * (sum(col_widths) + 5)
    double_sep = "=" * (sum(col_widths) + 5)

    print(f"\n{double_sep}")
    print(
        f"{header_file:<{col_widths[0]}} "
        f"{header_rows:>{col_widths[1]}} "
        f"{header_domain:>{col_widths[2]}} "
        f"{header_non_latin:>{col_widths[3]}} "
        f"{header_postal:>{col_widths[4]}} "
        f"{header_time:>{col_widths[5]}}"
    )
    print(sep_line)

    total_rows = 0
    total_time = 0.0

    for st in summary_stats:
        total_rows += st["rows"]
        total_time += st["elapsed"]
        print(
            f"{st['file']:<{col_widths[0]}} "
            f"{st['rows']:>{col_widths[1]},d} "
            f"{st['pct_domain']:>{col_widths[2]-1}.2f}% "
            f"{st['pct_non_latin']:>{col_widths[3]-1}.2f}% "
            f"{st['pct_postal']:>{col_widths[4]-1}.2f}% "
            f"{st['elapsed']:>{col_widths[5]-1}.2f}s"
        )

    print(sep_line)
    print(
        f"{'Total':<{col_widths[0]}} "
        f"{total_rows:>{col_widths[1]},d} "
        f"{'':>{col_widths[2]}} "
        f"{'':>{col_widths[3]}} "
        f"{'':>{col_widths[4]}} "
        f"{total_time:>{col_widths[5]-1}.2f}s"
    )
    print(f"{double_sep}\n")


def print_sample_before_after(processed_files: list[str], n_samples: int = 10) -> None:
    """Load sample rows from processed parquet files and print before/after comparisons."""
    records = []
    for fp in processed_files:
        p = Path(fp)
        if p.exists():
            df_sample = pd.read_parquet(p)
            for row in df_sample.itertuples(index=False):
                records.append(row)

    if not records:
        return

    random.seed(42)
    chosen = random.sample(records, min(n_samples, len(records)))

    print(f"=== {len(chosen)} SAMPLE BEFORE / AFTER NORMALIZATION ROWS ===\n")
    for i, r in enumerate(chosen, 1):
        print(f"[{i:2d}] {r.entity_id} (Country: {r.country})")
        print(f"     Name (Before)   : {r.business_name}")
        print(f"     Name (Clean)    : {r.name_clean}")
        print(f"       -> Legal Suffix : {r.name_legal_suffix!r}")
        print(f"       -> Is Domain    : {r.name_is_domain}")
        print(f"     Address (Before): {r.business_address}")
        print(f"     Address (Clean) : {r.address_clean}")
        print(f"       -> Postal Code  : {r.postal_code!r}")
        print(f"       -> Non-Latin    : {r.has_non_latin_script}")
        print("-" * 75)


def run_pipeline(is_sample: bool = False) -> None:
    """Run the normalization pipeline over sample fixture or full dataset."""
    repo_root = get_repo_root()

    if is_sample:
        print("=== NORMALIZATION PIPELINE: SAMPLE FIXTURE MODE ===")
        input_dir = repo_root / "data" / "sample"
        output_dir = repo_root / "data" / "processed" / "sample"

        file_pairs = [
            (input_dir / "sample_source1.tsv", output_dir / "sample_source1.parquet"),
            (input_dir / "sample_source2.tsv", output_dir / "sample_source2.parquet"),
            (input_dir / "sample_source3.tsv", output_dir / "sample_source3.parquet"),
        ]
    else:
        print("=== NORMALIZATION PIPELINE: FULL DATASET MODE ===")
        dataset_dir = repo_root / "student_resource" / "dataset"
        output_dir = repo_root / "data" / "processed"

        file_pairs = [
            (dataset_dir / "train" / "train_source1.tsv", output_dir / "train_source1.parquet"),
            (dataset_dir / "train" / "train_source2.tsv", output_dir / "train_source2.parquet"),
            (dataset_dir / "train" / "train_source3.tsv", output_dir / "train_source3.parquet"),
            (dataset_dir / "test" / "test_source1.tsv", output_dir / "test_source1.parquet"),
            (dataset_dir / "test" / "test_source2.tsv", output_dir / "test_source2.parquet"),
            (dataset_dir / "test" / "test_source3.tsv", output_dir / "test_source3.parquet"),
        ]

    summary_stats = []
    processed_paths = []

    for in_file, out_file in file_pairs:
        st = process_file(str(in_file), str(out_file))
        if st is not None:
            summary_stats.append(st)
            processed_paths.append(st["output_path"])

    print_summary_table(summary_stats)

    if is_sample and processed_paths:
        print_sample_before_after(processed_paths, n_samples=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize business entity names and addresses.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Process sample fixture dataset (data/sample/) instead of the full dataset.",
    )
    args = parser.parse_args()

    run_pipeline(is_sample=args.sample)
