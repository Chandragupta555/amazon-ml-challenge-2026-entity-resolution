# Business Entity Resolution

This project addresses the Amazon ML Challenge 2026 business entity resolution task by identifying and linking records across three independent data sources (Source 1 reference source, Source 2, and Source 3) that refer to the same real-world business entity. The pipeline performs end-to-end entity resolution comprising text normalization, candidate pair blocking, pairwise similarity and contextual feature extraction, LightGBM classification trained on ground truth labels, constraint-enforced one-to-one match assignment, and macro-averaged $F_{0.5}$ evaluation.

## Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Location

The dataset is expected at:
- `../../student_resource/dataset/` relative to this folder (`code/business_entity_resolution/`), or
- `../../../student_resource/dataset/` relative to the `src/` directory (three levels up: `src` → `business_entity_resolution` → `code` → repo root).

## Reproduction Pipeline

To reproduce the solution end-to-end, execute the following scripts in order from the `code/business_entity_resolution/` directory:

1. **Normalize text fields:**
   ```bash
   python src/normalize.py
   ```
2. **Generate candidate pairs (blocking):**
   ```bash
   python src/blocking.py
   ```
3. **Compute pairwise features:**
   ```bash
   python src/features.py
   ```
4. **Train matching model:**
   ```bash
   python src/train_model.py
   ```
5. **Assign matches and format output:**
   ```bash
   python src/assign_matches.py
   ```
6. **Evaluate on validation split:**
   ```bash
   python src/evaluate.py
   ```
