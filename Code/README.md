# Resource Paradox in Multilingual Propaganda Detection — Code Release (UMAP 2026)

This repository contains the **companion code** for the paper:

**“Resource Paradox and Cultural-Linguistic Adaptation in Propaganda Detection in Slavic Languages vs. English”** (UMAP ’26).

It implements a **two-stage pipeline** for persuasion/propaganda technique detection:

- **Stage 1 (Paragraph-level):** multi-label classification over a shared **23-technique** taxonomy, comparing:
  - **Sup-FT**: supervised fine-tuning with an XLM-RoBERTa-based classifier (including an explanation-aware dual-encoder variant)
  - **Prompt-A**: zero-shot prompted *multi-agent* detection (one agent per technique + coordinator)
  - **Iter-Ens**: multi-round sampling + voting to improve robustness/recall
- **Stage 2 (Character-level):** LLM-based span localization within paragraphs predicted positive in Stage 1. It prompts GPT-4o-mini to output JSON spans and applies confidence filtering (≥ 0.7) + post-processing.

---

## Repository Contents

### Paragraph-level (Stage 1)

- `single_encoding.py`  
  A supervised multi-label classifier pipeline (XLM-RoBERTa) with optional **LLM-generated explanation** augmentation (text + explanation), supporting BCE / Focal loss and S3 model upload.

- `dual_encoding.py`  
  An explanation-aware **dual-encoder** architecture: text encoder + explanation encoder, fused via **bidirectional cross-attention** and an MLP fusion head, then a 23-label classifier head (BCE / Focal supported), with S3 checkpointing.

- `Multi-Agent_Propaganda.ipynb`  
  “Flexible Multi-Agent Propaganda Detection System” — Prompt-A / Iter-Ens style experiments (AutoGen-based multi-agent prompting, voting). (See paper method description for the conceptual design.)

- `autogen_propaganda_analysis.ipynb`  
  AutoGen multi-agent prompt/keyword analysis and iterative refinement workflow (coordinator + language-technique agents + reviewer).

### Character-level localization (Stage 2)

- `Promotional_technology_character-level.ipynb`  
  “AI vs Rule-based propaganda technique character-level span localization comparison system” (LLM span localization vs heuristic baseline).

Stage 2 uses GPT-4o-mini to output JSON spans:
```json
{"positions":[{"start":int,"end":int,"confidence":float,"text":str}]}
```
and filters by `confidence >= 0.7` with offset validation + deduplication.

### Data augmentation / preprocessing utilities

- `text_translator_english.py`  
  A long-text translation helper that chunks documents (default 3000 chars), calls OpenAI ChatCompletion (legacy style), and writes outputs to configurable folders. Requires `OPENAI_API_KEY`.

### Explanation generation / analysis

- `LLM_Explained.ipynb`  
  Generates/collects LLM explanations and stores results in a SQLite DB (used as “technique-focused explanations” in supervised experiments).

### Paper

- `UMAP_2026_paper_0692.pdf` — the paper describing datasets, hypotheses (H1–H3), methods, and evaluation protocol.

---

## Environment Setup

Recommended: Python 3.10+ with a CUDA-capable PyTorch install for Sup-FT.

Core libraries used across scripts/notebooks include:
- `torch`, `pytorch-lightning`, `transformers`
- `pandas`, `numpy`, `tqdm`
- `boto3` (if using CNRS Pagoda S3 uploads)
- `openai` / HTTP requests (for LLM calls in notebooks/utilities)
- `autogen` (for multi-agent notebooks)

A minimal setup (adjust to your CUDA/PyTorch needs):
```bash
pip install torch pytorch-lightning transformers pandas numpy tqdm boto3 openai
# For multi-agent notebooks:
pip install pyautogen
```

---

## Data Layout & Formats

### 1) Article folders (per split)
Both `single_encoding.py` and `dual_encoding.py` expect **folders of `.txt` files**, one file per article:
- filename pattern: `<id>.txt`
- content: multi-line text (scripts enumerate lines, starting at 1)

In `make_dataframe(data_type=...)`, you must edit:
- `your_train_articles_folder/`
- `your_dev_articles_folder/`
- `your_train_labels_file.txt`
- `your_dev_labels_file.txt`

### 2) Label file
The loaders robustly parse a tab/space-separated label file. Conceptually each row maps to:
- `id`, `line`, `labels`

Where `labels` is a comma-separated string of technique names (multi-label).

### 3) Technique taxonomy file
Both training scripts look for:
- `techniques.txt` (one technique name per line; sorted after read).

### 4) Explanation TSV (optional, for explanation-aware training)
Both scripts support loading explanation TSVs with columns like:
- `id`, `text`, `analysis`
and key entries by `(id, text)` → `analysis`.

---

## Running: Supervised Fine-Tuning (Sup-FT)

### A) `single_encoding.py`
This script supports CLI mode (argparse) and a “Jupyter default” mode.

Example CLI:
```bash
python single_encoding.py   --explanations_file path/to/explanations.tsv   --loss_type bce   --gpu 0   --batch_size 8   --epochs 10   --learning_rate 1e-5   --model xlm-roberta-base   --max_length 256
```

Key options include:
- `--explanations_file` (required)
- `--loss_type {bce,focal}` + focal params
- GPU / batch / epochs / LR / max length
- optional S3 bucket/prefix parameters for uploading checkpoints

### B) `dual_encoding.py` (recommended for “explanation fusion”)
Same idea, but with dual encoders + cross-attention fusion.

Example CLI:
```bash
python dual_encoding.py   --explanations_file path/to/explanations.tsv   --loss_type focal   --focal_gamma 2.0   --focal_alpha 1.0   --model xlm-roberta-base
```

---

## Running: Prompt-A / Iter-Ens (Zero-shot, Multi-Agent)

The paper defines:

- **Prompt-A**: one expert agent per technique, each receives technique definition/cues + context with paragraph marked, outputs conservative binary decision; coordinator aggregates into multi-label output.
- **Iter-Ens**: repeat Prompt-A for *k* rounds at temperature *τ*, then majority vote; improves robustness/recall at higher inference cost.

Use:
- `Multi-Agent_Propaganda.ipynb` for the runnable multi-agent pipeline (Prompt-A / Iter-Ens style).
- `autogen_propaganda_analysis.ipynb` for agent/prompt analysis and iterative optimization.

---

## Running: Stage 2 Character-level Span Localization

Stage 2 is standardized across methods in the paper to keep comparisons controlled:
1) build structured prompt (persona + paragraph + technique definition excerpt + criteria + keyword references),
2) call GPT-4o-mini for JSON-formatted char spans,
3) filter `confidence >= 0.7`, validate offsets, deduplicate overlaps.

Use:
- `Promotional_technology_character-level.ipynb` to run and compare LLM-based localization vs a rule-based heuristic baseline.

---

## Translation-based Augmentation Utility

If you need to reproduce the translation augmentation workflow, `text_translator_english.py` provides a folder-level translation runner:
- reads files from `input_folder`
- chunks long files (`max_length=3000`)
- calls OpenAI ChatCompletion and writes translated outputs
- moves failed files to a “failed” folder

Minimal usage:
```bash
export OPENAI_API_KEY="..."
python text_translator_english.py
```
(Then edit `set_folders(...)` in the script or uncomment the custom paths block.)

---

## Notes on Reproducibility

- Span-level results are sensitive to Stage 1 recall, strict overlap matching, and boundary shifts; Stage 2 is intentionally held constant across methods.
- If you evaluate across languages (EN/PL/RU), keep thresholds and mapping consistent with the shared 23-technique space to match the reported comparisons.

---

## Citation

If you use this code, please cite the UMAP ’26 paper (see `UMAP_2026_paper_0692.pdf`).

---

## License / Disclaimer

This repository is released as research code accompanying the paper.  
Datasets are not included; please obtain them from their official sources and follow the corresponding licenses.
