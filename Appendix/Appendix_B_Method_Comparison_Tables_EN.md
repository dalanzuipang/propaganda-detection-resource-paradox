# B SUPPLEMENTARY METHOD COMPARISON TABLES

**Corresponding position in main paper:**
- **Table 1** (Section 5.1) - This appendix Table 9 is a complete expanded version of Table 1, adding all evaluation metrics
- **Table 4** (Section 6.3) - This appendix Table 10 is identical to main text Table 4, showing comparison of strengths and weaknesses of three methods
- **Section 6.3** "Validation of H1, H2, and H3" - Uses this appendix data to validate three hypotheses
- Main text cites: "see Table 1" (multiple references to performance data)

**Content description:**
This appendix provides two core comparison tables:
1. Table 9: Panoramic performance comparison of three methods across three languages (9 rows × 6 column metrics)
2. Table 10: Comparison of strengths and weaknesses of Sup-FT, Prompt-A, and Iter-Ens across 7 dimensions

**Corresponding main text position:**
- **Table 1 "Performance comparison of three methods"** (page 5): Main text table only shows core metrics; this appendix Table 9 provides complete 9 rows × 6 columns panoramic comparison
- **Table 4 "Strengths and weaknesses comparison"** (page 7): Main text already includes this table; this appendix is a complete preserved version
- **Section 6.3 "Cross-Linguistic Patterns"** (page 6): Main text mentions "H1, H2, H3 validation"; this appendix provides data support

**Note:** Table 9 shows all evaluation metrics for three methods (Sup-FT, Prompt-A, Iter-Ens) across three languages; Table 10 compares strengths and weaknesses of the three methods across 7 dimensions. These tables are expanded versions of main text Tables 1 and 4, providing more comprehensive method comparison data.

## Table 9: Cross-Linguistic Panoramic Performance Comparison of Three Methods

| Language | Method | Macro F1 | Micro F1 | Binary F1 | Span F1 | Hallucination@Gold0 |
|----------|--------|----------|----------|-----------|---------|---------------------|
| Polish | Sup-FT | 52.23% | 66.75% | 96.77% | 8.51% | 100% (3/3) |
| Polish | Prompt-A | 49.97% | 58.88% | 98.88% | 5.98% | 0% (0/3) |
| Polish | Iter-Ens | 50.55% | 58.65% | 96.77% | 7.54% | 0% (0/3) |
| English | Sup-FT | 41.76% | 59.15% | 97.11% | 11.82% | 66.7% (2/3) |
| English | Prompt-A | 38.47% | 51.84% | 94.55% | 7.81% | 0% (0/3) |
| English | Iter-Ens | 44.88% | 57.48% | 96.43% | 10.85% | 0% (0/3) |
| Russian | Sup-FT | 39.57% | 57.19% | 97.87% | 11.17% | 100% (1/1) |
| Russian | Prompt-A | 35.25% | 44.48% | 92.13% | 6.13% | 0% (0/1) |
| Russian | Iter-Ens | 38.77% | 48.05% | 100.00% | 6.16% | 0% (0/1) |

## Table 10: Strengths and Weaknesses Comparison of Three Methods

| Dimension | Sup-FT | Prompt-A | Iter-Ens |
|-----------|--------|----------|----------|
| Macro F1 | Highest (Polish 52.23%, Russian 39.57%) | Lowest (2-4 percentage points lower per language) | Medium (English highest 44.88%) |
| Interpretability | Black box, no reasoning process | Provides evidence fragments | Provides evidence + voting statistics |
| Hallucination Risk | Has hallucinations on gold=0 (2.22%-6.12%) | Zero hallucinations (0%) | Zero hallucinations (0%) |
| Deployment Cost | Low (one-time training, fast inference) | Medium (API calls, 1× per document) | High (API calls, 3× per document) |
| Training Requirements | Requires GPU + labeled data | Zero-shot, no training needed | Zero-shot, no training needed |
| False Positive Control | English Loaded_Language FP accounts for 48.7% | More balanced distribution | Total FP increases (Russian FP 4708 vs Sup-FT 2051) |
| Recall | Medium | Lowest | Higher recall tendency, but with increased false positives |

Gold=0 counts are extremely small; interpret Hallucination@Gold0 as a risk indicator.
