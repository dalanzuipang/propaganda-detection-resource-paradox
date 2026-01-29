# Q RUSSIAN: DETAILED TABLES

**Corresponding position in main paper:**
- **Section 6.2 "Russian: Low-Resource Specialization"** (page 6): Main text mentions "baseline Macro F1=0.00%... Sup-FT reaches 39.57%... distinctive advantage on Doubt: F1=0.907... conservative on Loaded_Language... FP distribution most dispersed (top-2 concentration 36.0%)"; this appendix provides supporting data
- **Table 1** (page 5): Main text Table 1 shows core metrics; this appendix Tables 18-23 provide complete Russian analysis

**Note:** This appendix contains 6 detailed tables for Russian language:
- **Table 18**: Baseline complete collapse (0.00%) vs. recovery with augmented methods
- **Table 19**: Comprehensive performance comparison of three methods
- **Table 20**: Cross-linguistic performance of Doubt (Russian achieves highest 0.907)
- **Table 21**: Conservative strategy on Loaded_Language (precision 1.000 but recall 0.761)
- **Table 22**: Most dispersed FP distribution (Top-2 only 36.0%, compared to English's 60.1%)
- **Table 23**: List of failed techniques under extreme scarcity

## Table 18: Russian Baseline vs. Augmented Methods (Task 2: technique multi-label classification)

| Method | Training Data | Macro F1 |
|--------|---------------|----------|
| Baseline | SlavicNLP seed only (77 seed articles; multilingual mix) | 0.00% |
| Sup-FT | SemEval + translation (+ seed) | 39.57% |
| Prompt-A | Zero-shot (no training) | 35.25% |
| Iter-Ens | Zero-shot (no training) | 38.77% |

## Table 19: Performance Comparison of Three Detection Methods on Russian

| Metric | Sup-FT | Prompt-A | Iter-Ens | Best Method |
|--------|--------|----------|----------|-------------|
| **Paragraph-level Classification** | | | | |
| Macro F1 | 39.57% | 35.25% | 38.77% | Sup-FT |
| Micro F1 | 57.19% | 44.48% | 48.05% | Sup-FT |
| Binary F1 | 97.87% | 92.13% | 100.00% | Iter-Ens |
| **Character-level Localization** | | | | |
| Span F1 | 11.17% | 6.13% | 6.16% | Sup-FT |
| Precision | 7.45% | 4.01% | 3.54% | Sup-FT |
| Recall | 22.33% | 12.99% | 23.41% | Iter-Ens |

## Table 20: Cross-linguistic Performance Comparison of Doubt (Best Method per Language)

| Language | Best F1 | Method | Precision | Recall | Support | Gap vs Russian |
|----------|---------|--------|-----------|--------|---------|----------------|
| Russian | 0.907 | Iter-Ens | 0.944 | 0.872 | 39 | – |
| Polish | 0.857 | Prompt-A | – | – | 32 | -0.050 |
| English | 0.750 | Sup-FT/Iter-Ens | 0.714 | 0.789 | 57 | -0.157 |

## Table 21: Cross-linguistic Comparison of Loaded_Language (Paragraph-level)

| Language | F1 | Precision | Recall | Support |
|----------|-----|-----------|--------|---------|
| English | 0.938 | 0.905 | 0.974 | 78 |
| Polish | 0.883 | 0.791 | 1.000 | 34 |
| Russian | 0.864 | 1.000 | 0.761 | 46 |

## Table 22: Top 5 False Positives in Russian (Sup-FT, FP-only)

| Rank | Technique | FP Count | % of Total FP |
|------|-----------|----------|---------------|
| 1 | Loaded_Language | 400 | 19.5% |
| 2 | Doubt | 338 | 16.5% |
| 3 | Questioning_Reputation | 212 | 10.3% |
| 4 | Obfuscation-Vagueness | 185 | 9.0% |
| 5 | Conversation_Killer | 157 | 7.7% |
| **Top 2 Concentration** | | 738 | 36.0% |

## Table 23: Russian weak/failed techniques under extreme scarcity (support and qualitative note)

| Technique | Support | Note |
|-----------|---------|------|
| Appeal_to_Authority | 1 | Complete failure under low support |
| Red_Herring | 1 | Complete failure under low support |
| Appeal_to_Time | 1 | Complete failure under low support |
| Appeal_to_Popularity | 2 | Unstable boundary under low support |
| Whataboutism | 4 | Culture/context dependent; scarce evidence |
