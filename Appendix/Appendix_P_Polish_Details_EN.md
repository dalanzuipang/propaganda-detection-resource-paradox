# P POLISH: DETAILED TABLES

**Corresponding position in main paper:**
- **Section 6.1** "Polish: Translation Augmentation Success" - This appendix provides detailed data support for this subsection
- Main text cites multiple specific data points:
  - "Baseline degenerates on Task 2 (Macro F1=0.53%)" → Table 12
  - "Name_Calling-Labeling (F1=0.904), exceeding RU (0.649)" → Table 14
  - "Loaded_Language contributes 31.1% of total FP, and the top five techniques contribute 66.8%" → Table 15
  - "Weak techniques... with F1 < 0.3 are dominated by low support" → Table 16
- **Table 4** - Table 17 is the detailed version of the Polish portion

**Content description:**
This appendix provides 6 detailed tables for Polish language:
1. **Table 12**: Baseline vs. Augmented method comparison (validates H1)
2. **Table 13**: Performance comparison of three detection methods (paragraph-level + span-level)
3. **Table 14**: Cross-linguistic performance comparison of Name_Calling (Polish advantage clear)
4. **Table 15**: Top 5 False Positives and concentration analysis
5. **Table 16**: List of weak techniques (F1<0.3, mainly affected by low support)
6. **Table 17**: Strengths, weaknesses, and applicable scenarios of three methods

**Corresponding main text position:**
- **Section 6.1 "Polish: Translation Augmentation Success"** (page 6): Main text mentions "baseline Macro F1=0.53%... Sup-FT reaches 52.23%... Name_Calling F1=0.904... top five techniques contribute 66.8%"; this appendix provides 6 detailed tables supporting these conclusions
- **Table 1** (page 5): Main text Table 1 shows core metrics; this appendix Tables 12-17 provide complete Polish analysis

**Note:** This appendix contains 6 detailed tables for Polish language:
- **Table 12**: Baseline vs. augmented methods comparison (showing 98.6× performance improvement)
- **Table 13**: Comprehensive performance comparison of three methods
- **Table 14**: Cross-linguistic performance of Name_Calling (Polish advantage clear)
- **Table 15**: Top-5 FP concentration analysis (66.8%)
- **Table 16**: List of weak techniques (F1<0.3, mainly due to insufficient samples)
- **Table 17**: Strengths, weaknesses, and applicable scenarios of three methods

## Table 12: Polish Baseline vs. Augmented Methods (Task 2: technique multi-label classification)

| Method | Training Data | Macro F1 |
|--------|---------------|----------|
| Baseline | SlavicNLP seed only (77 seed articles; multilingual mix) | 0.53% |
| Sup-FT | SemEval + translation (+ seed) | 52.23% |
| Prompt-A | Zero-shot (no training) | 49.97% |
| Iter-Ens | Zero-shot (no training) | 50.55% |

## Table 13: Performance Comparison of Three Detection Methods on Polish

| Evaluation Dimension | Sup-FT | Prompt-A | Iter-Ens | Best Method |
|----------------------|--------|----------|----------|-------------|
| **Paragraph-level Classification** | | | | |
| Macro F1 | 52.23% | 49.97% | 50.55% | Sup-FT |
| Micro F1 | 66.75% | 58.88% | 58.65% | Sup-FT |
| Binary F1 | 96.77% | 98.88% | 96.77% | Prompt-A |
| **Character-level Localization** | | | | |
| Span F1 | 8.51% | 5.98% | 7.54% | Sup-FT |
| Precision | 6.22% | 4.75% | 4.76% | Sup-FT |
| Recall | 13.50% | 8.07% | 18.12% | Iter-Ens |

## Table 14: Cross-linguistic Performance of Name_Calling Technique

| Metric | Polish | English | Russian |
|--------|--------|---------|---------|
| F1 | 0.904 | 0.833 | 0.649 |
| Precision | 0.917 | 0.778 | 0.706 |
| Recall | 0.892 | 0.889 | 0.600 |
| Support | 37 | 63 | 20 |
| Gap vs Polish | – | -0.071 | -0.255 |

## Table 15: Top 5 False Positives in Polish (Sup-FT, FP-only)

| Rank | Technique | FP Count | % of Total FP |
|------|-----------|----------|---------------|
| 1 | Loaded_Language | 1,248 | 31.1% |
| 2 | Obfuscation-Vagueness | 464 | 11.6% |
| 3 | Questioning_Reputation | 354 | 8.8% |
| 4 | Doubt | 318 | 7.9% |
| 5 | Appeal_to_Values | 295 | 7.4% |
| **Top 2 Concentration** | | 1,712 | 42.7% |
| **Top 5 Concentration** | | 2,679 | 66.8% |

## Table 16: Polish Weak Techniques (F1<0.3)

| Technique | F1 | Precision | Recall | Support | Main Issue |
|-----------|-----|-----------|--------|---------|------------|
| Straw_Man | 0.000 | 0.000 | 0.000 | 1 | Extremely few samples |
| Appeal_to_Pity | 0.000 | 0.000 | 0.000 | 0 | No test samples |
| Appeal_to_Time | 0.182 | 0.143 | 0.250 | 4 | Insufficient samples |
| Whataboutism | 0.222 | 0.143 | 0.500 | 2 | Extremely low frequency |
| Causal_Oversimplification | 0.267 | 0.154 | 1.000 | 4 | Insufficient samples |

## Table 17: Advantages and Disadvantages of Three Methods on Polish

| Method | Advantages | Disadvantages | Applicable Scenarios |
|--------|------------|---------------|----------------------|
| Sup-FT | Highest Macro F1 (52.23%)<br>Highest Span F1 (8.51%)<br>Balanced technique distribution<br>Lower false positive risk | | Precision priority<br>Comprehensive performance |
| Prompt-A | Highest Binary F1 (98.88%)<br>Zero hallucination<br>Strong interpretability | Lower Macro F1 (49.97%)<br>Lowest Span F1 (5.98%)<br>High API cost | Interpretability priority<br>Zero-shot deployment |
| Iter-Ens | Highest recall (18.12%)<br>Zero hallucination<br>Strong robustness | Increased false positives<br>High inference cost<br>Medium Span F1 (7.54%) | Recall priority<br>Robustness needs |
