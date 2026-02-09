# Appendices README

## Overview

This document provides a comprehensive guide to the supplementary appendices for the work:

**"Resource Paradox and Cultural-Linguistic Adaptation in Propaganda Detection in Slavic Languages vs. English"**

These appendices provide detailed technical specifications, experimental configurations, statistical analyses, and complete data tables that support the findings presented in the main paper. Each appendix is carefully cross-referenced to specific sections of the paper.

---

## Paper Structure Reference

The main paper is organized as follows:
- **Section 1**: Introduction
- **Section 2**: Related Work
- **Section 3**: Methodology (Data augmentation, Detection granularity, Three methods)
- **Section 4**: Evaluation Protocol
- **Section 5**: Systematic Bias of High-Resourced English
- **Section 6**: Performance on Slavic Languages
- **Section 7**: English vs. Polish: Performance Comparison
- **Section 8**: Error Patterns and User Perception
- **Section 9**: Discussion

---

## Appendix Organization

### Group 1: Core Methodology Details (Appendices A-F)

#### **Appendix A: Supplementary FP Distributions**
- **File**: `Appendix_A_Supplementary_FP_Distributions_EN.md`
- **Corresponds to**: Section 5.2 "FP Concentration and Hallucination"
- **Content**: 
  - Complete Top-10 False Positive technique tables for all three languages
  - Polish: Total FP=4,012 (Top technique: Loaded_Language 31.1%)
  - English: Total FP=4,697 (Top technique: Loaded_Language 48.7%)
  - Russian: Total FP=2,051 (Top technique: Loaded_Language 19.5%)
- **Key Finding**: English shows abnormally high FP concentration (48.7% on single technique) compared to Polish (31.1%) and Russian (19.5%)

#### **Appendix B: Method Comparison Tables**
- **File**: `Appendix_B_Method_Comparison_Tables_EN.md`
- **Corresponds to**: Table 1 (Section 5.1) and Table 4 (Section 6.3)
- **Content**:
  - **Table 9**: Complete panoramic performance comparison (9 rows × 6 metrics)
  - **Table 10**: Strengths and weaknesses across 7 dimensions
- **Key Tables**:
  - Macro/Micro F1, Binary F1, Span F1, Hallucination@Gold0 for all method-language combinations
  - Trade-off analysis: Performance vs. Interpretability vs. Cost

#### **Appendix C: Sup-FT Implementation Details**
- **File**: `Appendix_C_Sup_FT_Implementation_EN.md`
- **Corresponds to**: Section 3.3.1 "Supervised Fine-Tuning (Sup-FT)" and Figure 1
- **Content**:
  - Complete mathematical formulas for dual-encoder architecture
  - Cross-attention mechanism: Attn_T→E and Attn_E→T
  - Feature fusion MLP specifications
  - Training configuration: AdamW, learning rate, batch size, gradient clipping
  - Weighted BCE loss and class weight calculation
- **Critical for**: Experimental reproduction

#### **Appendix D-E-F: Data and Translation**
- **File**: `Appendix_D_E_F_Data_and_Translation_EN.md`
- **Corresponds to**: Section 3.1 "Translation-Based Augmentation Strategy" and Section 3.2 "Detection Granularity"
- **Content**:
  - **Appendix D**: Training corpus composition
    - Polish: 145 original (21%) + 517 translated (76%) + 15 SlavicNLP (2%) = 677 total
    - Russian: 143 original (21%) + 517 translated (75%) + 27 SlavicNLP (4%) = 687 total
    - English: 536 original (100%), no translation augmentation
  - **Appendix E**: Unified Stage 2 span localization procedure (3 steps)
  - **Appendix F**: Three translation artifacts risks (literalness, syntactic interference, cultural localization gap)

---

### Group 2: Evaluation & Statistical Analysis (Appendices G-K)

#### **Appendix G-H-I: Baseline and Metrics**
- **File**: `Appendix_G_H_I_Baseline_and_Metrics_EN.md`
- **Corresponds to**: Section 4 "Evaluation Protocol"
- **Content**:
  - **Appendix G**: Seed-only baseline configuration (77 articles)
    - Polish baseline: Macro F1 = 0.53% (28 predicted vs. 348 gold)
    - Russian baseline: Macro F1 = 0.00% (5 predicted vs. 278 gold)
  - **Appendix H**: Three ablation configurations
    - Config-1: Base Translation
    - Config-2: Concat w/ Explanations
    - Config-3: Dual-Encoder (full Sup-FT)
  - **Appendix I**: Mathematical definition of technique-level FP/FN metrics

#### **Appendix K: Statistical Significance**
- **File**: `Appendix_K_Statistical_Significance_EN.md`
- **Corresponds to**: Section 5.1 "High-Resource Paradox: Overview"
- **Content**:
  - Support-controlled significance tests (techniques with support ≥5)
  - Paired t-test: t=0.90, df=13, p=0.384
  - Wilcoxon signed-rank test: p=0.715
  - Bootstrap 95% CI: [-6.38%, 20.22%], p=0.508
  - Effect size: Cohen's d=0.24
- **Key Finding**: Polish's advantage narrows but remains directional; weak statistical significance suggests practical rather than systematic advantage

---

### Group 3: Language-Specific Analysis (Appendices P-Q-S-U-V)

#### **Appendix P: Polish Details**
- **File**: `Appendix_P_Polish_Details_EN.md`
- **Corresponds to**: Section 6.1 "Polish: Translation Augmentation Success"
- **Content** (6 detailed tables):
  - **Table 12**: Baseline vs. augmented methods (98.6× performance improvement)
  - **Table 13**: Three methods comparison (paragraph + span level)
  - **Table 14**: Cross-linguistic Name_Calling performance (Polish F1=0.904 vs. Russian 0.649)
  - **Table 15**: Top-5 FP concentration (66.8%)
  - **Table 16**: Weak techniques list (F1<0.3, mainly due to insufficient samples)
  - **Table 17**: Method advantages/disadvantages and applicable scenarios

#### **Appendix Q: Russian Details**
- **File**: `Appendix_Q_Russian_Details_EN.md`
- **Corresponds to**: Section 6.2 "Russian: Low-Resource Specialization"
- **Content** (6 detailed tables):
  - **Table 18**: Baseline complete collapse (0.00%) vs. recovery
  - **Table 19**: Three methods comparison
  - **Table 20**: Cross-linguistic Doubt performance (Russian achieves highest 0.907)
  - **Table 21**: Conservative Loaded_Language strategy (precision 1.000, recall 0.761)
  - **Table 22**: Most dispersed FP distribution (Top-2 only 36.0%)
  - **Table 23**: Failed techniques under extreme scarcity

#### **Appendix S-U: Cross-Linguistic & Cultural Analysis**
- **File**: `Appendix_S_U_Cross_Linguistic_Cultural_EN.md`
- **Corresponds to**: Section 6.3 "Cross-Linguistic Patterns" and Section 8.2 "Personalized Calibration"
- **Content**:
  - **Appendix S (Table 24)**: Cross-linguistic F1 comparison for 10 major techniques
    - Language-sensitive techniques: Name_Calling (max gap=0.255), Appeal_to_Hypocrisy (gap=0.661)
    - Robust techniques: Doubt (gap=0.099), Loaded_Language (gap=0.074)
  - **Appendix U**: Exploratory cultural dimension-based personalization framework
    - Knob 1: Personal targeting salience
    - Knob 2: Uncertainty tolerance
    - **Important**: Hypothesis-generating, not empirical evidence; requires user study validation

#### **Appendix V: Adaptation Strategy**
- **File**: `Appendix_V_Adaptation_Strategy_EN.md`
- **Corresponds to**: Section 8.3 "Multi-Method Strategy Selection Guide"
- **Content**:
  - Technique-language-method routing strategies
  - Default method recommendations per language
  - Specialized routing for specific techniques (e.g., Doubt in Russian → Iter-Ens)
  - Confidence-based fallback mechanisms

---

## Key Findings Summary by Appendix

### Hypothesis Validation

**H1 (Augmentation Viability)**: ✓ Strongly Supported
- Evidence: Appendices D, G, P (Table 12), Q (Table 18)
- Baseline collapses (PL: 0.53%, RU: 0.00%) → Augmented achieves 52.23% and 39.57%

**H2 (Zero-shot Feasibility)**: ✓ Partially Supported
- Evidence: Appendix B (Table 9)
- Prompt-A approaches Sup-FT in Polish (gap: 2.26 points), larger gaps in EN/RU

**H3 (Ensembling Robustness)**: ✓ Consistently Supported
- Evidence: Appendix B (Tables 9-10)
- Iter-Ens outperforms Prompt-A across all languages (+0.58 to +6.41 points)

### Resource Paradox Evidence

**Core Paradox**: English (536 articles) underperforms Polish (145 articles) by 10.47 Macro F1 points
- **FP Concentration**: Appendix A shows English 48.7% vs. Polish 31.1%
- **Ablation Analysis**: Appendix H (Table 5) shows paradox persists across all configurations
- **Statistical Test**: Appendix K shows weak effect size (Cohen's d=0.24)

### Language-Specific Strengths

**Polish**:
- Strongest paragraph-level performance (Macro F1: 52.23%)
- Excellent Name_Calling detection (F1: 0.904) - Appendix P (Table 14)
- Balanced FP distribution - Appendix A (Table 6)

**English**:
- Highest span localization (Span F1: 11.82%)
- Severe FP concentration on Loaded_Language - Appendix A (Table 7)
- Long-tail technique distribution - Section 5.3

**Russian**:
- Best Doubt detection (F1: 0.907) - Appendix Q (Table 20)
- Most dispersed FP profile (Top-2: 36.0%) - Appendix Q (Table 22)
- Conservative Loaded_Language strategy - Appendix Q (Table 21)

---

## Usage Guide

### For Experimental Reproduction
1. Start with **Appendix C**: Implementation details for Sup-FT
2. Refer to **Appendix D**: Data corpus composition
3. Check **Appendix G**: Baseline configuration
4. Use **Appendix H**: Ablation configurations

### For Performance Analysis
1. Review **Appendix B (Table 9)**: Complete performance metrics
2. Examine **Appendices P/Q**: Language-specific detailed tables
3. Compare **Appendix A**: FP distribution patterns

### For Statistical Validation
1. **Appendix K**: Significance tests for English vs. Polish comparison
2. **Appendix B (Table 10)**: Method trade-off analysis
3. **Appendix G (Appendix I)**: Metric definitions

### For Deployment Guidance
1. **Appendix B (Table 10)**: Method strengths/weaknesses
2. **Appendix V**: Technique-language-method routing
3. **Appendices P/Q (Table 17/23)**: Applicable scenarios by method

---

## Cross-Reference Matrix

| Main Paper Section | Primary Appendices | Key Tables/Figures |
|-------------------|-------------------|-------------------|
| 3.1 Translation Augmentation | D | Corpus composition |
| 3.2 Detection Granularity | E | Stage 2 localization |
| 3.3.1 Sup-FT Method | C | Architecture formulas |
| 4 Evaluation Protocol | G, H, I | Baseline, ablations, metrics |
| 5.1 High-Resource Paradox | K | Statistical tests |
| 5.2 FP Concentration | A | Tables 6-8 |
| 6.1 Polish Performance | P | Tables 12-17 |
| 6.2 Russian Performance | Q | Tables 18-23 |
| 6.3 Cross-Linguistic Patterns | S | Table 24 |
| 7.1 Ablation Study | H | Table 5 comparison |
| 8.2 Personalized Calibration | U | Cultural knobs |
| 8.3 Method Selection | V | Routing strategies |

---

## Important Notes

1. **All Chinese text has been translated to English**: The "_EN" suffix indicates English-translated versions

2. **Statistical Rigor**: Appendix K demonstrates that while Polish shows directional advantage, the effect size is small (Cohen's d=0.24), suggesting the paradox is practical rather than definitively systematic

3. **Exploratory Framework**: Appendix U's cultural dimension framework is explicitly hypothesis-generating and requires user study validation

4. **Hallucination Metrics**: Gold=0 counts are very small (EN: 3, PL: 3, RU: 1), interpret Hallucination@Gold0 as risk indicator, not robust estimate

5. **Support Effects**: Many techniques have support=0 or very low support, indicated by * in tables. F1=0.000 reflects lack of evaluable samples rather than model failure

---

## Citation

When referencing specific appendix content, please use:

```
[Author]. (2026). Resource Paradox and Cultural-Linguistic Adaptation 
in Propaganda Detection in Slavic Languages vs. English. 
In Proceedings of UMAP '26. Appendix [Letter].
```

---

## Version Information

- **Paper Version**: UMAP 2026 Conference Submission
- **Appendix Language**: English (translated from Chinese annotations)
- **Last Updated**: January 29, 2026

---

## Contact

For questions about specific appendices or data access, please refer to the anonymous repository:
https://anonymous.4open.science/r/propaganda-detection-resource-paradox-F871/

---

## Quick Navigation

- **Need implementation details?** → Appendix C
- **Need performance tables?** → Appendices A, B, P, Q
- **Need statistical tests?** → Appendix K
- **Need data composition?** → Appendix D
- **Need method comparison?** → Appendix B (Tables 9-10)
- **Need language-specific analysis?** → Appendices P (Polish), Q (Russian)
- **Need deployment guidance?** → Appendices V, P/Q (Table 17)
