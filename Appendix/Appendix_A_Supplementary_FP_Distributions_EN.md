# A SUPPLEMENTARY FP DISTRIBUTIONS

**Corresponding position in main paper:**
- **Section 5.2** "FP Concentration and Hallucination" - Provides detailed FP distribution data for three languages, extending Table 3
- **Section 6.1** "Polish: Translation Augmentation Success" - Supports Polish FP concentration analysis
- **Section 6.2** "Russian: Low-Resource Specialization" - Supports Russian FP dispersion analysis
- Main text briefly mentions: "EN approaches a single-peak regime: total FP is 4,697, with Loaded_Language contributing 2,287 FPs (48.7%)"; this appendix provides complete Top-10 lists

**Content description:**
This appendix provides detailed Top-10 False Positive technique statistics for three languages (English, Polish, Russian) using Sup-FT method in span-level detection, including FP counts, percentages, and corresponding F1 scores.

**Corresponding main text position:**
- **Section 5.2 "FP Concentration and Hallucination"** (page 5): Main text Table 3 only shows Top-10 FP technique rankings and partial data; this appendix provides complete FP distribution tables for all three languages
- **Section 7 "English vs. Polish: Performance Comparison"** (page 7): Main text mentions "EN shows markedly higher FP concentration: Loaded_Language accounts for 48.7%"; this appendix provides detailed data support

**Note:** This appendix lists in detail the Top-10 False Positive techniques for three languages (English, Polish, Russian) under the Sup-FT method, including FP counts, percentages, and F1 scores. These tables support the conclusion in the main text that "English FP concentration is abnormally high (48.7%) while Polish and Russian are more dispersed."

## Table 6: Polish Top-10 False Positive Techniques (Sup-FT, Total FP=4,012)

| Rank | Technique | FP | FP-only Share |
|------|-----------|-----|---------------|
| 1 | Loaded_Language | 1,248 | 31.1% |
| 2 | Obfuscation-Vagueness-Confusion | 464 | 11.6% |
| 3 | Questioning_the_Reputation | 354 | 8.8% |
| 4 | Doubt | 318 | 7.9% |
| 5 | Appeal_to_Values | 295 | 7.4% |
| 6 | Exaggeration-Minimisation | 268 | 6.7% |
| 7 | Name_Calling-Labeling | 249 | 6.2% |
| 8 | Appeal_to_Fear-Prejudice | 233 | 5.8% |
| 9 | Conversation_Killer | 157 | 3.9% |
| 10 | Appeal_to_Hypocrisy | 131 | 3.3% |

## Table 7: English Top-10 False Positive Techniques (Sup-FT, Total FP=4,697)

| Rank | Technique | FP | FP-only Share |
|------|-----------|-----|---------------|
| 1 | Loaded_Language | 2,287 | 48.7% |
| 2 | Doubt | 535 | 11.4% |
| 3 | Name_Calling-Labeling | 367 | 7.8% |
| 4 | Appeal_to_Fear-Prejudice | 272 | 5.8% |
| 5 | Conversation_Killer | 263 | 5.6% |
| 6 | Exaggeration-Minimisation | 239 | 5.1% |
| 7 | Questioning_the_Reputation | 232 | 4.9% |
| 8 | Repetition | 141 | 3.0% |
| 9 | Obfuscation-Vagueness-Confusion | 124 | 2.6% |
| 10 | Flag_Waving | 123 | 2.6% |

## Table 8: Russian Top-10 False Positive Techniques (Sup-FT, Total FP=2,051)

| Rank | Technique | FP | FP-only Share |
|------|-----------|-----|---------------|
| 1 | Loaded_Language | 400 | 19.5% |
| 2 | Doubt | 338 | 16.5% |
| 3 | Questioning_the_Reputation | 212 | 10.3% |
| 4 | Obfuscation-Vagueness-Confusion | 185 | 9.0% |
| 5 | Conversation_Killer | 157 | 7.7% |
| 6 | Appeal_to_Values | 122 | 5.9% |
| 7 | Appeal_to_Fear-Prejudice | 115 | 5.6% |
| 8 | Appeal_to_Hypocrisy | 113 | 5.5% |
| 9 | Exaggeration-Minimisation | 103 | 5.0% |
| 10 | Straw_Man | 92 | 4.5% |

Hallucination on gold=0. We define hallucination as predicting at least one span in an article annotated with no techniques (gold=0). We report hallucination statistics only when they are directly produced by the evaluation script and archived with the experiment outputs.
