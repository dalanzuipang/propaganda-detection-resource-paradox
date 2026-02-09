# propaganda-detection-resource-paradox

**Resource Paradox in Multilingual Propaganda Detection: Cross-linguistic Analysis of English and Slavic Languages (Polish, Russian)**

This repository provides the open-source materials accompanying the paper. It includes **reproducible experimental code** and **complete supplementary analysis materials**, supporting the paper’s cross-linguistic comparisons of propaganda/persuasion technique detection across English and Slavic languages (Polish, Russian).

---

## Repository Structure (Key)

```
.
├── Code/          # Runnable code and experiment scripts (training / inference / multi-agent / span localization, etc.)
├── Appendix/      # Supplementary materials: analysis tables, additional results, statistics, visualizations, etc. (content not included in the main paper)
└── README.md
```

### Appendix/ (Supplementary Materials)

The `Appendix/` folder contains all materials that **cannot be fully included in the main paper**, for example:

* Detailed result tables by language / method / setting (per-class, per-language, per-configuration)
* Ablation studies and additional comparison experiment tables
* Error analysis, case studies, confusion statistics, and visualization data
* Generated explanations, prompts, and intermediate artifacts

> Reading suggestion: If you want a quick understanding of the paper’s conclusions and evidence, read the main paper first. If you want to verify specific statistics or need more fine-grained metrics, go directly to `Appendix/`.

### Code/ (Codebase)

The `Code/` folder contains runnable implementations (depending on the actual files), typically covering:

* **Paragraph-level multi-label classification (Stage 1):** supervised fine-tuning (Sup-FT) and explanation-enhanced variants
* **Prompt-A / Iter-Ens:** multi-agent / prompted detection and iterative ensembling (voting) workflows
* **Character-level span localization (Stage 2):** locating the exact text spans 
* Data loading, format conversion, visualization, and utility scripts

---

## Quick Start (Recommended Workflow)

1. **Read the paper / key findings:** understand the task definition, language settings, metrics, and main conclusions.
2. **Validate supplementary evidence:** check `Appendix/` for full tables and detailed statistics.
3. **Reproduce experiments / run code:** go to `Code/` and follow the instructions in the scripts or notebooks.

---
