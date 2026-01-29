# G SEED-ONLY BASELINE CONFIGURATION

**Corresponding position in main paper:**
- **Section 4** "Baseline" - This appendix provides detailed configuration description for this paragraph
- Main text cites baseline results: "for Polish, Macro F1 = 0.53% (28 predicted labels vs. 348 gold labels), for Russian: Macro F1 = 0.00%"

**Content description:**
Provides complete training configuration for seed-only baseline, demonstrating that 77 articles are insufficient to support stable multi-label learning.

---

# H ABLATION CONFIGURATIONS FOR SUP-FT

**Corresponding position in main paper:**
- **Section 4** "Ablation Study Design" - This appendix details three ablation configurations
- **Section 7.1** "Ablation Study: Architecture vs. Data" - Uses these three configurations for experiments; results shown in Table 5
- Main text mentions: "We evaluate three controlled architectures: (1) Config-1... (2) Config-2... (3) Config-3"

**Content description:**
Detailed definition of three ablation experiment configurations to isolate the effects of translation, label semantics, and dual-encoder architecture.

---

# I TECHNIQUE-LEVEL FP/FN METRICS

**Corresponding position in main paper:**
- **Section 4** "Span-Level Error Analysis" - This appendix provides mathematical definition of FP-only share
- Main text uses this formula: "Share_t = (FP_t / Σ_k FP_k) × 100%"
- **Table 3** - Uses results calculated with this formula

**Content description:**
Defines technique-level FP/FN calculation methods and FP-only share formula.

**Corresponding main text position:**
- **Section 4 "Evaluation Protocol" - Baseline section** (page 4): Main text mentions "Baseline yields near-zero performance (Polish: 0.53%, Russian: 0.00%)"; Appendix G provides complete baseline training configuration
- **Section 4 "Evaluation Protocol" - Ablation Study Design section** (page 4): Main text describes the design of three Configs; Appendix H provides detailed explanation
- **Section 4 "Evaluation Protocol" - Span-Level Error Analysis section** (page 4): Main text defines FP-only share formula; Appendix I provides complete metric definition

**Note:** This document contains three appendix sections:
- **Appendix G**: Complete configuration of seed-only baseline (77 articles, XLM-RoBERTa-base, training parameters)
- **Appendix H**: Three configurations for ablation experiments (Config-1: Base Translation, Config-2: Concat + Expl., Config-3: Dual-Encoder)
- **Appendix I**: Mathematical definition of technique-level FP/FN metrics

We fine-tune XLM-RoBERTa-base on the SlavicNLP seed data (77 articles) for paragraph-level multi-label classification using sigmoid outputs. Input paragraphs are tokenized with padding/truncation to a maximum length of 256 tokens. We use a fixed decision threshold of 0.3 to predict all techniques whose probabilities exceed the threshold.

Training uses a 75%/25% train/validation split on the seed set, batch size 8, learning rate 5 × 10^-5, 8 epochs, and mixed precision (FP16). Due to the absence of Croatian training data, we report baseline results only for Polish and Russian.

# H ABLATION CONFIGURATIONS FOR SUP-FT

We evaluate three controlled architectures:

• Config-1 (Base Translation): XLM-RoBERTa trained on translated text only, without label semantics or dual-encoder structure.

• Config-2 (Concat w/ Explanations): Single-encoder model concatenating paragraph text with technique definitions/explanations.

• Config-3 (Dual-Encoder): Full dual-encoder with cross-attention fusion (reported as Sup-FT in main experiments).

These ablations isolate (i) label semantics (Config-2 vs. Config-1), (ii) dual-encoder fusion (Config-3 vs. Config-2), and (iii) the combined effect (Config-3 vs. Config-1).

# I TECHNIQUE-LEVEL FP/FN METRICS

For span-level error analysis, we compute per-technique false positives (FP) and false negatives (FN). To quantify error concentration, we define the FP-only share for technique t as:

Share_t = (FP_t / Σ_k FP_k) × 100%,

where the denominator is the total FP count for a given language/method setting.
