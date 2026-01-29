# D AUGMENTED TRAINING CORPUS COMPOSITION

**Corresponding position in main paper:**
- **Section 3.1** "Translation-Based Augmentation Strategy" - This appendix provides precise statistics on corpus composition
- Main text mentions: "(a) PL: 145 original (21%) + 517 translated (76%) + 15 SlavicNLP (2%) = 677 total"; this appendix explains the source of this data

**Content description:**
Detailed explanation of the training corpus composition ratios for three languages (original/translated/SlavicNLP seed).

---

# E STAGE 2 UNIFIED SPAN LOCALIZATION PROCEDURE

**Corresponding position in main paper:**
- **Section 3.2** "Detection Granularity: Paragraphs to Spans" - This appendix provides detailed process description for this subsection
- Main text briefly describes the three-step process: "(1) Prompt construction... (2) Span generation... (3) Post-processing"; this appendix provides complete technical details and error propagation analysis

**Content description:**
Detailed explanation of the LLM-based span localization process shared by all methods, including:
- Unified LLM-based Localization (unified GPT-4o-mini calling process)
- Error Propagation and Span-Level Difficulty (error propagation mechanism and span-level challenges)

---

# F TRANSLATION ARTIFACTS: RISKS AND THREATS TO VALIDITY

**Corresponding position in main paper:**
- **End of Section 3.1** - Main text mentions: "Translation-based expansion can introduce systematic artifacts... We treat translation as an empirical factor"
- **Section 7** "English vs. Polish: Performance Comparison" - Uses risk analysis from this appendix to explain performance differences

**Content description:**
Systematic discussion of three types of risks introduced by machine translation:
- Literalness and stylistic flattening
- Syntactic interference
- Cultural localization gap

**Corresponding main text position:**
- **Section 3.1 "Translation-Based Augmentation Strategy"** (pages 2-3): Main text mentions "PL: 145 original (21%) + 517 translated (76%)"; Appendix D provides complete corpus composition
- **Section 3.2 "Detection Granularity: Paragraphs to Spans"** (page 3): Main text briefly describes the two-stage pipeline; Appendix E provides detailed Stage 2 span localization process
- **Section 7 "English vs. Polish"** (page 7): Main text mentions translation artifacts; Appendix F details three translation risks

**Note:** This document contains three appendix sections:
- **Appendix D**: Detailed composition of training corpus (ratios of original, translated, and SlavicNLP seed data)
- **Appendix E**: Unified LLM-based span localization procedure (three-step process: prompt construction, span generation, post-processing)
- **Appendix F**: Three systematic risks of translation artifacts (literalness, syntactic interference, cultural localization gap)

We construct the augmented training corpus from SemEval-2023 Task 3 plus the SlavicNLP seed data (in-domain supplement for overlapping languages). For supervised training, the corpus composition is:

• Polish: 145 original (21%) + 517 translated (76%) + 15 SlavicNLP (2%) = 677 total
• Russian: 143 original (21%) + 517 translated (75%) + 27 SlavicNLP (4%) = 687 total
• English: 536 original (100%), no translation augmentation

# E STAGE 2 UNIFIED SPAN LOCALIZATION PROCEDURE

## Unified LLM-based Localization

All methods share the same LLM-based span localization procedure to ensure that span-level differences mainly reflect Stage 1 paragraph-level recall rather than boundary modeling choices. For each propaganda-positive paragraph P with predicted techniques T_P, we run the following steps for each t ∈ T_P:

(1) Prompt construction: We build a structured prompt containing (i) a language-specific expert persona, (ii) the paragraph text, (iii) a concise definition excerpt for technique t, (iv) detection criteria, and (v) keyword references distilled from the annotation guidelines.

(2) Span generation: We call GPT-4o-mini to output JSON-formatted character spans:

{"positions": [{"start": int, "end": int, "confidence": float, "text": str}]}

(3) Post-processing: We filter spans by confidence threshold (≥ 0.7), validate character offsets, and deduplicate overlapping spans with identical technique labels.

## Error Propagation and Span-Level Difficulty

Span F1 is substantially lower than paragraph-level Macro F1 due to (i) cascading errors when Stage 1 misses propaganda paragraphs, (ii) strict character-level overlap matching that penalizes small boundary shifts, and (iii) frequent cross-sentence gold spans that are harder to reproduce from paragraph-based prompting.

# F TRANSLATION ARTIFACTS: RISKS AND THREATS TO VALIDITY

Translation-based augmentation can introduce systematic artifacts that shift surface realizations away from native usage. We summarize three key risks:

• Literalness and stylistic flattening: idioms, metaphors, and emotional intensity may be paraphrased or weakened, potentially diluting cues for techniques such as Loaded_Language.

• Syntactic interference: source-language word order patterns may leak into Slavic targets, creating a mismatch between translated training and native test distributions.

• Cultural localization gap: culture-specific references and allusions may lose salience after translation, weakening technique-specific rhetorical signals.

We therefore treat translation as an empirical factor and quantify its impact through controlled comparisons and technique-level error analysis in later chapters.
