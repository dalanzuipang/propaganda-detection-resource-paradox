# V EXTENDED NOTES ON TECHNIQUE–LANGUAGE–METHOD ADAPTATION

**Corresponding position in main paper:**
- **Section 8.3 "Multi-Method Strategy Selection Guide" - Method Adaptability Insight** (page 8): Main text mentions "Language–technique routing can be used: use a default method per language, but switch methods for a small set of techniques where another paradigm is consistently better (e.g., Doubt in Russian)"
- **Section 9 "Discussion"** (pages 8-9): Main text mentions "technique-level routing strategies"

**Note:** This appendix provides specific examples of technique-language-method routing strategies, including: setting default methods for Polish, English, and Russian respectively, switching methods for specific techniques (e.g., using Iter-Ens for Doubt in Russian), and confidence-based fallback mechanisms. These strategies require validation of generalization capabilities and cost-benefit trade-offs in new domains.

As a practical direction, multilingual personalization can explore a "technique–language–method" adaptation strategy. For example, one may prioritize Sup-FT for Polish as a balanced default, while applying stricter thresholds for Obfuscation-Vagueness-Confusion; for English, retain Sup-FT where it is strong (e.g., Loaded_Language) but suppress over-triggering via post-processing; for Russian, treat Doubt as a candidate for specialized routing (e.g., Iter-Ens) while using Sup-FT as the default for other techniques. Such rules require validation on new domains and explicit evaluation of inference cost, especially for multi-round ensembling.
