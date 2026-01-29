# S CROSS-LINGUISTIC TECHNIQUE TABLES

**Corresponding position in main paper:**
- **Section 6.3 "Cross-Linguistic Patterns"** (page 6): Main text mentions "language-sensitive techniques with large fluctuations (e.g., Name_Calling, Appeal_to_Hypocrisy)... comparatively robust techniques (e.g., Doubt, Loaded_Language)"; this appendix Table 24 provides cross-linguistic F1 comparison for 10 major techniques
- **Section 8.2 "Personalized Calibration" - Optional Personalization Signals** (page 8): Main text mentions "culture-related factors as hypothesis generators"; Appendix U discusses detailed personalized knobs based on cultural dimensions

**Note:** This document contains two appendix sections:
- **Appendix S (Table 24)**: Cross-linguistic F1 comparison for 10 major techniques, annotating maximum gaps and their locations (e.g., Name_Calling: PL vs RU gap=0.255)
- **Appendix U**: Exploratory cultural dimension personalization framework, proposing two potential calibration knobs (Knob 1: personal attack salience, Knob 2: uncertainty tolerance), emphasizing this is hypothesis-generating rather than empirical evidence and requires user study validation

## Table 24: Cross-linguistic F1 comparison of major techniques (Sup-FT)

| Technique | PL | EN | RU | Max Gap | Gap Occurs Between |
|-----------|----|----|----|---------|--------------------|
| Questioning_Reputation | 0.746 | 0.000* | 0.800 | 0.800 | RU vs EN |
| Consequential_Oversimplification | 0.385 | 0.000* | 0.720 | 0.720 | RU vs EN |
| Appeal_to_Authority | 0.714 | 0.333 | 0.000 | 0.714 | PL vs RU |
| Appeal_to_Hypocrisy | 0.786 | 0.125 | 0.595 | 0.661 | PL vs EN |
| Conversation_Killer | 0.727 | 0.344 | 0.611 | 0.383 | PL vs EN |
| Name_Calling | 0.904 | 0.833 | 0.649 | 0.255 | PL vs RU |
| Exaggeration | 0.710 | 0.659 | 0.514 | 0.196 | PL vs RU |
| Flag_Waving | 0.556 | 0.746 | 0.667 | 0.190 | EN vs PL |
| Doubt | 0.831 | 0.750 | 0.849 | 0.099 | RU vs EN |
| Loaded_Language | 0.883 | 0.938 | 0.864 | 0.074 | EN vs RU |

* Indicates support=0 for this technique in the test set; F1=0.000 reflects lack of evaluable samples rather than model failure.

# U CULTURAL DIMENSION-BASED PERSONALIZATION (EXPLORATORY)

We treat culture-related factors as hypothesis generators rather than explanatory evidence. As a lightweight direction for future work, one may explore using broad cultural-dimension descriptors (e.g., Hofstede-style indices) as optional configuration signals for user-facing calibration—not to claim causal effects, but to suggest where user sensitivity might plausibly differ. Any such mapping must be validated with controlled user studies.

• Potential knob 1: salience of personal targeting. For user groups that may place higher salience on individualized blame/attack, the system could adopt a slightly lower decision threshold (or higher display priority) for Name_Calling-Labeling; for user groups that may focus more on collective framing, the system could instead emphasize group-oriented techniques.

• Potential knob 2: tolerance for uncertainty cues. For users who may be more risk-averse to ambiguity, the system could increase sensitivity to Doubt-like uncertainty markers (with stricter FP control); for users who may be less sensitive to such cues, thresholds could be kept conservative to avoid over-triggering.
