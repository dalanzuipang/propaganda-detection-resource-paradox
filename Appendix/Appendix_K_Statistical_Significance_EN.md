# K SUPPORT-CONTROLLED SIGNIFICANCE CHECKS FOR ENGLISH VS. POLISH

**Corresponding position in main paper:**
- **Section 5.1** "High-Resource Paradox: Overview" - This appendix provides detailed data for the statistical tests in this paragraph
- Main text directly cites: "paired t-test: t = 0.90, df = 13, p = 0.384; Wilcoxon signed-rank test: p = 0.715"
- Main text cites: "A bootstrap estimate yields a 95% confidence interval of [−6.38%, 20.22%] with p = 0.508. The effect size is small (Cohen's d = 0.24)"

**Content description:**
This appendix provides complete statistical significance tests for English vs. Polish performance differences. By controlling for technique support (≥5), it avoids interference from split-specific support effects, including:
- Paired t-test results
- Wilcoxon signed-rank test results
- Bootstrap 95% confidence interval
- Cohen's d effect size

Conclusion: After controlling for support, Polish's advantage narrows but directional differences remain. Statistical significance is weak, but practical evidence is clear.

**Corresponding main text position:**
- **Section 5.1 "High-Resource Paradox: Overview"** (pages 4-5): Main text mentions "A robustness check controlling for technique support... paired t-test: t=0.90, df=13, p=0.384; Wilcoxon: p=0.715; Bootstrap CI: [-6.38%, 20.22%], p=0.508; Cohen's d=0.24"
- **Section 7 "English vs. Polish: Performance Comparison"** (page 7): Main text analyzes statistical significance of EN-PL gap

**Note:** This appendix provides complete statistical significance test results under the support≥5 condition. To avoid the influence of split-specific label coverage, only 14 techniques with sufficient samples in both languages underwent paired tests. Results show that while Polish is directionally higher, the difference does not reach conventional statistical significance levels (small effect size, Cohen's d=0.24), indicating that part of the headline gap is driven by sample coverage differences.

To assess whether the English–Polish Macro F1 gap reflects systematic cross-technique advantages or split-specific support effects, we additionally restrict the comparison to techniques with adequate support in both languages (support ≥ 5 on the evaluated split). Under this restriction, the gap narrows relative to the headline comparison.

Across the 14 well-supported techniques, Polish remains directionally higher, but the difference does not reach conventional statistical significance under paired tests (paired t-test: t = 0.90, df = 13, p = 0.384; Wilcoxon signed-rank test: p = 0.715). A bootstrap estimate yields a 95% confidence interval of −6.38%, 20.22% with p = 0.508. The effect size is small (Cohen's d = 0.24), suggesting practical but weak evidence given limited test support.
