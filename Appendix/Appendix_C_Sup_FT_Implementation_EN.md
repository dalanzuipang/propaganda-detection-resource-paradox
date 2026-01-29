# C SUP-FT IMPLEMENTATION DETAILS

**Corresponding position in main paper:**
- **Section 3.3.1** "Supervised Fine-Tuning (Sup-FT)" - This appendix provides detailed technical implementation for this subsection
- **Figure 1** - Shows the dual-encoder architecture; this appendix provides mathematical formulas and hyperparameter details
- Brief description in main text: "we use a dual-encoder design that separately encodes the paragraph and technique-focused explanations and fuses them via bidirectional cross-attention"; this appendix provides complete mathematical definitions

**Content description:**
This appendix provides complete implementation details for the Sup-FT method, including:
- Model Architecture (dual-encoder structure)
- Cross-Attention mechanism (mathematical formulas)
- Feature Fusion (feature fusion MLP)
- Training Configuration (optimizer, learning rate, batch size, etc.)
- Loss and Class Weights (weighted BCE loss function)
- Sequence Lengths and Early Stopping (sequence length and early stopping strategy)

**Corresponding main text position:**
- **Section 3.3.1 "Supervised Fine-Tuning (Sup-FT)"** (page 3): Main text briefly describes the dual-encoder architecture and cross-attention; this appendix provides complete mathematical formulas and implementation details
- **Figure 1 "Sup-FT pipeline"** (page 3): Main text illustrates the model structure; this appendix supplements the mathematical definition and parameter configuration for each component

**Note:** This appendix details the complete technical implementation of the Sup-FT method, including: model architecture (dual-encoder, cross-attention, feature fusion), training configuration (optimizer, batch size, learning rate strategy), loss function (weighted BCE and class weight calculation), sequence length and early stopping strategy. These details are crucial for experimental reproduction.

## Model Architecture

Sup-FT employs two independent XLM-RoBERTa-base encoders (125M parameters each) to process (i) paragraph text and (ii) LLM-generated technique-focused explanations. The two streams are fused via bidirectional cross-attention and a feature-fusion MLP before the final 23-label classification head.

## Cross-Attention

We use bidirectional multi-head attention with dmodel = 768 and h = 8 heads. Let H ∈ R^(B×L×768) denote encoder hidden states. Text-to-explanation attention is:

Attn_T→E(Q = H_text, K = H_expl, V = H_expl),

and explanation-to-text attention is:

Attn_E→T(Q = H_expl, K = H_text, V = H_text).

## Feature Fusion

We concatenate four [CLS] embeddings h_text; h_expl; h_T→E^[CLS]; h_E→T^[CLS] and pass them through a two-layer MLP with ReLU activations. Dropout rates are 0.2 (first layer) and 0.1 (second layer). The final classifier is a linear projection W_c ∈ R^(23×768).

## Training Configuration

We use AdamW (β1 = 0.9, β2 = 0.999, weight decay 10^-3) with learning rate 1 × 10^-5, 1000-step linear warmup, and linear decay. Physical batch size is 8 with 4-step gradient accumulation (effective batch size 32). We apply gradient clipping (0.5) and enable gradient checkpointing.

## Loss and Class Weights

We optimize weighted binary cross-entropy. Technique-specific weights are computed as:

w_t = min(3 · n_t_neg / n_t_pos, 30),

where n_t_pos and n_t_neg are positive/negative counts for technique t.

## Sequence Lengths and Early Stopping

Maximum sequence length is 256 tokens for text and 128 tokens for explanations. We train up to 10 epochs with early stopping on validation Micro F1 (patience 3). Experiments run on NVIDIA A40 GPUs (48GB).
