# Hebbian Fast-Weight MLP (10K+ Token Fact Verification)

**Field**: Inference-Time Learning  
**Not ML. Not DL. No gradients. No training.**

---

## The Problem It Solves

Standard MLPs in transformer-like architectures are frozen at inference time. This project asks: what if the weights learned during the forward pass itself, with no gradients, no backpropagation, and no KV cache? This is **Inference-Time Learning**. By augmenting standard feed-forward projections with sequence-specific plastic fast-weights, we can store and retrieve specific associations dynamically over long contexts (10,000+ tokens) with constant O(1) memory space.

---

## Architectural Overview

Standard MLPs do not change when processing sequences. The **Hebbian Fast-Weight MLP** incorporates dynamic, sequence-specific, and real-time plastic fast-weights (`W_plastic`) that are updated in-place during the forward pass. These weights act as a high-capacity, training-free associative memory.

### Forward Pass Mechanism
During each time-step `t`:
1. The input `x` is normalized using an input RMSNorm layer:
   `x_norm = RMSNorm(x)`
2. The output pre-activations are computed by summing the static projection and the dynamic plastic projection:
   `y_raw = W_static * x_norm + bias_static + W_plastic * x_norm`
3. The raw pre-activations are normalized via an output RMSNorm layer to ensure spectral containment and control magnitude growth:
   `y = RMSNorm(y_raw)`

---

## Contrastive Novelty Gating

Our core algorithmic contribution is **Contrastive Novelty Gating**, which solves the problem of memory saturation under continuous noise. 

During background sequence processing (distractor phases), before writing to the weight matrix, we measure the cosine divergence between the static projection output (`y_static`) and the total projection output (`y`):
`novelty = 1.0 - abs(cos_sim(y_static, y))`

Where:
`cos_sim(a, b) = dot_product(a, b) / (L2_norm(a) * L2_norm(b))`

### The Cosine Divergence Logic:
* If the incoming input is **familiar** or carries no new association, the plastic weights will not trigger a meaningful projection change. The static and total output vectors remain highly aligned, yielding a low novelty score, and the update is **bypassed**.
* If the input is **novel**, the plastic weights generate a projection that shifts the total output vector away from the static prior, producing high novelty. The update gate is then triggered to write this association into the fast-weights.
This is a self-calibrating, parameter-free write protection system.

---

## The Failure and the Fix (Narrative)

### The Failure: Catastrophic Forgetting
During our initial trials, when we enabled active distractor updates (`update_gate = 0.1` continuously for all 10,000+ background tokens), the model suffered from complete catastrophic forgetting. Because the weight norm is strictly bounded (`max_norm = 2.0`) to guarantee stability, the sequential updates from random noise completely overwrote and washed out the target association injected at step 0, resulting in **0% retrieval accuracy** at retrieval step 10,004.

### The Fix: Contrastive Sparsification
To fix this, we integrated **Contrastive Novelty Gating** alongside **Stochastic Sparsification**:
1. **Sparsification**: We filter out 98% of active distractor steps stochastically (`sparsity_update_rate = 0.02`), bypassing the updates entirely to prevent CPU computational lag.
2. **Contrastive Filtering**: For the remaining 2% of distractor tokens, we compute the novelty index. If it is below a threshold (`novelty_threshold = 0.5`), the update is skipped.

This combination of filters protects the original target association, yielding **100% retrieval accuracy** and reducing sequential execution time to just **~10.4 seconds** on a standard CPU.

---

## Experimental Results

Below are the raw results compiled from the 10,005-token needle-in-a-haystack verification runs.

### Table 1: Retrieval Performance (10,005 Tokens)

| Configuration | Model | Retrieved Token | Target Probability | Cosine Similarity | Success |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Config 1: Oja, Passive** | Hebbian MLP | 999 | 100.00% | 0.9067 | **TRUE** |
| | Static Control | 506 | 0.00% | 0.0672 | **FALSE** |
| **Config 2: Oja, Active+Gate** | Hebbian MLP | 999 | 100.00% | 0.8930 | **TRUE** |
| | Static Control | 894 | 0.00% | -0.0685 | **FALSE** |
| **Config 3: Decay, Passive** | Hebbian MLP | 999 | 100.00% | 0.8959 | **TRUE** |
| | Static Control | 975 | 0.00% | 0.0857 | **FALSE** |

### Table 2: Frobenius Norm Stability Audit (Target max_norm = 2.0)

| Configuration | Min Norm | Max Norm | Mean Norm | Std Dev | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Config 1: Oja, Passive** | 2.0000 | 2.0000 | 2.0000 | 0.0000 | **STABLE** |
| **Config 2: Oja, Active+Gate**| 2.0000 | 2.0000 | 2.0000 | 0.0000 | **STABLE** |
| **Config 3: Decay, Passive** | 2.0000 | 2.0000 | 2.0000 | 0.0000 | **STABLE** |

---

## Mathematical Formulations

To prevent spectral explosion over long sequence lengths, we implement multiple learning rules alongside Frobenius norm projection.

### 1. Oja's Multi-dimensional Rule
Enforces convergence toward the principal components of the inputs, naturally bounding weight scale.
`dW = eta * (y * x_norm.T - (y * y.T) * W_plastic)`

### 2. Oja's Scalar-decay Rule
Reduces the decay term calculation cost from O(d^2) to O(d) while preserving self-normalization.
`dW = eta * (y * x_norm.T - (norm(y)^2) * W_plastic)`

### 3. Gated Decay Rule
Classic fast-weight update with explicit exponential forgetting.
`dW = -decay * W_plastic + eta * (y * x_norm.T)`

### Frobenius Norm Containment
At each update step, if the Frobenius norm of `W_plastic` exceeds `max_norm`:
`W_plastic = W_plastic * (max_norm / Frobenius_norm(W_plastic))`

---

## Running the Verification

To run the verification suite containing the optimized Contrastive Gating configurations:

```powershell
python hebbian_mlp/verify_10k.py
```
