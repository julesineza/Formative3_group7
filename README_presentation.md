# Formative Assessment 3
## Probability Distributions (EM) · Bayesian Inference · Gradient Descent

**Group 7:** Jonathan Ishimwe · Angel Ingabire · Jules Ineza
**Tools:** Python, NumPy, SciPy, Matplotlib, pandas.

| Part | Topic | Method | Where |
|------|-------|--------|-------|
| 1 | Mothers vs. Children heights | Gaussian Mixture fitted with **EM from scratch** | Notebook, Part 1 |
| 2 | IMDb sentiment | **Bayes' theorem**, basic Python only | Notebook, Part 2 |
| 3 | Gradient descent by hand | Matrix form, 3 updates (one per member) | Handwritten PDF |
| 4 | Gradient descent in code | NumPy + **SciPy** derivative check + Matplotlib | Notebook, Part 4 |
---

# Part 1: Probability Distributions and EM

## The problem
- Heights of **mothers** and **children** (GaltonFamilies dataset), pooled with **no labels**.
- Treat the pooled data as a **mixture of two Gaussians** and let the algorithm find the groups.

## Data preparation
| Sample | Size | How built |
|--------|------|-----------|
| Mothers | **205** | one record per family (a mother repeats once per child, so we avoid double counting) |
| Children | **934** | one record per child |
| **Pooled, shuffled** | **1,139** | mean 66.252 in, variance 12.580, std 3.547 |

Mother and child distributions **overlap heavily**, so a height alone cannot tell you the group with certainty.

## Model
Each height is drawn from component *k* with weight πₖ:

```
p(x) = Σₖ πₖ · N(x | μₖ, σₖ²)          Σ πₖ = 1
N(x | μ, σ²) = 1/√(2πσ²) · exp( −(x−μ)² / 2σ² )
```
Parameters: **means μₖ**, **variances σₖ²**, **mixing weights πₖ**.
Convention: Component 1 = higher mean ("Children"), Component 2 = lower mean ("Mothers").

## The EM algorithm

**Initialization:** μ₁ = 75th percentile (69.0), μ₂ = 25th percentile (63.5), σ₁² = σ₂² = global variance (12.58), π₁ = π₂ = 0.5.

**E-step:** responsibility of component *k* for each height
```
γᵢₖ = πₖ N(xᵢ | μₖ, σₖ²) / Σⱼ πⱼ N(xᵢ | μⱼ, σⱼ²)
```
This is a soft assignment: each row of γ sums to 1.

**M-step:** re-estimate parameters using γ as weights
```
Nₖ = Σᵢ γᵢₖ        μₖ = Σ γᵢₖ xᵢ / Nₖ
σₖ² = Σ γᵢₖ (xᵢ − μₖ)² / Nₖ        πₖ = Nₖ / n
```

**Stopping rule:** log-likelihood
```
ln L = Σᵢ ln( Σₖ πₖ N(xᵢ | μₖ, σₖ²) )
```
Stop when the change in ln L is below **1e-6**. EM never decreases ln L.

## Tracking table (from our code)
| Iteration | μ₁ (Children) | μ₂ (Mothers) | σ₁² | σ₂² | π₁ | π₂ | Log-likelihood |
|-----------|------|------|-------|------|-------|-------|----------|
| 0 (init) | 69.0000 | 63.5000 | 12.5802 | 12.5802 | 0.5000 | 0.5000 | −3115.4014 |
| 1 | 68.2536 | 64.3037 | 9.8201 | 7.5712 | 0.4933 | 0.5067 | −3046.1412 |
| 2 | 68.3165 | 64.2587 | 10.0323 | 6.9517 | 0.4913 | 0.5087 | −3044.0230 |
| **386 (converged)** | **69.7856** | **64.2993** | **5.4747** | **5.7935** | **0.3559** | **0.6441** | **−3035.7484** |

## Final fitted mixture
| | Mean | Variance | Weight |
|---|------|----------|--------|
| Component 1 (Children) | 69.786 in | 5.475 | 0.356 |
| Component 2 (Mothers) | 64.299 in | 5.793 | 0.644 |

Equal-posterior decision boundary: **67.65 in** (global mean: 66.25 in).

## Classifying a test height (Bayes' theorem)
```
P(component k | h) = πₖ N(h | μₖ, σₖ²) / p(h)
```
Example, **h = 63 in**: P(Children) = **0.97%**, P(Mothers) = **99.03%**.

**Live demo:** change `TEST_HEIGHT` (and `UNIT` = "in" or "cm") in the notebook cell of Section 9 and re-run that cell. It prints both posterior probabilities, which sum to 1.

## Why not just split at the global mean?
1. **Ignores the variances:** assumes both groups have equal spread and symmetric shape.
2. **Ignores the mixing weights:** if one group is bigger, the best boundary moves away from the overall mean (EM boundary 67.65 in vs. mean 66.25 in).
3. **Hard assignment:** heights near the cut get a firm yes/no with no uncertainty, which risks misclassification. EM gives **probabilities** (soft assignment).
4. **Means of the two piles are biased:** each pile is cut off at the split, so its mean is pulled toward the cutoff. EM uses every point in both components, weighted by responsibility.

## Part 1 conclusions
- EM recovered two overlapping groups from unlabeled data, using only the heights.
- Log-likelihood rose from −3115.40 to −3035.75. Most of the gain comes in the first update.
- Component names are only labels. The children sample contains males and females, so the components are overlapping height groups, not a clean demographic split.

---

# Part 2: Bayesian Probability for Sentiment Analysis

## Setup
- **Dataset:** IMDb Movie Reviews, **50,000 reviews**, columns `review` and `sentiment`.
- **Prior:** P(Positive) = **0.50** (25,000 positive of 50,000).
- **We compute P(Positive | keyword) only** (as the assignment requires, not both).
- **Positive keywords:** excellent, amazing, wonderful
- **Negative keywords:** terrible, awful, worst
- **Implementation:** basic Python only (`csv`, `re`, `math`). No ML libraries. Whole-word, case-insensitive match; a keyword counts once per review.

## Bayes' theorem
```
P(Positive | W) = P(W | Positive) × P(Positive) / P(W)
```
| Term | Meaning | How we compute it |
|------|---------|-------------------|
| Prior P(Positive) | share of positive reviews | positive reviews / total reviews |
| Likelihood P(W\|Positive) | how often W appears in positive reviews | positive reviews containing W / positive reviews |
| Marginal P(W) | how often W appears in any review | reviews containing W / total reviews |
| **Posterior P(Positive\|W)** | chance a review is positive given W | likelihood × prior / marginal |

## Results
| Keyword | Prior | Likelihood P(W\|Pos) | Marginal P(W) | **Posterior P(Pos\|W)** | Positive / all matches |
|---------|-------|------|------|-----------|------------|
| excellent | 0.5 | 0.114720 | 0.071040 | **0.807432** | 2868 / 3552 |
| amazing | 0.5 | 0.067160 | 0.043160 | **0.778035** | 1679 / 2158 |
| wonderful | 0.5 | 0.090320 | 0.055600 | **0.812230** | 2258 / 2780 |
| terrible | 0.5 | 0.015320 | 0.054020 | **0.141799** | 383 / 2701 |
| awful | 0.5 | 0.011360 | 0.057660 | **0.098508** | 284 / 2883 |
| worst | 0.5 | 0.016440 | 0.088680 | **0.092693** | 411 / 4434 |

**Worked example (excellent):** 0.114720 × 0.5 / 0.071040 = **0.807432 (80.74%)**

## Findings
- Positive keywords move the probability **above the 0.5 prior** (0.78 to 0.81).
- Negative keywords move it **far below** (0.09 to 0.14).
- **Check:** each posterior equals (positive matches / all matches). The code asserts this.
- **Limits:** the method ignores negation, sarcasm, word order and context. These are dataset associations, not guaranteed predictions.

---

# Part 3: Gradient Descent by Hand (on paper)

**Model:** ŷ = Xm + b, with X = [[1, 3], [4, 10]], y = [5, 6]ᵀ, initial m = [−1, 2]ᵀ, b = [1, 1]ᵀ.
**Matrix form throughout. Three updates, one per member. See the handwritten PDF.**

**Cost:** J = (1/n) eᵀe, where e = Xm + b − y

**Chain rule:**
```
∂J/∂e = (2/n) e
∂e/∂m = X     →   ∂J/∂m = (2/n) Xᵀ e
∂e/∂b = I     →   ∂J/∂b = (2/n) e
```
**Update:** m ← m − α ∂J/∂m, b ← b − α ∂J/∂b, with α = 0.01.

---

# Part 4: Gradient Descent in Code

## Same problem, in Python
| Item | Value |
|------|-------|
| X | [[1, 3], [4, 10]] |
| y | [[5], [6]] |
| initial m | [[−1], [2]] |
| initial b | [[1], [1]] (each observation has its own bias) |
| learning rate α | 0.01 |
| updates | 3 |

## Steps visible in the code
1. Predict: `y_hat = X @ m + b`
2. Error: `error = y_hat - y`
3. Cost: `J = (error.T @ error) / n`
4. Gradients: `grad_m = (2/n) * X.T @ error`, `grad_b = (2/n) * error`
5. **SciPy check:** `scipy.differentiate.derivative` gives a numerical gradient for each parameter. We compare it with the analytic one.
6. Update: `m = m - α * grad_m`, `b = b - α * grad_b`

## Results
| Iter | m | b | ŷ | error | MSE J |
|------|---|---|---|-------|-------|
| 0 | [−1.00, 2.00] | [1.00, 1.00] | [6.00, 17.00] | [1.00, 11.00] | **61.0000** |
| 1 | [−1.45, 0.87] | [0.99, 0.89] | [2.15, 3.79] | [−2.85, −2.21] | **6.5033** |
| 2 | [−1.3331, 1.1765] | [1.0185, 0.9121] | [3.2149, 7.3447] | [−1.7851, 1.3447] | **2.4974** |
| 3 | [−1.3690, 1.0956] | [1.0364, 0.8987] | [2.9541, 6.3783] | [−2.0459, 0.3783] | **2.1645** |

**Gradients and SciPy check**
| Iter | dJ/dm (analytic) | dJ/db (analytic) | SciPy (dm₁, dm₂, db₁, db₂) |
|------|------|------|------------|
| 0 | [45, 113] | [1, 11] | 45.0, 113.0, 1.0, 11.0 ✓ |
| 1 | [−11.69, −30.65] | [−2.85, −2.21] | −11.69, −30.65, −2.85, −2.21 ✓ |
| 2 | [3.5937, 8.0917] | [−1.7851, 1.3447] | 3.5937, 8.0917, −1.7851, 1.3447 ✓ |

**Final predictions:** [2.954, 6.378] vs. actual y = [5, 6].

## Plots
- Parameters (m₁, m₂, b₁, b₂) over iterations
- MSE over iterations: **61.00 → 6.50 → 2.50 → 2.16**

## Trend
- The error falls at every update, so m and b are moving in a direction that reduces it.
- The biggest drop is the first step, because the initial gradients are large (45 and 113).
- Steps shrink as the gradient shrinks. After 3 updates the model has not converged yet, and more updates would keep lowering J.
- The numerical (SciPy) and analytical gradients agree, which confirms the derivation.

---

# Summary
| Part | Result |
|------|--------|
| 1 | EM separated unlabeled heights into two Gaussians (69.79 in, weight 0.36 and 64.30 in, weight 0.64) after 386 iterations; a 63 in test height is 99.03% Component 2 |
| 2 | One keyword moves belief from 50% to about 80% (positive words) or about 10% (negative words) |
| 3 | Manual matrix-form gradient descent, with the chain rule shown for ∂J/∂m and ∂J/∂b |
| 4 | Same descent in code: MSE 61.00 → 2.16 in 3 updates, SciPy confirms every gradient |

# Running the notebook
1. Place `GaltonFamilies.csv` next to the notebook (Part 1).
2. Part 2 uploads the IMDb CSV through Colab's file upload (`google.colab.files`).
3. Run all cells top to bottom. To test a new height, edit `TEST_HEIGHT` in Part 1, Section 9.
