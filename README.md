# Formative3_group7
## Part 1: Probability Distributions — Expectation-Maximization (EM) Algorithm

This part implements the Expectation-Maximization (EM) algorithm from scratch in Python to model a mixture of two Gaussian distributions using mothers' and children's height data from the `GaltonFamilies.csv` dataset.

The implementation combines the heights into an unlabeled dataset and estimates the Gaussian mixture parameters: means (\(\mu\)), variances (\(\sigma^2\)), and mixing coefficients (\(\pi\)).

The algorithm alternates between two steps:
- **Expectation (E-step):** Calculates the posterior probabilities (responsibilities) that each height belongs to each Gaussian component.
- **Maximization (M-step):** Updates the means, variances, and mixing coefficients using the calculated responsibilities.

The process continues until the log-likelihood converges. The results include a parameter-tracking table for initialization and the first two iterations, visualizations of the height distribution and fitted Gaussian mixture, and posterior probabilities for a test height.

The implementation also explains why simply splitting the data at the global mean is not a reliable alternative to EM, since hard assignments can misclassify observations that could plausibly belong to either component.

**Tools and libraries:** Python, NumPy, Pandas, Matplotlib, and Seaborn.