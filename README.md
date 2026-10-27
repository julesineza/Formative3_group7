# Formative 3 — Probability, Bayes, and Gradient Descent

This repository contains Group 7's submission for the Mathematics for Machine
Learning formative assignment. It is organized as a presentation guide: a
facilitator can start with the overview below and then open the relevant
notebook, dataset, or supporting document for each part.

## Quick navigation

| Area | Main resource | What to review |
| --- | --- | --- |
| Part 1 — Gaussian mixtures and EM | [`notebooks/Part1.ipynb`](./notebooks/Part1.ipynb) | EM from scratch, convergence, mixture plots, and posterior classification |
| Part 2 — Bayesian probability | `notebooks/Part2.ipynb` *(to be added)* | Keyword probabilities and Bayes' theorem |
| Part 3 — Manual gradient descent | [`docs/manual-gradient-descent.pdf`](./docs/manual-gradient-descent.pdf) *(to be added)* | Handwritten matrix calculations and member-by-member updates |
| Part 4 — Gradient descent in code | [`Gradient_descent/gradient_descent.ipynb`](./Gradient_descent/gradient_descent.ipynb) | SciPy derivative check, updates, predictions, and plots |
| Manual-calculation backup | [`docs/manual-calculations-supplement.pdf`](./docs/manual-calculations-supplement.pdf) *(to be added)* | Second PDF containing the supplementary/manual working |
| Contribution evidence | [`docs/contributions.pdf`](./docs/contributions.pdf) *(to be added)* | Team contribution record and evidence |
| Data | [`data/GaltonFamilies.csv`](./data/GaltonFamilies.csv) | Galton parent/child height data used in Part 1 |

> **Submission status:** Files marked “to be added” are intentional placeholders
> for the final deliverables. Add the files at the displayed paths so the links
> become active before submission.

## Assignment objectives

The assignment develops practical understanding of:

1. Probability distributions and the difference between hard and probabilistic
   assignments.
2. Bayesian probability applied to a real-world text-classification problem.
3. Manual, matrix-based gradient-descent calculations.
4. A Python implementation of gradient descent for linear regression using
   NumPy, SciPy, and Matplotlib.

All code should remain modular and follow the DRY principle. During the
presentation, run the notebooks and explain the mathematics rather than
showing only final screenshots.

## Part 1 — Probability distributions and EM

### Question

The Galton dataset contains heights from two populations. For this project,
the selected populations are **mothers and children** (the same workflow can
be adapted to fathers and children). The observations are treated as an
unlabelled mixture of two Gaussian distributions.

Dataset source: [Galton parents-versus-children heights on Kaggle](https://www.kaggle.com/datasets/jacopoferretti/parents-heights-vs-children-heights-galton-data).

### Implementation

Open [`notebooks/Part1.ipynb`](./notebooks/Part1.ipynb). The notebook should
show:

- data loading and preparation from [`data/GaltonFamilies.csv`](./data/GaltonFamilies.csv);
- an Expectation-Maximization algorithm implemented from scratch;
- the E-step responsibilities and M-step parameter updates;
- estimated means (`μ`), variances (`σ²`), and mixing proportions (`π`);
- log-likelihood values for initialization and at least the first two
  iterations;
- a completed parameter-tracking table;
- mixture/distribution visualizations; and
- posterior probabilities for a supplied test height.

### Presentation question: split at the global mean?

Do **not** simply split the observations into two piles at the global mean and
then calculate the mean of each pile as the primary model. That approach makes
hard assignments before accounting for uncertainty, can be distorted by
unequal mixture proportions or overlapping distributions, and does not
estimate the variances or mixing proportions coherently. EM instead assigns a
responsibility to each component, updates all parameters together, and
maximizes the mixture log-likelihood. A global-mean split can be shown as a
baseline, but its resulting group means must be compared with the EM result.

### Required live demonstration

Run the classifier with a random height provided by the facilitator and print
the exact posterior probabilities:

- `P(Child | height)`
- `P(Mother | height)`

Use the population names consistently in the notebook and presentation. Do not
label the second class “basketball player” unless a separate basketball-player
dataset and model have actually been added.

## Part 2 — Bayesian probability

Use the [IMDb 50K movie reviews dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

Open [`notebooks/Part2.ipynb`](./notebooks/Part2.ipynb) once it has been added.
The notebook and presentation must document:

1. Two to four keywords associated with positive sentiment and two to four
   associated with negative sentiment.
2. One selected conditional direction: either `P(Positive | keyword)` **or**
   `P(Negative | keyword)`, not both.
3. A probability table for every selected keyword containing:
   - prior: `P(Positive)` (or the selected class prior);
   - likelihood: `P(keyword | Positive)`;
   - marginal: `P(keyword)`; and
   - posterior: `P(Positive | keyword)`.
4. A basic-Python implementation of Bayes' theorem. External machine-learning
   libraries must not be used for this part.

Explain how the chosen probabilities are counted from the dataset and state
any handling of repeated words, case, punctuation, and zero counts.

## Part 3 — Manual calculation of gradient descent

The required handwritten work belongs in
[`docs/manual-gradient-descent.pdf`](./docs/manual-gradient-descent.pdf) and
the supplementary calculations belong in
[`docs/manual-calculations-supplement.pdf`](./docs/manual-calculations-supplement.pdf).

Use matrix multiplication throughout; do not silently replace the vectors by
unrelated scalar calculations. The exercise uses:

```text
y = m1*x1 + m2*x2 + b
m = [-1, 2]
b = [1, 1]
X = [[1, 3],
     [4, 10]]
y = [5, 6]
```

The PDFs should visibly include:

- predictions `ŷ` from the current parameters;
- the Mean Squared Error cost;
- the derivation of the gradients with respect to `m` and `b`;
- the complete intermediate arithmetic for every update;
- a number of updates equal to the number of group members, with every member
  completing at least one iteration; and
- an explanation of how `m`, `b`, and the error change over time.

## Part 4 — Gradient descent in code

Open [`Gradient_descent/gradient_descent.ipynb`](./Gradient_descent/gradient_descent.ipynb)
for the executable implementation. Its local guide is
[`Gradient_descent/README.md`](./Gradient_descent/README.md).

The notebook should make each step visible:

1. calculate predictions;
2. calculate the MSE;
3. calculate the matrix gradients;
4. use SciPy to check the derivative of the equation/cost function;
5. update `m` and `b`;
6. repeat for the required number of iterations; and
7. calculate final predictions.

It must also generate:

- a plot of `m` and `b` over the iterations; and
- a separate plot of error/MSE over the iterations.

Current implementation dependencies are Python, NumPy, SciPy, and Matplotlib.
Run the notebook from top to bottom using the project environment.

## Deliverables checklist

- [x] Root README used as the facilitator's navigation guide.
- [x] Part 1 EM notebook.
- [ ] Part 2 Bayesian-probability notebook: [`notebooks/Part2.ipynb`](./notebooks/Part2.ipynb).
- [ ] Neatly written Part 3 manual-calculation PDF:
      [`docs/manual-gradient-descent.pdf`](./docs/manual-gradient-descent.pdf).
- [ ] Supplementary/manual-calculation PDF:
      [`docs/manual-calculations-supplement.pdf`](./docs/manual-calculations-supplement.pdf).
- [x] Part 4 gradient-descent notebook.
- [ ] Contribution-evidence PDF:
      [`docs/contributions.pdf`](./docs/contributions.pdf).
- [ ] Relevant images/plots embedded in the notebooks.
- [ ] Git history showing each member's contribution.

## Suggested presentation route

1. Introduce the assignment and repository map.
2. Open Part 1 and explain why EM is preferable to a global-mean split.
3. Run a facilitator-supplied height and interpret both posterior probabilities.
4. Present Part 2's keyword table and explain the selected Bayes calculation.
5. Walk through the handwritten matrix calculations in the two PDFs.
6. Run Part 4, compare the code output with the manual work, and interpret both
   plots.
7. Finish with the contribution evidence and invite questions.

## Team contribution evidence

See [`docs/contributions.pdf`](./docs/contributions.pdf) *(to be added)* for
the contribution record. The repository's commit history is the source of
truth for code changes; each member should use their own account when making
commits and should be prepared to explain the work they contributed.
