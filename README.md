# Formative 3 — Probability, Bayes, and Gradient Descent

This repository contains Group 7's submission for the Mathematics for Machine
Learning formative assignment. It is organized as a facilitator-friendly
presentation guide: start with the navigation table, then open the relevant
notebook, Python implementation, dataset, or supporting document for each
part.

## Quick navigation

| Area | Main resource | What to review |
| --- | --- | --- |
| Part 1 — Gaussian mixtures and EM | [`notebooks/Part1.ipynb`](./notebooks/Part1.ipynb) | EM from scratch, convergence, mixture plots, and posterior classification |
| Part 2 — Bayesian probability | [`bayesian_sentiment.py`](./bayesian_sentiment.py) | Keyword selection, Bayes' theorem, probabilities, and formatted results |
| Part 2 tests | [`test_bayesian_sentiment.py`](./test_bayesian_sentiment.py) | Standard-library tests for the Bayesian implementation |
| Part 3 — Manual gradient descent | [`formative3 part 3 group7.pdf`](./formative3%20part%203%20group7.pdf) | Handwritten matrix calculations and member-by-member updates |
| Part 4 — Gradient descent in code | [`Gradient_descent/gradient_descent.ipynb`](./Gradient_descent/gradient_descent.ipynb) | SciPy derivative check, updates, predictions, and plots |
| Part 4 guide | [`Gradient_descent/README.md`](./Gradient_descent/README.md) | Equations, recorded results, and instructions for running the notebook |
| Combined assignment notebook | [`Formative_Assessment_3_Probability,_Bayesian_Inference,_and_Gradient_Descent_Group7.ipynb`](./Formative_Assessment_3_Probability,_Bayesian_Inference,_and_Gradient_Descent_Group7.ipynb) | Consolidated notebook for the assignment |
| Presentation guide | [`README_presentation.md`](./README_presentation.md) | Presentation-focused explanations and results |
| Assignment task sheet | [`BSE Group Assignments - Task Sheet`](./BSE%20Group%20Assignments%20_%20Task%20Sheet_%5BMathematics_for_Machine_Learning_Formative3_Cohort%201_Team7%5D%20-%201.pdf) | Original requirements and marking context |
| Contribution evidence | [`docs/contributions.pdf`](./docs/contributions.pdf) *(to be added)* | Team contribution record and evidence |
| Galton data | [`data/GaltonFamilies.csv`](./data/GaltonFamilies.csv) | Parent/child height data used in Part 1 |
| IMDb data | [`IMDB Dataset.csv`](./IMDB%20Dataset.csv) | 50,000 movie reviews used in Part 2 |

> **Submission status:** The manual-calculation PDF and combined assignment
> notebook are present in the repository. The contribution PDF is still a
> placeholder; add it at the displayed path so its link becomes active before
> submission.

## Assignment objectives

The assignment develops practical understanding of:

1. Probability distributions and the difference between hard and probabilistic
   assignments.
2. Bayesian probability applied to a real-world text-classification problem.
3. Manual, matrix-based gradient-descent calculations.
4. A Python implementation of gradient descent for linear regression using
   NumPy, SciPy, and Matplotlib.

All code should remain modular and follow the DRY principle. During the
presentation, run the notebooks and Python implementation and explain the
mathematics rather than showing only final screenshots.

## Part 1 — Probability distributions and EM

### Question

The Galton dataset contains heights from two populations. This project models
**mothers and children** as an unlabelled mixture of two Gaussian
distributions. The same workflow can be adapted to fathers and children.

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

The implementation uses the [IMDb 50K movie reviews dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews),
stored in the repository as [`IMDB Dataset.csv`](./IMDB%20Dataset.csv).

Run [`bayesian_sentiment.py`](./bayesian_sentiment.py) from the repository root:

```bash
python bayesian_sentiment.py
```

To use another location for the dataset:

```bash
python bayesian_sentiment.py "/path/to/IMDB Dataset.csv"
```

The CSV must contain `review` and `sentiment` columns, with `positive` or
`negative` sentiment values. The script reports **only**
`P(Positive | keyword)`, as required by the assignment.

### Keywords and calculation

The selected keywords are:

- Positive indicators: `excellent`, `amazing`, `wonderful`
- Negative indicators: `terrible`, `awful`, `worst`

A keyword is counted at most once per review, case-insensitively, using whole
word matching. For each keyword, the script reports:

- prior: `P(Positive)`;
- likelihood: `P(keyword | Positive)`;
- marginal: `P(keyword)`; and
- posterior: `P(Positive | keyword)`.

With `N` as the total number of reviews, `N_positive` as positive reviews,
`N_keyword` as reviews containing the keyword, and `N_keyword_positive` as
positive reviews containing it:

```text
P(Positive) = N_positive / N
P(keyword | Positive) = N_keyword_positive / N_positive
P(keyword) = N_keyword / N
P(Positive | keyword) =
    P(keyword | Positive) * P(Positive) / P(keyword)
```

The implementation uses only basic Python operations and streams the CSV
instead of loading all review text into memory. It explicitly reports invalid
headers, invalid sentiment values, missing files, and keywords that never
occur.

Run the tests with:

```bash
python -m unittest -v test_bayesian_sentiment.py
```

## Part 3 — Manual calculation of gradient descent

The required handwritten work is provided in
[`formative3 part 3 group7.pdf`](./formative3%20part%203%20group7.pdf). If a
second manual-calculation PDF is required, add it at
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
- complete intermediate arithmetic for every update;
- a number of updates equal to the number of group members, with every member
  completing at least one iteration; and
- an explanation of how `m`, `b`, and the error change over time.

## Part 4 — Gradient descent in code

Open [`Gradient_descent/gradient_descent.ipynb`](./Gradient_descent/gradient_descent.ipynb)
for the executable implementation. Its local guide is
[`Gradient_descent/README.md`](./Gradient_descent/README.md).

The notebook makes each step visible:

1. calculate predictions;
2. calculate the MSE;
3. calculate matrix gradients;
4. use SciPy to check the derivative of the equation/cost function;
5. update `m` and `b`;
6. repeat for the required number of iterations; and
7. calculate final predictions.

It also generates:

- a plot of `m` and `b` over the iterations; and
- a separate plot of error/MSE over the iterations.

The implementation uses Python, NumPy, SciPy, and Matplotlib. See
[`Gradient_descent/requirements.txt`](./Gradient_descent/requirements.txt) for
the listed dependencies.

## Deliverables checklist

- [x] Root README used as the facilitator's navigation guide.
- [x] Part 1 EM notebook.
- [x] Part 2 Bayesian implementation.
- [x] Part 2 automated tests.
- [x] Neatly written Part 3 manual-calculation PDF:
      [`formative3 part 3 group7.pdf`](./formative3%20part%203%20group7.pdf).
- [ ] Supplementary/manual-calculation PDF:
      [`docs/manual-calculations-supplement.pdf`](./docs/manual-calculations-supplement.pdf).
- [x] Combined assignment notebook:
      [`Formative_Assessment_3_Probability,_Bayesian_Inference,_and_Gradient_Descent_Group7.ipynb`](./Formative_Assessment_3_Probability,_Bayesian_Inference,_and_Gradient_Descent_Group7.ipynb).
- [x] Part 4 gradient-descent notebook.
- [ ] Contribution-evidence PDF:
      [`docs/contributions.pdf`](./docs/contributions.pdf).
- [ ] Relevant images/plots embedded in the notebooks.
- [ ] Git history showing each member's contribution.

## Suggested presentation route

1. Introduce the assignment and repository map.
2. Open Part 1 and explain why EM is preferable to a global-mean split.
3. Run a facilitator-supplied height and interpret both posterior probabilities.
4. Run Part 2, review the keyword probability output, and explain the selected
   Bayes calculation.
5. Walk through the handwritten matrix calculations in the two PDFs.
6. Run Part 4, compare the code output with the manual work, and interpret both
   plots.
7. Finish with the contribution evidence and invite questions.

## Team contribution evidence

See [`docs/contributions.pdf`](./docs/contributions.pdf) *(to be added)* for
the contribution record. The repository's commit history is the source of
truth for code changes; each member should use their own account when making
commits and should be prepared to explain the work they contributed.
