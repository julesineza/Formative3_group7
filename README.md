# Formative 3 — Probability, Bayes, and Gradient Descent

## Quick navigation

| Area | Main resource | What to review |
| --- | --- | --- |
| Part 1 — Gaussian mixtures and EM | [`notebooks/Part1.ipynb`](./notebooks/Part1.ipynb) | EM from scratch, convergence, mixture plots, and posterior classification |
| Part 2 — Bayesian probability | [`bayesian_sentiment.py`](./bayesian_sentiment.py) | Keyword selection, Bayes' theorem, probabilities, and formatted results |
| Part 2 tests | [`test_bayesian_sentiment.py`](./test_bayesian_sentiment.py) | Standard-library tests for the Bayesian implementation |
| Part 3 — Manual gradient descent | [`formative3 part 3 group7.pdf`](./formative3%20part%203%20group7.pdf) | Handwritten matrix calculations and member-by-member updates |
| Part 4 — Gradient descent in code | [`Gradient_descent/gradient_descent.ipynb`](./Gradient_descent/gradient_descent.ipynb) | SciPy derivative check, updates, predictions, and plots |

| Resource | File / Link | Description |
|---|---|---|
| Combined Assignment Notebook | [Formative_Assessment_3_Probability,_Bayesian_Inference,_and_Gradient_Descent_Group7.ipynb](./Formative_Assessment_3_Probability,_Bayesian_Inference,_and_Gradient_Descent_Group7.ipynb) | Consolidated notebook for the assignment |
| Presentation Guide | [README_presentation.md](./README_presentation.md) | Presentation-focused explanations and results |
| Assignment Task Sheet | [BSE Group Assignments - Task Sheet](./BSE%20Group%20Assignments%20_%20Task%20Sheet_%5BMathematics_for_Machine_Learning_Formative3_Cohort%201_Team7%5D%20-%201.pdf) | task sheet |
| Galton Data | [GaltonFamilies.csv](./data/GaltonFamilies.csv) | Parent/child height data used in Part 1 |
| IMDb Data | [IMDB Dataset.csv](./IMDB%20Dataset.csv) | 50,000 movie reviews used in Part 2 |
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

