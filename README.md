# Formative3_group7

## Bayesian Sentiment Analysis

This part of the assignment estimates **only** `P(Positive | keyword)` from the
Kaggle [IMDb Movie Reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
It uses only Python's standard library—no machine-learning packages.

## Run it

Download the Kaggle file named `IMDB Dataset.csv` and place it in the repository
root, then run:

```powershell
python bayesian_sentiment.py
```

To use a different location:

```powershell
python bayesian_sentiment.py "C:\path\to\IMDB Dataset.csv"
```

The CSV must have `review` and `sentiment` columns, with sentiment values
`positive` or `negative`. The script prints the prior, likelihood, marginal,
posterior, and the underlying review counts for each selected keyword. The
selected positive-sentiment keywords are `excellent`, `amazing`, and
`wonderful`; the negative-sentiment keywords are `terrible`, `awful`, and
`worst`. For consistency with the assignment, the conditional being reported
for every keyword (including the negative-indicating words) is
`P(Positive | keyword)`, not `P(Negative | keyword)`.

## Probability calculation

A keyword is present in a review if it occurs at least once as a whole word;
repeating it in the same review does not increase its count. Let `N` be the
number of reviews, `N_positive` the number labeled positive, `N_keyword` the
number containing the keyword, and `N_keyword_positive` the number that both
contain it and are labeled positive. For each keyword:

- Prior: `P(Positive) = N_positive / N`
- Likelihood: `P(keyword | Positive) = N_keyword_positive / N_positive`
- Marginal: `P(keyword) = N_keyword / N`
- Posterior (Bayes' theorem):
  `P(Positive | keyword) = P(keyword | Positive) * P(Positive) / P(keyword)`

The implementation computes the posterior from the equivalent count ratio
`N_keyword_positive / N_keyword`. Values are calculated from the actual input
dataset; no fabricated sample probabilities are included here.

## Results from the provided dataset

The following results were generated from the 50,000-review Kaggle CSV in this
repository. The dataset contains 25,000 positive reviews, so the prior is
`P(Positive) = 0.5`. Counts are review-level: a review contributes at most once
to a keyword, even if the word appears repeatedly.

| Keyword | Prior: P(Positive) | Likelihood: P(keyword\|Positive) | Marginal: P(keyword) | Posterior: P(Positive\|keyword) | Positive keyword reviews / all keyword reviews |
|---|---:|---:|---:|---:|---:|
| excellent | 0.500000 | 0.114720 | 0.071040 | 0.807432 | 2,868 / 3,552 |
| amazing | 0.500000 | 0.067160 | 0.043160 | 0.778035 | 1,679 / 2,158 |
| wonderful | 0.500000 | 0.090320 | 0.055600 | 0.812230 | 2,258 / 2,780 |
| terrible | 0.500000 | 0.015320 | 0.054020 | 0.141799 | 383 / 2,701 |
| awful | 0.500000 | 0.011360 | 0.057660 | 0.098508 | 284 / 2,883 |
| worst | 0.500000 | 0.016440 | 0.088680 | 0.092693 | 411 / 4,434 |

These estimates show that the selected positive words are associated with a
positive label in about 78–81% of matching reviews. For the selected negative
words, `P(Positive | keyword)` is about 9–14%. These are associations in this
dataset, not proof that a keyword alone determines a review's sentiment.

## Tests

Run the standard-library tests with:

```powershell
python -m unittest -v test_bayesian_sentiment.py
```
