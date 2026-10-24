"""Calculate P(Positive | keyword) from the Kaggle IMDb reviews CSV."""

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


POSITIVE_KEYWORDS = ("excellent", "amazing", "wonderful")
NEGATIVE_KEYWORDS = ("terrible", "awful", "worst")
KEYWORDS = POSITIVE_KEYWORDS + NEGATIVE_KEYWORDS


@dataclass(frozen=True)
class KeywordResult:
    keyword: str
    prior_positive: float
    likelihood_keyword_given_positive: float
    marginal_keyword: float
    posterior_positive_given_keyword: float
    positive_review_count: int
    keyword_positive_review_count: int
    keyword_review_count: int
    total_review_count: int


def analyze_reviews(
    reviews: Iterable[Mapping[str, str]], keywords: Sequence[str] = KEYWORDS
) -> list[KeywordResult]:
    """Compute review-level keyword probabilities from labeled review records."""
    patterns = {
        keyword: re.compile(r"\b" + re.escape(keyword.casefold()) + r"\b")
        for keyword in keywords
    }
    positive_count = 0
    total_count = 0
    keyword_counts = {keyword: 0 for keyword in keywords}
    positive_keyword_counts = {keyword: 0 for keyword in keywords}

    for row_number, row in enumerate(reviews, start=1):
        if "review" not in row or "sentiment" not in row:
            raise ValueError(
                f"Review record {row_number} must contain 'review' and 'sentiment'."
            )

        review = row["review"]
        sentiment = row["sentiment"].strip().casefold()
        if sentiment not in ("positive", "negative"):
            raise ValueError(
                f"Review record {row_number} has unsupported sentiment "
                f"{row['sentiment']!r}; expected 'positive' or 'negative'."
            )

        total_count += 1
        is_positive = sentiment == "positive"
        if is_positive:
            positive_count += 1

        normalized_review = review.casefold()
        for keyword, pattern in patterns.items():
            if pattern.search(normalized_review):
                keyword_counts[keyword] += 1
                if is_positive:
                    positive_keyword_counts[keyword] += 1

    if total_count == 0:
        raise ValueError("The CSV contains no review records.")
    if positive_count == 0:
        raise ValueError("The dataset contains no positive reviews.")

    prior_positive = positive_count / total_count
    results = []
    for keyword in keywords:
        keyword_count = keyword_counts[keyword]
        if keyword_count == 0:
            raise ValueError(
                f"Keyword {keyword!r} does not occur in any review; "
                "its posterior cannot be estimated from this dataset."
            )

        keyword_positive_count = positive_keyword_counts[keyword]
        results.append(
            KeywordResult(
                keyword=keyword,
                prior_positive=prior_positive,
                likelihood_keyword_given_positive=(
                    keyword_positive_count / positive_count
                ),
                marginal_keyword=keyword_count / total_count,
                posterior_positive_given_keyword=(
                    keyword_positive_count / keyword_count
                ),
                positive_review_count=positive_count,
                keyword_positive_review_count=keyword_positive_count,
                keyword_review_count=keyword_count,
                total_review_count=total_count,
            )
        )
    return results


def read_reviews(csv_path: Path) -> list[dict[str, str]]:
    """Read the standard Kaggle IMDb CSV, requiring its review and sentiment columns."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is empty or has no CSV header.")

        columns = {name.strip().casefold(): name for name in reader.fieldnames}
        missing = {"review", "sentiment"} - columns.keys()
        if missing:
            raise ValueError(
                f"{csv_path} is missing required column(s): "
                + ", ".join(sorted(missing))
            )
        return [
            {
                "review": row[columns["review"]] or "",
                "sentiment": row[columns["sentiment"]] or "",
            }
            for row in reader
        ]


def format_results(results: Sequence[KeywordResult]) -> str:
    """Format the Bayes calculations as a presentation-ready Markdown table."""
    lines = [
        "| Keyword | Prior: P(Positive) | "
        "Likelihood: P(keyword|Positive) | Marginal: P(keyword) | "
        "Posterior: P(Positive|keyword) |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.keyword} "
            f"| {result.prior_positive:.6f} "
            f"| {result.likelihood_keyword_given_positive:.6f} "
            f"| {result.marginal_keyword:.6f} "
            f"| {result.posterior_positive_given_keyword:.6f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate P(Positive | keyword) from the IMDb reviews CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path("IMDB Dataset.csv"),
        help="path to the Kaggle IMDb CSV (default: IMDB Dataset.csv)",
    )
    args = parser.parse_args()

    try:
        results = analyze_reviews(read_reviews(args.csv_path))
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print("Chosen conditional probability: P(Positive | keyword)")
    print("Keyword is counted once per review, regardless of how often it appears.")
    print(format_results(results))
    print("\nCounts used:")
    for result in results:
        print(
            f"{result.keyword}: {result.keyword_positive_review_count} positive "
            f"matches / {result.keyword_review_count} total matches; "
            f"{result.positive_review_count} positive / "
            f"{result.total_review_count} reviews overall"
        )


if __name__ == "__main__":
    main()
