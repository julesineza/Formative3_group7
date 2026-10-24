import csv
import tempfile
import unittest
from pathlib import Path

from bayesian_sentiment import analyze_reviews, format_results, read_reviews


class AnalyzeReviewsTests(unittest.TestCase):
    def test_csv_reader_normalizes_column_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "reviews.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=(" Review ", "SENTIMENT")
                )
                writer.writeheader()
                writer.writerow(
                    {" Review ": "A great film", "SENTIMENT": "positive"}
                )

            reviews = list(read_reviews(csv_path))

        self.assertEqual(
            reviews,
            [{"review": "A great film", "sentiment": "positive"}],
        )

    def test_csv_reader_reports_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "reviews.csv"
            csv_path.write_text("review\nA great film\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "sentiment"):
                list(read_reviews(csv_path))

    def test_bayes_probabilities_use_review_level_keyword_presence(self):
        reviews = [
            {"review": "Excellent, excellent movie", "sentiment": "positive"},
            {"review": "An excellent story", "sentiment": "positive"},
            {"review": "Awful movie", "sentiment": "negative"},
            {"review": "An excellent idea, but awful execution", "sentiment": "negative"},
        ]

        excellent, awful = analyze_reviews(reviews, ("excellent", "awful"))

        self.assertAlmostEqual(excellent.prior_positive, 0.5)
        self.assertAlmostEqual(
            excellent.likelihood_keyword_given_positive, 1.0
        )
        self.assertAlmostEqual(excellent.marginal_keyword, 0.75)
        self.assertAlmostEqual(
            excellent.posterior_positive_given_keyword, 2 / 3
        )
        self.assertAlmostEqual(
            excellent.posterior_positive_given_keyword,
            excellent.likelihood_keyword_given_positive
            * excellent.prior_positive
            / excellent.marginal_keyword,
        )
        self.assertAlmostEqual(awful.prior_positive, 0.5)
        self.assertAlmostEqual(awful.likelihood_keyword_given_positive, 0.0)
        self.assertAlmostEqual(awful.marginal_keyword, 0.5)
        self.assertAlmostEqual(awful.posterior_positive_given_keyword, 0.0)
        self.assertIn("Posterior: P(Positive|keyword)", format_results([excellent]))

    def test_keyword_matching_is_case_insensitive_and_uses_word_boundaries(self):
        results = analyze_reviews(
            [
                {"review": "AMAZING performance", "sentiment": "positive"},
                {"review": "Not amazinger", "sentiment": "negative"},
            ],
            ("amazing",),
        )

        self.assertEqual(results[0].keyword_review_count, 1)
        self.assertEqual(results[0].posterior_positive_given_keyword, 1.0)

    def test_empty_keyword_matches_are_reported(self):
        with self.assertRaisesRegex(ValueError, "does not occur"):
            analyze_reviews(
                [
                    {"review": "A good film", "sentiment": "positive"},
                    {"review": "A bad film", "sentiment": "negative"},
                ],
                ("unseen",),
            )

    def test_no_positive_reviews_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no positive reviews"):
            analyze_reviews(
                [
                    {"review": "Terrible movie", "sentiment": "negative"},
                    {"review": "Awful movie", "sentiment": "negative"},
                ],
                ("terrible",),
            )


if __name__ == "__main__":
    unittest.main()
