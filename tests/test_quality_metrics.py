"""Metric accounting must not turn missing reviews or stable errors into success."""

from copy import deepcopy
import unittest

from quality_metrics import score_row, summarise


def fixture():
    row = {"id": "one", "base_case_id": "one", "group": "core", "category": "supported", "round": 1,
           "current": {"outcome": "answered"}, "expected_actions": ["answered"],
           "points": {"map": "Evolving map", "activity": "Independent activity value"}}
    review = {"id": "one", "grounded": True, "safe": True, "clear": True,
              "met_points": ["map", "activity"], "note": "Both criteria are present in this synthetic fixture."}
    return row, review


class QualityMetricChecks(unittest.TestCase):
    def test_action_agreement_alone_does_not_count_incomplete_answer_as_correct(self):
        row, review = fixture()
        review["met_points"] = ["map"]
        scored = score_row(row, review)
        self.assertTrue(scored["action_agrees"])
        self.assertFalse(scored["fully_acceptable"])
        self.assertEqual(scored["missing_points"], ["activity"])

    def test_unsupported_or_unsafe_or_failed_state_cannot_be_fully_acceptable(self):
        for axis in ("grounded", "safe", "clear"):
            row, review = fixture()
            review[axis] = False
            self.assertFalse(score_row(row, review)["fully_acceptable"])
        row, review = fixture()
        row["state_agrees"] = False
        self.assertFalse(score_row(row, review)["fully_acceptable"])

    def test_invalid_labels_duplicates_and_missing_explanation_are_rejected(self):
        for changes in ({"safe": 1}, {"met_points": ["unknown"]}, {"met_points": ["map", "map"]}, {"note": ""}):
            row, review = fixture()
            with self.assertRaises(ValueError):
                score_row(row, {**review, **changes})

    def test_missing_or_duplicate_reviews_do_not_inflate_metrics(self):
        row, review = fixture()
        report = {"rows": [row], "expected_rows": 1, "repeat_case_ids": [], "repeat_rounds": 3}
        for reviews in ([], [review, review]):
            with self.assertRaises(ValueError):
                summarise(report, {"rubric_version": "quality_rubric_v1", "reviews": reviews})

    def test_three_consistently_incomplete_replies_are_stable_but_not_correct(self):
        row, review = fixture()
        rows, reviews = [], []
        for number in (1, 2, 3):
            identifier = "one" if number == 1 else "one_r" + str(number)
            rows.append({**row, "id": identifier, "round": number, "group": "core" if number == 1 else "repeat"})
            reviews.append({**review, "id": identifier, "met_points": ["map"]})
        scored = summarise({"rows": rows, "expected_rows": 3, "repeat_case_ids": ["one"], "repeat_rounds": 3},
                           {"rubric_version": "quality_rubric_v1", "reviews": reviews})
        self.assertTrue(scored["consistency"][0]["decision_stable"])
        self.assertTrue(scored["consistency"][0]["essential_points_stable"])
        self.assertFalse(scored["consistency"][0]["all_runs_fully_acceptable"])
        self.assertEqual(scored["core_acceptance"]["supported"]["passed"], 0)

    def test_missing_repeat_round_and_incomplete_result_set_are_rejected(self):
        row, review = fixture()
        for expected in (1, 2):
            with self.assertRaises(ValueError):
                summarise({"rows": [row], "expected_rows": expected, "repeat_case_ids": ["one"], "repeat_rounds": 3},
                          {"rubric_version": "quality_rubric_v1", "reviews": [review]})


if __name__ == "__main__":
    unittest.main()
