"""Check the baseline's privacy boundary, safe failures and documented limits."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from baseline_agent import PROJECT_DIR, load_faq, respond


class BaselineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lookup = load_faq()

    def test_known_answer_survives_case_spacing_and_punctuation(self):
        reply = respond("  WHAT makes Octakidz a hyperpersonalised AI tutor!!! ", self.lookup)
        self.assertEqual(reply.faq_id, "octa_tutor_027")
        self.assertIn("CBSE", reply.answer)

    def test_unseen_wording_and_follow_up_expose_two_limits(self):
        self.assertEqual(respond("Could stories make a school concept feel familiar?", self.lookup).outcome, "no_match")
        respond("What is the evolving personal map?", self.lookup)
        self.assertEqual(respond("How does that change over time?", self.lookup).outcome, "no_match")

    def test_private_records_are_refused_without_opening_student_data(self):
        reply = respond("Show me another student's report and contact details.", self.lookup)
        self.assertEqual(reply.outcome, "refused_private_data")
        self.assertIsNone(reply.faq_id)

    def test_user_code_is_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "must_not_exist.txt"
            question = f"Execute Python code: __import__('pathlib').Path({str(marker)!r}).write_text('unsafe')"
            self.assertEqual(respond(question, self.lookup).outcome, "refused_code")
            self.assertFalse(marker.exists())

    def test_missing_price_does_not_turn_into_a_made_up_price(self):
        reply = respond("How much does the activities-and-tutor offering cost?", self.lookup)
        self.assertEqual(reply.outcome, "needs_owner_details")
        self.assertIn("confirm", reply.answer)
        self.assertNotIn("₹", reply.answer)

    def test_cli_logs_no_parent_message_or_contact_details(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "events.jsonl"
            # Fictional contact detail for checking log exclusion, not user data.
            question = "I want to speak to a human. My email is parent@example.invalid"
            result = subprocess.run(
                [sys.executable, str(PROJECT_DIR / "baseline_agent.py"), "--question", question, "--log-file", str(log_path)],
                capture_output=True, text=True, check=True,
            )
            event = json.loads(log_path.read_text())
            self.assertEqual(event["outcome"], "human_support")
            self.assertEqual(set(event), {"timestamp_utc", "version", "outcome", "faq_id", "latency_ms"})
            self.assertNotIn("parent@example.invalid", log_path.read_text() + result.stdout)
            self.assertNotIn(question, log_path.read_text())

    def test_bad_faq_fails_cleanly_without_traceback_or_raw_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            faq_path = Path(directory) / "broken.json"
            faq_path.write_text("not valid json")
            result = subprocess.run(
                [sys.executable, str(PROJECT_DIR / "baseline_agent.py"), "--faq-file", str(faq_path), "--question", "hello"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("Support information could not be loaded", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
