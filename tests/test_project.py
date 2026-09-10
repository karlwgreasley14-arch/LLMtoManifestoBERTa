import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Ensure src/ and project root are in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import init_db, get_db_connection, HARDCODED_MODELS
from collector import run_collection, scorer


class TestLLMAlignmentTracker(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.test_dir.name, "test_thesis_data.db")
        self.env_path = os.path.join(self.test_dir.name, ".env")
        self.prompts_path = os.path.join(self.test_dir.name, "prompts.json")

        with open(self.env_path, "w", encoding="utf-8") as f:
            f.write("OPENROUTER_API_KEY=test_api_key\n")
            f.write(f"DB_PATH={self.db_path}\n")

        with open(self.prompts_path, "w", encoding="utf-8") as f:
            f.write("""[
              {
                "prompt_id": "TEST_P101",
                "marpor_domain": "100 - Foreign Relations",
                "text": "What is the optimal international security strategy?"
              },
              {
                "prompt_id": "TEST_P401",
                "marpor_domain": "400 - Economy",
                "text": "What are the effects of taxation on market dynamics?"
              }
            ]""")

        os.environ["DB_PATH"] = self.db_path

    def tearDown(self):
        self.test_dir.cleanup()

    def test_01_db_schema_initialization(self):
        """Verify config.py initializes thesis_data.db with daily_llm_outputs schema."""
        init_db(self.db_path)
        self.assertTrue(os.path.exists(self.db_path))

        conn = get_db_connection(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_llm_outputs'")
            table = cursor.fetchone()
            self.assertIsNotNone(table)

            cursor.execute("PRAGMA table_info(daily_llm_outputs)")
            columns = [col["name"] for col in cursor.fetchall()]
            expected_columns = [
                "id", "run_id", "timestamp_utc", "day_number", "requested_model",
                "resolved_model", "prompt_id", "marpor_domain", "system_prompt",
                "user_prompt", "raw_response", "finish_reason", "temperature",
                "seed", "prompt_tokens", "completion_tokens", "cost_usd", "status"
            ]
            for col in expected_columns:
                self.assertIn(col, columns)
        finally:
            conn.close()

    @patch("requests.post")
    def test_02_pipeline_execution_with_mock_and_error_trapping(self, mock_post):
        """Verify collector.py runs, calls OpenRouter mock API, handles errors, and inserts rows into DB."""

        def mock_api_call(url, headers, json, timeout):
            model_slug = json.get("model", "")
            if "grok" in model_slug:
                raise Exception("Simulated API Connection Timeout")

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {
                "id": "gen-mock-123",
                "model": f"{model_slug}-resolved",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": f"Mock analysis response for model {model_slug}."
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 70,
                    "cost": 0.00125
                }
            }
            return mock_resp

        mock_post.side_effect = mock_api_call

        with patch("collector.scorer.score_response", return_value={"rile": 10.5, "evasion": 0.0}), \
             patch("analytics.aggregate_graphs") as mock_agg:
            run_id, count = run_collection(
                db_path=self.db_path,
                prompts_path=self.prompts_path,
                env_path=self.env_path
            )
            mock_agg.assert_called_once_with(db_path=self.db_path)

        expected_total = len(HARDCODED_MODELS) * 2
        expected_success = (len(HARDCODED_MODELS) - 1) * 2
        expected_error = 2

        self.assertEqual(count, expected_total)

        conn = get_db_connection(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS c FROM daily_llm_outputs WHERE run_id = ?", (run_id,))
            self.assertEqual(cursor.fetchone()["c"], expected_total)

            cursor.execute("SELECT COUNT(*) AS c FROM daily_llm_outputs WHERE status = 'success'")
            self.assertEqual(cursor.fetchone()["c"], expected_success)

            cursor.execute("SELECT COUNT(*) AS c FROM daily_llm_outputs WHERE status = 'error'")
            self.assertEqual(cursor.fetchone()["c"], expected_error)

            cursor.execute(
                "SELECT requested_model, resolved_model, status, cost_usd FROM daily_llm_outputs WHERE requested_model = 'gpt-mini-latest' LIMIT 1"
            )
            row = cursor.fetchone()
            self.assertEqual(row["requested_model"], "gpt-mini-latest")
            self.assertIn("resolved", row["resolved_model"])
            self.assertEqual(row["status"], "success")
            self.assertAlmostEqual(row["cost_usd"], 0.00125)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
