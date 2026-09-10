import base64
import os
import sys
import sqlite3
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Ensure src/ and project root are in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server
from config import init_db, get_db_connection, HARDCODED_MODELS
from collector import run_collection, scorer


class TestLLMAlignmentTracker(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for isolated database and file operations
        self.test_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.test_dir.name, "test_thesis_data.db")
        self.env_path = os.path.join(self.test_dir.name, ".env")
        self.prompts_path = os.path.join(self.test_dir.name, "prompts.json")

        # Write test .env file
        with open(self.env_path, "w", encoding="utf-8") as f:
            f.write("OPENROUTER_API_KEY=test_api_key\n")
            f.write("ADMIN_USERNAME=admin\n")
            f.write("ADMIN_PASSWORD=admin\n")
            f.write(f"DB_PATH={self.db_path}\n")

        # Write test prompts.json file (2 test prompts)
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

        # Configure Flask app for testing
        os.environ["DB_PATH"] = self.db_path
        os.environ["ADMIN_USERNAME"] = "admin"
        os.environ["ADMIN_PASSWORD"] = "admin"
        server.app.config["TESTING"] = True
        self.client = server.app.test_client()

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

            # Inspect columns
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

        # Define mock API side effect (succeed for most, simulate failure for grok)
        def mock_api_call(url, headers, json, timeout):
            model_slug = json.get("model", "")
            if "grok" in model_slug:
                # Simulate single model failure
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

        # Mock scorer and aggregate_graphs to isolate pipeline logic and protect production static JSON
        with patch("collector.scorer.score_response", return_value={"rile": 10.5, "evasion": 0.0}), \
             patch("analytics.aggregate_graphs") as mock_agg:
            # Run pipeline collection
            run_id, count = run_collection(
                db_path=self.db_path,
                prompts_path=self.prompts_path,
                env_path=self.env_path
            )
            mock_agg.assert_called_once_with(db_path=self.db_path)

        # Expected inserted count: 2 prompts * 5 models = 10 rows
        expected_total = len(HARDCODED_MODELS) * 2
        expected_success = (len(HARDCODED_MODELS) - 1) * 2
        expected_error = 2

        self.assertEqual(count, expected_total)

        # Verify DB insertions
        conn = get_db_connection(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS c FROM daily_llm_outputs WHERE run_id = ?", (run_id,))
            total_rows = cursor.fetchone()["c"]
            self.assertEqual(total_rows, expected_total)

            # Verify success rows
            cursor.execute("SELECT COUNT(*) AS c FROM daily_llm_outputs WHERE status = 'success'")
            success_count = cursor.fetchone()["c"]
            self.assertEqual(success_count, expected_success)

            # Verify error trapped rows
            cursor.execute("SELECT COUNT(*) AS c FROM daily_llm_outputs WHERE status = 'error'")
            error_count = cursor.fetchone()["c"]
            self.assertEqual(error_count, expected_error)

            # Check detailed record content
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

    def test_03_flask_basic_auth_unauthorized(self):
        """Verify Flask GET / returns 401 Unauthorized without valid credentials."""
        init_db(self.db_path)

        # 1. No auth header
        res_no_auth = self.client.get("/")
        self.assertEqual(res_no_auth.status_code, 401)
        self.assertIn("WWW-Authenticate", res_no_auth.headers)

        # 2. Invalid auth header
        invalid_auth = "Basic " + base64.b64encode(b"admin:wrongpass").decode("utf-8")
        res_bad_auth = self.client.get("/", headers={"Authorization": invalid_auth})
        self.assertEqual(res_bad_auth.status_code, 401)

    def test_04_flask_basic_auth_authorized_and_dashboard_render(self):
        """Verify Flask GET / returns 200 OK with valid HTTP Basic Auth credentials."""
        init_db(self.db_path)

        # Insert a sample row to test rendering with data
        conn = get_db_connection(self.db_path)
        try:
            with conn:
                conn.execute("""
                    INSERT INTO daily_llm_outputs (
                        run_id, timestamp_utc, day_number, requested_model, resolved_model,
                        prompt_id, marpor_domain, system_prompt, user_prompt, raw_response,
                        finish_reason, temperature, seed, prompt_tokens, completion_tokens,
                        cost_usd, status
                    ) VALUES (
                        'run_test_001', '2026-08-05T19:00:00Z', 1, 'gpt-4o', 'openai/gpt-4o',
                        'TEST_P1', '100 - Foreign Relations', 'sys', 'user', 'Sample LLM output',
                        'stop', 0.0, 42, 20, 50, 0.0025, 'success'
                    )
                """)
        finally:
            conn.close()

        valid_auth = "Basic " + base64.b64encode(b"admin:admin").decode("utf-8")
        response = self.client.get("/", headers={"Authorization": valid_auth})

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("LLM Political Alignment Tracker", html)
        self.assertIn("Total Observations", html)
        self.assertIn("Cumulative Cost (USD)", html)
        self.assertIn("gpt-4o", html)
        self.assertIn("SUCCESS", html)

    def test_05_flask_pagination(self):
        """Verify pagination query parameters work correctly."""
        init_db(self.db_path)
        conn = get_db_connection(self.db_path)
        try:
            with conn:
                for i in range(55):
                    conn.execute("""
                        INSERT INTO daily_llm_outputs (
                            run_id, timestamp_utc, day_number, requested_model, resolved_model,
                            prompt_id, marpor_domain, system_prompt, user_prompt, raw_response,
                            finish_reason, temperature, seed, prompt_tokens, completion_tokens,
                            cost_usd, status
                        ) VALUES (
                            ?, '2026-08-05T19:00:00Z', 1, 'gpt-4o', 'openai/gpt-4o',
                            'P1', '100', 'sys', 'user', 'res', 'stop', 0.0, 42, 10, 10, 0.001, 'success'
                        )
                    """, (f"run_page_{i}",))
        finally:
            conn.close()

        valid_auth = "Basic " + base64.b64encode(b"admin:admin").decode("utf-8")
        
        # Page 1
        res1 = self.client.get("/?page=1", headers={"Authorization": valid_auth})
        self.assertEqual(res1.status_code, 200)
        self.assertIn("Page 1 of 2", res1.get_data(as_text=True))

        # Page 2
        res2 = self.client.get("/?page=2", headers={"Authorization": valid_auth})
        self.assertEqual(res2.status_code, 200)
        self.assertIn("Page 2 of 2", res2.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
