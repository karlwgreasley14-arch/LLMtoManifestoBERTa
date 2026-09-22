#!/usr/bin/env python3
"""
Pipeline Health & Data Verification Checker.
Runs 30 minutes after scheduled collection runs (02:30 UTC and 13:30 UTC).
Audits SQLite database records in thesis_data.db for:
- Collection completeness (expected vs actual successful samples for current day)
- Missing or error states by model
- OpenRouter vs Local Ollama operational status
- Delivers a structured status digest to Telegram via the Tinknet Telegram Gateway.
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timezone, date
import requests

# Ensure src modules can be resolved if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "/app/src")

TELEGRAM_GATEWAY_URL = os.getenv("TELEGRAM_GATEWAY_URL", "http://atlas-telegram-gateway/api/ping")


def get_db_path():
    env_db = os.getenv("DB_PATH")
    if env_db and os.path.exists(env_db):
        return env_db
    candidates = [
        "/app/data/thesis_data.db",
        "/home/karl/tinknet/tinknet_data/llm_alignment_tracker/thesis_data.db",
        os.path.join(os.path.dirname(__file__), "..", "data", "thesis_data.db"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return "/app/data/thesis_data.db"


def calculate_current_day_number():
    """Calculates thesis day number starting from 2026-08-25."""
    start_date = date(2026, 8, 25)
    today = datetime.now(timezone.utc).date()
    return (today - start_date).days + 1


def verify_pipeline():
    db_path = get_db_path()
    if not os.path.exists(db_path):
        send_alert(
            level="critical",
            title="Database Missing",
            message=f"Pipeline verification failed: database file not found at {db_path}."
        )
        return False

    current_day = calculate_current_day_number()
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Check rows recorded for today (either matching day_number or timestamp_utc date)
        cursor.execute("""
            SELECT requested_model, status, COUNT(*) as count,
                   COALESCE(SUM(cost_usd), 0.0) as total_cost,
                   MAX(resolved_model) as sample_resolved
            FROM daily_llm_outputs
            WHERE day_number = ? OR date(timestamp_utc) = ?
            GROUP BY requested_model, status
            ORDER BY requested_model, status;
        """, (current_day, today_utc))
        rows = cursor.fetchall()

        # Overall counts
        cursor.execute("""
            SELECT 
                COUNT(*) as total_rows,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_rows,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_rows,
                COALESCE(SUM(cost_usd), 0.0) as day_cost
            FROM daily_llm_outputs
            WHERE day_number = ? OR date(timestamp_utc) = ?;
        """, (current_day, today_utc))
        summary = cursor.fetchone()

        total = summary["total_rows"] or 0
        successes = summary["success_rows"] or 0
        errors = summary["error_rows"] or 0
        cost = summary["day_cost"] or 0.0

        # Model breakdown dictionary
        models_data = {}
        for r in rows:
            m = r["requested_model"]
            if m not in models_data:
                models_data[m] = {"success": 0, "error": 0, "cost": 0.0, "version": None}
            models_data[m][r["status"]] += r["count"]
            models_data[m]["cost"] += r["total_cost"]
            if r["sample_resolved"]:
                models_data[m]["version"] = r["sample_resolved"]

        # Expected samples per day: 180 (6 models * 30 prompts)
        expected_total = 180
        is_healthy = (successes >= expected_total) and (errors == 0)

        lines = []
        if total == 0:
            level = "error"
            title = f"No Data Collected — Day {current_day}"
            lines.append(f"⚠️ Zero records were found in the database for today ({today_utc}, Day {current_day}).")
            lines.append("The scheduled collection job may have failed to execute.")
        elif errors > 0 or successes < expected_total:
            level = "warn"
            title = f"Data Collection Warning — Day {current_day}"
            lines.append(f"Pipeline collected <b>{successes}/{expected_total}</b> expected samples ({errors} errors).")
            lines.append(f"Today's compute cost: <code>${cost:.4f} USD</code>\n")
            lines.append("<b>Model Breakdown:</b>")
            for m_name, d in sorted(models_data.items()):
                status_icon = "✅" if d["success"] >= 30 and d["error"] == 0 else "❌"
                ver_str = f" ({d['version']})" if d['version'] else ""
                lines.append(f"{status_icon} <b>{m_name}</b>{ver_str}: {d['success']}/30 ok (errors: {d['error']})")
        else:
            level = "success"
            title = f"Daily Run Verified — Day {current_day}"
            lines.append(f"✅ All <b>{successes}/{expected_total}</b> prompt samples collected and scored successfully.")
            lines.append(f"Total cost: <code>${cost:.4f} USD</code>\n")
            lines.append("<b>Active Model Versions:</b>")
            for m_name, d in sorted(models_data.items()):
                ver_str = d['version'] or 'local'
                lines.append(f"• <b>{m_name}</b>: <code>{ver_str}</code>")

        message_body = "\n".join(lines)
        metadata = {
            "Day Number": current_day,
            "Date UTC": today_utc,
            "Success Rows": successes,
            "Error Rows": errors,
            "Day Cost USD": f"${cost:.4f}",
            "Database": db_path
        }

        send_alert(level=level, title=title, message=message_body, metadata=metadata)
        return is_healthy

    except Exception as e:
        send_alert(
            level="error",
            title="Verifier Exception",
            message=f"Error occurred while verifying database: {str(e)}"
        )
        return False
    finally:
        conn.close()


def send_alert(level, title, message, metadata=None):
    payload = {
        "source": "llm-alignment",
        "level": level,
        "title": title,
        "message": message,
        "metadata": metadata or {}
    }

    try:
        resp = requests.post(TELEGRAM_GATEWAY_URL, json=payload, timeout=10)
        if resp.status_code in (200, 201):
            print(f"Telegram notification delivered: {title}")
        else:
            print(f"Telegram gateway responded with HTTP {resp.status_code}: {resp.text}")
    except Exception as ex:
        print(f"Failed to deliver notification to Telegram gateway ({TELEGRAM_GATEWAY_URL}): {ex}")


if __name__ == "__main__":
    verify_pipeline()
